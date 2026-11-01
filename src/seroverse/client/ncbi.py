"""NCBI Entrez E-Utilities and NCBI Datasets v2 REST API client, query builders, and models."""

from __future__ import annotations

import io
import json
import os
import re
import threading
import urllib.parse
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Self

from seroverse.client.base import Client
from seroverse.client.progress import current_progress

NCBI_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2"

_BAD_METADATA_VALUES: frozenset[str] = frozenset(
    {
        "",
        "-",
        "?",
        "na",
        "n/a",
        "nd",
        "nt",
        "none",
        "missing",
        "unknown",
        "not known",
        "not available",
        "not available: to be reported later",
        "not applicable",
        "not collected",
        "not determined",
        "not provided",
        "restricted access",
        "untyped",
        "untypeable",
        "non-typeable",
        "nontypeable",
    }
)

_BIOSAMPLE_STRAIN_KEYS: tuple[str, ...] = (
    "strain",
    "culture_collection",
    "type-material",
    "bio_material",
    "isolate_name_alias",
    "isolate-name-alias-2",
    "strain_synonym",
    "other_strain_names",
    "alternate-name-alias",
    "alternate strain name",
    "alias",
    "insdc center alias",
    "isolate",
)

_BIOSAMPLE_SEROTYPE_KEYS: tuple[str, ...] = (
    "serotype",
    "serovar",
    "serogroup",
    "antigenic_formula",
    "o_type",
    "o-type",
    "k_type",
    "k-type",
    "capsule_type",
    "kl_type",
    "ocl_type",
    "subtype",
    "subgroup",
    "genotype",
    "strain_note",
    "phenotype",
)

_CULTURE_COLLECTION_RE = re.compile(
    r"\b(?:ATCC|NCTC|DSM|CIP|CCUG|LMG|JCM|NBRC|CNCTC|PCM|CECT|SSI|SSISP|NIPH|IAM|IFO|NCIMB|CBS|BCRC)\b",
    re.IGNORECASE,
)

_ASSEMBLY_LEVEL_RANK: dict[str, int] = {
    "complete genome": 4,
    "chromosome": 3,
    "scaffold": 2,
    "contig": 1,
}

_GB_ACC_VERSION_RE = re.compile(r"^VERSION\s+(\S+)", re.MULTILINE)
_GB_ACCESSION_RE = re.compile(r"^ACCESSION\s+(\S+)", re.MULTILINE)
_GB_DEFINITION_RE = re.compile(r"^DEFINITION\s+(.+?)(?=\n[A-Z]+\s+|\Z)", re.MULTILINE | re.DOTALL)
_GB_ORGANISM_RE = re.compile(r"^\s+ORGANISM\s+([^\n]+)", re.MULTILINE)
_GB_PUBMED_RE = re.compile(r"^\s+PUBMED\s+(\d+)", re.MULTILINE)
_GB_TAXID_RE = re.compile(r'/db_xref="taxon:(\d+)"')
_GB_QUALIFIER_RE = re.compile(r'/([a-zA-Z0-9_]+)="([^"]*)"', re.DOTALL)


def _clean_meta_value(val: Any) -> str:
    """Return a cleaned metadata string, or `""` if it is a placeholder/missing value."""
    if val is None or isinstance(val, bool):
        return ""
    s = " ".join(str(val).strip().split())
    if not s or s.lower() in _BAD_METADATA_VALUES or s.lower().startswith("not available"):
        return ""
    return s


@dataclass(frozen=True, slots=True)
class NuccoreQuery:
    """Composable object-oriented query builder for NCBI Entrez `nuccore` locus discovery.

    Attributes:
        taxon: Scientific organism name (e.g. `"Klebsiella pneumoniae"`) or integer NCBI `tax_id`.
        title_terms: Keywords matched in `[Title]` (combined with `OR`).
        text_terms: Additional keywords matched in `[All Fields]` (combined with `OR`).
        slen_range: `(min_bp, max_bp)` sequence length filter (`min:max[SLEN]`).
        exclude_title_terms: Terms excluded from `[Title]` (`NOT (...)`) so whole-genome contigs,
            plasmids, phages, and rRNA entries do not saturate `retmax`.
        retmax: Maximum number of Entrez UIDs to retrieve for this query.
    """

    taxon: str | int
    title_terms: tuple[str, ...] = ()
    text_terms: tuple[str, ...] = ()
    slen_range: tuple[int, int] | None = (2500, 120000)
    exclude_title_terms: tuple[str, ...] = (
        "shotgun",
        "whole genome",
        "complete genome",
        "chromosome",
        "plasmid",
        "bacteriophage",
        "phage",
        "16S",
        "rRNA",
        "tRNA",
    )
    retmax: int = 500

    def with_title_terms(self, *terms: str) -> Self:
        """Return a copy of this query with updated `title_terms`."""
        return replace(self, title_terms=tuple(t.strip() for t in terms if t.strip()))

    def with_text_terms(self, *terms: str) -> Self:
        """Return a copy of this query with updated `text_terms`."""
        return replace(self, text_terms=tuple(t.strip() for t in terms if t.strip()))

    def with_slen_range(self, min_bp: int, max_bp: int) -> Self:
        """Return a copy of this query with updated `(min_bp, max_bp)` sequence length bounds."""
        return replace(self, slen_range=(int(min_bp), int(max_bp)))

    def to_term(self) -> str:
        """Compile this specification into a valid NCBI Entrez Boolean query string."""
        clauses: list[str] = []
        if isinstance(self.taxon, int) or (isinstance(self.taxon, str) and self.taxon.strip().isdigit()):
            clauses.append(f"txid{int(self.taxon)}[Organism:exp]")
        else:
            clean_tax = str(self.taxon).strip().strip('"')
            if clean_tax:
                clauses.append(f'"{clean_tax}"[Organism]')

        if self.title_terms:
            t_parts = [f'"{t}"[Title]' if " " in t or "-" in t else f"{t}[Title]" for t in self.title_terms if t]
            if len(t_parts) == 1:
                clauses.append(t_parts[0])
            elif t_parts:
                clauses.append("(" + " OR ".join(t_parts) + ")")

        if self.text_terms:
            x_parts = [
                f'"{t}"[All Fields]' if " " in t or "-" in t else f"{t}[All Fields]" for t in self.text_terms if t
            ]
            if len(x_parts) == 1:
                clauses.append(x_parts[0])
            elif x_parts:
                clauses.append("(" + " OR ".join(x_parts) + ")")

        if self.slen_range is not None:
            lo, hi = int(self.slen_range[0]), int(self.slen_range[1])
            clauses.append(f"{lo}:{hi}[SLEN]")

        if self.exclude_title_terms:
            ex_parts = [
                f'"{t}"[Title]' if " " in t or "-" in t else f"{t}[Title]" for t in self.exclude_title_terms if t
            ]
            if ex_parts:
                clauses.append("NOT (" + " OR ".join(ex_parts) + ")")

        return " AND ".join(clauses)


@dataclass(frozen=True, slots=True)
class NuccoreSummary:
    """Immutable NCBI `nuccore` `esummary` record."""

    uid: str
    accession_version: str
    title: str = ""
    subname: str = ""
    subtype: str = ""
    slen: int = 0

    @property
    def base_accession(self) -> str:
        return self.accession_version.split(".")[0] if self.accession_version else ""

    @property
    def blob_text(self) -> str:
        return f"{self.title} | {self.subname}"

    @property
    def sub_pairs(self) -> list[tuple[str, str]]:
        return list(zip(self.subtype.split("|"), self.subname.split("|"), strict=False))


@dataclass(frozen=True, slots=True)
class GenBankRecord:
    """Parsed GenBank flat-file record with `/source` qualifiers, PubMed IDs, and nucleotide sequence."""

    accession_version: str
    definition: str = ""
    organism: str = ""
    tax_id: int = 0
    strain: str = ""
    isolate: str = ""
    culture_collections: tuple[str, ...] = ()
    serotype: str = ""
    serovar: str = ""
    serogroup: str = ""
    notes: tuple[str, ...] = ()
    pmids: tuple[int, ...] = ()
    sequence: bytes = b""

    @property
    def base_accession(self) -> str:
        """Uppercase accession stem without `.version` suffix."""
        return self.accession_version.split(".")[0].upper() if self.accession_version else ""

    @property
    def strain_candidates(self) -> tuple[str, ...]:
        """Ordered tuple of non-empty strain, culture-collection, and isolate strings."""
        raw = [self.strain, *self.culture_collections, self.isolate]
        return tuple(dict.fromkeys(c for x in raw if (c := _clean_meta_value(x))))

    @property
    def serotype_candidates(self) -> tuple[str, ...]:
        """Ordered tuple of non-empty serotype, serovar, serogroup, note, and definition strings."""
        raw = [self.serotype, self.serovar, self.serogroup, *self.notes, self.definition]
        return tuple(dict.fromkeys(c for x in raw if (c := _clean_meta_value(x))))


def parse_genbank_flatfile(raw: bytes | str) -> list[GenBankRecord]:
    """Parse one or more concatenated GenBank flat-file records (`gb` or `gbwithparts`)."""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    records: list[GenBankRecord] = []
    for chunk in text.split("\n//"):
        entry = chunk.strip()
        if not entry or not entry.startswith("LOCUS"):
            continue
        m_ver = _GB_ACC_VERSION_RE.search(entry)
        m_acc = _GB_ACCESSION_RE.search(entry)
        acc_ver = m_ver.group(1).strip() if m_ver else (m_acc.group(1).strip() if m_acc else "")
        if not acc_ver:
            continue

        m_def = _GB_DEFINITION_RE.search(entry)
        definition = " ".join(m_def.group(1).split()) if m_def else ""
        m_org = _GB_ORGANISM_RE.search(entry)
        organism = m_org.group(1).strip() if m_org else ""
        m_tax = _GB_TAXID_RE.search(entry)
        tax_id = int(m_tax.group(1)) if m_tax else 0
        pmids = tuple(dict.fromkeys(int(m.group(1)) for m in _GB_PUBMED_RE.finditer(entry) if int(m.group(1)) > 0))

        # Parse /source feature block (up to the next feature or ORIGIN)
        feat_idx = entry.find("\nFEATURES")
        orig_idx = entry.find("\nORIGIN")
        feat_section = entry[feat_idx : orig_idx if orig_idx != -1 else len(entry)] if feat_idx != -1 else entry
        # Extract the source feature slice before the first CDS/gene/tRNA/repeat_region feature
        m_next_feat = re.search(r"\n     (?!source\b)[a-zA-Z_0-9]+\s+", feat_section)
        source_block = feat_section[: m_next_feat.start()] if m_next_feat else feat_section

        strain = ""
        isolate = ""
        cc_list: list[str] = []
        serotype = ""
        serovar = ""
        serogroup = ""
        notes: list[str] = []
        for q_key, q_val in _GB_QUALIFIER_RE.findall(source_block):
            k_low = q_key.lower()
            v_clean = _clean_meta_value(q_val)
            if not v_clean:
                continue
            if k_low == "strain" and not strain:
                strain = v_clean
            elif k_low == "isolate" and not isolate:
                isolate = v_clean
            elif k_low == "culture_collection":
                cc_norm = v_clean.replace(":", " ").strip()
                if cc_norm and cc_norm not in cc_list:
                    cc_list.append(cc_norm)
            elif k_low == "serotype" and not serotype:
                serotype = v_clean
            elif k_low == "serovar" and not serovar:
                serovar = v_clean
            elif k_low == "serogroup" and not serogroup:
                serogroup = v_clean
            elif k_low == "note" and v_clean not in notes:
                notes.append(v_clean)

        seq_bytes = b""
        if orig_idx != -1:
            origin_body = entry[orig_idx + 7 :]
            seq_chars = re.sub(r"[^a-zA-Z]", "", origin_body).upper()
            seq_bytes = seq_chars.encode("ascii")

        records.append(
            GenBankRecord(
                accession_version=acc_ver,
                definition=definition,
                organism=organism,
                tax_id=tax_id,
                strain=strain,
                isolate=isolate,
                culture_collections=tuple(cc_list),
                serotype=serotype,
                serovar=serovar,
                serogroup=serogroup,
                notes=tuple(notes),
                pmids=pmids,
                sequence=seq_bytes,
            )
        )
    return records


@dataclass(frozen=True, slots=True)
class PubMedArticleId:
    """Article identifier entry inside a PubMed `esummary` record."""

    idtype: str = ""
    value: str = ""


@dataclass(frozen=True, slots=True)
class PubMedSummary:
    """Immutable PubMed `esummary` record."""

    uid: str
    title: str = ""
    pubdate: str = ""
    doi: str = ""
    articleids: tuple[PubMedArticleId, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetAssemblyReport:
    """Immutable NCBI Datasets v2 genome assembly report summary."""

    accession: str
    organism_name: str = ""
    tax_id: int = 0
    strain: str = ""
    serovar: str = ""
    isolate: str = ""
    biosample_accession: str = ""
    assembly_level: str = ""
    paired_accession: str = ""


@dataclass(frozen=True, slots=True)
class AssemblySummaryRecord:
    """Structured NCBI Datasets v2 genome assembly record parsed from `ncbi_summary.jsonl`.

    Extracts all strain, culture-collection, alias, and serotype/antigen candidates across
    `organism.infraspecific_names`, `organism.organism_name`, and `assembly_info.biosample`.
    """

    accession: str
    paired_accession: str = ""
    organism_name: str = ""
    tax_id: int = 0
    assembly_level: str = ""
    refseq_category: str = ""
    contig_n50: int = 0
    num_contigs: int = 0
    total_length: int = 0
    biosample_accession: str = ""
    bioproject_accession: str = ""
    type_material: str = ""
    strain_candidates: tuple[str, ...] = ()
    serotype_candidates: tuple[str, ...] = ()

    @property
    def base_accession(self) -> str:
        """Accession stem without version suffix (e.g. `'GCF_000005845'`)."""
        return self.accession.split(".")[0].upper() if self.accession else ""

    @property
    def paired_base_accession(self) -> str:
        """Paired GenBank/RefSeq accession stem without version suffix."""
        return self.paired_accession.split(".")[0].upper() if self.paired_accession else ""

    @property
    def primary_strain(self) -> str:
        """Primary strain or culture-collection designation if present."""
        return self.strain_candidates[0] if self.strain_candidates else ""

    @property
    def has_culture_collection(self) -> bool:
        """Return `True` if any strain candidate contains a recognized culture-collection code."""
        return bool(self.type_material) or any(bool(_CULTURE_COLLECTION_RE.search(s)) for s in self.strain_candidates)

    @property
    def is_complete_level(self) -> bool:
        """Return `True` if `assembly_level` is `'Complete Genome'` or `'Chromosome'`."""
        return self.assembly_level.lower() in ("complete genome", "chromosome")

    @property
    def authority_tier(self) -> int:
        """Integer authority score (`0..4`) prioritizing reference/type/culture-collection strains."""
        score = 0
        if self.refseq_category:
            score += 2
        if self.type_material:
            score += 1
        if self.has_culture_collection:
            score += 1
        return score

    @property
    def quality_key(self) -> tuple[int, int, int, int]:
        """Composite ranking key `(authority_tier, assembly_level_rank, contig_n50, -num_contigs)`."""
        lvl_rank = _ASSEMBLY_LEVEL_RANK.get(self.assembly_level.lower(), 0)
        return (self.authority_tier, lvl_rank, self.contig_n50, -self.num_contigs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AssemblySummaryRecord | None:
        """Parse a single JSON object from `ncbi_summary.jsonl` into an `AssemblySummaryRecord`."""
        acc = str(data.get("accession") or "").strip()
        if not acc:
            return None
        paired = str(data.get("paired_accession") or "").strip()
        org = data.get("organism") or {}
        if not isinstance(org, dict):
            org = {}
        org_name = _clean_meta_value(org.get("organism_name"))
        tax_id = int(org.get("tax_id") or 0) if str(org.get("tax_id") or "").isdigit() else 0
        infra = org.get("infraspecific_names") or {}
        if not isinstance(infra, dict):
            infra = {}

        ainfo = data.get("assembly_info") or {}
        if not isinstance(ainfo, dict):
            ainfo = {}
        level = _clean_meta_value(ainfo.get("assembly_level"))
        refseq_cat = _clean_meta_value(ainfo.get("refseq_category"))
        bioproj = _clean_meta_value(ainfo.get("bioproject_accession"))

        astats = data.get("assembly_stats") or {}
        if not isinstance(astats, dict):
            astats = {}
        contig_n50 = int(astats.get("contig_n50") or 0)
        num_contigs = int(astats.get("number_of_contigs") or 0)
        total_len = int(astats.get("total_sequence_length") or 0)

        bs = ainfo.get("biosample") or {}
        if not isinstance(bs, dict):
            bs = {}
        bs_acc = _clean_meta_value(bs.get("accession"))
        attrs_raw = bs.get("attributes") or ()
        attrs: dict[str, list[str]] = {}
        if isinstance(attrs_raw, list):
            for item in attrs_raw:
                if isinstance(item, dict):
                    k = str(item.get("name") or "").strip().lower()
                    v = _clean_meta_value(item.get("value"))
                    if k and v:
                        attrs.setdefault(k, []).append(v)

        type_mat = ""
        for tk in ("type-material", "type_material", "type_status", "type_strain"):
            if tk in attrs:
                type_mat = attrs[tk][0]
                break

        # Collect strain / culture-collection / alias candidates in priority order
        strain_list: list[str] = []

        def _add_strain(raw_val: str) -> None:
            v = _clean_meta_value(raw_val)
            if not v:
                return
            # Normalize colon-separated culture collection codes like "ATCC:13883" -> "ATCC 13883"
            if ":" in v and _CULTURE_COLLECTION_RE.match(v.split(":", 1)[0]):
                v = " ".join(p.strip() for p in v.split(":") if p.strip())
            if v and v not in strain_list:
                strain_list.append(v)

        _add_strain(str(infra.get("strain") or ""))
        _add_strain(str(bs.get("strain") or ""))
        for sk in _BIOSAMPLE_STRAIN_KEYS:
            for sv in attrs.get(sk, ()):
                _add_strain(sv)
        _add_strain(str(infra.get("isolate") or ""))
        _add_strain(str(bs.get("isolate") or ""))

        # Collect serotype / serovar / antigen candidates in priority order
        sero_list: list[str] = []

        def _add_sero(raw_val: str) -> None:
            v = _clean_meta_value(raw_val)
            if v and v not in sero_list:
                sero_list.append(v)

        for k in ("serovar", "serotype"):
            _add_sero(str(infra.get(k) or ""))
            _add_sero(str(bs.get(k) or ""))
        for sk in _BIOSAMPLE_SEROTYPE_KEYS:
            for sv in attrs.get(sk, ()):
                _add_sero(sv)

        # Also check organism_name suffix beyond binomial species name (e.g. "Actinobacillus pleuropneumoniae serovar 1")
        org_parts = org_name.split()
        if len(org_parts) > 2:
            suffix = " ".join(org_parts[2:])
            _add_sero(suffix)

        return cls(
            accession=acc,
            paired_accession=paired,
            organism_name=org_name,
            tax_id=tax_id,
            assembly_level=level,
            refseq_category=refseq_cat,
            contig_n50=contig_n50,
            num_contigs=num_contigs,
            total_length=total_len,
            biosample_accession=bs_acc,
            bioproject_accession=bioproj,
            type_material=type_mat,
            strain_candidates=tuple(strain_list),
            serotype_candidates=tuple(sero_list),
        )


@dataclass(frozen=True, slots=True)
class AssemblySummaryReader:
    """Streaming reader over an NCBI Datasets v2 `ncbi_summary.jsonl` file."""

    path: Path
    allowed_taxids: frozenset[int] | None = None
    allowed_binomials: frozenset[str] | None = None

    def __init__(
        self,
        path: str | Path,
        *,
        allowed_taxids: Iterable[int] | None = None,
        allowed_binomials: Iterable[str] | None = None,
    ) -> None:
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(
            self,
            "allowed_taxids",
            frozenset(int(t) for t in allowed_taxids if int(t) > 0) if allowed_taxids is not None else None,
        )
        object.__setattr__(
            self,
            "allowed_binomials",
            frozenset(" ".join(s.lower().split()[:2]) for s in allowed_binomials if s.strip())
            if allowed_binomials is not None
            else None,
        )

    def __iter__(self) -> Iterator[AssemblySummaryRecord]:
        """Stream matching `AssemblySummaryRecord` instances from `self.path`."""
        if not self.path.is_file():
            return
        taxids = self.allowed_taxids
        binomials = self.allowed_binomials
        with self.path.open("rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except Exception:
                    continue
                if not isinstance(raw, dict):
                    continue
                if taxids is not None or binomials is not None:
                    org = raw.get("organism") or {}
                    tid = int(org.get("tax_id") or 0) if isinstance(org, dict) else 0
                    matched_tax = taxids is not None and tid in taxids
                    if not matched_tax and binomials is not None and isinstance(org, dict):
                        oname = str(org.get("organism_name") or "").lower().split()
                        matched_tax = len(oname) >= 2 and f"{oname[0]} {oname[1]}" in binomials
                    if not matched_tax:
                        continue
                rec = AssemblySummaryRecord.from_dict(raw)
                if rec is not None:
                    yield rec


class NCBIClient(Client):
    """Client for NCBI Entrez E-Utilities and NCBI Datasets v2."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 40.0,
        max_concurrency: int | None = None,
        polite_delay: float = 0.20,
        recorder: Any | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("NCBI_API_KEY", "")
        conc = max_concurrency if max_concurrency is not None else (5 if resolved_key else 3)
        super().__init__(
            base_url=NCBI_EUTILS_URL,
            api_key_header="api-key" if resolved_key else None,
            api_key=resolved_key or None,
            user_agent="seroverse-ncbi-client/0.1",
            timeout=timeout,
            max_concurrency=conc,
            polite_delay=polite_delay,
            recorder=recorder,
        )

    def _eutils_params(self, extra: dict[str, str]) -> dict[str, str]:
        base = {"retmode": "json"}
        if self.api_key:
            base["api_key"] = self.api_key
        base.update(extra)
        return base

    def esearch(self, db: str, term: str, retmax: int = 100) -> list[str]:
        """Run rate-limited `esearch.fcgi` and return matching UID strings."""
        params = self._eutils_params({"db": db, "term": term, "retmax": str(retmax)})
        payload = self.get_data("esearch.fcgi", params=params, allow_404=True)
        if (
            isinstance(payload, dict)
            and isinstance(res := payload.get("esearchresult"), dict)
            and isinstance(idlist := res.get("idlist"), list)
        ):
            return [str(u) for u in idlist]
        return []

    def search_nuccore(
        self,
        queries: NuccoreQuery | Sequence[NuccoreQuery],
        *,
        chunk_size: int = 200,
    ) -> dict[str, NuccoreSummary]:
        """Execute one or more `NuccoreQuery` specifications and return `{uid: NuccoreSummary}`."""
        q_list = [queries] if isinstance(queries, NuccoreQuery) else list(queries)
        uids: list[str] = []
        seen: set[str] = set()
        min_lo: int | None = None
        max_hi: int | None = None
        for q in q_list:
            if q.slen_range is not None:
                min_lo = q.slen_range[0] if min_lo is None else min(min_lo, q.slen_range[0])
                max_hi = q.slen_range[1] if max_hi is None else max(max_hi, q.slen_range[1])
            for u in self.esearch("nuccore", q.to_term(), retmax=q.retmax):
                if u not in seen:
                    seen.add(u)
                    uids.append(u)
        if not uids:
            return {}
        slen_bounds = (min_lo, max_hi) if (min_lo is not None and max_hi is not None) else None
        return self.fetch_nuccore_summaries(uids, slen_range=slen_bounds, chunk_size=chunk_size)

    def esummary(self, db: str, uids: Sequence[str], chunk_size: int = 200) -> dict[str, dict[str, Any]]:
        """Run batched `esummary.fcgi` POST requests and return `{uid: info_dict}`."""
        out: dict[str, dict[str, Any]] = {}
        chunks = [uids[i : i + chunk_size] for i in range(0, len(uids), chunk_size)]
        if not chunks:
            return out
        prog = current_progress()
        completed = 0
        total = len(chunks)
        lock = threading.Lock()

        def _fetch_chunk(chunk: Sequence[str]) -> None:
            nonlocal completed
            try:
                data = self._eutils_params({"db": db, "id": ",".join(chunk)})
                payload = self.post_data("esummary.fcgi", data=data, allow_404=True)
                if isinstance(payload, dict) and isinstance(res := payload.get("result"), dict):
                    with lock:
                        for u in chunk:
                            if isinstance(info := res.get(u), dict):
                                out[u] = info
            finally:
                with lock:
                    completed += 1
                    done = completed
                if prog.is_active:
                    prog.report(done / total, detail=f"NCBI esummary ({db}): batch {done}/{total}")

        self.map_concurrent(_fetch_chunk, chunks)
        return out

    def fetch_nuccore_summaries(
        self,
        uids: Sequence[str],
        *,
        slen_range: tuple[int, int] | None = None,
        chunk_size: int = 200,
    ) -> dict[str, NuccoreSummary]:
        """Fetch `NuccoreSummary` records for `uids`, optionally filtering by sequence length `slen_range`."""
        raw_map = self.esummary("nuccore", uids, chunk_size=chunk_size)
        summaries: dict[str, NuccoreSummary] = {}
        for u, info in raw_map.items():
            slen = int(info.get("slen") or 0)
            if slen_range is not None and not (slen_range[0] <= slen <= slen_range[1]):
                continue
            acc_ver = str(info.get("accessionversion") or info.get("caption") or "").strip()
            if acc_ver:
                summaries[u] = NuccoreSummary(
                    uid=u,
                    accession_version=acc_ver,
                    title=str(info.get("title") or ""),
                    subname=str(info.get("subname") or ""),
                    subtype=str(info.get("subtype") or ""),
                    slen=slen,
                )
        return summaries

    def fetch_genbank_records(
        self,
        uids: Sequence[str],
        *,
        rettype: str = "gb",
        batch_size: int = 40,
    ) -> list[GenBankRecord]:
        """Download and parse full `GenBankRecord` objects (qualifiers, PMIDs, sequences) in batches."""
        records: list[GenBankRecord] = []
        for i in range(0, len(uids), batch_size):
            batch = uids[i : i + batch_size]
            raw_gb = self.efetch_genbank(batch, db="nuccore", rettype=rettype)
            if raw_gb:
                records.extend(parse_genbank_flatfile(raw_gb))
        return records

    def fetch_pubmed_summaries(self, pmids: Sequence[int | str], chunk_size: int = 150) -> dict[int, PubMedSummary]:
        """Fetch `PubMedSummary` records (including DOIs) for `pmids`."""
        uid_strs = [str(p) for p in pmids]
        raw_map = self.esummary("pubmed", uid_strs, chunk_size=chunk_size)
        out: dict[int, PubMedSummary] = {}
        for u_s, info in raw_map.items():
            if not u_s.isdigit():
                continue
            aids: list[PubMedArticleId] = []
            doi = ""
            for aid in info.get("articleids") or ():
                if isinstance(aid, dict):
                    idt = str(aid.get("idtype") or "")
                    val = str(aid.get("value") or "").strip()
                    aids.append(PubMedArticleId(idtype=idt, value=val))
                    if idt == "doi" and val and not doi:
                        doi = val
            out[int(u_s)] = PubMedSummary(
                uid=u_s,
                title=str(info.get("title") or ""),
                pubdate=str(info.get("pubdate") or ""),
                doi=doi,
                articleids=tuple(aids),
            )
        return out

    def elink_pubmed_to_nuccore(self, pmids: Sequence[int], batch_size: int = 40) -> dict[int, list[str]]:
        """Map PubMed IDs to linked `nuccore` UIDs via `elink.fcgi` (`pubmed_nuccore`)."""
        out: dict[int, list[str]] = {}
        batches = [pmids[i : i + batch_size] for i in range(0, len(pmids), batch_size)]
        if not batches:
            return out
        prog = current_progress()
        total = len(batches)
        for idx, batch in enumerate(batches, start=1):
            try:
                params: list[tuple[str, str]] = [
                    ("dbfrom", "pubmed"),
                    ("db", "nuccore"),
                    ("linkname", "pubmed_nuccore"),
                    ("retmode", "json"),
                ]
                if self.api_key:
                    params.append(("api_key", self.api_key))
                for p in batch:
                    params.append(("id", str(p)))
                payload = self.get_data("elink.fcgi", params=params, allow_404=True)
                if not isinstance(payload, dict):
                    continue
                for lset in payload.get("linksets") or ():
                    if not isinstance(lset, dict):
                        continue
                    ids = lset.get("ids") or ()
                    if not ids:
                        continue
                    pmid = int(ids[0])
                    uids: list[str] = []
                    for ldb in lset.get("linksetdbs") or ():
                        if isinstance(ldb, dict) and ldb.get("linkname") == "pubmed_nuccore":
                            uids.extend(str(x) for x in (ldb.get("links") or ()))
                    if uids:
                        out[pmid] = uids
            finally:
                if prog.is_active:
                    prog.report(idx / total, detail=f"NCBI elink: batch {idx}/{total}")
        return out

    def _efetch_text(self, uids: Sequence[str], db: str, rettype: str) -> bytes:
        if not uids:
            return b""
        params = {"db": db, "id": ",".join(uids), "rettype": rettype, "retmode": "text"}
        if self.api_key:
            params["api_key"] = self.api_key
        return self.stream_bytes("GET", "efetch.fcgi", params=params, allow_404=True)

    def efetch_fasta(self, uids: Sequence[str], db: str = "nuccore") -> bytes:
        """Download multi-FASTA bytes for `uids` via `efetch.fcgi`."""
        return self._efetch_text(uids, db, "fasta")

    def efetch_genbank(
        self,
        uids: Sequence[str],
        db: str = "nuccore",
        rettype: str = "gbwithparts",
    ) -> bytes:
        """Download GenBank flat-file bytes (`gbwithparts` or `gb`) for `uids` via `efetch.fcgi`."""
        return self._efetch_text(uids, db, rettype)

    def fetch_taxon_assemblies(self, taxon: str, page_size: int = 1000) -> list[DatasetAssemblyReport]:
        """Query NCBI Datasets v2 `genome/taxon/{taxon}/dataset_report` and return assembly reports."""
        encoded = urllib.parse.quote(taxon)
        url = f"{NCBI_DATASETS_URL}/genome/taxon/{encoded}/dataset_report"
        payload = self.get_data(url, params={"page_size": str(page_size)}, allow_404=True)
        if not isinstance(payload, dict):
            return []
        reports: list[DatasetAssemblyReport] = []
        for rep in payload.get("reports") or ():
            if not isinstance(rep, dict):
                continue
            acc = str(rep.get("accession") or "").strip()
            if not acc:
                continue
            org = rep.get("organism") or {}
            infraspec = (org.get("infraspecific_names") or {}) if isinstance(org, dict) else {}
            asm_info = rep.get("assembly_info") or {}
            biosample = (asm_info.get("biosample") or {}) if isinstance(asm_info, dict) else {}
            strain = str(infraspec.get("strain") or biosample.get("strain") or "").strip()
            serovar = str(infraspec.get("serovar") or biosample.get("serovar") or "").strip()
            isolate = str(infraspec.get("isolate") or biosample.get("isolate") or "").strip()
            reports.append(
                DatasetAssemblyReport(
                    accession=acc,
                    organism_name=str(org.get("organism_name") or "").strip() if isinstance(org, dict) else "",
                    tax_id=int(org.get("tax_id") or 0) if isinstance(org, dict) else 0,
                    strain=strain,
                    serovar=serovar,
                    isolate=isolate,
                    biosample_accession=str(biosample.get("accession") or "").strip(),
                    assembly_level=str(asm_info.get("assembly_level") or "").strip()
                    if isinstance(asm_info, dict)
                    else "",
                    paired_accession=str(rep.get("paired_accession") or "").strip(),
                )
            )
        return reports

    def download_assembly_fasta(self, accession: str) -> bytes:
        """Download genome FASTA bytes for `accession` via NCBI Datasets v2 archive endpoint."""
        url = f"{NCBI_DATASETS_URL}/genome/accession/{accession}/download"
        zip_bytes = self.stream_bytes(
            "GET",
            url,
            params={"include_annotation_type": "GENOME_FASTA"},
            allow_404=True,
        )
        if not zip_bytes:
            return b""
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            fna_chunks = [
                zf.read(zi.filename) for zi in zf.infolist() if zi.filename.endswith((".fna", ".fasta", ".fa"))
            ]
            return b"\n".join(fna_chunks)
