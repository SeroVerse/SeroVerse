"""Client and metadata models for the ATCC Genome Portal REST API."""

import gzip
import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from seroverse.client.base import Client
from seroverse.client.progress import current_progress
from seroverse.core.io.fasta import FastaReader
from seroverse.core.seq.seq import Sequences

ATCC_GENOME_PORTAL_URL = "https://genomes.atcc.org/api"

_ATCC_PREFIX_RE = re.compile(r"^atcc[\s\-_:]*", re.IGNORECASE)
_GZIP_MAGIC = b"\x1f\x8b"


def _normalize_product_id(product_id: str) -> str:
    """Normalize an ATCC catalog number by stripping any leading ``'ATCC'`` prefix."""
    return _ATCC_PREFIX_RE.sub("", product_id.strip()).strip()


def _coerce_int(val: Any, default: int = 0) -> int:
    if val is None or isinstance(val, bool):
        return default
    try:
        return int(val)
    except TypeError, ValueError:
        return default


def _coerce_opt_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool) or val == "":
        return None
    try:
        return int(val)
    except TypeError, ValueError:
        return None


def _coerce_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        return float(val)
    except TypeError, ValueError:
        return default


@dataclass(frozen=True, slots=True)
class ATCCAssembly:
    """Immutable ATCC Genome Portal assembly metadata and quality metrics record.

    Attributes:
        id: Unique assembly UUID/identifier on the ATCC Genome Portal.
        genome_id: Parent genome UUID/identifier.
        version: Assembly version string.
        assembly_level: Assembly completion status or level.
        genome_size: Total assembly sequence length in base pairs.
        n_contigs: Number of contigs/scaffolds in the assembly.
        n50: Contig N50 length in base pairs.
        gc_content: Assembly GC percentage or fraction.
        sequencing_technology: Sequencing platform/technology summary.
        attributes: Additional raw portal metadata and quality metrics.
    """

    id: str
    genome_id: str = ""
    version: str = ""
    assembly_level: str = ""
    genome_size: int = 0
    n_contigs: int = 0
    n50: int = 0
    gc_content: float = 0.0
    sequencing_technology: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ATCCAssembly:
        """Instantiate an [`ATCCAssembly`][seroverse.client.atcc.ATCCAssembly] from a raw portal dictionary."""
        stats = (
            data.get("metadata")
            or data.get("assembly_statistics")
            or data.get("assembly_metrics")
            or data.get("statistics")
            or data.get("metrics")
            or {}
        )
        if not isinstance(stats, dict):
            stats = {}

        asm_id = str(data.get("id") or data.get("uuid") or "").strip()
        genome_id = str(data.get("genome_id") or data.get("genome_uuid") or "").strip()
        version = str(data.get("version") or data.get("assembly_version") or "").strip()
        level = str(data.get("assembly_level") or data.get("status") or stats.get("assembly_level") or "").strip()
        genome_size = _coerce_int(
            data.get("genome_size")
            or data.get("total_length")
            or data.get("size")
            or stats.get("total_length")
            or stats.get("genome_size")
            or stats.get("size")
        )
        n_contigs = _coerce_int(
            data.get("n_contigs")
            or data.get("contig_count")
            or data.get("contigs")
            or stats.get("n_contigs")
            or stats.get("contig_count")
            or stats.get("number_of_contigs")
            or stats.get("contigs")
            or stats.get("scaffolds")
        )
        n50 = _coerce_int(data.get("n50") or stats.get("n50") or stats.get("contig_n50") or stats.get("scaffold_n50"))
        gc_content = _coerce_float(
            data.get("gc_content")
            or data.get("gc_percent")
            or data.get("gc")
            or stats.get("gc_content")
            or stats.get("gc_percent")
            or stats.get("gc_Per")
            or stats.get("gc")
        )
        seq_tech = str(
            data.get("sequencing_technology")
            or data.get("sequencing_platform")
            or stats.get("sequencing_technology")
            or ""
        ).strip()

        known_keys = {
            "id",
            "uuid",
            "genome_id",
            "genome_uuid",
            "version",
            "assembly_version",
            "assembly_level",
            "status",
            "genome_size",
            "total_length",
            "size",
            "n_contigs",
            "contig_count",
            "contigs",
            "n50",
            "gc_content",
            "gc_percent",
            "gc",
            "sequencing_technology",
            "sequencing_platform",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            id=asm_id,
            genome_id=genome_id,
            version=version,
            assembly_level=level,
            genome_size=genome_size,
            n_contigs=n_contigs,
            n50=n50,
            gc_content=gc_content,
            sequencing_technology=seq_tech,
            attributes=extra,
        )


@dataclass(frozen=True, slots=True)
class ATCCGenome:
    """Immutable ATCC Genome Portal genome record.

    Attributes:
        id: Unique genome UUID/identifier on the ATCC Genome Portal.
        product_id: Normalized ATCC catalog number (e.g. ``"25922"`` or ``"BAA-1716"``).
        taxon_name: Scientific/organism name.
        ncbi_taxid: Optional NCBI taxonomy ID.
        strain: Strain name or designation string.
        serotype: Serotype or serovar designation if available.
        isolation_source: Biological or environmental isolation source.
        geographic_location: Country or geographic origin of the isolate.
        collection_date: Collection date or year string.
        primary_assembly_id: Identifier of the primary assembly if available.
        primary_assembly: Parsed [`ATCCAssembly`][seroverse.client.atcc.ATCCAssembly] when embedded in the response.
        attributes: Additional raw portal metadata (such as ``atcc_metadata``).
    """

    id: str
    product_id: str = ""
    taxon_name: str = ""
    ncbi_taxid: int | None = None
    strain: str = ""
    serotype: str = ""
    isolation_source: str = ""
    geographic_location: str = ""
    collection_date: str = ""
    primary_assembly_id: str = ""
    primary_assembly: ATCCAssembly | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def catalog_label(self) -> str:
        """Return ``'ATCC <product_id>'`` when `product_id` is set, otherwise `id`."""
        return f"ATCC {self.product_id}" if self.product_id else self.id

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ATCCGenome:
        """Instantiate an [`ATCCGenome`][seroverse.client.atcc.ATCCGenome] from a raw portal dictionary."""
        raw_attrs = data.get("attributes")
        attrs: dict[str, Any] = raw_attrs if isinstance(raw_attrs, dict) else {}
        atcc_meta = data.get("atcc_metadata") or attrs.get("atcc_metadata") or data.get("metadata") or attrs or {}
        if not isinstance(atcc_meta, dict):
            atcc_meta = {}
        raw_tax = attrs.get("taxonomy")
        taxonomy_meta: dict[str, Any] = raw_tax if isinstance(raw_tax, dict) else {}

        genome_id = str(data.get("id") or data.get("uuid") or "").strip()
        raw_prod = str(
            data.get("product_id")
            or data.get("catalog_number")
            or data.get("atcc_id")
            or attrs.get("product_id")
            or atcc_meta.get("product_id")
            or atcc_meta.get("catalog_number")
            or ""
        ).strip()
        product_id = _normalize_product_id(raw_prod) if raw_prod else ""

        taxon_name = str(
            data.get("taxon_name")
            or data.get("scientific_name")
            or data.get("organism_name")
            or data.get("name")
            or attrs.get("name")
            or taxonomy_meta.get("name")
            or atcc_meta.get("taxon_name")
            or atcc_meta.get("scientific_name")
            or ""
        ).strip()

        ncbi_taxid = _coerce_opt_int(
            data.get("ncbi_taxid")
            or data.get("tax_id")
            or data.get("taxid")
            or taxonomy_meta.get("tax_id")
            or taxonomy_meta.get("ncbi_taxid")
            or atcc_meta.get("ncbi_taxid")
            or atcc_meta.get("tax_id")
        )

        raw_strain = (
            data.get("strain")
            or data.get("strain_name")
            or atcc_meta.get("strain")
            or atcc_meta.get("strain_name")
            or atcc_meta.get("strain_designations")
            or ""
        )
        if isinstance(raw_strain, (list, tuple)):
            strain = ", ".join(str(x).strip() for x in raw_strain if str(x).strip())
        else:
            strain = str(raw_strain).strip()

        serotype = str(
            data.get("serotype")
            or data.get("serovar")
            or atcc_meta.get("serotype")
            or atcc_meta.get("serovar")
            or atcc_meta.get("antigenic_prop")
            or ""
        ).strip()

        isolation_source = str(
            data.get("isolation_source") or atcc_meta.get("isolation_source") or atcc_meta.get("source") or ""
        ).strip()

        geographic_location = str(
            data.get("geographic_location")
            or data.get("country")
            or atcc_meta.get("geographic_location")
            or atcc_meta.get("geographic_info")
            or atcc_meta.get("isolation_country")
            or atcc_meta.get("country")
            or ""
        ).strip()

        collection_date = str(
            data.get("collection_date")
            or data.get("collection_year")
            or atcc_meta.get("collection_date")
            or atcc_meta.get("collection_year")
            or atcc_meta.get("year_of_isolation")
            or ""
        ).strip()

        raw_asm = data.get("primary_assembly") or data.get("assembly")
        primary_asm: ATCCAssembly | None = None
        primary_asm_id = str(data.get("primary_assembly_id") or data.get("assembly_id") or "").strip()

        if isinstance(raw_asm, dict):
            asm_payload = dict(raw_asm)
            if genome_id and "genome_id" not in asm_payload:
                asm_payload["genome_id"] = genome_id
            primary_asm = ATCCAssembly.from_dict(asm_payload)
            if not primary_asm_id and primary_asm.id:
                primary_asm_id = primary_asm.id
        elif isinstance(raw_asm, str) and raw_asm.strip() and not primary_asm_id:
            primary_asm_id = raw_asm.strip()

        known_keys = {
            "id",
            "uuid",
            "product_id",
            "catalog_number",
            "atcc_id",
            "taxon_name",
            "scientific_name",
            "organism_name",
            "name",
            "ncbi_taxid",
            "tax_id",
            "taxid",
            "strain",
            "strain_name",
            "serotype",
            "serovar",
            "isolation_source",
            "geographic_location",
            "country",
            "collection_date",
            "collection_year",
            "primary_assembly_id",
            "assembly_id",
            "primary_assembly",
            "assembly",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            id=genome_id,
            product_id=product_id,
            taxon_name=taxon_name,
            ncbi_taxid=ncbi_taxid,
            strain=strain,
            serotype=serotype,
            isolation_source=isolation_source,
            geographic_location=geographic_location,
            collection_date=collection_date,
            primary_assembly_id=primary_asm_id,
            primary_assembly=primary_asm,
            attributes=extra,
        )


class ATCCGenomePortalClient(Client):
    """REST client for the ATCC Genome Portal (`https://genomes.atcc.org/api`).

    Requires an ATCC Genome Portal API key (sent via the ``X-API-Key`` HTTP header,
    or ``Authorization`` if a ``Bearer <token>`` string is supplied).
    """

    DEFAULT_BASE_URL: ClassVar[str] = ATCC_GENOME_PORTAL_URL

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        timeout: float = 30.0,
        max_concurrency: int = 5,
        polite_delay: float = 0.05,
        recorder: Any | None = None,
    ) -> None:
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if not clean_key:
            raise ValueError("A non-empty ATCC Genome Portal API key is required.")

        resolved_url = base_url.rstrip("/")
        if not resolved_url.endswith("/api"):
            resolved_url = f"{resolved_url}/api"

        is_bearer = clean_key.lower().startswith("bearer ")
        header_name = "Authorization" if is_bearer else "X-API-Key"

        super().__init__(
            base_url=resolved_url,
            api_key_header=header_name,
            api_key=clean_key,
            user_agent="seroverse-atcc-client/0.1",
            timeout=timeout,
            max_concurrency=max_concurrency,
            polite_delay=polite_delay,
            recorder=recorder,
        )

    @staticmethod
    def _extract_page_items(payload: Any, page_size: int) -> tuple[list[dict[str, Any]], bool]:
        """Extract genome dict records and a `has_more` flag from a portal response."""
        if isinstance(payload, list):
            items = [x for x in payload if isinstance(x, dict)]
            return items, len(items) >= page_size > 0
        if isinstance(payload, dict):
            raw_list = (
                payload.get("results") or payload.get("genomes") or payload.get("items") or payload.get("data") or []
            )
            items = [x for x in raw_list if isinstance(x, dict)] if isinstance(raw_list, list) else []
            if "has_next" in payload:
                has_more = bool(payload["has_next"])
            elif "has_more" in payload:
                has_more = bool(payload["has_more"])
            elif "next" in payload:
                has_more = bool(payload["next"])
            elif "total_pages" in payload and "page" in payload:
                has_more = _coerce_int(payload["page"]) < _coerce_int(payload["total_pages"])
            else:
                has_more = len(items) >= page_size > 0
            return items, has_more
        return [], False

    def list_genomes(
        self,
        *,
        page: int = 1,
        page_size: int = 100,
        all_pages: bool = False,
    ) -> list[ATCCGenome]:
        """List available genomes from ``GET /api/genomes``."""
        genomes: list[ATCCGenome] = []
        current_page = max(1, page)
        prog = current_progress()

        while True:
            params = {"page": str(current_page), "page_size": str(page_size)}
            payload = self.get_data("genomes", params=params)
            items, has_more = self._extract_page_items(payload, page_size)
            genomes.extend(ATCCGenome.from_dict(item) for item in items)

            if not all_pages or not has_more or not items:
                break

            if prog.is_active:
                prog.report(
                    min(0.95, 1.0 - 1.0 / (current_page + 1.0)),
                    detail=f"ATCC genomes: page {current_page} ({len(genomes):,} fetched)",
                )
            current_page += 1

        return genomes

    def search_genomes(
        self,
        *,
        product_id: str | None = None,
        text: str | None = None,
        page: int = 1,
        page_size: int = 100,
        all_pages: bool = False,
    ) -> list[ATCCGenome]:
        """Search genomes via ``POST /api/genomes/search`` by `product_id` or free `text`."""
        if not product_id and not text:
            raise ValueError("At least one of 'product_id' or 'text' must be provided to search_genomes.")

        base_body: dict[str, Any] = {}
        if product_id is not None:
            norm_id = _normalize_product_id(product_id)
            if not norm_id:
                raise ValueError("'product_id' cannot be empty.")
            base_body["product_id"] = norm_id
        if text is not None:
            clean_text = text.strip()
            if not clean_text:
                raise ValueError("'text' cannot be empty.")
            base_body["text"] = clean_text

        genomes: list[ATCCGenome] = []
        current_page = max(1, page)
        prog = current_progress()

        while True:
            params = {"page": str(current_page)} if current_page > 1 else None
            payload = self.post_data("genomes/search", json=base_body, params=params)
            items, has_more = self._extract_page_items(payload, page_size)
            genomes.extend(ATCCGenome.from_dict(item) for item in items)

            if not all_pages or not has_more or not items:
                break

            if prog.is_active:
                prog.report(
                    min(0.95, 1.0 - 1.0 / (current_page + 1.0)),
                    detail=f"ATCC search: page {current_page} ({len(genomes):,} matches)",
                )
            current_page += 1

        return genomes

    def search_by_product_id(self, product_id: str) -> list[ATCCGenome]:
        """Search genomes matching an ATCC catalog number (e.g. ``'25922'`` or ``'ATCC BAA-1716'``)."""
        return self.search_genomes(product_id=product_id)

    def get_genome(self, genome_id: str) -> ATCCGenome:
        """Fetch full metadata for a single genome from ``GET /api/genomes/{id}``."""
        clean_id = genome_id.strip()
        if not clean_id:
            raise ValueError("A non-empty 'genome_id' is required.")
        payload = self.get_data(f"genomes/{clean_id}")
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected response payload for genome '{clean_id}'.")
        return ATCCGenome.from_dict(payload)

    def get_assembly(self, assembly_id: str) -> ATCCAssembly:
        """Fetch metadata and quality metrics for an assembly from ``GET /api/assemblies/{id}``."""
        clean_id = assembly_id.strip()
        if not clean_id:
            raise ValueError("A non-empty 'assembly_id' is required.")
        payload = self.get_data(f"assemblies/{clean_id}")
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected response payload for assembly '{clean_id}'.")
        return ATCCAssembly.from_dict(payload)

    def _fetch_presigned_bytes(self, signed_url: str) -> bytes:
        """Download raw bytes from a pre-signed cloud storage URL without leaking the API key."""
        return self.fetch_external_bytes(signed_url)

    def _download_genome_artifact(
        self,
        genome_id: str,
        action: str,
        *,
        decompress: bool = True,
    ) -> bytes:
        """Resolve a two-step ATCC Genome Portal download endpoint and return file bytes."""
        clean_id = genome_id.strip()
        if not clean_id:
            raise ValueError("A non-empty 'genome_id' is required.")

        raw = self.stream_bytes("GET", f"genomes/{clean_id}/{action}")
        if raw.startswith(b"{"):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                signed_url = str(
                    decoded.get("url") or decoded.get("download_url") or decoded.get("signed_url") or ""
                ).strip()
                if not signed_url:
                    raise ValueError(
                        f"ATCC download endpoint 'genomes/{clean_id}/{action}' returned JSON without a download URL."
                    )
                raw = self._fetch_presigned_bytes(signed_url)

        if decompress and raw.startswith(_GZIP_MAGIC):
            return gzip.decompress(raw)
        return raw

    def download_assembly(self, genome_id: str, *, decompress: bool = True) -> bytes:
        """Download assembly FASTA bytes for `genome_id` via ``GET /api/genomes/{id}/download_assembly``."""
        return self._download_genome_artifact(
            genome_id,
            "download_assembly",
            decompress=decompress,
        )

    def download_annotations(self, genome_id: str, *, decompress: bool = True) -> bytes:
        """Download annotation file bytes (GenBank/GFF) for `genome_id` via ``GET /api/genomes/{id}/download_annotations``."""
        return self._download_genome_artifact(
            genome_id,
            "download_annotations",
            decompress=decompress,
        )

    def fetch_assembly_sequences(
        self,
        genome_id: str,
        *,
        truncate_id: bool = True,
        uppercase: bool = True,
    ) -> Sequences:
        """Download the assembly FASTA for `genome_id` and parse it into a [`Sequences`][seroverse.core.seq.seq.Sequences] container."""
        fasta_bytes = self.download_assembly(genome_id, decompress=True)
        if not fasta_bytes:
            return Sequences.empty()
        with FastaReader.from_bytes(
            fasta_bytes,
            truncate_id=truncate_id,
            uppercase=uppercase,
        ) as reader:
            return reader.load()
