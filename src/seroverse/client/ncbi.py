"""NCBI Entrez E-Utilities and NCBI Datasets v2 REST API client and models."""

from __future__ import annotations

import asyncio
import io
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import msgspec

from seroverse.client.base import Client
from seroverse.progress import current_progress

NCBI_EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_DATASETS_URL = "https://api.ncbi.nlm.nih.gov/datasets/v2"


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
class _ESearchResultInner:
    idlist: tuple[str, ...] = ()
    count: str = "0"


@dataclass(frozen=True, slots=True)
class _ESearchEnvelope:
    esearchresult: _ESearchResultInner | None = None


class NCBIClient(Client):
    """Asynchronous client for NCBI Entrez E-Utilities and NCBI Datasets v2."""

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

    async def esearch(self, db: str, term: str, retmax: int = 100) -> list[str]:
        """Run rate-limited `esearch.fcgi` and return matching UID strings."""
        try:
            params = self._eutils_params({"db": db, "term": term, "retmax": str(retmax)})
            env = await self.get_json("esearch.fcgi", type_=_ESearchEnvelope, params=params)
            if env.esearchresult and env.esearchresult.idlist:
                return [str(u) for u in env.esearchresult.idlist]
        except Exception:
            pass
        return []

    async def esummary(self, db: str, uids: Sequence[str], chunk_size: int = 200) -> dict[str, dict[str, Any]]:
        """Run batched `esummary.fcgi` POST requests and return `{uid: info_dict}`."""
        out: dict[str, dict[str, Any]] = {}
        chunks = [uids[i : i + chunk_size] for i in range(0, len(uids), chunk_size)]
        if not chunks:
            return out
        prog = current_progress()
        completed = 0
        total = len(chunks)

        async def _fetch_chunk(chunk: Sequence[str]) -> None:
            nonlocal completed
            try:
                data = self._eutils_params({"db": db, "id": ",".join(chunk)})
                raw = await self.stream_bytes("POST", "esummary.fcgi", data=data)
                payload = msgspec.json.decode(raw)
                if isinstance(payload, dict) and isinstance(res := payload.get("result"), dict):
                    for u in chunk:
                        if isinstance(info := res.get(u), dict):
                            out[u] = info
            except Exception:
                pass
            finally:
                completed += 1
                if prog.is_active:
                    prog.report(completed / total, detail=f"NCBI esummary ({db}): batch {completed}/{total}")

        await asyncio.gather(*[_fetch_chunk(c) for c in chunks])
        return out

    async def fetch_nuccore_summaries(
        self,
        uids: Sequence[str],
        *,
        slen_range: tuple[int, int] | None = None,
        chunk_size: int = 200,
    ) -> dict[str, NuccoreSummary]:
        """Fetch `NuccoreSummary` records for `uids`, optionally filtering by sequence length `slen_range`."""
        raw_map = await self.esummary("nuccore", uids, chunk_size=chunk_size)
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

    async def fetch_pubmed_summaries(
        self, pmids: Sequence[int | str], chunk_size: int = 150
    ) -> dict[int, PubMedSummary]:
        """Fetch `PubMedSummary` records (including DOIs) for `pmids`."""
        uid_strs = [str(p) for p in pmids]
        raw_map = await self.esummary("pubmed", uid_strs, chunk_size=chunk_size)
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

    async def elink_pubmed_to_nuccore(self, pmids: Sequence[int], batch_size: int = 40) -> dict[int, list[str]]:
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
                raw = await self.stream_bytes("GET", "elink.fcgi", params=params)
                payload = msgspec.json.decode(raw)
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
            except Exception:
                pass
            finally:
                if prog.is_active:
                    prog.report(idx / total, detail=f"NCBI elink: batch {idx}/{total}")
        return out

    async def efetch_fasta(self, uids: Sequence[str], db: str = "nuccore") -> bytes:
        """Download multi-FASTA bytes for `uids` via `efetch.fcgi`."""
        if not uids:
            return b""
        try:
            params = {"db": db, "id": ",".join(uids), "rettype": "fasta", "retmode": "text"}
            if self.api_key:
                params["api_key"] = self.api_key
            return await self.stream_bytes("GET", "efetch.fcgi", params=params)
        except Exception:
            return b""

    async def fetch_taxon_assemblies(self, taxon: str, page_size: int = 1000) -> list[DatasetAssemblyReport]:
        """Query NCBI Datasets v2 `genome/taxon/{taxon}/dataset_report` and return assembly reports."""
        import urllib.parse

        encoded = urllib.parse.quote(taxon)
        url = f"{NCBI_DATASETS_URL}/genome/taxon/{encoded}/dataset_report"
        try:
            raw = await self.stream_bytes("GET", url, params={"page_size": str(page_size)})
            payload = msgspec.json.decode(raw)
        except Exception:
            return []
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

    async def download_assembly_fasta(self, accession: str) -> bytes:
        """Download genome FASTA bytes for `accession` via NCBI Datasets v2 archive endpoint."""
        url = f"{NCBI_DATASETS_URL}/genome/accession/{accession}/download"
        try:
            zip_bytes = await self.stream_bytes("GET", url, params={"include_annotation_type": "GENOME_FASTA"})
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                fna_chunks = [
                    zf.read(zi.filename) for zi in zf.infolist() if zi.filename.endswith((".fna", ".fasta", ".fa"))
                ]
                return b"\n".join(fna_chunks)
        except Exception:
            return b""
