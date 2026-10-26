"""DSMZ StrainInfo v2 and BacDive v2 REST API clients and culture-collection models."""

from __future__ import annotations

import asyncio
import os
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import msgspec

from seroverse.client.base import Client

STRAININFO_V2_URL = "https://api.straininfo.dsmz.de/v2"
BACDIVE_V2_URL = "https://api.bacdive.dsmz.de/v2"


@dataclass(frozen=True, slots=True)
class StrainLiterature:
    """Immutable bibliographic entry linked to a StrainInfo culture deposit."""

    id: int = 0
    pubmed: int = 0
    doi: str = ""
    title: str = ""
    year: int = 0


@dataclass(frozen=True, slots=True)
class StrainPassport:
    """Consolidated DSMZ StrainInfo v2 passport with culture-collection synonyms, PMIDs, and INSDC accessions."""

    si_id: int
    taxon_name: str = ""
    ncbi_taxid: int = 0
    bacdive_id: int = 0
    type_strain: bool = False
    designations: tuple[str, ...] = ()
    pmids: tuple[int, ...] = ()
    dois: tuple[str, ...] = ()
    sequence_accessions: tuple[str, ...] = ()
    genome_accessions: tuple[str, ...] = ()

    @classmethod
    def from_raw_dict(cls, raw: dict[str, Any]) -> StrainPassport | None:
        """Parse a raw StrainInfo `/v2/data/strain/max/` JSON object into a `StrainPassport`."""
        si_obj = raw.get("strain") if isinstance(raw.get("strain"), dict) else raw
        if not isinstance(si_obj, dict):
            return None
        raw_id = si_obj.get("siID") if si_obj.get("siID") is not None else si_obj.get("id")
        si_id = int(raw_id or 0) if str(raw_id or "").isdigit() else 0
        if si_id <= 0:
            return None

        taxon = si_obj.get("taxon") or {}
        taxon_name = str(taxon.get("name") or "").strip() if isinstance(taxon, dict) else ""
        ncbi_taxid = (
            int(taxon.get("ncbi") or 0) if isinstance(taxon, dict) and str(taxon.get("ncbi") or "").isdigit() else 0
        )
        raw_bd = si_obj.get("bdID") if si_obj.get("bdID") is not None else si_obj.get("bacdive")
        bacdive_id = int(raw_bd or 0) if str(raw_bd or "").isdigit() else 0
        type_strain = bool(si_obj.get("typeStrain") if "typeStrain" in si_obj else si_obj.get("type_strain"))

        des_list: list[str] = []
        rel = si_obj.get("relation") or {}
        if isinstance(rel, dict):
            for c_item in (rel.get("deposit") or ()) or (rel.get("culture") or ()):
                if isinstance(c_item, dict):
                    sn = str(c_item.get("designation") or c_item.get("strain_number") or "").strip()
                    if sn and sn not in des_list:
                        des_list.append(sn)
            for d_item in rel.get("designation") or ():
                sn_d = str(d_item.get("designation") if isinstance(d_item, dict) else d_item or "").strip()
                if sn_d and sn_d not in des_list:
                    des_list.append(sn_d)
        for dep_item in raw.get("deposits") or ():
            if isinstance(dep_item, dict) and (sn_dep := str(dep_item.get("designation") or "").strip()):
                if sn_dep not in des_list:
                    des_list.append(sn_dep)

        pmids: list[int] = []
        dois: list[str] = []
        for lit in si_obj.get("literature") or ():
            if isinstance(lit, dict):
                pm_raw = str(lit.get("pubmed") or "").strip()
                if pm_raw.isdigit() and (pm_int := int(pm_raw)) > 0 and pm_int not in pmids:
                    pmids.append(pm_int)
                doi_raw = str(lit.get("doi") or "").strip()
                if doi_raw and doi_raw not in dois:
                    dois.append(doi_raw)

        seq_accs: list[str] = []
        gen_accs: list[str] = []
        for seq in (si_obj.get("sequence") or ()) or (si_obj.get("sequences") or ()):
            if isinstance(seq, dict) and (acc := str(seq.get("accessionNumber") or seq.get("accession") or "").strip()):
                s_type = str(seq.get("type") or "").lower()
                if "genome" in s_type or acc.startswith(("GCA_", "GCF_")):
                    if acc not in gen_accs:
                        gen_accs.append(acc)
                elif s_type != "rrnaop" and acc not in seq_accs:
                    seq_accs.append(acc)

        return cls(
            si_id=si_id,
            taxon_name=taxon_name,
            ncbi_taxid=ncbi_taxid,
            bacdive_id=bacdive_id,
            type_strain=type_strain,
            designations=tuple(des_list),
            pmids=tuple(pmids),
            dois=tuple(dois),
            sequence_accessions=tuple(seq_accs),
            genome_accessions=tuple(gen_accs),
        )


class StrainInfoClient(Client):
    """Asynchronous client for the public DSMZ StrainInfo v2 REST API."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_concurrency: int = 5,
        polite_delay: float = 0.05,
        recorder: Any | None = None,
    ) -> None:
        super().__init__(
            base_url=STRAININFO_V2_URL,
            user_agent="seroverse-straininfo-client/0.1",
            timeout=timeout,
            max_concurrency=max_concurrency,
            polite_delay=polite_delay,
            recorder=recorder,
        )

    async def _search_ids(self, path: str) -> list[int]:
        try:
            raw_ids = await self.get_json(path, type_=list[int])
            return [int(x) for x in raw_ids if int(x) > 0]
        except Exception:
            return []

    async def search_by_taxon(self, taxon_name: str) -> list[int]:
        """Return all StrainInfo IDs (`SI-ID`s) for a species or genus name."""
        return await self._search_ids(f"search/strain/tax_name/{urllib.parse.quote(taxon_name.strip())}")

    async def search_by_cc_no(self, cc_no: str) -> list[int]:
        """Return matching `SI-ID`s for a culture collection number (e.g. `ATCC 25238`, `NCTC 9001`)."""
        return await self._search_ids(f"search/strain/cc_no/{urllib.parse.quote(cc_no.strip())}")

    async def search_by_designation(self, designation: str) -> list[int]:
        """Return matching `SI-ID`s for an arbitrary strain designation (e.g. `Prague 1550/57`)."""
        return await self._search_ids(f"search/strain/des/{urllib.parse.quote(designation.strip())}")

    async def search_by_collection(self, collection_code: str) -> list[int]:
        """Return all `SI-ID`s deposited in a culture collection (e.g. `CNCTC`, `NCTC`, `ATCC`)."""
        return await self._search_ids(f"search/strain/cc/{urllib.parse.quote(collection_code.strip())}")

    async def fetch_passports(self, si_ids: Sequence[int], batch_size: int = 50) -> list[StrainPassport]:
        """Batch-fetch `StrainPassport` records for `si_ids` via `/v2/data/strain/max/{ids}`."""
        unique_ids = list(dict.fromkeys(int(x) for x in si_ids if int(x) > 0))
        if not unique_ids:
            return []
        passports: list[StrainPassport] = []

        async def _fetch_chunk(chunk: Sequence[int]) -> list[StrainPassport]:
            joined = ",".join(str(x) for x in chunk)
            try:
                raw = await self.stream_bytes("GET", f"data/strain/max/{joined}")
                items = msgspec.json.decode(raw)
                if isinstance(items, list):
                    return [
                        p
                        for item in items
                        if isinstance(item, dict) and (p := StrainPassport.from_raw_dict(item)) is not None
                    ]
            except Exception:
                pass
            return []

        chunks = [unique_ids[i : i + batch_size] for i in range(0, len(unique_ids), batch_size)]
        for res in await asyncio.gather(*[_fetch_chunk(c) for c in chunks]):
            passports.extend(res)
        return passports

    async def fetch_taxon_passports(self, taxon_name: str, batch_size: int = 50) -> list[StrainPassport]:
        """Search all strains for `taxon_name` and batch-fetch their complete `StrainPassport` records."""
        si_ids = await self.search_by_taxon(taxon_name)
        return await self.fetch_passports(si_ids, batch_size=batch_size)


class BacDiveClient(Client):
    """Asynchronous client for the DSMZ BacDive v2 REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
        max_concurrency: int = 4,
        polite_delay: float = 0.05,
        recorder: Any | None = None,
    ) -> None:
        token = api_key if api_key is not None else os.environ.get("BACDIVE_API_KEY", "")
        super().__init__(
            base_url=BACDIVE_V2_URL,
            api_key_header="Authorization" if token else None,
            api_key=f"Bearer {token}" if token and not token.startswith("Bearer ") else (token or None),
            user_agent="seroverse-bacdive-client/0.1",
            timeout=timeout,
            max_concurrency=max_concurrency,
            polite_delay=polite_delay,
            recorder=recorder,
        )

    async def fetch_strain(self, bacdive_id: int) -> dict[str, Any]:
        """Fetch a BacDive strain record by its integer `bacdive_id`."""
        try:
            res = await self.get(f"fetch/{int(bacdive_id)}")
            return res if isinstance(res, dict) else {}
        except Exception:
            return {}
