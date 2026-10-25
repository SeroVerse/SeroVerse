"""Asynchronous client for the EMBL-EBI ENA Portal REST API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from seroverse.client.base import Client

ENA_PORTAL_URL = "https://www.ebi.ac.uk/ena/portal/api"


@dataclass(frozen=True, slots=True)
class ENASampleRecord:
    """Immutable ENA Portal `sample` record decoded via `msgspec`."""

    sample_accession: str = ""
    scientific_name: str = ""
    strain: str = ""
    serotype: str = ""
    serovar: str = ""
    tax_id: str = ""


@dataclass(frozen=True, slots=True)
class ENAAssemblyRecord:
    """Immutable ENA Portal `assembly` record decoded via `msgspec`."""

    accession: str = ""
    sample_accession: str = ""
    scientific_name: str = ""
    strain: str = ""
    tax_id: str = ""


class ENAClient(Client):
    """Asynchronous client for the EMBL-EBI ENA Portal API."""

    def __init__(
        self,
        *,
        timeout: float = 40.0,
        max_concurrency: int = 4,
        polite_delay: float = 0.05,
        recorder: Any | None = None,
    ) -> None:
        super().__init__(
            base_url=ENA_PORTAL_URL,
            user_agent="seroverse-ena-client/0.1",
            timeout=timeout,
            max_concurrency=max_concurrency,
            polite_delay=polite_delay,
            recorder=recorder,
        )

    async def search_samples(
        self,
        query: str,
        fields: Sequence[str] = ("sample_accession", "scientific_name", "strain", "serotype", "serovar", "tax_id"),
        limit: int = 0,
    ) -> list[ENASampleRecord]:
        """Query `search?result=sample` and decode directly into `list[ENASampleRecord]`."""
        params = {
            "result": "sample",
            "query": query,
            "fields": ",".join(fields),
            "limit": str(limit),
            "format": "json",
        }
        try:
            return await self.get_json("search", type_=list[ENASampleRecord], params=params)
        except Exception:
            return []

    async def search_assemblies(
        self,
        query: str,
        fields: Sequence[str] = ("accession", "sample_accession", "scientific_name", "strain", "tax_id"),
        limit: int = 0,
    ) -> list[ENAAssemblyRecord]:
        """Query `search?result=assembly` and decode directly into `list[ENAAssemblyRecord]`."""
        params = {
            "result": "assembly",
            "query": query,
            "fields": ",".join(fields),
            "limit": str(limit),
            "format": "json",
        }
        try:
            return await self.get_json("search", type_=list[ENAAssemblyRecord], params=params)
        except Exception:
            return []
