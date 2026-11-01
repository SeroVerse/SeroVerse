"""Client for the EMBL-EBI ENA Portal REST API."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from seroverse.client.base import Client

ENA_PORTAL_URL = "https://www.ebi.ac.uk/ena/portal/api"


@dataclass(frozen=True, slots=True)
class ENASampleRecord:
    """Immutable ENA Portal `sample` record."""

    sample_accession: str = ""
    scientific_name: str = ""
    strain: str = ""
    isolate: str = ""
    culture_collection: str = ""
    serotype: str = ""
    serovar: str = ""
    description: str = ""
    tax_id: str = ""

    @property
    def taxid_int(self) -> int:
        """Numeric NCBI Taxonomy ID, or `0` if absent."""
        s = self.tax_id.strip()
        return int(s) if s.isdigit() else 0

    @property
    def strain_candidates(self) -> tuple[str, ...]:
        """Ordered tuple of non-empty strain/isolate/culture-collection strings."""
        return tuple(
            dict.fromkeys(x.strip() for x in (self.strain, self.culture_collection, self.isolate) if x.strip())
        )

    @property
    def serotype_candidates(self) -> tuple[str, ...]:
        """Ordered tuple of non-empty serotype/serovar/description strings."""
        return tuple(dict.fromkeys(x.strip() for x in (self.serotype, self.serovar, self.description) if x.strip()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ENASampleRecord:
        return cls(
            sample_accession=str(data.get("sample_accession") or ""),
            scientific_name=str(data.get("scientific_name") or ""),
            strain=str(data.get("strain") or ""),
            isolate=str(data.get("isolate") or ""),
            culture_collection=str(data.get("culture_collection") or ""),
            serotype=str(data.get("serotype") or ""),
            serovar=str(data.get("serovar") or ""),
            description=str(data.get("description") or ""),
            tax_id=str(data.get("tax_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class ENAAssemblyRecord:
    """Immutable ENA Portal `assembly` record."""

    accession: str = ""
    sample_accession: str = ""
    scientific_name: str = ""
    strain: str = ""
    tax_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ENAAssemblyRecord:
        return cls(
            accession=str(data.get("accession") or ""),
            sample_accession=str(data.get("sample_accession") or ""),
            scientific_name=str(data.get("scientific_name") or ""),
            strain=str(data.get("strain") or ""),
            tax_id=str(data.get("tax_id") or ""),
        )


class ENAClient(Client):
    """Client for the EMBL-EBI ENA Portal API."""

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

    def _search_portal(self, result: str, query: str, fields: Sequence[str], limit: int) -> list[dict[str, Any]]:
        params = {
            "result": result,
            "query": query,
            "fields": ",".join(fields),
            "limit": str(limit),
            "format": "json",
        }
        res = self.get_data("search", params=params, allow_404=True)
        if isinstance(res, list):
            return [x for x in res if isinstance(x, dict)]
        return []

    def search_samples(
        self,
        query: str,
        fields: Sequence[str] = ("sample_accession", "scientific_name", "strain", "serotype", "serovar", "tax_id"),
        limit: int = 0,
    ) -> list[ENASampleRecord]:
        """Query `search?result=sample` and return `list[ENASampleRecord]`."""
        return [ENASampleRecord.from_dict(x) for x in self._search_portal("sample", query, fields, limit)]

    def search_assemblies(
        self,
        query: str,
        fields: Sequence[str] = ("accession", "sample_accession", "scientific_name", "strain", "tax_id"),
        limit: int = 0,
    ) -> list[ENAAssemblyRecord]:
        """Query `search?result=assembly` and return `list[ENAAssemblyRecord]`."""
        return [ENAAssemblyRecord.from_dict(x) for x in self._search_portal("assembly", query, fields, limit)]
