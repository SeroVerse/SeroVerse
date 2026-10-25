"""Pathogenwatch REST API client and collection metadata models."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from seroverse.client.base import Client

PATHOGENWATCH_API_URL = "https://pathogen.watch/api"


@dataclass(frozen=True, slots=True)
class PathogenwatchCollection:
    """Dataclass representing a Pathogenwatch collection metadata object.

    Attributes:
        uuid: Short UUID identifier of the collection.
        name: Human-readable title of the collection.
        id: Optional numeric or string internal Pathogenwatch collection ID.
        description: Optional collection description text.
        size: Total number of genomes in the collection, or ``None``.
        organism_id: Taxonomic organism identifier.
        organism_name: Human-readable species/organism name.
        owner: Ownership tag (e.g., ``"me"`` for user-owned collections).
        access: Visibility/access level (e.g., ``"PUBLIC"`` or ``"PRIVATE"``).
        binned: Whether the collection is binned/archived.
        created_at: ISO timestamp string when the collection was created.
    """

    uuid: str
    name: str
    id: int | str | None = None
    description: str | None = None
    size: int | None = None
    organism_id: str | None = None
    organism_name: str | None = None
    owner: str | None = None
    access: str | None = None
    binned: bool | None = None
    created_at: str | None = None

    ORGANISM_TAXID_MAP: ClassVar[dict[str, str]] = {
        "197": "Campylobacter jejuni",
        "210": "Helicobacter pylori",
        "287": "Pseudomonas aeruginosa",
        "470": "Acinetobacter baumannii",
        "485": "Neisseria gonorrhoeae",
        "487": "Neisseria meningitidis",
        "562": "Escherichia coli",
        "573": "Klebsiella pneumoniae",
        "590": "Salmonella",
        "620": "Shigella",
        "666": "Vibrio cholerae",
        "670": "Vibrio parahaemolyticus",
        "1280": "Staphylococcus aureus",
        "1311": "Streptococcus agalactiae",
        "1313": "Streptococcus pneumoniae",
        "1314": "Streptococcus pyogenes",
        "1336": "Streptococcus equi",
        "1351": "Enterococcus faecalis",
        "1352": "Enterococcus faecium",
        "1639": "Listeria monocytogenes",
        "1646": "Renibacterium salmoninarum",
        "1773": "Mycobacterium tuberculosis",
        "28901": "Salmonella enterica",
        "64320": "Zika virus",
        "90370": "Salmonella enterica serovar Typhi",
        "2697049": "SARS-CoV-2",
        "3390273": "Klebsiella pneumoniae species complex",
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PathogenwatchCollection:
        """Instantiates a ``PathogenwatchCollection`` from a raw API dictionary payload."""
        uuid_val = str(data.get("uuid") or data.get("id") or "")
        name_val = str(data.get("name") or "")
        size_val = data.get("size")
        if size_val is None:
            size_val = data.get("genomeCount") if data.get("genomeCount") is not None else data.get("genome_count")
        org_id_raw = data.get("organismId") or data.get("organism_id")
        org_id = str(org_id_raw).strip() if org_id_raw is not None and str(org_id_raw).strip() else None
        org_name_raw = data.get("organismName") or data.get("organism_name")
        org_name = str(org_name_raw).strip() if org_name_raw is not None and str(org_name_raw).strip() else None
        if org_name is None and org_id is not None:
            org_name = cls.ORGANISM_TAXID_MAP.get(org_id)

        return cls(
            uuid=uuid_val,
            name=name_val,
            id=data.get("id"),
            description=data.get("description"),
            size=int(size_val) if size_val is not None else None,
            organism_id=org_id,
            organism_name=org_name,
            owner=data.get("owner"),
            access=data.get("access"),
            binned=data.get("binned"),
            created_at=data.get("createdAt") or data.get("created_at"),
        )

    @property
    def organism_label(self) -> str | None:
        """Returns the resolved human-readable species/organism label for grouping, or ``None``."""
        if self.organism_name and self.organism_name.strip():
            return self.organism_name.strip()
        if self.organism_id and self.organism_id.strip():
            oid = self.organism_id.strip()
            return self.ORGANISM_TAXID_MAP.get(oid, f"Organism ({oid})")
        return None

    @property
    def genome_count(self) -> int | None:
        """Returns the collection genome count."""
        return self.size

    @property
    def is_owned(self) -> bool:
        """Returns whether this collection belongs to the authenticated user (``"me"``)."""
        return self.owner == "me"

    @property
    def display_label(self) -> str:
        """Returns a human-readable selector label including genome count when available."""
        title = self.name.strip() or self.uuid
        if self.size is not None:
            return f"{title} (n={self.size:,})"
        return title

    @staticmethod
    def group_choices(collections: Sequence[PathogenwatchCollection]) -> dict[str, Any]:
        """Builds grouped selectize choices organized by species/organism (with owned collections first)."""
        valid = [c for c in collections if c.uuid and (c.size is None or c.size > 0)]
        by_organism: dict[str, list[PathogenwatchCollection]] = {}
        untyped_owned: dict[str, str] = {}
        untyped_shared: dict[str, str] = {}

        for c in valid:
            org_label = c.organism_label
            if org_label is not None:
                by_organism.setdefault(org_label, []).append(c)
            elif c.is_owned:
                untyped_owned[c.uuid] = c.display_label
            else:
                untyped_shared[c.uuid] = c.display_label

        grouped: dict[str, Any] = {}
        sorted_org_keys = sorted(
            by_organism.keys(),
            key=lambda k: (not any(c.is_owned for c in by_organism[k]), -len(by_organism[k]), k),
        )
        for org_key in sorted_org_keys:
            items = sorted(by_organism[org_key], key=lambda c: (not c.is_owned, c.name.lower()))
            grouped[org_key] = {c.uuid: f"★ {c.display_label}" if c.is_owned else c.display_label for c in items}

        if untyped_owned:
            grouped["My Collections"] = untyped_owned
        if untyped_shared:
            grouped["Public / Shared"] = untyped_shared
        return grouped


class PathogenwatchBaseClient(Client):
    """Asynchronous and synchronous client for interacting with the Pathogenwatch REST API."""

    DEFAULT_BASE_URL: ClassVar[str] = PATHOGENWATCH_API_URL

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initializes the Pathogenwatch API client."""
        clean_key = api_key.strip() if isinstance(api_key, str) else ""
        if not clean_key:
            raise ValueError("A non-empty Pathogenwatch API key is required.")
        resolved_url = base_url.rstrip("/")
        if not resolved_url.endswith("/api"):
            resolved_url = f"{resolved_url}/api"

        super().__init__(
            base_url=resolved_url,
            api_key_header="X-API-Key",
            api_key=clean_key,
            timeout=timeout,
        )

    async def get_collections(
        self,
        owner: str | None = None,
        exclude: str | None = None,
        binned: bool | None = None,
        limit: int | None = None,
    ) -> list[PathogenwatchCollection]:
        """Fetches collections available to the authenticated account or public collections."""
        params: dict[str, str] = {}
        if owner is not None:
            params["owner"] = owner
        if exclude is not None:
            params["exclude"] = exclude
        if binned is not None:
            params["binned"] = "true" if binned else "false"
        if limit is not None:
            params["limit"] = str(limit)

        res = await self.get("collections/list", params=params or None)
        if isinstance(res, list):
            return [PathogenwatchCollection.from_dict(c) for c in res]
        return []

    async def get_collection(self, uuid: str) -> dict[str, Any]:
        """Fetches complete metadata and analysis jobs for a specific collection."""
        res = await self.get("collections/details", params={"uuid": uuid})
        return res if isinstance(res, dict) else {}

    async def get_collection_genomes(
        self,
        uuid: str,
        share_id: str | None = None,
        page: int = 1,
        all_pages: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetches genome analysis records for a collection."""
        all_records: list[dict[str, Any]] = []
        current_page = page

        while True:
            params: dict[str, str] = {"uuid": uuid, "page": str(current_page)}
            if share_id:
                params["shareId"] = share_id

            res = await self.get("collections/genomes", params=params)
            if not isinstance(res, dict):
                break

            records = res.get("genomes") or res.get("genome") or []
            all_records.extend(records)

            if not all_pages or not res.get("hasMore") or not records:
                break

            current_page += 1

        return all_records

    async def download_distance_matrix(
        self,
        uuid: str,
        matrix_type: str = "difference",
    ) -> str:
        """Downloads pairwise genomic distance matrix CSV for a collection."""
        res = await self.get(f"downloads/matrix/{matrix_type}", params={"uuid": uuid})
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from matrix endpoint, got {type(res)}")
        return res

    async def download_analysis(
        self,
        uuid: str,
        download_type: str,
        job: str | None = None,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads discrete analysis CSV results (e.g. Kleborate, Kaptive) for a collection."""
        if job is None:
            details = await self.get_collection(uuid)
            downloads = details.get("downloads", [])
            for d in downloads:
                if d.get("name", "").lower() == download_type.lower():
                    job = d.get("job")
                    break
            if not job:
                raise ValueError(
                    f"Analysis download '{download_type}' not found in collection '{uuid}'. "
                    f"Available downloads: {[d.get('name') for d in downloads]}"
                )

        if genome_ids is None:
            records = await self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        ids_param = ",".join(genome_ids)
        params: dict[str, str] = {"job": job}
        if prefix:
            params["prefix"] = prefix

        res = await self.post(
            f"downloads/{download_type.lower()}",
            params=params,
            json={"ids": ids_param},
        )
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from downloads/{download_type}, got {type(res)}")
        return res

    async def download_kleborate(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads Kleborate analysis CSV for a collection."""
        return await self.download_analysis(
            uuid=uuid,
            download_type="kleborate",
            genome_ids=genome_ids,
            prefix=prefix,
        )

    async def download_kaptive(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads Kaptive analysis CSV for a collection."""
        return await self.download_analysis(
            uuid=uuid,
            download_type="kaptive",
            genome_ids=genome_ids,
            prefix=prefix,
        )

    async def download_metadata(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads isolate metadata CSV for a collection."""
        if genome_ids is None:
            records = await self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        ids_param = ",".join(genome_ids)
        params: dict[str, str] = {}
        if prefix:
            params["prefix"] = prefix

        res = await self.post(
            "downloads/metadata",
            params=params or None,
            json={"ids": ids_param},
        )
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from downloads/metadata, got {type(res)}")
        return res

    async def download_summary(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Downloads summary CSV for a collection."""
        if genome_ids is None:
            records = await self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        ids_param = ",".join(genome_ids)
        params: dict[str, str] = {}
        if prefix:
            params["prefix"] = prefix

        res = await self.post(
            "downloads/summary",
            params=params or None,
            json={"ids": ids_param},
        )
        if not isinstance(res, str):
            raise ValueError(f"Expected CSV string response from downloads/summary, got {type(res)}")
        return res

    async def download_collection_files(
        self,
        uuid: str,
        output_dir: str | Path,
        analysis_types: Sequence[str] = ("kleborate", "kaptive"),
        include_metadata: bool = True,
        include_matrix: bool = True,
        matrix_type: str = "difference",
        prefix: str | None = None,
    ) -> dict[str, Path]:
        """Downloads discrete collection files to an output directory on disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        async def _try_matrix_raw() -> str | None:
            if not include_matrix:
                return None
            return await self.download_distance_matrix(uuid=uuid, matrix_type=matrix_type)

        details, records, matrix_csv = await asyncio.gather(
            self.get_collection(uuid),
            self.get_collection_genomes(uuid, all_pages=True),
            _try_matrix_raw(),
        )
        genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")
        available_downloads = {d.get("name", "").lower(): d.get("job") for d in details.get("downloads", [])}

        tasks = []
        task_names = []

        for at in analysis_types:
            at_lower = at.lower()
            if at_lower in available_downloads:
                tasks.append(
                    self.download_analysis(
                        uuid=uuid,
                        download_type=at_lower,
                        job=available_downloads[at_lower],
                        genome_ids=genome_ids,
                        prefix=prefix,
                    )
                )
                task_names.append(at_lower)

        if include_metadata:
            tasks.append(self.download_metadata(uuid=uuid, genome_ids=genome_ids, prefix=prefix))
            task_names.append("metadata")

        results = list(await asyncio.gather(*tasks)) if tasks else []
        if include_matrix and matrix_csv is not None:
            task_names.append(f"{matrix_type}_matrix")
            results.append(matrix_csv)

        async def _write_csv(name: str, csv_text: str) -> tuple[str, Path]:
            fpath = out / f"{name}.csv"
            await asyncio.to_thread(fpath.write_text, csv_text, encoding="utf-8")
            return name, fpath

        written = await asyncio.gather(*(_write_csv(n, txt) for n, txt in zip(task_names, results, strict=True)))
        for name, fpath in written:
            paths[name] = fpath

        return paths

    @staticmethod
    def default_analysis_types(downloads: Sequence[dict[str, Any]]) -> tuple[str, ...]:
        """Selects the default genotype analysis types from a collection's ``downloads`` list."""
        names = [
            str(d["name"]).strip().lower()
            for d in downloads
            if isinstance(d, dict) and d.get("name") and str(d["name"]).strip()
        ]
        if "kleborate" in names:
            return ("kleborate",)
        if "kaptive" in names:
            return ("kaptive",)
        if names:
            return (names[0],)
        return ()

    # Synchronous wrappers
    def get_collections_sync(
        self,
        owner: str | None = None,
        exclude: str | None = None,
        binned: bool | None = None,
        limit: int | None = None,
    ) -> list[PathogenwatchCollection]:
        """Synchronously fetches collections available to the authenticated account or public collections."""
        return self.run_sync(self.get_collections(owner=owner, exclude=exclude, binned=binned, limit=limit))

    def get_collection_sync(self, uuid: str) -> dict[str, Any]:
        """Synchronously fetches complete metadata and analysis jobs for a collection."""
        return self.run_sync(self.get_collection(uuid))

    def get_collection_genomes_sync(
        self,
        uuid: str,
        share_id: str | None = None,
        page: int = 1,
        all_pages: bool = True,
    ) -> list[dict[str, Any]]:
        """Synchronously fetches genome analysis records for a collection."""
        return self.run_sync(self.get_collection_genomes(uuid, share_id=share_id, page=page, all_pages=all_pages))

    def download_distance_matrix_sync(self, uuid: str, matrix_type: str = "difference") -> str:
        """Synchronously downloads the pairwise genomic distance matrix CSV for a collection."""
        return self.run_sync(self.download_distance_matrix(uuid, matrix_type=matrix_type))

    def download_analysis_sync(
        self,
        uuid: str,
        download_type: str,
        job: str | None = None,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads discrete analysis CSV results for a collection."""
        return self.run_sync(
            self.download_analysis(
                uuid=uuid,
                download_type=download_type,
                job=job,
                genome_ids=genome_ids,
                prefix=prefix,
            )
        )

    def download_kleborate_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads Kleborate analysis CSV for a collection."""
        return self.run_sync(self.download_kleborate(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_kaptive_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads Kaptive analysis CSV for a collection."""
        return self.run_sync(self.download_kaptive(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_metadata_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads isolate metadata CSV for a collection."""
        return self.run_sync(self.download_metadata(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_summary_sync(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> str:
        """Synchronously downloads summary CSV for a collection."""
        return self.run_sync(self.download_summary(uuid=uuid, genome_ids=genome_ids, prefix=prefix))

    def download_collection_files_sync(
        self,
        uuid: str,
        output_dir: str | Path,
        analysis_types: Sequence[str] = ("kleborate", "kaptive"),
        include_metadata: bool = True,
        include_matrix: bool = True,
        matrix_type: str = "difference",
        prefix: str | None = None,
    ) -> dict[str, Path]:
        """Synchronously downloads collection CSV files to disk."""
        return self.run_sync(
            self.download_collection_files(
                uuid=uuid,
                output_dir=output_dir,
                analysis_types=analysis_types,
                include_metadata=include_metadata,
                include_matrix=include_matrix,
                matrix_type=matrix_type,
                prefix=prefix,
            )
        )
