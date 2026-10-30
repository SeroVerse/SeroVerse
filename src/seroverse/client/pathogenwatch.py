"""Pathogenwatch REST API client and collection metadata models."""

import threading
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from seroverse.client.base import Client
from seroverse.client.progress import current_progress

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


@dataclass(frozen=True, slots=True)
class PathogenwatchCollectionBundle:
    """In-memory collection download bundle from Pathogenwatch.

    Attributes:
        uuid: Short UUID identifier of the collection.
        name: Resolved collection name.
        details: Raw collection details dictionary (`downloads`, `trees`, `organismName`).
        records: Paginated genome records tuple.
        analyses: Tuple of `(download_type, csv_bytes)` pairs in requested order.
        metadata_csv: Downloaded isolate metadata CSV bytes, or `None`.
        matrix_csv: Downloaded pairwise distance matrix CSV bytes, or `None`.
        fallback_newick: First tree Newick byte string from `details['trees']` if available.
    """

    uuid: str
    name: str
    details: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    analyses: tuple[tuple[str, bytes], ...]
    metadata_csv: bytes | None = None
    matrix_csv: bytes | None = None
    fallback_newick: bytes = b""


class PathogenwatchClient(Client):
    """Client for interacting with the Pathogenwatch REST API."""

    DEFAULT_BASE_URL: ClassVar[str] = PATHOGENWATCH_API_URL

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        *,
        max_concurrency: int = 8,
        polite_delay: float = 0.0,
        recorder: Any | None = None,
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
            user_agent="seroverse-pathogenwatch-client/0.1",
            timeout=timeout,
            max_concurrency=max_concurrency,
            polite_delay=polite_delay,
            recorder=recorder,
        )

    def get_collections(
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

        res = self.get_data("collections/list", params=params or None)
        if isinstance(res, list):
            return [PathogenwatchCollection.from_dict(c) for c in res]
        return []

    def get_collection(self, uuid: str) -> dict[str, Any]:
        """Fetches complete metadata and analysis jobs for a specific collection."""
        res = self.get_data("collections/details", params={"uuid": uuid})
        return res if isinstance(res, dict) else {}

    def get_collection_genomes(
        self,
        uuid: str,
        share_id: str | None = None,
        page: int = 1,
        all_pages: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetches genome analysis records for a collection."""
        all_records: list[dict[str, Any]] = []
        current_page = page
        prog = current_progress()

        while True:
            params: dict[str, str] = {"uuid": uuid, "page": str(current_page)}
            if share_id:
                params["shareId"] = share_id

            res = self.get_data("collections/genomes", params=params)
            if not isinstance(res, dict):
                break

            records = res.get("genomes") or res.get("genome") or []
            all_records.extend(records)

            if not all_pages or not res.get("hasMore") or not records:
                break

            if prog.is_active:
                prog.report(
                    min(0.95, 1.0 - 1.0 / (current_page + 1.0)),
                    detail=f"Fetched page {current_page} ({len(all_records):,} genomes)...",
                )
            current_page += 1

        return all_records

    def download_distance_matrix(
        self,
        uuid: str,
        matrix_type: str = "difference",
    ) -> bytes:
        """Downloads pairwise genomic distance matrix CSV bytes for a collection."""
        return self.stream_bytes("GET", f"downloads/matrix/{matrix_type}", params={"uuid": uuid})

    def download_analysis(
        self,
        uuid: str,
        download_type: str,
        job: str | None = None,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> bytes:
        """Downloads discrete analysis CSV bytes (e.g. Kleborate, Kaptive) for a collection."""
        if job is None:
            details = self.get_collection(uuid)
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
            records = self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        ids_param = ",".join(genome_ids)
        params: dict[str, str] = {"job": job}
        if prefix:
            params["prefix"] = prefix

        return self.stream_bytes(
            "POST",
            f"downloads/{download_type.lower()}",
            params=params,
            json={"ids": ids_param},
        )

    def download_kleborate(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> bytes:
        """Downloads Kleborate analysis CSV bytes for a collection."""
        return self.download_analysis(
            uuid=uuid,
            download_type="kleborate",
            genome_ids=genome_ids,
            prefix=prefix,
        )

    def download_kaptive(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> bytes:
        """Downloads Kaptive analysis CSV bytes for a collection."""
        return self.download_analysis(
            uuid=uuid,
            download_type="kaptive",
            genome_ids=genome_ids,
            prefix=prefix,
        )

    def _download_by_ids(
        self,
        endpoint: str,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
        extra_params: dict[str, str] | None = None,
    ) -> bytes:
        if genome_ids is None:
            records = self.get_collection_genomes(uuid, all_pages=True)
            genome_ids = [str(r["id"]) for r in records if "id" in r]
        if not genome_ids:
            raise ValueError(f"No genomes found for collection '{uuid}'.")

        params: dict[str, str] = dict(extra_params) if extra_params else {}
        if prefix:
            params["prefix"] = prefix

        return self.stream_bytes(
            "POST",
            endpoint,
            params=params or None,
            json={"ids": ",".join(genome_ids)},
        )

    def download_metadata(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> bytes:
        """Downloads isolate metadata CSV bytes for a collection."""
        return self._download_by_ids("downloads/metadata", uuid, genome_ids=genome_ids, prefix=prefix)

    def download_summary(
        self,
        uuid: str,
        genome_ids: Sequence[str] | None = None,
        prefix: str | None = None,
    ) -> bytes:
        """Downloads summary CSV bytes for a collection."""
        return self._download_by_ids("downloads/summary", uuid, genome_ids=genome_ids, prefix=prefix)

    def fetch_collection_bundle(
        self,
        uuid: str,
        *,
        with_distances: bool = True,
        with_metadata: bool = True,
        matrix_type: str = "difference",
        dataset_name: str | None = None,
        strict_distances: bool = False,
        analysis_types: Sequence[str] | str | None = None,
    ) -> PathogenwatchCollectionBundle:
        """Downloads collection details, genomes, analyses, metadata, and distance matrix in memory."""
        prog = current_progress()
        prog.report(
            0.05,
            message="Fetching Pathogenwatch collection",
            detail=f"Metadata and isolate list for {uuid}...",
            force=True,
        )

        def _try_matrix() -> tuple[bytes | None, Exception | None]:
            if not with_distances:
                return None, None
            try:
                return self.download_distance_matrix(uuid=uuid, matrix_type=matrix_type), None
            except Exception as exc:
                return None, exc

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            matrix_fut = pool.submit(_try_matrix)
            try:
                with prog.subrange(0.05, 0.25, message="Fetching Pathogenwatch collection"):
                    details_fut = pool.submit(self.get_collection, uuid)
                    records_fut = pool.submit(self.get_collection_genomes, uuid, all_pages=True)
                    details = details_fut.result()
                    records = records_fut.result()
            except Exception:
                matrix_fut.cancel()
                raise

            resolved_name = dataset_name or str(details.get("name") or "") or f"Pathogenwatch {uuid}"
            genome_ids = [str(r["id"]) for r in records if "id" in r]
            if not genome_ids:
                matrix_fut.cancel()
                raise ValueError(f"Collection '{resolved_name}' ({uuid}) contains no genomes.")

            raw_downloads = details.get("downloads", [])
            available_downloads = {
                str(d.get("name", "")).strip().lower(): d.get("job")
                for d in raw_downloads
                if isinstance(d, dict) and d.get("name")
            }

            if analysis_types is None:
                requested_analyses = [a for a in ("kleborate", "kaptive") if a in available_downloads][:1]
            elif isinstance(analysis_types, str):
                clean_a = analysis_types.strip().lower()
                requested_analyses = [clean_a] if clean_a and clean_a in available_downloads else []
            else:
                requested_analyses = []
                for a in analysis_types:
                    clean_a = str(a).strip().lower()
                    if clean_a and clean_a in available_downloads and clean_a not in requested_analyses:
                        requested_analyses.append(clean_a)

            prog.report(
                0.25,
                message=f"Downloading '{resolved_name}' ({len(genome_ids):,} isolates)",
                detail="Requesting genotypes, metadata, and distance matrix...",
                force=True,
            )
            completed_downloads = 0
            total_downloads = max(1, len(requested_analyses)) + int(with_metadata) + int(with_distances)
            lock = threading.Lock()

            def _bump_download(stage_label: str) -> None:
                nonlocal completed_downloads
                with lock:
                    completed_downloads += 1
                    frac = 0.25 + 0.75 * (completed_downloads / max(1, total_downloads))
                prog.report(frac, message=f"Downloading '{resolved_name}'", detail=stage_label, force=True)

            def _fetch_single_analysis(kind: str) -> tuple[str, bytes]:
                raw_bytes = self.download_analysis(
                    uuid=uuid,
                    download_type=kind,
                    job=available_downloads[kind],
                    genome_ids=genome_ids,
                )
                _bump_download(f"Genotype calls ({kind.capitalize()}) downloaded")
                return kind, raw_bytes

            def _fetch_meta() -> bytes | None:
                if with_metadata:
                    res = self.download_metadata(uuid=uuid, genome_ids=genome_ids)
                    _bump_download("Spatiotemporal metadata downloaded")
                    return res
                return None

            trees_list = details.get("trees") or []
            fallback_newick = (
                str(trees_list[0].get("newick", "")).strip().encode("utf-8")
                if (isinstance(trees_list, list) and trees_list and isinstance(trees_list[0], dict))
                else b""
            )

            def _fetch_distances() -> bytes | None:
                res, exc = matrix_fut.result()
                if not with_distances:
                    return None
                if exc is not None:
                    if strict_distances:
                        raise exc
                    if fallback_newick:
                        _bump_download("Using collection Newick tree for pairwise distances")
                        return None
                    warnings.warn(
                        f"Could not download '{matrix_type}' distance matrix for Pathogenwatch collection "
                        f"'{uuid}' ({exc}); proceeding without pairwise distances.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    _bump_download("Distance matrix unavailable (skipped)")
                    return None
                _bump_download("Pairwise distance matrix downloaded")
                return res

            analysis_futs = [pool.submit(_fetch_single_analysis, k) for k in requested_analyses]
            if not requested_analyses:
                _bump_download("No genotype export required")
            meta_fut = pool.submit(_fetch_meta)
            dist_fut = pool.submit(_fetch_distances)

            geno_items = [f.result() for f in analysis_futs]
            meta_csv = meta_fut.result()
            dist_csv = dist_fut.result()

        if not geno_items and meta_csv is None:
            raise ValueError(f"Collection '{uuid}' has no genotype or metadata exports available.")

        return PathogenwatchCollectionBundle(
            uuid=uuid,
            name=resolved_name,
            details=details,
            records=tuple(records),
            analyses=tuple(geno_items),
            metadata_csv=meta_csv,
            matrix_csv=dist_csv,
            fallback_newick=fallback_newick,
        )

    def download_collection_files(
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
        prog = current_progress()
        prog.report(0.05, message=f"Downloading collection {uuid}", detail="Fetching collection details...")

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            details_fut = pool.submit(self.get_collection, uuid)
            records_fut = pool.submit(self.get_collection_genomes, uuid, all_pages=True)
            matrix_fut = (
                pool.submit(self.download_distance_matrix, uuid=uuid, matrix_type=matrix_type)
                if include_matrix
                else None
            )
            details = details_fut.result()
            records = records_fut.result()
            matrix_csv = matrix_fut.result() if matrix_fut is not None else None

            genome_ids = [str(r["id"]) for r in records if "id" in r]
            if not genome_ids:
                raise ValueError(f"No genomes found for collection '{uuid}'.")
            available_downloads = {d.get("name", "").lower(): d.get("job") for d in details.get("downloads", [])}

            futs = []
            task_names = []
            for at in analysis_types:
                at_lower = at.lower()
                if at_lower in available_downloads:
                    futs.append(
                        pool.submit(
                            self.download_analysis,
                            uuid=uuid,
                            download_type=at_lower,
                            job=available_downloads[at_lower],
                            genome_ids=genome_ids,
                            prefix=prefix,
                        )
                    )
                    task_names.append(at_lower)

            if include_metadata:
                futs.append(pool.submit(self.download_metadata, uuid=uuid, genome_ids=genome_ids, prefix=prefix))
                task_names.append("metadata")

            prog.report(0.35, message=f"Downloading collection {uuid}", detail="Downloading analysis exports...")
            results = [f.result() for f in futs]

        if include_matrix and matrix_csv is not None:
            task_names.append(f"{matrix_type}_matrix")
            results.append(matrix_csv)

        for name, csv_bytes in zip(task_names, results, strict=True):
            fpath = out / f"{name}.csv"
            fpath.write_bytes(csv_bytes)
            paths[name] = fpath

        prog.report(1.0, message=f"Downloaded collection {uuid}", detail=f"{len(paths)} files written", force=True)
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
