"""Streaming and batch client for the UniProt REST API (`uniprotkb`, `uniref`, `uniparc`)."""

import zlib
from collections.abc import Iterator
from collections.abc import Sequence as PySequence
from dataclasses import dataclass
from io import RawIOBase
from itertools import islice
from typing import Any, Literal

from seroverse.client.base import Client
from seroverse.core.io.fasta import FastaReader
from seroverse.core.seq.seq import Sequence, Sequences

UNIPROT_API_URL = "https://rest.uniprot.org"

type UniProtDatabase = Literal["uniprotkb", "uniref", "uniparc"]


class _ChunkStream(RawIOBase):
    """Readable binary stream adapter over a byte-chunk iterator."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = chunks

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            return b"".join(self._chunks)
        for chunk in self._chunks:
            if chunk:
                return chunk
        return b""

    def close(self) -> None:
        if callable(close_fn := getattr(self._chunks, "close", None)):
            close_fn()
        super().close()


def _iter_maybe_gzip_chunks(chunks: Iterator[bytes]) -> Iterator[bytes]:
    """Transparently decompress a gzip-encoded chunk stream if the gzip magic header (`\\x1f\\x8b`) is present."""
    first = b""
    for chunk in chunks:
        if chunk:
            first = chunk
            break
    if not first:
        return

    if not first.startswith(b"\x1f\x8b"):
        yield first
        for chunk in chunks:
            if chunk:
                yield chunk
        return

    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = decompressor.decompress(first)
    if out:
        yield out
    for chunk in chunks:
        if chunk:
            out = decompressor.decompress(chunk)
            if out:
                yield out
    tail = decompressor.flush()
    if tail:
        yield tail


@dataclass(frozen=True, slots=True)
class UniProtEntry:
    """Immutable summary record for a UniProtKB, UniRef, or UniParc entry."""

    accession: str = ""
    id: str = ""
    protein_name: str = ""
    gene_names: tuple[str, ...] = ()
    organism_name: str = ""
    organism_id: int | None = None
    length: int = 0
    reviewed: bool = False
    is_fragment: bool = False
    sequence: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniProtEntry:
        """Construct a [`UniProtEntry`][seroverse.client.uniprot.UniProtEntry] from a UniProt REST JSON dictionary."""
        accession = str(
            data.get("primaryAccession") or data.get("id") or data.get("uniParcId") or data.get("accession") or ""
        )
        entry_id = str(data.get("uniProtkbId") or data.get("id") or data.get("uniParcId") or accession)

        # Protein name (UniProtKB nested `proteinDescription` or UniRef `name` / flat `protein_name`)
        protein_name = str(data.get("protein_name") or data.get("name") or "")
        if not protein_name and isinstance(desc := data.get("proteinDescription"), dict):
            rec_name = desc.get("recommendedName")
            if not isinstance(rec_name, dict):
                sub_names = desc.get("submissionNames") or desc.get("alternativeNames")
                rec_name = sub_names[0] if isinstance(sub_names, list) and sub_names else None
            if isinstance(rec_name, dict) and isinstance(full := rec_name.get("fullName"), dict):
                protein_name = str(full.get("value") or "")
            if not protein_name and desc.get("flag") == "Fragment":
                is_frag_flag = True
            else:
                is_frag_flag = desc.get("flag") == "Fragment"
        else:
            is_frag_flag = bool(data.get("is_fragment") or data.get("fragment") == "fragment")

        # Gene names
        genes_list: list[str] = []
        if isinstance(raw_genes := data.get("genes"), list):
            for g in raw_genes:
                if isinstance(g, dict):
                    if isinstance(gn := g.get("geneName"), dict) and (val := gn.get("value")):
                        genes_list.append(str(val))
                    for syn in g.get("synonyms") or ():
                        if isinstance(syn, dict) and (sval := syn.get("value")):
                            genes_list.append(str(sval))
        elif isinstance(raw_gene_names := data.get("gene_names"), (list, tuple)):
            genes_list.extend(str(x) for x in raw_gene_names if x)
        elif isinstance(raw_gene_str := data.get("gene_names"), str) and raw_gene_str.strip():
            genes_list.extend(raw_gene_str.split())

        # Organism
        organism_name = str(data.get("organism_name") or "")
        organism_id: int | None = None
        if isinstance(org := data.get("organism"), dict):
            organism_name = organism_name or str(org.get("scientificName") or org.get("commonName") or "")
            if (raw_tax := org.get("taxonId")) is not None:
                try:
                    organism_id = int(raw_tax)
                except ValueError, TypeError:
                    organism_id = None
        elif (raw_tax := data.get("organism_id")) is not None:
            try:
                organism_id = int(raw_tax)
            except ValueError, TypeError:
                organism_id = None

        # Sequence & length
        seq_str = ""
        length = 0
        if isinstance(seq_obj := data.get("sequence"), dict):
            seq_str = str(seq_obj.get("value") or "")
            try:
                length = int(seq_obj.get("length") or len(seq_str))
            except ValueError, TypeError:
                length = len(seq_str)
        elif isinstance(rep := data.get("representativeMember"), dict) and isinstance(
            rep_seq := rep.get("sequence"), dict
        ):
            seq_str = str(rep_seq.get("value") or "")
            try:
                length = int(rep_seq.get("length") or len(seq_str))
            except ValueError, TypeError:
                length = len(seq_str)
            if not organism_name:
                organism_name = str(rep.get("organismName") or "")
            if organism_id is None and (rep_tax := rep.get("organismTaxId")) is not None:
                try:
                    organism_id = int(rep_tax)
                except ValueError, TypeError:
                    organism_id = None
        else:
            if isinstance(raw_seq := data.get("sequence"), str):
                seq_str = raw_seq
            try:
                length = int(data.get("length") or len(seq_str))
            except ValueError, TypeError:
                length = len(seq_str)

        entry_type = str(data.get("entryType") or "")
        reviewed = bool(data.get("reviewed")) or (
            "reviewed" in entry_type.lower() and "unreviewed" not in entry_type.lower()
        )

        return cls(
            accession=accession,
            id=entry_id,
            protein_name=protein_name,
            gene_names=tuple(dict.fromkeys(genes_list)),
            organism_name=organism_name,
            organism_id=organism_id,
            length=length,
            reviewed=reviewed,
            is_fragment=is_frag_flag,
            sequence=seq_str,
        )


class UniProtClient(Client):
    """Streaming and batch HTTP client for the UniProt REST API (`uniprotkb`, `uniref`, `uniparc`).

    Uses `/<database>/search?size=limit` when `limit <= 500` for fast server-side page caps,
    and `/<database>/stream` when `limit > 500` or `limit is None`, streaming raw FASTA chunks
    directly through [`FastaReader`][seroverse.core.io.fasta.FastaReader].
    """

    _MAX_SEARCH_SIZE: int = 500

    def __init__(
        self,
        base_url: str = UNIPROT_API_URL,
        *,
        timeout: float = 40.0,
        max_concurrency: int = 4,
        polite_delay: float = 0.05,
        recorder: Any | None = None,
        user_agent: str = "seroverse-uniprot-client/0.1",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            user_agent=user_agent,
            timeout=timeout,
            max_concurrency=max_concurrency,
            polite_delay=polite_delay,
            recorder=recorder,
            **kwargs,
        )

    @staticmethod
    def _format_query(
        query: str,
        database: UniProtDatabase,
        include_fragments: bool,
    ) -> str:
        """Normalize a UniProt query string, appending `AND fragment:false` for `uniprotkb` when requested."""
        q = query.strip()
        if not q:
            raise ValueError("UniProt query string must not be empty.")
        if database == "uniprotkb" and not include_fragments and "fragment:" not in q.lower():
            return f"({q}) AND fragment:false"
        return q

    def _build_fasta_request(
        self,
        query: str,
        *,
        database: UniProtDatabase,
        limit: int | None,
        include_fragments: bool,
        compressed: bool,
        include_isoform: bool = False,
    ) -> tuple[str, dict[str, str]]:
        """Resolve the optimal endpoint (`search` vs `stream`) and query parameters for FASTA retrieval."""
        formatted_query = self._format_query(query, database, include_fragments)
        use_search = limit is not None and 0 < limit <= self._MAX_SEARCH_SIZE
        endpoint = f"{database}/search" if use_search else f"{database}/stream"
        params: dict[str, str] = {
            "query": formatted_query,
            "format": "fasta",
        }
        if use_search and limit is not None:
            params["size"] = str(limit)
        if compressed:
            params["compressed"] = "true"
        if database == "uniprotkb" and include_isoform:
            params["includeIsoform"] = "true"
        return endpoint, params

    def stream_sequences(
        self,
        query: str,
        *,
        database: UniProtDatabase = "uniprotkb",
        limit: int | None = None,
        include_fragments: bool = False,
        compressed: bool = False,
        include_isoform: bool = False,
        truncate_id: bool = True,
        uppercase: bool = True,
    ) -> Iterator[Sequence]:
        """Stream [`Sequence`][seroverse.core.seq.seq.Sequence] records from UniProt in $O(1)$ memory.

        Automatically uses `/<database>/search?size=limit` when `limit <= 500` and `/<database>/stream`
        when `limit > 500` or `limit is None`, closing the HTTP stream as soon as `limit` sequences
        have been yielded.

        Args:
            query: UniProt search query (e.g. `'gene:wzx AND taxonomy_id:2'`).
            database: Target UniProt database (`"uniprotkb"`, `"uniref"`, or `"uniparc"`).
            limit: Optional maximum number of sequences to yield.
            include_fragments: If `False` (default), appends `AND fragment:false` for `"uniprotkb"` queries.
            compressed: Request gzip-compressed stream from UniProt and decompress on the fly.
            include_isoform: Include isoform sequences for `"uniprotkb"` queries.
            truncate_id: If `True`, truncates FASTA header IDs at the first whitespace character.
            uppercase: If `True`, normalizes sequence bytes to uppercase ASCII.

        Yields:
            Sequence: Parsed [`Sequence`][seroverse.core.seq.seq.Sequence] record.
        """
        if limit is not None and limit <= 0:
            return
        endpoint, params = self._build_fasta_request(
            query,
            database=database,
            limit=limit,
            include_fragments=include_fragments,
            compressed=compressed,
            include_isoform=include_isoform,
        )
        chunk_gen = self.stream_chunks(
            "GET",
            endpoint,
            params=params,
            headers={"Accept": "text/plain; format=fasta"},
            allow_404=True,
        )
        raw_stream = _ChunkStream(_iter_maybe_gzip_chunks(chunk_gen))
        try:
            with FastaReader(raw_stream, truncate_id=truncate_id, uppercase=uppercase) as reader:
                records: Iterator[Sequence] = islice(reader, limit) if limit is not None else reader
                yield from records
        finally:
            raw_stream.close()
            if callable(close_fn := getattr(chunk_gen, "close", None)):
                close_fn()

    def fetch_sequences(
        self,
        query: str,
        *,
        database: UniProtDatabase = "uniprotkb",
        limit: int | None = 200,
        include_fragments: bool = False,
        compressed: bool = False,
        include_isoform: bool = False,
        truncate_id: bool = True,
        uppercase: bool = True,
    ) -> Sequences:
        """Fetch matching UniProt sequences into a contiguous [`Sequences`][seroverse.core.seq.seq.Sequences] SoA batch.

        Args:
            query: UniProt search query.
            database: Target UniProt database (`"uniprotkb"`, `"uniref"`, or `"uniparc"`).
            limit: Maximum number of sequences to return (`None` to stream all matching sequences). Defaults to `200`.
            include_fragments: If `False` (default), appends `AND fragment:false` for `"uniprotkb"` queries.
            compressed: Request gzip-compressed stream from UniProt and decompress on the fly.
            include_isoform: Include isoform sequences for `"uniprotkb"` queries.
            truncate_id: If `True`, truncates FASTA header IDs at the first whitespace character.
            uppercase: If `True`, normalizes sequence bytes to uppercase ASCII.

        Returns:
            Sequences: Packed Structure-of-Arrays container of fetched sequences.
        """
        if limit is not None and limit <= 0:
            return Sequences.empty()

        # When limit > 500, `/stream` is used; stop reading chunks immediately after `limit` records
        if limit is not None and limit > self._MAX_SEARCH_SIZE:
            items = list(
                self.stream_sequences(
                    query,
                    database=database,
                    limit=limit,
                    include_fragments=include_fragments,
                    compressed=compressed,
                    include_isoform=include_isoform,
                    truncate_id=truncate_id,
                    uppercase=uppercase,
                )
            )
            return Sequences.from_list(items)

        endpoint, params = self._build_fasta_request(
            query,
            database=database,
            limit=limit,
            include_fragments=include_fragments,
            compressed=compressed,
            include_isoform=include_isoform,
        )
        chunk_gen = self.stream_chunks(
            "GET",
            endpoint,
            params=params,
            headers={"Accept": "text/plain; format=fasta"},
            allow_404=True,
        )
        raw_stream = _ChunkStream(_iter_maybe_gzip_chunks(chunk_gen))
        try:
            with FastaReader(raw_stream, truncate_id=truncate_id, uppercase=uppercase) as reader:
                return reader.load()
        finally:
            raw_stream.close()
            if callable(close_fn := getattr(chunk_gen, "close", None)):
                close_fn()

    def get_entry(
        self,
        accession: str,
        *,
        database: UniProtDatabase = "uniprotkb",
    ) -> UniProtEntry | None:
        """Fetch a single [`UniProtEntry`][seroverse.client.uniprot.UniProtEntry] by accession (`None` on HTTP 404)."""
        clean = accession.strip()
        if not clean:
            return None
        data = self.get_data(f"{database}/{clean}.json", allow_404=True)
        if isinstance(data, dict):
            return UniProtEntry.from_dict(data)
        return None

    def search(
        self,
        query: str,
        *,
        database: UniProtDatabase = "uniprotkb",
        limit: int = 25,
        include_fragments: bool = False,
        fields: PySequence[str] | None = None,
    ) -> list[UniProtEntry]:
        """Search UniProt metadata records and return a list of [`UniProtEntry`][seroverse.client.uniprot.UniProtEntry] objects."""
        if limit <= 0:
            return []
        formatted_query = self._format_query(query, database, include_fragments)
        use_search = limit <= self._MAX_SEARCH_SIZE
        endpoint = f"{database}/search" if use_search else f"{database}/stream"
        params: dict[str, str] = {
            "query": formatted_query,
            "format": "json",
        }
        if use_search:
            params["size"] = str(limit)
        if fields:
            params["fields"] = ",".join(fields)
        data = self.get_data(endpoint, params=params, allow_404=True)
        if isinstance(data, dict) and isinstance(results := data.get("results"), list):
            return [UniProtEntry.from_dict(x) for x in results[:limit] if isinstance(x, dict)]
        return []
