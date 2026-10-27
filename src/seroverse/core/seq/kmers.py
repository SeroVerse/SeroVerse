r"""K-mer indexing, FracMinHash, and Randstrobe sketch construction and search.

This module provides sequence sketch indices ([`BaseKmerIndex`][seroverse.core.seq.kmers.BaseKmerIndex],
[`FracMinHashIndex`][seroverse.core.seq.kmers.FracMinHashIndex],
[`RandstrobeIndex`][seroverse.core.seq.kmers.RandstrobeIndex]) and Numba JIT parallel search kernels
emitting [`Seeds`][seroverse.core.align.seeds.Seeds].
"""

from dataclasses import dataclass, replace
from typing import Self

import numpy as np
from numba import njit, prange

from seroverse.core.align.score import SubstitutionMatrix, _score_ungapped_diagonal
from seroverse.core.align.seeds import Seeds
from seroverse.core.seq.alphabet import Alphabet
from seroverse.core.seq.seq import Sequences

type _SketchSoA = tuple[
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
]


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class BaseKmerIndex:
    r"""Data-Oriented Structure-of-Arrays (SoA) container for sequence k-mer sketches.

    Attributes:
        hashes (np.ndarray[tuple[int], np.dtype[np.uint64]]): 1D array of 64-bit sketch hashes.
        seq_indices (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of source sequence indices.
        positions (np.ndarray[tuple[int], np.dtype[np.uint32]]): 1D array of start positions in sequences.
        offsets (np.ndarray[tuple[int], np.dtype[np.int32]]): 1D array of sequence record start offsets.
        lengths (np.ndarray[tuple[int], np.dtype[np.int32]]): 1D array of record counts per sequence.
        n_seqs (int): Total number of indexed sequences.
        is_sorted (bool): True if records are sorted by hash.
        k (int): K-mer length parameter.
        alphabet (Alphabet): Alphabet used for sequence encoding.
        reduce (bool): True if using a reduced alphabet mapping.
        lut (np.ndarray[tuple[int], np.dtype[np.uint8]] | None): Optional custom lookup table.
    """

    hashes: np.ndarray[tuple[int], np.dtype[np.uint64]]
    seq_indices: np.ndarray[tuple[int], np.dtype[np.uint32]]
    positions: np.ndarray[tuple[int], np.dtype[np.uint32]]
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]]
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]]
    n_seqs: int = 0
    is_sorted: bool = False
    k: int = 10
    alphabet: Alphabet = Alphabet.DNA
    reduce: bool = False
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None

    def __len__(self) -> int:
        r"""Return the number of indexed sequences."""
        return self.n_seqs

    @property
    def total_records(self) -> int:
        r"""Total number of k-mer / sketch records across all sequences."""
        return len(self.hashes)

    @property
    def indptr(self) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
        r"""CSR-compatible $(N + 1)$-length index pointer array (`[0, *cumsum(lengths)]`)."""
        n = len(self.offsets)
        if n == 0:
            return np.zeros(1, dtype=np.uint32)
        out = np.empty(n + 1, dtype=np.uint32)
        out[:n] = self.offsets
        out[n] = self.offsets[-1] + self.lengths[-1]
        return out

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty index instance using subclass field defaults."""
        return cls(
            hashes=np.empty(0, dtype=np.uint64),
            seq_indices=np.empty(0, dtype=np.uint32),
            positions=np.empty(0, dtype=np.uint32),
            offsets=np.empty(0, dtype=np.int32),
            lengths=np.empty(0, dtype=np.int32),
        )

    def _extract_soa(self, batch: Sequences) -> _SketchSoA:
        r"""Extract `(offsets, lengths, hashes, seq_indices, positions)` SoA arrays from `batch`."""
        raise NotImplementedError

    def _populate(self, batch: Sequences, *, sort_by_hash: bool = False) -> Self:
        r"""Populate a new index instance from `batch` using this instance's configured parameters."""
        if len(batch) == 0:
            return replace(
                self,
                hashes=np.empty(0, dtype=np.uint64),
                seq_indices=np.empty(0, dtype=np.uint32),
                positions=np.empty(0, dtype=np.uint32),
                offsets=np.empty(0, dtype=np.int32),
                lengths=np.empty(0, dtype=np.int32),
                n_seqs=0,
                is_sorted=False,
            )
        out_offsets, counts, out_hashes, out_seq_indices, out_positions = self._extract_soa(batch)
        if len(out_hashes) == 0:
            return replace(
                self,
                hashes=out_hashes,
                seq_indices=out_seq_indices,
                positions=out_positions,
                offsets=out_offsets,
                lengths=counts,
                n_seqs=0,
                is_sorted=False,
            )
        if sort_by_hash:
            out_hashes, out_seq_indices, out_positions = _radix_sort_soa(out_hashes, out_seq_indices, out_positions)
        return replace(
            self,
            hashes=out_hashes,
            seq_indices=out_seq_indices,
            positions=out_positions,
            offsets=out_offsets,
            lengths=counts,
            n_seqs=len(batch),
            is_sorted=sort_by_hash,
        )

    def to_sorted(self) -> Self:
        r"""Return a copy of the index with records sorted by hash."""
        if self.is_sorted:
            return self
        sorted_h, sorted_s, sorted_p = _radix_sort_soa(self.hashes, self.seq_indices, self.positions)
        return replace(self, hashes=sorted_h, seq_indices=sorted_s, positions=sorted_p, is_sorted=True)

    def _build_queries(self, queries: Sequences) -> Self:
        r"""Build unsorted query index with parameters matching `self`."""
        return self._populate(queries, sort_by_hash=False)

    def _prep_queries(self, queries: BaseKmerIndex | Sequences) -> BaseKmerIndex:
        r"""Prepare and validate query index against target index state."""
        if not self.is_sorted:
            raise ValueError("Target index must be sorted by hash for binary search. Build it with sort_by_hash=True.")
        if isinstance(queries, Sequences):
            queries = self._build_queries(queries)
        if queries.is_sorted:
            raise ValueError("Query index must NOT be sorted by hash. Build it with sort_by_hash=False.")
        return queries

    def _prep_self_hits(
        self,
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.uint64]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint64]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
        np.ndarray[tuple[int], np.dtype[np.uint32]],
    ]:
        r"""Prepare query and target records for self-hits."""
        if self.is_sorted:
            order = np.argsort(self.seq_indices, kind="stable")
            q_offsets = _compute_query_offsets(self.seq_indices[order], self.n_seqs)
            return (
                self.hashes[order],
                self.positions[order],
                self.hashes,
                self.seq_indices,
                self.positions,
                q_offsets,
            )
        t_hashes, t_seq_indices, t_positions = _radix_sort_soa(self.hashes, self.seq_indices, self.positions)
        return self.hashes, self.positions, t_hashes, t_seq_indices, t_positions, self.indptr

    def _default_matrix(self) -> SubstitutionMatrix:
        r"""Return the default SubstitutionMatrix matching this index's alphabet."""
        return SubstitutionMatrix.DNA if self.alphabet.supports_canonical else SubstitutionMatrix.BLOSUM62

    def _search(
        self,
        queries: BaseKmerIndex | Sequences | None,
        min_score: int,
        mode: int,
        query_seqs: Sequences | None,
        target_seqs: Sequences | None,
        matrix: SubstitutionMatrix | None,
    ) -> Seeds:
        r"""Execute hash-range intersection and seed extraction (`mode`: 0=top_hits, 1=aggregated, 2=raw anchors)."""
        if (queries is not None and len(queries) == 0) or len(self.hashes) == 0:
            return Seeds.empty()

        q_seqs_eff = queries if isinstance(queries, Sequences) else query_seqs
        if queries is None:
            q_hashes, q_positions, t_hashes, t_seq_indices, t_positions, q_offsets = self._prep_self_hits()
            num_queries = self.n_seqs
            if q_seqs_eff is None and target_seqs is not None:
                q_seqs_eff = target_seqs
        else:
            queries_idx = self._prep_queries(queries)
            q_hashes, q_positions = queries_idx.hashes, queries_idx.positions
            t_hashes, t_seq_indices, t_positions = self.hashes, self.seq_indices, self.positions
            num_queries = queries_idx.n_seqs
            q_offsets = queries_idx.indptr

        has_seqs = q_seqs_eff is not None and target_seqs is not None
        mat = matrix if matrix is not None else self._default_matrix()
        empty_u8 = np.empty(0, dtype=np.uint8)
        empty_i32 = np.empty(0, dtype=np.int32)

        match_starts, match_ends = _find_hash_ranges_kernel(q_hashes, t_hashes)
        out_q, out_t, out_s, out_o, out_qp, out_tp = _search_seeds_kernel(
            q_positions,
            q_offsets,
            num_queries,
            match_starts,
            match_ends,
            t_seq_indices,
            t_positions,
            min_score,
            mode,
            has_seqs,
            q_seqs_eff.seqs if q_seqs_eff is not None else empty_u8,
            q_seqs_eff.offsets if q_seqs_eff is not None else empty_i32,
            q_seqs_eff.lengths if q_seqs_eff is not None else empty_i32,
            target_seqs.seqs if target_seqs is not None else empty_u8,
            target_seqs.offsets if target_seqs is not None else empty_i32,
            target_seqs.lengths if target_seqs is not None else empty_i32,
            mat.lut,
        )
        if len(out_q) == 0:
            return Seeds.empty()
        return Seeds(out_q, out_t, out_s, out_o, out_qp, out_tp, np.full(len(out_q), self.k, dtype=np.uint16))

    def top_hits(
        self,
        queries: BaseKmerIndex | Sequences | None = None,
        min_score: int = 1,
        *,
        query_seqs: Sequences | None = None,
        target_seqs: Sequences | None = None,
        matrix: SubstitutionMatrix | None = None,
    ) -> Seeds:
        r"""Find the single best-matching target sequence for each query sequence.

        Args:
            queries (BaseKmerIndex | Sequences | None): Query sequences or index. If None, queries itself.
            min_score (int): Minimum match score threshold. Defaults to 1.
            query_seqs (Sequences | None): Optional query sequences for ungapped diagonal tie-breaking.
            target_seqs (Sequences | None): Optional target sequences for ungapped diagonal tie-breaking.
            matrix (SubstitutionMatrix | None): Substitution matrix for ungapped diagonal scoring.

        Returns:
            Seeds: Matched [`Seeds`][seroverse.core.align.seeds.Seeds] collection.
        """
        return self._search(queries, min_score, 0, query_seqs, target_seqs, matrix)

    def hits(
        self,
        queries: BaseKmerIndex | Sequences | None = None,
        min_score: int = 1,
        *,
        aggregate: bool = True,
        query_seqs: Sequences | None = None,
        target_seqs: Sequences | None = None,
        matrix: SubstitutionMatrix | None = None,
    ) -> Seeds:
        r"""Find all matching target sequences or individual anchors for each query sequence.

        When `aggregate=True` (default), uses single-pass binary search over `self.hashes` followed
        by per-query sorted run-length grouping and modal/consensus diagonal voting to return one
        consensus [`Seed`][seroverse.core.align.seeds.Seed] per qualifying `(query, target)` pair.
        When `aggregate=False`, emits every individual k-mer / strobe anchor hit for qualifying
        `(query, target)` pairs sorted by `(query_index, target_index, t_pos, q_pos)` for direct
        consumption by [`Chainer`][seroverse.core.align.chaining.Chainer].

        Args:
            queries (BaseKmerIndex | Sequences | None): Query sequence index or raw sequences. If None, queries itself.
            min_score (int): Minimum shared k-mer count required for a `(query, target)` pair. Defaults to 1.
            aggregate (bool): If True, aggregate hits per `(query, target)` pair into a consensus diagonal seed.
                If False, return all individual `(q_pos, t_pos, span)` anchor occurrences sorted for collinear
                chaining. Defaults to True.
            query_seqs (Sequences | None): Optional query sequences for ungapped diagonal tie-breaking.
            target_seqs (Sequences | None): Optional target sequences for ungapped diagonal tie-breaking.
            matrix (SubstitutionMatrix | None): Substitution matrix for ungapped diagonal scoring.

        Returns:
            Seeds: Matched [`Seeds`][seroverse.core.align.seeds.Seeds] collection.
        """
        return self._search(queries, min_score, 1 if aggregate else 2, query_seqs, target_seqs, matrix)


@dataclass(frozen=True, slots=True, kw_only=True)
class FracMinHashIndex(BaseKmerIndex):
    r"""Specialized index for fast sequence comparisons using FracMinHash.

    Attributes:
        k (int): K-mer length parameter. Defaults to 21.
        scaled (int): FracMinHash scale factor (e.g. 100).
        canonical (bool): True if canonical k-mers (min of fwd/rev) are hashed.
        bits_per_char (int): Alphabet bits per character.
    """

    k: int = 21
    scaled: int = 100
    canonical: bool = True
    bits_per_char: int = 2

    def _extract_soa(self, batch: Sequences) -> _SketchSoA:
        r"""Extract FracMinHash SoA buffers from `batch`."""
        kernel_lut = self.lut if self.lut is not None else self.alphabet.kmer_lut
        return _extract_fracminhash_kernel(
            batch.seqs,
            batch.offsets,
            batch.lengths,
            kernel_lut,
            self.k,
            self.scaled,
            self.canonical,
            self.bits_per_char,
        )

    @classmethod
    def build(
        cls,
        batch: Sequences,
        k: int | None = None,
        alphabet: Alphabet = Alphabet.DNA,
        scaled: int = 100,
        canonical: bool | None = None,
        seed: int = 42,
        sort_by_hash: bool = False,
        lut: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None,
        bits_per_char: int | None = None,
        reduce: bool | None = None,
        **kwargs: object,
    ) -> FracMinHashIndex:
        r"""Build a FracMinHashIndex from sequence batch.

        Args:
            batch (Sequences): Input sequence collection.
            k (int | None): K-mer length. Defaults to alphabet default_k.
            alphabet (Alphabet): Alphabet instance. Defaults to `Alphabet.DNA`.
            scaled (int): Sampling scale factor. Defaults to 100.
            canonical (bool | None): True if using canonical k-mers. Defaults to alphabet.supports_canonical.
            seed (int): Seed for hashing. Defaults to 42.
            sort_by_hash (bool): True to sort output records by hash. Defaults to False.
            lut (np.ndarray | None): Optional custom lookup table.
            bits_per_char (int | None): Bits per character. Defaults to alphabet.bits_per_char.
            reduce (bool | None): True to reduce alphabet.
            **kwargs (object): Additional parameters.

        Returns:
            FracMinHashIndex: Built [`FracMinHashIndex`][seroverse.core.seq.kmers.FracMinHashIndex] instance.
        """
        if canonical is True and not alphabet.supports_canonical:
            raise ValueError(f"Canonical k-mers are not supported for alphabet {alphabet}.")

        resolved_canonical = alphabet.supports_canonical if canonical is None else canonical
        resolved_k = alphabet.default_k if k is None else k
        resolved_bits = alphabet.bits_per_char if bits_per_char is None else bits_per_char
        resolved_reduce = alphabet is Alphabet.MMSEQS12 if reduce is None else reduce

        if resolved_k * resolved_bits > 64:
            raise ValueError(
                f"k-mer size ({resolved_k}) with {resolved_bits} bits/char exceeds 64-bit integer capacity."
            )

        template = replace(
            cls.empty(),
            k=resolved_k,
            alphabet=alphabet,
            scaled=scaled,
            canonical=resolved_canonical,
            bits_per_char=resolved_bits,
            reduce=resolved_reduce,
            lut=lut,
        )
        return template._populate(batch, sort_by_hash=sort_by_hash)


@dataclass(frozen=True, slots=True, kw_only=True)
class RandstrobeIndex(BaseKmerIndex):
    r"""Specialized index for fast sequence comparisons using syncmer-linked randstrobes.

    Attributes:
        alphabet (Alphabet): Alphabet used for encoding. Defaults to `Alphabet.MMSEQS12`.
        s (int): Sub-k-mer size for open syncmer evaluation. Defaults to 5.
        w_min (int): Minimum syncmer offset window bound. Defaults to 1.
        w_max (int): Maximum syncmer offset window bound. Defaults to 5.
        reduce (bool): True if using a reduced alphabet mapping. Defaults to True.
    """

    alphabet: Alphabet = Alphabet.MMSEQS12
    s: int = 5
    w_min: int = 1
    w_max: int = 5
    reduce: bool = True

    def _extract_soa(self, batch: Sequences) -> _SketchSoA:
        r"""Extract Randstrobe SoA buffers from `batch`."""
        kernel_lut = self.lut if self.lut is not None else self.alphabet.kmer_lut
        return _extract_randstrobes_kernel(
            batch.seqs, batch.offsets, batch.lengths, kernel_lut, self.k, self.s, self.w_min, self.w_max
        )

    @classmethod
    def build(
        cls,
        batch: Sequences,
        k: int = 10,
        s: int = 5,
        w_min: int = 1,
        w_max: int = 5,
        alphabet: Alphabet = Alphabet.MMSEQS12,
        seed: int = 42,
        sort_by_hash: bool = False,
        lut: np.ndarray[tuple[int], np.dtype[np.uint8]] | None = None,
        reduce: bool | None = None,
        **kwargs: object,
    ) -> RandstrobeIndex:
        r"""Build a RandstrobeIndex from sequence batch.

        Args:
            batch (Sequences): Input sequence collection.
            k (int): K-mer length. Defaults to 10.
            s (int): Sub-k-mer size. Defaults to 5.
            w_min (int): Min syncmer window bound. Defaults to 1.
            w_max (int): Max syncmer window bound. Defaults to 5.
            alphabet (Alphabet): Alphabet instance. Defaults to `Alphabet.MMSEQS12`.
            seed (int): Random seed. Defaults to 42.
            sort_by_hash (bool): True to sort records by hash. Defaults to False.
            lut (np.ndarray | None): Optional alphabet lookup table.
            reduce (bool | None): True to reduce alphabet.
            **kwargs (object): Additional parameters.

        Returns:
            RandstrobeIndex: Constructed [`RandstrobeIndex`][seroverse.core.seq.kmers.RandstrobeIndex].

        Raises:
            ValueError: If `s >= k`.
        """
        if s >= k:
            raise ValueError("Sub-k-mer size (s) must be strictly less than k-mer size (k).")

        resolved_reduce = alphabet is Alphabet.MMSEQS12 if reduce is None else reduce
        template = replace(
            cls.empty(), k=k, s=s, w_min=w_min, w_max=w_max, alphabet=alphabet, reduce=resolved_reduce, lut=lut
        )
        return template._populate(batch, sort_by_hash=sort_by_hash)


# Kernels --------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, inline="always")
def _splitmix64(x: np.uint64) -> np.uint64:
    r"""Fast, invertible 64-bit integer mixing function for randomizing hashes."""
    x = np.uint64(x + np.uint64(0x9E3779B97F4A7C15))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@njit(cache=True, nogil=True)
def _radix_sort_soa(
    hashes: np.ndarray,
    seq_indices: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""In-place radix sort of SoA arrays by 64-bit hash values."""
    n = len(hashes)
    src_h, dst_h = hashes.copy(), np.empty(n, dtype=np.uint64)
    src_s, dst_s = seq_indices.copy(), np.empty(n, dtype=np.uint32)
    src_p, dst_p = positions.copy(), np.empty(n, dtype=np.uint32)
    hist = np.zeros(256, dtype=np.uint32)
    pos = np.zeros(256, dtype=np.uint32)

    for p in range(8):
        shift = np.uint64(p * 8)
        hist.fill(0)
        for i in range(n):
            hist[(src_h[i] >> shift) & np.uint64(0xFF)] += 1

        curr_pos = np.uint32(0)
        for i in range(256):
            pos[i] = curr_pos
            curr_pos += hist[i]

        for i in range(n):
            byte_val = (src_h[i] >> shift) & np.uint64(0xFF)
            write_idx = pos[byte_val]
            dst_h[write_idx] = src_h[i]
            dst_s[write_idx] = src_s[i]
            dst_p[write_idx] = src_p[i]
            pos[byte_val] += 1

        src_h, dst_h = dst_h, src_h
        src_s, dst_s = dst_s, src_s
        src_p, dst_p = dst_p, src_p

    return src_h, src_s, src_p


@njit(cache=True, nogil=True)
def _alloc_sketch_buffers(
    counts: np.ndarray[tuple[int], np.dtype[np.int32]],
) -> tuple[
    int,
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
]:
    r"""Compute sequence offsets and allocate flat SoA sketch output arrays inside Numba."""
    n_seqs = len(counts)
    out_offsets = np.zeros(n_seqs, dtype=np.int32)
    total = 0
    for i in range(n_seqs):
        out_offsets[i] = total
        total += int(counts[i])
    return (
        total,
        out_offsets,
        np.empty(total, dtype=np.uint64),
        np.empty(total, dtype=np.uint32),
        np.empty(total, dtype=np.uint32),
    )


@njit(cache=True, nogil=True, inline="always")
def _scan_minhashes_seq(
    seqs: np.ndarray,
    seq_start: int,
    seq_len: int,
    lut: np.ndarray,
    k: int,
    scaled: int,
    canonical: bool,
    bits_per_char: int,
    seq_idx: int,
    out_s: int,
    out_hashes: np.ndarray,
    out_seq_indices: np.ndarray,
    out_positions: np.ndarray,
    populate: bool,
) -> int:
    r"""Scan rolling k-mers for a single sequence; count or populate records."""
    if seq_len < k:
        return 0

    total_bits = bits_per_char * k
    mask = ~np.uint64(0) if total_bits >= 64 else (np.uint64(1) << np.uint64(total_bits)) - np.uint64(1)
    max_val = np.uint64(1) << np.uint64(bits_per_char)
    threshold = ~np.uint64(0) // np.uint64(scaled)

    k_val_fwd = np.uint64(0)
    k_val_rev = np.uint64(0)
    valid_len = 0
    count = 0

    for j in range(seq_len):
        val = np.uint64(lut[seqs[seq_start + j]])
        if val < max_val:
            k_val_fwd = ((k_val_fwd << np.uint64(bits_per_char)) & mask) | val
            valid_len += 1
            if canonical:
                comp = val ^ np.uint64(2)
                k_val_rev = (k_val_rev >> np.uint64(bits_per_char)) | (comp << np.uint64(bits_per_char * (k - 1)))

            if valid_len >= k:
                h = _splitmix64(min(k_val_fwd, k_val_rev) if canonical else k_val_fwd)
                if h <= threshold:
                    if populate:
                        out_hashes[out_s + count] = h
                        out_seq_indices[out_s + count] = seq_idx
                        out_positions[out_s + count] = j - k + 1
                    count += 1
        else:
            valid_len = 0
            k_val_fwd = np.uint64(0)
            k_val_rev = np.uint64(0)

    return count


@njit(parallel=True, cache=True, nogil=True)
def _extract_fracminhash_kernel(
    seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
    k: int,
    scaled: int,
    canonical: bool,
    bits_per_char: int,
) -> _SketchSoA:
    r"""Two-pass parallel FracMinHash extraction returning `(offsets, lengths, hashes, seq_indices, positions)`."""
    n_seqs = len(offsets)
    counts = np.zeros(n_seqs, dtype=np.int32)
    dummy_u64, dummy_u32 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint32)

    for idx in prange(n_seqs):  # type: ignore[not-iterable]
        counts[idx] = _scan_minhashes_seq(
            seqs,
            offsets[idx],
            lengths[idx],
            lut,
            k,
            scaled,
            canonical,
            bits_per_char,
            idx,
            0,
            dummy_u64,
            dummy_u32,
            dummy_u32,
            False,
        )

    total, out_offsets, out_hashes, out_seq_indices, out_positions = _alloc_sketch_buffers(counts)
    if total == 0:
        return out_offsets, counts, out_hashes, out_seq_indices, out_positions

    for idx in prange(n_seqs):  # type: ignore[not-iterable]
        if counts[idx] > 0:
            _scan_minhashes_seq(
                seqs,
                offsets[idx],
                lengths[idx],
                lut,
                k,
                scaled,
                canonical,
                bits_per_char,
                idx,
                out_offsets[idx],
                out_hashes,
                out_seq_indices,
                out_positions,
                True,
            )
    return out_offsets, counts, out_hashes, out_seq_indices, out_positions


@njit(cache=True, nogil=True, inline="always")
def _is_open_syncmer(seqs: np.ndarray, pos: int, lut: np.ndarray, k: int, s: int) -> bool:
    r"""Return True if the k-mer starting at `pos` is an open syncmer (minimum s-mer at offset 0 or k - s)."""
    min_s_hash = ~np.uint64(0)
    min_s_idx = -1
    for j in range(k - s + 1):
        s_val = np.uint64(0)
        for c in range(s):
            s_val = s_val * np.uint64(12) + np.uint64(lut[seqs[pos + j + c]])
        h = _splitmix64(s_val)
        if h < min_s_hash:
            min_s_hash = h
            min_s_idx = j
    return min_s_idx == 0 or min_s_idx == (k - s)


@njit(cache=True, nogil=True, inline="always")
def _scan_randstrobes_seq(
    seqs: np.ndarray,
    seq_start: int,
    seq_len: int,
    lut: np.ndarray,
    k: int,
    s: int,
    w_min: int,
    w_max: int,
    seq_idx: int,
    out_s: int,
    out_hashes: np.ndarray,
    out_seq_indices: np.ndarray,
    out_positions: np.ndarray,
    populate: bool,
) -> int:
    r"""Scan open syncmers and link order-2 randstrobes for a single sequence."""
    if seq_len < k:
        return 0

    max_syncmers = seq_len - k + 1
    if not populate:
        n_syncmers = 0
        for i in range(max_syncmers):
            if _is_open_syncmer(seqs, seq_start + i, lut, k, s):
                n_syncmers += 1
        return n_syncmers - w_min if n_syncmers > w_min else 0

    syncmer_pos = np.empty(max_syncmers, dtype=np.uint32)
    syncmer_hash = np.empty(max_syncmers, dtype=np.uint64)
    n_syncmers = 0

    for i in range(max_syncmers):
        if _is_open_syncmer(seqs, seq_start + i, lut, k, s):
            syncmer_pos[n_syncmers] = i
            k_val = np.uint64(0)
            for c in range(k):
                k_val = k_val * np.uint64(12) + np.uint64(lut[seqs[seq_start + i + c]])
            syncmer_hash[n_syncmers] = _splitmix64(k_val)
            n_syncmers += 1

    out_idx = out_s
    for i in range(n_syncmers - w_min):
        h1 = syncmer_hash[i]
        min_randstrobe_hash = ~np.uint64(0)
        best_j = -1
        for j in range(i + w_min, min(i + w_max + 1, n_syncmers)):
            combined = _splitmix64(h1 ^ _splitmix64(syncmer_hash[j]))
            if combined < min_randstrobe_hash:
                min_randstrobe_hash = combined
                best_j = j
        if best_j != -1:
            out_hashes[out_idx] = min_randstrobe_hash
            out_seq_indices[out_idx] = seq_idx
            out_positions[out_idx] = syncmer_pos[i]
            out_idx += 1

    return out_idx - out_s


@njit(parallel=True, cache=True, nogil=True)
def _extract_randstrobes_kernel(
    seqs: np.ndarray[tuple[int], np.dtype[np.uint8]],
    offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    lut: np.ndarray[tuple[int], np.dtype[np.uint8]],
    k: int,
    s: int,
    w_min: int,
    w_max: int,
) -> _SketchSoA:
    r"""Two-pass parallel Randstrobe extraction returning `(offsets, lengths, hashes, seq_indices, positions)`."""
    n_seqs = len(offsets)
    counts = np.zeros(n_seqs, dtype=np.int32)
    dummy_u64, dummy_u32 = np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint32)

    for idx in prange(n_seqs):  # type: ignore[not-iterable]
        counts[idx] = _scan_randstrobes_seq(
            seqs, offsets[idx], lengths[idx], lut, k, s, w_min, 0, idx, 0, dummy_u64, dummy_u32, dummy_u32, False
        )

    total, out_offsets, out_hashes, out_seq_indices, out_positions = _alloc_sketch_buffers(counts)
    if total == 0:
        return out_offsets, counts, out_hashes, out_seq_indices, out_positions

    for idx in prange(n_seqs):  # type: ignore[not-iterable]
        if counts[idx] > 0:
            _scan_randstrobes_seq(
                seqs,
                offsets[idx],
                lengths[idx],
                lut,
                k,
                s,
                w_min,
                w_max,
                idx,
                out_offsets[idx],
                out_hashes,
                out_seq_indices,
                out_positions,
                True,
            )
    return out_offsets, counts, out_hashes, out_seq_indices, out_positions


@njit(cache=True, nogil=True)
def _compute_query_offsets(q_seq_indices: np.ndarray, num_queries: int) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
    r"""Find boundary indices for each query sequence in flat hash array."""
    q_offsets = np.zeros(num_queries + 1, dtype=np.uint32)
    n = len(q_seq_indices)
    if n == 0:
        return q_offsets

    current_query = 0
    for i in range(n):
        q = q_seq_indices[i]
        while current_query < q:
            current_query += 1
            q_offsets[current_query] = i
    while current_query < num_queries:
        current_query += 1
        q_offsets[current_query] = n
    return q_offsets


@njit(cache=True, nogil=True, inline="always")
def _unpack_diag(key: np.uint64) -> np.int32:
    r"""Unpack signed int32 diagonal from lower 32 bits of a sorted uint64 key."""
    return np.int32(np.uint32(key & np.uint64(0xFFFFFFFF)) ^ np.uint32(0x80000000))


@njit(cache=True, nogil=True, inline="always")
def _modal_diagonal_from_run(
    keys: np.ndarray[tuple[int], np.dtype[np.uint64]],
    sorted_q: np.ndarray[tuple[int], np.dtype[np.int32]],
    sorted_t: np.ndarray[tuple[int], np.dtype[np.int32]],
    run_start: int,
    run_end: int,
    tol: int,
    has_seqs: bool,
    q_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    s1: int,
    l1: int,
    t_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    s2: int,
    l2: int,
    lut: np.ndarray[tuple[int, int], np.dtype[np.int8]],
) -> tuple[np.int32, np.int32, np.int32]:
    r"""Select consensus diagonal and representative `(q_pos, t_pos)` from a sorted run of `(t_id, diag)` keys."""
    d_first = _unpack_diag(keys[run_start])
    if run_end - run_start == 1 or _unpack_diag(keys[run_end - 1]) == d_first:
        return d_first, sorted_q[run_start], sorted_t[run_start]

    best_d, best_q, best_t = d_first, sorted_q[run_start], sorted_t[run_start]
    best_votes, best_sc, best_abs = 0, -1, abs(int(d_first))
    min_span = min(12, l1, l2) if has_seqs else 12

    left, right = run_start, run_start
    for a in range(run_start, run_end):
        d_a = _unpack_diag(keys[a])
        if a > run_start and d_a == _unpack_diag(keys[a - 1]):
            continue
        while left < run_end and int(d_a) - int(_unpack_diag(keys[left])) > tol:
            left += 1
        while right < run_end and int(_unpack_diag(keys[right])) - int(d_a) <= tol:
            right += 1
        votes = right - left
        sc = _score_ungapped_diagonal(q_data, s1, l1, t_data, s2, l2, int(d_a), lut, min_span) if has_seqs else 0
        abs_d = abs(int(d_a))
        better = (
            (sc > best_sc or (sc == best_sc and (votes > best_votes or (votes == best_votes and abs_d < best_abs))))
            if has_seqs
            else (votes > best_votes or (votes == best_votes and abs_d < best_abs))
        )
        if better:
            best_sc, best_votes, best_abs = sc, votes, abs_d
            best_d, best_q, best_t = d_a, sorted_q[a], sorted_t[a]

    return best_d, best_q, best_t


@njit(parallel=True, cache=True, nogil=True)
def _find_hash_ranges_kernel(
    q_hashes: np.ndarray[tuple[int], np.dtype[np.uint64]],
    t_hashes: np.ndarray[tuple[int], np.dtype[np.uint64]],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.uint32]], np.ndarray[tuple[int], np.dtype[np.uint32]]]:
    r"""Find `[start, end)` range in sorted `t_hashes` for every hash in `q_hashes` in a single pass."""
    n_q, t_len = len(q_hashes), len(t_hashes)
    match_starts = np.empty(n_q, dtype=np.uint32)
    match_ends = np.empty(n_q, dtype=np.uint32)

    for i in prange(n_q):  # type: ignore[not-iterable]
        h = q_hashes[i]
        low, high = 0, t_len
        while low < high:
            mid = (low + high) // 2
            if t_hashes[mid] < h:
                low = mid + 1
            else:
                high = mid
        curr = low
        while curr < t_len and t_hashes[curr] == h:
            curr += 1
        match_starts[i] = np.uint32(low)
        match_ends[i] = np.uint32(curr)

    return match_starts, match_ends


@njit(cache=True, nogil=True, inline="always")
def _pack_sorted_query_hits(
    start: int,
    end: int,
    q_positions: np.ndarray,
    match_starts: np.ndarray[tuple[int], np.dtype[np.uint32]],
    match_ends: np.ndarray[tuple[int], np.dtype[np.uint32]],
    t_seq_indices: np.ndarray,
    t_positions: np.ndarray,
    by_diag: bool,
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint64]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    r"""Pack all seed matches for a single query into sorted `(keys, q_pos, t_pos)` arrays."""
    raw_count = 0
    for i in range(start, end):
        raw_count += int(match_ends[i] - match_starts[i])
    if raw_count == 0:
        return np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    keys = np.empty(raw_count, dtype=np.uint64)
    raw_q = np.empty(raw_count, dtype=np.int32)
    raw_t = np.empty(raw_count, dtype=np.int32)
    k = 0
    for i in range(start, end):
        q_p = np.int32(q_positions[i])
        for curr in range(match_starts[i], match_ends[i]):
            t_id = t_seq_indices[curr]
            t_p = np.int32(t_positions[curr])
            low_bits = (np.uint32(q_p - t_p) ^ np.uint32(0x80000000)) if by_diag else np.uint32(t_p)
            keys[k] = (np.uint64(t_id) << np.uint64(32)) | np.uint64(low_bits)
            raw_q[k] = q_p
            raw_t[k] = t_p
            k += 1

    order = np.argsort(keys)
    sorted_keys, sorted_q, sorted_t = keys[order], raw_q[order], raw_t[order]

    if not by_diag:
        for a in range(1, raw_count):
            if sorted_keys[a] == sorted_keys[a - 1] and sorted_q[a] < sorted_q[a - 1]:
                j = a
                while j > 0 and sorted_keys[j] == sorted_keys[j - 1] and sorted_q[j] < sorted_q[j - 1]:
                    sorted_q[j], sorted_q[j - 1] = sorted_q[j - 1], sorted_q[j]
                    j -= 1

    return sorted_keys, sorted_q, sorted_t


@njit(parallel=True, cache=True, nogil=True)
def _search_seeds_kernel(
    q_positions: np.ndarray,
    q_offsets: np.ndarray[tuple[int], np.dtype[np.uint32]],
    num_queries: int,
    match_starts: np.ndarray[tuple[int], np.dtype[np.uint32]],
    match_ends: np.ndarray[tuple[int], np.dtype[np.uint32]],
    t_seq_indices: np.ndarray,
    t_positions: np.ndarray,
    min_score: int,
    mode: int,
    has_seqs: bool,
    q_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    q_seq_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    q_seq_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_data: np.ndarray[tuple[int], np.dtype[np.uint8]],
    t_seq_offsets: np.ndarray[tuple[int], np.dtype[np.int32]],
    t_seq_lengths: np.ndarray[tuple[int], np.dtype[np.int32]],
    lut: np.ndarray[tuple[int, int], np.dtype[np.int8]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
]:
    r"""Unified two-pass parallel kernel for `top_hits` (`mode=0`), aggregated `hits` (`mode=1`), and raw anchors (`mode=2`)."""
    counts = np.zeros(num_queries, dtype=np.uint32)
    for q_idx in prange(num_queries):  # type: ignore[not-iterable]
        start, end = int(q_offsets[q_idx]), int(q_offsets[q_idx + 1])
        raw_count = 0
        for i in range(start, end):
            raw_count += int(match_ends[i] - match_starts[i])

        if mode == 0 and min_score <= 0:
            counts[q_idx] = np.uint32(1)
            continue
        if raw_count < min_score or raw_count == 0:
            continue
        if min_score <= 1 and mode != 1:
            counts[q_idx] = np.uint32(1 if mode == 0 else raw_count)
            continue

        t_ids = np.empty(raw_count, dtype=np.uint32)
        k = 0
        for i in range(start, end):
            for curr in range(match_starts[i], match_ends[i]):
                t_ids[k] = t_seq_indices[curr]
                k += 1
        t_ids.sort()

        count = 0
        run_start = 0
        while run_start < raw_count:
            t_id = t_ids[run_start]
            run_end = run_start + 1
            while run_end < raw_count and t_ids[run_end] == t_id:
                run_end += 1
            run_len = run_end - run_start
            if run_len >= min_score:
                if mode == 0:
                    count = 1
                    break
                count += 1 if mode == 1 else run_len
            run_start = run_end
        counts[q_idx] = np.uint32(count)

    out_offsets = np.zeros(num_queries, dtype=np.uint32)
    total = 0
    for i in range(num_queries):
        out_offsets[i] = np.uint32(total)
        total += int(counts[i])

    if total == 0:
        empty_u32, empty_i32 = np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.int32)
        return empty_u32, empty_u32, empty_u32, empty_i32, empty_i32, empty_i32

    out_q = np.zeros(total, dtype=np.uint32)
    out_t = np.zeros(total, dtype=np.uint32)
    out_s = np.ones(total, dtype=np.uint32) if mode == 2 else np.zeros(total, dtype=np.uint32)
    out_o = np.zeros(total, dtype=np.int32)
    out_qp = np.zeros(total, dtype=np.int32)
    out_tp = np.zeros(total, dtype=np.int32)

    for q_idx in prange(num_queries):  # type: ignore[not-iterable]
        if counts[q_idx] == 0:
            continue
        out_idx = int(out_offsets[q_idx])
        start, end = int(q_offsets[q_idx]), int(q_offsets[q_idx + 1])
        keys, sorted_q, sorted_t = _pack_sorted_query_hits(
            start, end, q_positions, match_starts, match_ends, t_seq_indices, t_positions, mode != 2
        )
        n_keys = len(keys)
        if n_keys == 0:
            out_q[out_idx] = np.uint32(q_idx)
            continue

        s1 = int(q_seq_offsets[q_idx]) if has_seqs else 0
        l1 = int(q_seq_lengths[q_idx]) if has_seqs else 0

        best_score, best_target, best_rs, best_re = 0, np.uint32(0), 0, 0
        run_start = 0
        while run_start < n_keys:
            t_id = np.uint32(keys[run_start] >> np.uint64(32))
            run_end = run_start + 1
            while run_end < n_keys and np.uint32(keys[run_end] >> np.uint64(32)) == t_id:
                run_end += 1
            score = run_end - run_start
            if mode == 0:
                if score > best_score:
                    best_score, best_target, best_rs, best_re = score, t_id, run_start, run_end
            elif score >= min_score:
                if mode == 1:
                    s2 = int(t_seq_offsets[int(t_id)]) if has_seqs else 0
                    l2 = int(t_seq_lengths[int(t_id)]) if has_seqs else 0
                    d_val, q_val, t_val = _modal_diagonal_from_run(
                        keys, sorted_q, sorted_t, run_start, run_end, 16, has_seqs, q_data, s1, l1, t_data, s2, l2, lut
                    )
                    out_q[out_idx], out_t[out_idx], out_s[out_idx] = np.uint32(q_idx), t_id, np.uint32(score)
                    out_o[out_idx], out_qp[out_idx], out_tp[out_idx] = d_val, q_val, t_val
                    out_idx += 1
                else:
                    for a in range(run_start, run_end):
                        q_p, t_p = sorted_q[a], sorted_t[a]
                        out_q[out_idx], out_t[out_idx] = np.uint32(q_idx), t_id
                        out_o[out_idx], out_qp[out_idx], out_tp[out_idx] = q_p - t_p, q_p, t_p
                        out_idx += 1
            run_start = run_end

        if mode == 0:
            s2 = int(t_seq_offsets[int(best_target)]) if has_seqs else 0
            l2 = int(t_seq_lengths[int(best_target)]) if has_seqs else 0
            d_val, q_val, t_val = _modal_diagonal_from_run(
                keys, sorted_q, sorted_t, best_rs, best_re, 16, has_seqs, q_data, s1, l1, t_data, s2, l2, lut
            )
            out_q[out_idx], out_t[out_idx], out_s[out_idx] = np.uint32(q_idx), best_target, np.uint32(best_score)
            out_o[out_idx], out_qp[out_idx], out_tp[out_idx] = d_val, q_val, t_val

    return out_q, out_t, out_s, out_o, out_qp, out_tp
