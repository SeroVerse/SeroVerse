r"""Biological sequence data structures, SoA containers, and translation utilities.

This module provides high-performance data structures for storing, manipulating, and
translating biological sequences (DNA, RNA, protein). Sequences are stored either as individual
immutable [`Sequence`][seroverse.core.seq.seq.Sequence] instances or as contiguous, flat arrays in
[`Sequences`][seroverse.core.seq.seq.Sequences] Structure-of-Arrays (SoA) containers.

Key Classes:
    - [`Sequence`][seroverse.core.seq.seq.Sequence]: Immutable container for a single sequence record.
    - [`Sequences`][seroverse.core.seq.seq.Sequences]: Memory-efficient, Numba-accelerated SoA container.
    - [`GeneticCode`][seroverse.core.seq.alphabet.GeneticCode]: NCBI genetic code translator.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, overload

import numpy as np
from numba import njit, prange

from seroverse.core.collections import (
    BoolArray,
    BytesArray,
    Int8Array,
    Int32Array,
    RaggedArrayContainer,
    ScalarView,
    UInt8Array,
    UInt64Array,
)
from seroverse.core.seq.alphabet import Alphabet, GeneticCode


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Sequence(ScalarView):
    r"""A simple, immutable container for a single biological sequence.

    Attributes:
        id (bytes): Unique identifier or name of the sequence.
        seq (bytes): Raw sequence data (e.g. DNA, RNA, or protein).
    """

    id: bytes
    seq: bytes

    def __len__(self) -> int:
        r"""Return the length of the sequence in bytes.

        Returns:
            int: Sequence length in bytes.
        """
        return len(self.seq)

    @overload
    def __getitem__(self, item: int) -> int: ...

    @overload
    def __getitem__(self, item: slice) -> bytes: ...

    def __getitem__(self, item: int | slice) -> int | bytes:
        r"""Access sequence bytes by integer index or slice.

        Args:
            item (int | slice): Integer index or slice.

        Returns:
            int | bytes: Sub-sequence slice (bytes) or single byte character (int).
        """
        return self.seq[item]

    def to_fasta(self) -> bytes:
        r"""Format the sequence record as a standard FASTA byte string.

        Returns:
            bytes: FASTA formatted byte representation of the record.
        """
        return b">%s\n%s\n" % (self.id, self.seq)

    def translate(
        self,
        code: GeneticCode = GeneticCode.bacterial,
        frame: int = 0,
        to_stop: bool = False,
    ) -> Sequence:
        r"""Translate a nucleotide sequence into a protein sequence using the specified genetic code.

        Args:
            code (GeneticCode): Genetic code translation table to use. Defaults to `GeneticCode.bacterial`.
            frame (int): Reading frame offset (0, 1, or 2). Defaults to 0.
            to_stop (bool): If True, truncates translation at the first stop codon (*). Defaults to False.

        Returns:
            Sequence: New sequence record containing the translated protein sequence.
        """
        return Sequence(self.id, code.translate(self.seq, frame=frame, to_stop=to_stop).tobytes())


@dataclass(frozen=True, slots=True)
class Sequences(RaggedArrayContainer[Sequence]):
    r"""A high-performance SoA container for biological sequences.

    Stores multiple sequences in a flat, contiguous memory layout using a 1D NumPy array
    of unsigned 8-bit integers (`np.uint8`). Individual sequences are accessed via parallel arrays
    of offsets and lengths.

    Attributes:
        ids (BytesArray): 1D byte string array of identifiers for each sequence.
        seqs (UInt8Array): Single flat 1D array containing all sequence byte data concatenated.
        offsets (Int32Array): 1D array of start indices in `seqs`.
        lengths (Int32Array): 1D array of sequence lengths.
    """

    _ragged_fields: ClassVar[tuple[str, ...]] = ("seqs",)

    ids: BytesArray
    seqs: UInt8Array  # concatenated sequences
    offsets: Int32Array  # start index per sequence
    lengths: Int32Array  # length per sequence
    _lookup_cache: dict[bytes, int] | None = field(default=None, init=False, repr=False, compare=False, hash=False)

    def _get_scalar(self, idx: int) -> Sequence:
        r"""Materialize the [`Sequence`][seroverse.core.seq.seq.Sequence] at index `idx`."""
        offset_val = int(self.offsets[idx])
        return Sequence(
            self.ids[idx],
            bytes(self.seqs.data[offset_val : offset_val + int(self.lengths[idx])]),
        )

    def to_fasta(self, use_indices: bool = False) -> bytes:
        r"""Format the entire batch as a single FASTA byte string.

        Args:
            use_indices (bool): If True, uses 0-based integer index as FASTA header (`>0`, `>1`, ...)
                instead of byte string `ids`. Defaults to False.

        Returns:
            bytes: Complete FASTA-formatted byte string containing all sequences.
        """
        if len(self.ids) == 0:
            return b""

        mv = self.seqs.data
        if use_indices:
            return b"".join(
                b">%d\n%s\n" % (i, bytes(mv[o : o + length_val]))
                for i, (o, length_val) in enumerate(zip(self.offsets, self.lengths, strict=False))
            )
        return b"".join(
            b">%s\n%s\n" % (name, bytes(mv[o : o + length_val]))
            for name, o, length_val in zip(self.ids, self.offsets, self.lengths, strict=False)
        )

    @property
    def internal_stops(self) -> BoolArray:
        r"""Vectorized check for internal stop codons in protein sequences.

        Scans each sequence for the stop codon character ('*') before the final position using a Numba kernel.

        Returns:
            BoolArray: A 1D boolean array where True indicates presence of an internal stop.
        """
        return _internal_stops_kernel(self.seqs, self.offsets, self.lengths)

    def unique(self) -> Sequences:
        r"""Return a new Sequences container containing only unique sequences.

        Uses a Numba-accelerated 64-bit FNV-1a hash to identify unique sequences while preserving
        first-occurrence order.

        Returns:
            Sequences: A new container with duplicate sequences removed.
        """
        if len(self) <= 1:
            return self

        hashes = _hash_sequences_kernel(self.seqs, self.offsets, self.lengths)
        _, unique_indices = np.unique(hashes, return_index=True)
        unique_indices.sort()

        return self[unique_indices]

    @overload
    def index(self, key_or_keys: bytes | np.bytes_) -> int: ...

    @overload
    def index(self, key_or_keys: BytesArray | Iterable[bytes]) -> Int32Array: ...

    def index(
        self,
        key_or_keys: bytes | np.bytes_ | BytesArray | Iterable[bytes],
    ) -> int | Int32Array:
        r"""Resolve byte identifier(s) to 0-based integer index or `Int32Array` of indices.

        Builds and caches an $O(1)$ hash map on first lookup (preserving first-occurrence index for duplicate IDs).

        Args:
            key_or_keys (bytes | np.bytes_ | BytesArray | Iterable[bytes]): Single sequence ID
                or array/iterable of IDs.

        Returns:
            int | Int32Array: Integer index for scalar queries or 1D `int32` array of indices for batch queries.

        Raises:
            KeyError: If any requested sequence identifier is not found.
        """
        cache = self._lookup_cache
        if cache is None:
            cache = {bytes(self.ids[i]): i for i in range(len(self.ids) - 1, -1, -1)}
            object.__setattr__(self, "_lookup_cache", cache)

        if isinstance(key_or_keys, (bytes, np.bytes_)):
            target = bytes(key_or_keys)
            idx = cache.get(target)
            if idx is None:
                raise KeyError(f"Sequence identifier {target!r} not found")
            return idx

        keys_list = [bytes(k) for k in key_or_keys]
        if not keys_list:
            return np.empty(0, dtype=np.int32)

        out = np.empty(len(keys_list), dtype=np.int32)
        missing: list[bytes] = []
        for i, k in enumerate(keys_list):
            idx = cache.get(k)
            if idx is None:
                missing.append(k)
            else:
                out[i] = idx
        if missing:
            raise KeyError(f"Sequence identifiers not found: {missing[:5]!r}")
        return out

    @overload
    def __getitem__(self, item: int | np.integer[Any] | bytes | np.bytes_) -> Sequence: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[Any, Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[Any, Any] | list[Any] | bytes | np.bytes_,
    ) -> Sequence | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | slice | np.ndarray[Any, Any] | list[Any] | bytes | np.bytes_,
    ) -> Sequence | Self:
        r"""Access sequences by index, slice, boolean mask, or byte identifier(s)."""
        if isinstance(item, (bytes, np.bytes_)):
            return self._get_scalar(self.index(item))
        if (isinstance(item, np.ndarray) and item.dtype.kind in ("S", "a")) or (
            isinstance(item, list) and len(item) > 0 and isinstance(item[0], (bytes, np.bytes_))
        ):
            return self._get_slice(self.index(item))
        return super().__getitem__(item)

    @classmethod
    def from_bytes(
        cls,
        seqs: list[bytes] | tuple[bytes, ...] | Iterable[bytes],
        ids: tuple[bytes, ...] | BytesArray | list[bytes] | None = None,
    ) -> Sequences:
        r"""Construct Sequences from a collection of raw sequence byte strings.

        Args:
            seqs (Iterable[bytes]): Sequence byte strings.
            ids (tuple[bytes, ...] | BytesArray | list[bytes] | None): Sequence identifiers.
                Defaults to integer byte strings (b"0", b"1", ...).

        Returns:
            Sequences: Newly constructed sequences container.
        """
        seqs_list = list(seqs) if not isinstance(seqs, list) else seqs
        if not seqs_list:
            return cls.empty()

        if ids is None:
            ids_arr = np.array([b"%d" % i for i in range(len(seqs_list))], dtype="S")
        elif isinstance(ids, np.ndarray) and ids.dtype.kind in ("S", "a"):
            ids_arr = ids
        else:
            ids_arr = np.array(ids, dtype="S")

        # Fast C-level concatenation of raw byte buffers
        joined_bytes = b"".join(seqs_list)
        out_seqs = np.frombuffer(joined_bytes, dtype=np.uint8).copy()
        lengths = np.fromiter((len(s) for s in seqs_list), dtype=np.int32, count=len(seqs_list))
        offsets = cls.lengths_to_offsets(lengths)
        return cls(ids_arr, out_seqs, offsets, lengths)

    def extract(
        self,
        indices: Int32Array,
        starts: Int32Array,
        ends: Int32Array,
        strands: Int8Array,
        new_ids: tuple[bytes, ...] | BytesArray | list[bytes] | None = None,
    ) -> Sequences:
        r"""Vectorized sub-sequence extraction from the batch.

        Handles coordinates and reverse-complementation across multiple sequences in parallel using Numba.

        Args:
            indices (np.ndarray): 1D array specifying parent sequence index for each extraction.
            starts (np.ndarray): 0-based start coordinates relative to parent sequence.
            ends (np.ndarray): 0-based end coordinates relative to parent sequence.
            strands (np.ndarray): Strand orientations (1 for forward, -1 for reverse-complement).
            new_ids (tuple[bytes, ...] | np.ndarray | list[bytes] | None): Identifiers for extracted sequences.
                If None, auto-generates names.

        Returns:
            Sequences: New collection of extracted sub-sequences.
        """
        if len(indices) == 0:
            return self.empty()
        if new_ids is None:
            new_ids_arr = np.array(
                [
                    b"%s_%d_%d_%d" % (self.ids[i], x, y, z)
                    for i, x, y, z in zip(indices, starts, ends, strands, strict=False)
                ],
                dtype="S",
            )
        elif isinstance(new_ids, np.ndarray) and new_ids.dtype.kind in ("S", "a"):
            new_ids_arr = new_ids
        else:
            new_ids_arr = np.array(new_ids, dtype="S")

        out_seqs, offsets, lengths = _extract_ragged_kernel(
            self.seqs,
            self.offsets,
            indices,
            starts,
            ends,
            strands,
            Alphabet.DNA.complement_lut,
        )
        return Sequences(new_ids_arr, out_seqs, offsets, lengths)

    def encode(self, alphabet: Alphabet) -> UInt8Array:
        r"""Encode sequence bytes into dense integer values [0..K-1] according to the given Alphabet.

        Args:
            alphabet (Alphabet): Alphabet containing precomputed kmer lookup table.

        Returns:
            UInt8Array: Array of dense integer indices.
        """
        return alphabet.encode(self.seqs)

    def reduce(self, alphabet: Alphabet) -> Sequences:
        r"""Vectorized reduction of sequence characters using a reduced Alphabet.

        Args:
            alphabet (Alphabet): Reduced alphabet containing precomputed lookup tables (e.g. `Alphabet.MMSEQS12`).

        Returns:
            Sequences: New collection containing reduced ASCII sequence bytes.
        """
        if len(self) == 0:
            return self.empty()
        reduced_seqs = alphabet.reduce(self.seqs)
        return Sequences(self.ids, reduced_seqs, self.offsets.copy(), self.lengths.copy())  # type: ignore

    def translate(
        self,
        code: GeneticCode = GeneticCode.bacterial,
        frames: Int8Array | None = None,
        to_stop: bool = False,
    ) -> Sequences:
        r"""Vectorized translation of nucleotide sequences into protein sequences.

        Translates sequences according to the provided [`GeneticCode`][seroverse.core.seq.alphabet.GeneticCode]
        (defaults to NCBI Translation Table 11: `GeneticCode.bacterial`).

        Args:
            code (GeneticCode): Genetic code translation table to use. Defaults to `GeneticCode.bacterial`.
            frames (Int8Array | None): Reading frame offsets (0, 1, or 2) per sequence.
                Defaults to frame 0 for all.
            to_stop (bool): If True, truncates translation at the first stop codon (*). Defaults to False.

        Returns:
            Sequences: New collection containing translated protein sequences.
        """
        if len(self) == 0:
            return self.empty()
        out_seqs, offsets, lengths = code.translate_ragged(
            self.seqs, self.offsets, self.lengths, frames=frames, to_stop=to_stop
        )
        return Sequences(self.ids, out_seqs, offsets, lengths)

    @classmethod
    def random(
        cls,
        alphabet: Alphabet = Alphabet.DNA,
        n_seqs: int = 10,
        length: int | tuple[int, int] = (50, 250),
        rng: np.random.Generator | int | None = None,
    ) -> Self:
        r"""Generate a batch of random sequences sampled uniformly from an [`Alphabet`][seroverse.core.seq.alphabet.Alphabet].

        Args:
            alphabet (Alphabet): Alphabet whose primary symbols are sampled uniformly. Defaults to `Alphabet.DNA`.
            n_seqs (int): Number of sequences to generate (`n_seqs >= 0`). Defaults to `10`.
            length (int | tuple[int, int]): Either a fixed integer sequence length or an inclusive
                `(min_len, max_len)` 2-tuple sampled uniformly per sequence. Defaults to `(50, 250)`.
            rng (np.random.Generator | int | None): Optional random generator, integer seed, or `None`
                passed to `np.random.default_rng`.

        Returns:
            Self: A newly generated [`Sequences`][seroverse.core.seq.seq.Sequences] container.
        """
        if n_seqs < 0:
            raise ValueError(f"n_seqs must be >= 0; got {n_seqs}.")
        if n_seqs == 0:
            return cls.empty()

        gen = np.random.default_rng(rng)
        if isinstance(length, (int, np.integer)):
            if int(length) < 0:
                raise ValueError(f"length must be >= 0; got {length}.")
            lengths = np.full(n_seqs, int(length), dtype=np.int32)
        else:
            min_len, max_len = int(length[0]), int(length[1])
            if min_len < 0 or max_len < min_len:
                raise ValueError(f"Invalid length range ({min_len}, {max_len}); require 0 <= min_len <= max_len.")
            lengths = gen.integers(min_len, max_len + 1, size=n_seqs, dtype=np.int32)

        offsets = cls.lengths_to_offsets(lengths)
        total_len = int(offsets[-1] + lengths[-1])
        sym_bytes = np.frombuffer(alphabet.symbols, dtype=np.uint8)
        seqs = sym_bytes[gen.integers(0, len(sym_bytes), size=total_len)]
        ids = np.array([b"%d" % i for i in range(n_seqs)], dtype="S")
        return cls(ids, seqs, offsets, lengths)


# Kernels --------------------------------------------------------------------------------------------------------------
@njit(parallel=True, cache=True, nogil=True)
def _hash_sequences_kernel(
    seqs: UInt8Array,
    offsets: Int32Array,
    lengths: Int32Array,
) -> UInt64Array:
    r"""Compute a 64-bit FNV-1a hash for each sequence in parallel.

    Args:
        seqs (UInt8Array): Concatenated sequence bytes.
        offsets (Int32Array): Sequence start offsets.
        lengths (Int32Array): Sequence lengths.

    Returns:
        UInt64Array: 1D array of 64-bit hash values.
    """
    n = len(offsets)
    hashes = np.zeros(n, dtype=np.uint64)
    for i in prange(n):
        s = offsets[i]
        length_val = lengths[i]
        h = np.uint64(14695981039346656037)  # FNV offset basis
        for j in range(length_val):
            h = (h ^ np.uint64(seqs[s + j])) * np.uint64(1099511628211)  # FNV prime
        # Mix the length to further reduce collision risk
        h = (h ^ np.uint64(length_val)) * np.uint64(1099511628211)
        hashes[i] = h
    return hashes


@njit(cache=True, nogil=True, parallel=True)
def _extract_ragged_kernel(
    seqs: UInt8Array,
    offsets: Int32Array,
    indices: Int32Array,
    starts: Int32Array,
    ends: Int32Array,
    strands: Int8Array,
    comp_map: UInt8Array,
) -> tuple[UInt8Array, Int32Array, Int32Array]:
    r"""Parallel Numba kernel for multi-sequence sub-region extraction and reverse-complementation.

    Args:
        seqs (UInt8Array): Concatenated parent sequence array.
        offsets (Int32Array): Parent sequence start offsets.
        indices (Int32Array): Index of parent sequence for each extraction region.
        starts (Int32Array): 0-based extraction start coordinates.
        ends (Int32Array): 0-based extraction end coordinates.
        strands (Int8Array): Strand orientations (1 forward, -1 reverse).
        comp_map (UInt8Array): Complement character lookup map.

    Returns:
        tuple[UInt8Array, Int32Array, Int32Array]:
            Tuple of (extracted concatenated sequence bytes, new offsets, new lengths).
    """
    n = len(indices)
    out_lengths = np.empty(n, dtype=np.int32)
    total_len = 0
    for i in range(n):
        out_lengths[i] = ends[i] - starts[i]
        total_len += out_lengths[i]

    out_offsets = np.empty(n, dtype=offsets.dtype)
    if n > 0:
        out_offsets[0] = 0
        for i in range(1, n):
            out_offsets[i] = out_offsets[i - 1] + out_lengths[i - 1]

    out_seqs = np.empty(total_len, dtype=seqs.dtype)
    for i in prange(n):
        length_val = out_lengths[i]
        if length_val == 0:
            continue

        t_idx = indices[i]
        g_s = offsets[t_idx] + starts[i]
        g_e = offsets[t_idx] + ends[i]
        st = strands[i]
        out_s = out_offsets[i]

        if st >= 0:
            for c in range(length_val):
                out_seqs[out_s + c] = seqs[g_s + c]
        else:
            for c in range(length_val):
                out_seqs[out_s + c] = comp_map[seqs[g_e - 1 - c]]
    return out_seqs, out_offsets, out_lengths


@njit(cache=True, nogil=True, parallel=True)
def _internal_stops_kernel(
    seqs: UInt8Array,
    offsets: Int32Array,
    lengths: Int32Array,
) -> BoolArray:
    r"""Parallel Numba kernel for detecting internal stop codons in protein sequences.

    Args:
        seqs (np.ndarray): Concatenated protein sequence bytes.
        offsets (np.ndarray): Sequence start offsets.
        lengths (np.ndarray): Sequence lengths.

    Returns:
        np.ndarray[np.bool_]: 1D boolean array indicating presence of internal stop codons.
    """
    n = len(offsets)
    out = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        s = offsets[i]
        length_val = lengths[i]
        if length_val > 0:  # Check up to the second-to-last character (s + length_val - 1)
            for j in range(s, s + length_val - 1):
                if seqs[j] == 42:  # 42 is ord('*')
                    out[i] = True
                    break
    return out
