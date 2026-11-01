r"""Genomic feature annotations, key-value qualifiers, and Structure-of-Arrays feature containers.

This module provides scalar views and ragged Structure-of-Arrays (SoA) batch containers for
representing annotated sequence features (e.g., GFF3, GenBank, EMBL) with zero-copy coordinate
and qualifier slicing:
- [`FeatureKey`][seroverse.core.seq.feature.FeatureKey]: Standardized INSDC/GFF3 feature key ontology.
- [`Qualifier`][seroverse.core.seq.feature.Qualifier] and [`Qualifiers`][seroverse.core.seq.feature.Qualifiers]:
  Scalar view and ragged SoA container for key-value feature qualifiers without fixed-width `|S{max}` padding overhead.
- [`Feature`][seroverse.core.seq.feature.Feature] and [`Features`][seroverse.core.seq.feature.Features]:
  Scalar view and ragged SoA container pairing feature keys, [`Intervals`][seroverse.core.interval.Intervals],
  and variable-length [`Qualifiers`][seroverse.core.seq.feature.Qualifiers].
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, overload

import numpy as np

from seroverse.core.collections import RaggedBatch, SpecEnum, View
from seroverse.core.interval import Interval, Intervals, Strand
from seroverse.core.types import Types


# Helpers --------------------------------------------------------------------------------------------------------------
def _norm_key_bytes(key: bytes | str | np.bytes_) -> bytes:
    r"""Normalize a qualifier key or value to raw ASCII/UTF-8 bytes."""
    if isinstance(key, str):
        return key.encode("utf-8")
    return bytes(key)


# Enums ----------------------------------------------------------------------------------------------------------------
class FeatureKey(SpecEnum):
    r"""Standardized INSDC / GFF3 biological feature key ontology.

    Inherits $O(1)$ case-insensitive and delimiter-normalized resolution from
    [`SpecEnum`][seroverse.core.collections.SpecEnum] (`FeatureKey.from_name`, `FeatureKey.resolve`).
    """

    UNKNOWN = 0
    CDS = 1
    GENE = 2
    TRNA = 3
    RRNA = 4
    TMRNA = 5
    NCRNA = 6
    MISC_RNA = 7
    REGULATORY = 8
    REPEAT_REGION = 9
    MOBILE_ELEMENT = 10
    IS_ELEMENT = 11
    MISC_FEATURE = 12
    SOURCE = 13
    REGION = 14
    BIOSYNTHETIC_GENE_CLUSTER = 15


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class Qualifier(View):
    r"""Immutable scalar view of a single `(key, value)` feature qualifier.

    Attributes:
        key (bytes): Qualifier tag name (e.g. `b"locus_tag"`, `b"product"`).
        value (bytes): Qualifier value byte string.
    """

    key: bytes
    value: bytes


@dataclass(frozen=True, slots=True)
class Qualifiers(RaggedBatch[Qualifier]):
    r"""Ragged Structure-of-Arrays (SoA) batch of `(key, value)` feature qualifiers.

    Stores qualifier `keys` as a 1D byte-string array and variable-length `values` in a flat,
    contiguous `uint8` byte buffer indexed by `offsets` and `lengths`, avoiding `|S{max}` memory
    inflation when long qualifiers (such as `/translation` or `/note`) are present.

    Attributes:
        keys (Types.BytesArray): 1D byte-string array of qualifier tag names.
        values (Types.UInt8Array): Flat concatenated `uint8` buffer of qualifier value bytes.
        offsets (Types.Int32Array): 1D array of 0-based start offsets into `values`.
        lengths (Types.Int32Array): 1D array of byte lengths in `values` per qualifier.
    """

    _ragged_fields: ClassVar[tuple[str, ...]] = ("values",)

    keys: Types.BytesArray
    values: Types.UInt8Array
    offsets: Types.Int32Array
    lengths: Types.Int32Array

    def _value_at(self, idx: int) -> bytes:
        r"""Extract the raw `bytes` value for qualifier at 0-based index `idx`."""
        off = int(self.offsets[idx])
        return bytes(self.values.data[off : off + int(self.lengths[idx])])

    def get(self, key: bytes | str, default: bytes | None = None) -> bytes | None:
        r"""Return the first qualifier value matching `key`, or `default` if not present.

        Args:
            key (bytes | str): Qualifier tag name to look up.
            default (bytes | None): Fallback value if `key` is absent. Defaults to `None`.

        Returns:
            bytes | None: Matching qualifier byte string or `default`.
        """
        if len(self.keys) == 0:
            return default
        target = _norm_key_bytes(key)
        matches = np.nonzero(self.keys == target)[0]
        if len(matches) == 0:
            return default
        return self._value_at(int(matches[0]))

    def get_all(self, key: bytes | str) -> list[bytes]:
        r"""Return all qualifier values matching `key` in order of appearance.

        Args:
            key (bytes | str): Qualifier tag name to look up.

        Returns:
            list[bytes]: List of matching qualifier values (empty list if none match).
        """
        if len(self.keys) == 0:
            return []
        target = _norm_key_bytes(key)
        matches = np.nonzero(self.keys == target)[0]
        return [self._value_at(int(i)) for i in matches]

    def __contains__(self, item: object) -> bool:
        r"""Check whether a qualifier tag `key` (`bytes | str`) or [`Qualifier`][seroverse.core.seq.feature.Qualifier] is present."""
        if isinstance(item, (bytes, str, np.bytes_)):
            if len(self.keys) == 0:
                return False
            return bool(np.any(self.keys == _norm_key_bytes(item)))
        if isinstance(item, Qualifier):
            return any(q == item for q in self)
        return False

    @overload
    def __getitem__(self, item: int | np.integer[Any] | bytes | str) -> Qualifier: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[Any, Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | bytes | str | slice | np.ndarray[Any, Any] | list[Any],
    ) -> Qualifier | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | bytes | str | slice | np.ndarray[Any, Any] | list[Any],
    ) -> Qualifier | Self:
        r"""Access a [`Qualifier`][seroverse.core.seq.feature.Qualifier] by integer index or key, or sliced `Qualifiers`."""
        if isinstance(item, (bytes, str, np.bytes_)):
            target = _norm_key_bytes(item)
            if len(self.keys) > 0:
                matches = np.nonzero(self.keys == target)[0]
                if len(matches) > 0:
                    return self._get_scalar(int(matches[0]))
            raise KeyError(item)
        return super().__getitem__(item)

    def to_multidict(self) -> dict[bytes, list[bytes]]:
        r"""Convert qualifiers into a dictionary mapping each key to all of its values in order."""
        out: dict[bytes, list[bytes]] = {}
        for i in range(len(self.keys)):
            k = bytes(self.keys[i])
            out.setdefault(k, []).append(self._value_at(i))
        return out

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[bytes | str, bytes | str | Iterable[bytes | str]]
        | list[tuple[bytes | str, bytes | str]]
        | tuple[tuple[bytes | str, bytes | str], ...],
    ) -> Self:
        r"""Construct `Qualifiers` from a dict/mapping (supporting multi-valued iterables) or `(key, value)` pairs.

        Args:
            mapping: Mapping of `key -> value | Iterable[value]` or sequence of `(key, value)` pairs.

        Returns:
            Self: Packed `Qualifiers` container.
        """
        items: list[Qualifier] = []
        if isinstance(mapping, Mapping):
            for k, v in mapping.items():
                kb = _norm_key_bytes(k)
                if isinstance(v, (bytes, str, np.bytes_)):
                    items.append(Qualifier(kb, _norm_key_bytes(v)))
                else:
                    for sub_v in v:
                        items.append(Qualifier(kb, _norm_key_bytes(sub_v)))
        else:
            for k, v in mapping:
                items.append(Qualifier(_norm_key_bytes(k), _norm_key_bytes(v)))
        return cls.from_list(items)


@dataclass(slots=True, frozen=True)
class Feature(View):
    r"""Immutable scalar view of a single annotated genomic feature.

    Attributes:
        key (FeatureKey): Standardized feature type (e.g., `FeatureKey.CDS`, `FeatureKey.GENE`).
        interval (Interval): 0-based half-open genomic coordinate span and strand.
        qualifiers (Qualifiers): Ragged batch of `(key, value)` qualifiers attached to this feature.
    """

    key: FeatureKey
    interval: Interval
    qualifiers: Qualifiers = field(default_factory=Qualifiers.empty)

    @property
    def start(self) -> int:
        r"""0-based inclusive start coordinate of the feature."""
        return self.interval.start

    @property
    def end(self) -> int:
        r"""0-based exclusive end coordinate of the feature."""
        return self.interval.end

    @property
    def strand(self) -> Strand:
        r"""Genomic strand orientation of the feature."""
        return self.interval.strand

    def __len__(self) -> int:
        r"""Length of the feature interval in base pairs (`end - start`)."""
        return len(self.interval)

    def get_qualifier(self, key: bytes | str, default: bytes | None = None) -> bytes | None:
        r"""Return the first qualifier value matching `key`, or `default` if absent."""
        return self.qualifiers.get(key, default)

    def to_gff3(
        self,
        seqid: bytes | str = b"contig_1",
        source: bytes | str = b".",
        *,
        score: bytes | str | float | None = None,
        phase: bytes | str | int | None = None,
    ) -> bytes:
        r"""Format this feature as a single 9-column GFF3 line (1-based inclusive coordinates, no trailing newline)."""
        from seroverse.core.io.gff import GffWriter

        return GffWriter.to_bytes(
            self,
            seqid=seqid,
            source=source,
            include_header=False,
            score=score,
            phase=phase,
        ).rstrip(b"\n")


@dataclass(frozen=True, slots=True)
class Features(RaggedBatch[Feature]):
    r"""Ragged Structure-of-Arrays (SoA) container of annotated genomic features.

    Pairs per-feature `keys` (`Types.UInt8Array`) and 1-to-1 nested `intervals`
    ([`Intervals`][seroverse.core.interval.Intervals]) with a concatenated ragged
    [`Qualifiers`][seroverse.core.seq.feature.Qualifiers] container indexed by `offsets` and `lengths`.

    Attributes:
        keys (Types.UInt8Array): 1D `uint8` array of [`FeatureKey`][seroverse.core.seq.feature.FeatureKey] enum codes.
        intervals (Intervals): 1-to-1 SoA container of feature intervals (`starts`, `ends`, `strands`).
        qualifiers (Qualifiers): Concatenated [`Qualifiers`][seroverse.core.seq.feature.Qualifiers] across all features.
        offsets (Types.Int32Array): 1D array of start indices into `qualifiers` per feature.
        lengths (Types.Int32Array): 1D array of qualifier counts per feature.
    """

    _ragged_fields: ClassVar[tuple[str, ...]] = ("qualifiers",)

    keys: Types.UInt8Array
    intervals: Intervals
    qualifiers: Qualifiers
    offsets: Types.Int32Array
    lengths: Types.Int32Array

    @property
    def starts(self) -> Types.Int32Array:
        r"""1D array of 0-based inclusive feature start coordinates (`self.intervals.starts`)."""
        return self.intervals.starts

    @property
    def ends(self) -> Types.Int32Array:
        r"""1D array of 0-based exclusive feature end coordinates (`self.intervals.ends`)."""
        return self.intervals.ends

    @property
    def strands(self) -> Types.Int8Array:
        r"""1D array of feature strand orientations (`self.intervals.strands`)."""
        return self.intervals.strands

    @property
    def span_lengths(self) -> Types.Int32Array:
        r"""1D array of feature coordinate span lengths in base pairs (`self.intervals.lengths`)."""
        return self.intervals.lengths

    def filter_by_key(self, *keys: FeatureKey | int | bytes | str) -> Self:
        r"""Return a subset of features matching one or more [`FeatureKey`][seroverse.core.seq.feature.FeatureKey] values.

        Args:
            *keys (FeatureKey | int | bytes | str): Feature key(s) to retain.

        Returns:
            Self: Filtered `Features` container.
        """
        if len(self) == 0 or not keys:
            return self.empty()
        codes = np.array([int(FeatureKey.from_name(k)) for k in keys], dtype=np.uint8)
        if len(codes) == 1:
            mask: Types.BoolArray = self.keys == codes[0]
        else:
            mask = np.isin(self.keys, codes)
        return self._get_slice(mask)

    def to_gff3(
        self,
        seqid: bytes | str = b"contig_1",
        source: bytes | str = b".",
        *,
        include_header: bool = True,
    ) -> bytes:
        r"""Format all features in the batch as a GFF3 document (`bytes`)."""
        from seroverse.core.io.gff import GffWriter

        return GffWriter.to_bytes(self, seqid=seqid, source=source, include_header=include_header)
