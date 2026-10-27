r"""Unit tests for `seroverse.core.collections` SoA containers and ontology Spec registries."""

from dataclasses import dataclass

import numpy as np
import pytest

from seroverse.core.collections import (
    BoolArray,
    BytesArray,
    Float32Array,
    Int8Array,
    Int32Array,
    Spec,
    SpecContainer,
    SpecEnum,
)


class MockCategory(SpecEnum):
    r"""Simple `SpecEnum` using the default `SpecContainer[Spec, SpecContainer]` compilation."""

    UNKNOWN = 0
    INITIATOR = 1
    TRANSPORT_PATHWAY = 2


@dataclass(frozen=True, slots=True)
class MockRoleSpec(Spec):
    r"""Custom `Spec` subclass with secondary abbreviation lookup key and `SpecEnum` field."""

    abbreviation: bytes
    category: MockCategory
    is_machinery: bool = False
    null_prior: float = 0.005

    def lookup_keys(self) -> tuple[bytes, ...]:
        return (self.name.strip().lower(), self.abbreviation.strip().lower())


@dataclass(frozen=True, slots=True)
class MockRoleSpecs(SpecContainer[MockRoleSpec, "MockRoleSpecs"]):
    r"""Custom `SpecContainer` compiled from `MockRoleSpec` scalar records."""

    abbreviations: BytesArray
    categories: Int8Array
    is_machinery: BoolArray
    null_priors: Float32Array


class MockRole(SpecEnum):
    r"""Rich `SpecEnum` backed by `MockRoleSpecs`."""

    TRANSFERASE = 0
    FLK_TRANSPORTER = 1
    UNKNOWN = 2

    @classmethod
    def _compile_specs(cls) -> MockRoleSpecs:
        return MockRoleSpecs.from_list(
            (
                MockRoleSpec(
                    name=b"glycosyltransferase",
                    abbreviation=b"GT",
                    category=MockCategory.INITIATOR,
                    is_machinery=False,
                    null_prior=0.25,
                ),
                MockRoleSpec(
                    name=b"flippase",
                    abbreviation=b"Wzx",
                    category=MockCategory.TRANSPORT_PATHWAY,
                    is_machinery=True,
                    null_prior=0.05,
                ),
                MockRoleSpec(
                    name=b"unknown",
                    abbreviation=b"UNK",
                    category=MockCategory.UNKNOWN,
                    is_machinery=False,
                    null_prior=0.70,
                ),
            )
        )


def test_default_spec_enum() -> None:
    r"""Verify `SpecEnum` default compilation, lookup, and `UNKNOWN` fallback."""
    specs = MockCategory.specs()
    assert len(specs) == 3
    assert specs is MockCategory.specs()
    assert not specs.names.flags.writeable

    assert MockCategory.INITIATOR.spec == Spec(name=b"initiator")
    assert MockCategory.TRANSPORT_PATHWAY.name_bytes == b"transport_pathway"
    assert MockCategory.TRANSPORT_PATHWAY.name_str == "transport_pathway"

    # Case-insensitive and delimiter-normalized resolution
    assert MockCategory.from_name("INITIATOR") is MockCategory.INITIATOR
    assert MockCategory.from_name(b"transport pathway") is MockCategory.TRANSPORT_PATHWAY
    assert MockCategory.from_name("transport-pathway") is MockCategory.TRANSPORT_PATHWAY
    assert MockCategory.from_name(1) is MockCategory.INITIATOR
    assert MockCategory.from_name(MockCategory.UNKNOWN) is MockCategory.UNKNOWN

    assert MockCategory.resolve("nonexistent") is MockCategory.UNKNOWN
    assert MockCategory.resolve("nonexistent", default=MockCategory.INITIATOR) is MockCategory.INITIATOR


def test_custom_spec_container_and_enum() -> None:
    r"""Verify custom `Spec`, `SpecContainer`, and `SpecEnum` integration."""
    specs = MockRole.specs()
    assert isinstance(specs, MockRoleSpecs)
    assert len(specs) == 3

    # Verify all arrays are frozen read-only
    for field_name in specs._field_dtypes:
        arr = getattr(specs, field_name)
        assert not arr.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            arr[0] = arr[0]

    # Verify scalar materialization and automatic Enum conversion
    flk_spec = MockRole.FLK_TRANSPORTER.spec
    assert isinstance(flk_spec, MockRoleSpec)
    assert flk_spec.name == b"flippase"
    assert flk_spec.name_str == "flippase"
    assert flk_spec.abbreviation == b"Wzx"
    assert flk_spec.category is MockCategory.TRANSPORT_PATHWAY
    assert flk_spec.is_machinery is True
    assert flk_spec.null_prior == pytest.approx(0.05)

    # Verify O(1) container indexing by int, str, bytes, and slice/mask
    assert specs[1] == flk_spec
    assert specs["flippase"] == flk_spec
    assert specs[b"wzx"] == flk_spec
    assert "WZX" in specs
    assert b"glycosyltransferase" in specs
    assert "missing" not in specs
    assert specs.index("Wzx") == 1
    assert specs.resolve_index("missing", default=-1) == -1

    sliced = specs[np.array([True, False, True])]
    assert isinstance(sliced, MockRoleSpecs)
    assert len(sliced) == 2
    assert sliced[0].name == b"glycosyltransferase"
    assert sliced[1].name == b"unknown"

    # Verify SpecEnum resolution across member names, Spec names, and abbreviations
    assert MockRole.from_name("FLK_TRANSPORTER") is MockRole.FLK_TRANSPORTER
    assert MockRole.from_name("flk transporter") is MockRole.FLK_TRANSPORTER
    assert MockRole.from_name("flippase") is MockRole.FLK_TRANSPORTER
    assert MockRole.from_name(b"wzx") is MockRole.FLK_TRANSPORTER
    assert MockRole.from_name("GT") is MockRole.TRANSFERASE
    assert MockRole.resolve("nonexistent") is MockRole.UNKNOWN


def test_spec_enum_validation_errors() -> None:
    r"""Verify `SpecEnum.specs()` rejects non-contiguous IDs or mismatched spec counts."""

    class NonContiguousEnum(SpecEnum):
        A = 0
        B = 2

    with pytest.raises(ValueError, match="contiguous 0-based integers"):
        NonContiguousEnum.specs()

    class MismatchedCountEnum(SpecEnum):
        A = 0
        B = 1

        @classmethod
        def _compile_specs(cls) -> SpecContainer[Spec, SpecContainer]:
            return SpecContainer.from_list([Spec(name=b"only_one")])

    with pytest.raises(ValueError, match="returned 1 specs for 2 enum members"):
        MismatchedCountEnum.specs()

    class NoUnknownEnum(SpecEnum):
        A = 0
        B = 1

    with pytest.raises(KeyError):
        NoUnknownEnum.resolve("unknown_key")
    assert isinstance(np.zeros(1, dtype=np.int32), Int32Array.__value__.__origin__)


def test_container_to_dict_chunks_sort_and_pad() -> None:
    r"""Verify `to_dict`, `chunks`, `argsort`/`sort_by`, and `RaggedArrayContainer.pad`."""
    from seroverse.core.interval import Interval, Intervals, Strand
    from seroverse.core.seq.seq import Sequence, Sequences

    intervals = Intervals.from_list(
        [
            Interval(100, 150, Strand.REVERSE),  # len 50
            Interval(10, 80, Strand.FORWARD),  # len 70
            Interval(10, 40, Strand.FORWARD),  # len 30
            Interval(50, 120, Strand.UNSTRANDED),  # len 70
        ]
    )

    # 1. Enum coercion in _get_scalar and to_dict()
    assert intervals[0].strand is Strand.REVERSE
    d = intervals.to_dict()
    assert set(d) == {"starts", "ends", "strands"}
    assert np.array_equal(d["starts"], intervals.starts)

    # 2. chunks(size)
    chunks = list(intervals.chunks(2))
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert Intervals.concat(chunks) == intervals
    with pytest.raises(ValueError, match="positive"):
        list(intervals.chunks(0))

    # 3. sort_by / argsort (including computed property "lengths" and mixed directions)
    by_len_desc = intervals.sort_by("lengths", descending=True)
    # Tie on length 70 (idx 1 and idx 3) preserved stably; then 50 (idx 0), then 30 (idx 2)
    assert np.array_equal(by_len_desc.starts, np.array([10, 50, 100, 10], dtype=np.int32))
    assert np.array_equal(by_len_desc.lengths, np.array([70, 70, 50, 30], dtype=np.int32))

    # Multi-key sort: primary starts ascending, secondary ends descending
    by_start_asc_end_desc = intervals.sort_by("starts", "ends", descending=(False, True))
    assert np.array_equal(by_start_asc_end_desc.starts, np.array([10, 10, 50, 100], dtype=np.int32))
    assert np.array_equal(by_start_asc_end_desc.ends, np.array([80, 40, 120, 150], dtype=np.int32))

    with pytest.raises(ValueError, match="no field or property"):
        intervals.sort_by("nonexistent")
    with pytest.raises(ValueError, match="must match number of sort keys"):
        intervals.sort_by("starts", "ends", descending=(True,))

    # 4. RaggedArrayContainer.pad and chunks
    seqs = Sequences.from_list(
        [
            Sequence(b"s1", np.array([1, 2, 3], dtype=np.uint8)),
            Sequence(b"s2", np.array([4], dtype=np.uint8)),
            Sequence(b"s3", np.array([5, 6, 7, 8, 9], dtype=np.uint8)),
        ]
    )
    seq_chunks = list(seqs.chunks(2))
    assert len(seq_chunks) == 2
    assert Sequences.concat(seq_chunks) == seqs

    padded = seqs.pad(fill_value=255)
    expected = np.array(
        [
            [1, 2, 3, 255, 255],
            [4, 255, 255, 255, 255],
            [5, 6, 7, 8, 9],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(padded, expected)

    # Truncated pad
    truncated = seqs.pad("seqs", max_len=2, fill_value=0)
    assert np.array_equal(
        truncated,
        np.array([[1, 2], [4, 0], [5, 6]], dtype=np.uint8),
    )

    # Empty pad
    assert Sequences.empty().pad().shape == (0, 0)
    with pytest.raises(ValueError, match="not a ragged field"):
        seqs.pad("lengths")

    # 5. Multi-dimensional array fields (e.g. 2D feature_matrix on SpecContainer)
    @dataclass(frozen=True, slots=True)
    class FeatureSpec(Spec):
        weight: float

    @dataclass(frozen=True, slots=True)
    class FeatureSpecs(SpecContainer[FeatureSpec, "FeatureSpecs"]):
        weights: Float32Array
        feature_matrix: np.ndarray[tuple[int, int], np.dtype[np.float32]]

    empty_fs = FeatureSpecs.empty()
    assert len(empty_fs) == 0
    assert empty_fs.feature_matrix.shape == (0, 0)

    fs = FeatureSpecs(
        names=np.array([b"a", b"b"], dtype="S"),
        weights=np.array([1.0, 2.0], dtype=np.float32),
        feature_matrix=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
    )
    assert fs[1] == FeatureSpec(name=b"b", weight=2.0)
    assert fs[:1].feature_matrix.shape == (1, 3)
    assert FeatureSpecs.concat([fs[:1], fs[1:]]) == fs
