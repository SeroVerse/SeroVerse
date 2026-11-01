r"""Unit tests for `seroverse.core.collections` SoA containers and ontology Spec registries."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from seroverse.core.collections import (
    Batch,
    Spec,
    SpecEnum,
    Specs,
    View,
)
from seroverse.core.types import Types


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
class MockRoleSpecs(Specs[MockRoleSpec]):
    r"""Custom `SpecContainer` compiled from `MockRoleSpec` scalar records."""

    abbreviations: Types.BytesArray
    categories: Types.Int8Array
    is_machinery: Types.BoolArray
    null_priors: Types.Float32Array


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
        def _compile_specs(cls) -> Specs[Spec]:
            return Specs.from_list([Spec(name=b"only_one")])

    with pytest.raises(ValueError, match="returned 1 specs for 2 enum members"):
        MismatchedCountEnum.specs()

    class NoUnknownEnum(SpecEnum):
        A = 0
        B = 1

    with pytest.raises(KeyError):
        NoUnknownEnum.resolve("unknown_key")
    assert isinstance(np.zeros(1, dtype=np.int32), Types.Int32Array.__value__.__origin__)
    with pytest.raises(TypeError, match="static namespace"):
        Types()
    assert Types.resolve_dtype(Types.Int32Array) == np.dtype(np.int32)
    assert Types.resolve_dtype("Int32Array") == np.dtype(np.int32)
    assert Types.resolve_dtype("Types.Int32Array") == np.dtype(np.int32)
    assert Types.resolve_dtype("int32") == np.dtype(np.int32)
    assert Types.resolve_dtype("np.int32") == np.dtype(np.int32)
    assert Types.resolve_dtype("BytesArray") == np.dtype("S")


def test_container_to_dict_chunks_sort_and_pad() -> None:
    r"""Verify `to_dict`, `chunks`, `argsort`/`sort_by`, and `RaggedArrayContainer.pad`."""
    from seroverse.core.interval import Interval, Intervals, Strand
    from seroverse.core.seq.seq import SeqRecord, SeqRecords

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
    seqs = SeqRecords.from_list(
        [
            SeqRecord(b"s1", b"\x01\x02\x03"),
            SeqRecord(b"s2", b"\x04"),
            SeqRecord(b"s3", b"\x05\x06\x07\x08\x09"),
        ]
    )
    seq_chunks = list(seqs.chunks(2))
    assert len(seq_chunks) == 2
    assert SeqRecords.concat(seq_chunks) == seqs

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
    assert SeqRecords.empty().pad().shape == (0, 0)
    with pytest.raises(ValueError, match="not a ragged field"):
        seqs.pad("lengths")

    # Support nested BatchedContainer fields while rejecting arbitrary non-array/non-container types
    @dataclass(frozen=True, slots=True)
    class _NestedItem(View):
        interval: Interval
        seq: SeqRecord
        score: int

    @dataclass(frozen=True, slots=True)
    class _NestedBatch(Batch[_NestedItem]):
        intervals: Intervals
        seqs: SeqRecords
        scores: Types.Int32Array

    items = [
        _NestedItem(Interval(10, 40, Strand.FORWARD), SeqRecord(b"a", b"ATG"), 5),
        _NestedItem(Interval(50, 90, Strand.REVERSE), SeqRecord(b"b", b"TAA"), 9),
    ]
    batch = _NestedBatch.from_list(items)
    assert len(batch) == 2
    assert batch[0] == items[0]
    assert batch[1] == items[1]
    assert len(batch[:1]) == 1
    assert batch[:1][0] == items[0]
    assert _NestedBatch.concat([batch[:1], batch[1:]]) == batch
    assert len(_NestedBatch.empty()) == 0

    with pytest.raises(TypeError, match="must be annotated as a 1D NumPy array or Batch"):

        @dataclass(frozen=True, slots=True)
        class _InvalidScalarField(Batch[Spec]):
            bad_field: int


def test_features_and_ragged_nested_batches(tmp_path: Path) -> None:
    r"""Verify `Qualifier`, `Qualifiers`, `FeatureKey`, `Feature`, and `Features` ragged nested batch operations."""
    from seroverse.core.interval import Interval, Strand
    from seroverse.core.io.gff import GffReader, GffWriter
    from seroverse.core.seq.feature import Feature, FeatureKey, Features, Qualifier, Qualifiers

    # 1. FeatureKey SpecEnum resolution
    assert FeatureKey.from_name("CDS") is FeatureKey.CDS
    assert FeatureKey.from_name(b"mobile_element") is FeatureKey.MOBILE_ELEMENT
    assert FeatureKey.from_name("mobile-element") is FeatureKey.MOBILE_ELEMENT
    assert FeatureKey.resolve("unrecognized_feature") is FeatureKey.UNKNOWN

    # 2. Qualifiers ragged batch and key lookups
    q1 = Qualifiers.from_mapping(
        {
            b"locus_tag": b"b0001",
            "product": "O-antigen flippase Wzx",
            b"db_xref": [b"InterPro:IPR001", b"GO:0005886"],
        }
    )
    assert len(q1) == 4
    assert q1[0] == Qualifier(b"locus_tag", b"b0001")
    assert q1[b"locus_tag"] == Qualifier(b"locus_tag", b"b0001")
    assert q1["product"] == Qualifier(b"product", b"O-antigen flippase Wzx")
    assert q1.get(b"locus_tag") == b"b0001"
    assert q1.get("product") == b"O-antigen flippase Wzx"
    assert q1.get(b"missing", b"fallback") == b"fallback"
    assert q1.get_all(b"db_xref") == [b"InterPro:IPR001", b"GO:0005886"]
    assert b"locus_tag" in q1
    assert "product" in q1
    assert b"nonexistent" not in q1
    assert Qualifier(b"locus_tag", b"b0001") in q1
    assert q1.to_multidict() == {
        b"locus_tag": [b"b0001"],
        b"product": [b"O-antigen flippase Wzx"],
        b"db_xref": [b"InterPro:IPR001", b"GO:0005886"],
    }
    with pytest.raises(KeyError):
        _ = q1[b"nonexistent"]

    # 3. Features with ragged nested Qualifiers and 1-to-1 nested Intervals
    f0 = Feature(
        key=FeatureKey.GENE,
        interval=Interval(100, 1300, Strand.FORWARD),
        qualifiers=Qualifiers.from_mapping({b"gene": b"wzx"}),
    )
    f1 = Feature(
        key=FeatureKey.CDS,
        interval=Interval(100, 1300, Strand.FORWARD),
        qualifiers=q1,
    )
    f2 = Feature(
        key=FeatureKey.REPEAT_REGION,
        interval=Interval(20, 80, Strand.REVERSE),
    )

    feats = Features.from_list([f0, f1, f2])
    assert len(feats) == 3
    assert feats[0] == f0
    assert feats[1] == f1
    assert feats[2] == f2
    assert feats[1].start == 100
    assert feats[1].end == 1300
    assert feats[1].strand is Strand.FORWARD
    assert len(feats[1]) == 1200
    assert feats[1].get_qualifier("locus_tag") == b"b0001"

    # Contiguous slicing, non-contiguous fancy indexing, filtering, sorting, and concat
    assert feats[1:3][0] == f1
    assert feats[1:3][1] == f2
    assert feats[[2, 0]][0] == f2
    assert feats[[2, 0]][1] == f0
    assert Features.concat([feats[:1], feats[1:2], feats[2:]]) == feats

    cds_only = feats.filter_by_key(FeatureKey.CDS)
    assert len(cds_only) == 1
    assert cds_only[0] == f1

    sorted_feats = feats.sort_by("starts", "keys")
    assert sorted_feats[0] == f2
    assert sorted_feats[1] == f1
    assert sorted_feats[2] == f0

    with pytest.raises(TypeError, match="Cannot pad nested container field"):
        feats.pad()

    # 4. GFF3 serialization and GffReader/GffWriter streaming/batch/file round-trip
    f_parent = Feature(
        key=FeatureKey.BIOSYNTHETIC_GENE_CLUSTER,
        interval=Interval(0, 1500, Strand.FORWARD),
        qualifiers=Qualifiers.from_mapping(
            {
                b"ID": b"locus_1",
                b"Name": b"CPS;O1=K1",
                b"score": b"42.50",
            }
        ),
    )
    all_feats = Features.from_list([f_parent, f0, f1, f2])
    gff_doc = all_feats.to_gff3(seqid=b"contig_1", source=b"SeroRef")
    assert gff_doc.startswith(b"##gff-version 3\ncontig_1\tSeroRef\tbiosynthetic_gene_cluster\t1\t1500\t42.50\t+\t.\t")
    assert b"Name=CPS%3BO1%3DK1" in gff_doc

    parsed_batch = GffReader.from_bytes(gff_doc + b"##FASTA\n>contig_1\nACGT\n", chunk_size=19).load()
    assert parsed_batch == all_feats
    assert parsed_batch[0].get_qualifier(b"Name") == b"CPS;O1=K1"
    assert parsed_batch[0].get_qualifier(b"score") == b"42.50"

    gff_gz_path = tmp_path / "annotations.gff3.gz"
    GffWriter.to_file(gff_gz_path, all_feats, seqid=b"contig_1", source=b"SeroRef")
    assert GffReader.from_file(gff_gz_path).load() == all_feats


def test_types_njit_eager_signatures() -> None:
    """Verify @Types.njit derives eager Numba signatures and enforces concrete annotations."""

    @Types.njit
    def _scale_into(src: Types.Float64Array, dst: Types.Float64Array, factor: float) -> float:
        total = 0.0
        for i in range(len(src)):
            val = src[i] * factor
            dst[i] = val
            total += val
        return total

    # Eagerly compiled at decoration time before any call
    assert len(getattr(_scale_into, "overloads", {})) == 1

    # Read-only input array accepted alongside writable output array
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    x.setflags(write=False)
    y = np.zeros(3, dtype=np.float64)
    total = _scale_into(x, y, 2.5)
    assert np.isclose(total, 15.0)
    np.testing.assert_allclose(y, [2.5, 5.0, 7.5])

    # Missing or unresolvable annotations raise TypeError at decoration time
    with pytest.raises(TypeError, match="Cannot derive Numba signature"):

        @Types.njit
        def _unresolvable_ann(a: object, b: int) -> int:
            return b
