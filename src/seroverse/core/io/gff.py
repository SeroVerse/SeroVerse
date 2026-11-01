"""High-performance C-level and Structure-of-Arrays GFF3 I/O."""

from collections.abc import Iterator
from io import BufferedIOBase, RawIOBase
from typing import IO, ClassVar

import numpy as np

from seroverse.core.interval import Interval, Strand
from seroverse.core.io.base import BaseReader, BaseWriter
from seroverse.core.seq.feature import Feature, FeatureKey, Features, Qualifier, Qualifiers

_HEX_BYTES: frozenset[int] = frozenset(b"0123456789ABCDEFabcdef")
_GFF3_META_KEYS: frozenset[bytes] = frozenset({b"score", b"phase"})
_GFF3_KEY_ESCAPE: dict[int, bytes] = {
    ord("%"): b"%25",
    ord(";"): b"%3B",
    ord("="): b"%3D",
    ord("&"): b"%26",
    ord(","): b"%2C",
    ord("\t"): b"%09",
    ord("\n"): b"%0A",
    ord("\r"): b"%0D",
}
_FEATURE_KEY_GFF_TYPES: dict[FeatureKey, bytes] = {
    FeatureKey.UNKNOWN: b"region",
    FeatureKey.CDS: b"CDS",
    FeatureKey.GENE: b"gene",
    FeatureKey.TRNA: b"tRNA",
    FeatureKey.RRNA: b"rRNA",
    FeatureKey.TMRNA: b"tmRNA",
    FeatureKey.NCRNA: b"ncRNA",
    FeatureKey.MISC_RNA: b"misc_RNA",
    FeatureKey.REGULATORY: b"regulatory",
    FeatureKey.REPEAT_REGION: b"repeat_region",
    FeatureKey.MOBILE_ELEMENT: b"mobile_element",
    FeatureKey.IS_ELEMENT: b"IS_element",
    FeatureKey.MISC_FEATURE: b"misc_feature",
    FeatureKey.SOURCE: b"source",
    FeatureKey.REGION: b"region",
    FeatureKey.BIOSYNTHETIC_GENE_CLUSTER: b"biosynthetic_gene_cluster",
}


def _norm_bytes(val: bytes | str | np.bytes_) -> bytes:
    r"""Normalize a string or byte-like value to raw UTF-8 `bytes`."""
    if isinstance(val, str):
        return val.encode("utf-8")
    return bytes(val)


def _gff3_escape(raw: bytes) -> bytes:
    r"""Percent-encode reserved GFF3 characters (`%`, `;`, `=`, `&`, `,`, `\t`, `\n`, `\r`)."""
    if not any(b in _GFF3_KEY_ESCAPE for b in raw):
        return raw
    return b"".join(_GFF3_KEY_ESCAPE.get(b, bytes([b])) for b in raw)


def _gff3_unescape(raw: bytes) -> bytes:
    r"""Decode percent-escaped `%XX` sequences in GFF3 attribute keys or values."""
    if b"%" not in raw:
        return raw
    parts = raw.split(b"%")
    out = bytearray(parts[0])
    for part in parts[1:]:
        if len(part) >= 2 and part[0] in _HEX_BYTES and part[1] in _HEX_BYTES:
            out.append(int(part[:2], 16))
            out.extend(part[2:])
        else:
            out.append(ord("%"))
            out.extend(part)
    return bytes(out)


class GffReader(BaseReader[Feature, Features]):
    r"""High-performance stateful GFF3 stream iterator and batch loader.

    Streams binary GFF3 handles in chunks, skipping `#` comment/directive lines and stopping at
    `##FASTA` directives, yielding [`Feature`][seroverse.core.seq.feature.Feature] instances with
    0-based half-open [`Interval`][seroverse.core.interval.Interval] coordinates and parsed
    [`Qualifiers`][seroverse.core.seq.feature.Qualifiers], or loading the entire stream into a
    contiguous [`Features`][seroverse.core.seq.feature.Features] Structure-of-Arrays container
    via [`load`][seroverse.core.io.base.BaseReader.load].
    """

    _EXTENSIONS: ClassVar[tuple[str, ...]] = ("gff", "gff3", "gtf")

    __slots__ = ()

    @staticmethod
    def parse_qualifiers(raw: bytes | str) -> Qualifiers:
        r"""Parse a GFF3 column-9 attribute string into a [`Qualifiers`][seroverse.core.seq.feature.Qualifiers] batch."""
        attr = _norm_bytes(raw).strip()
        if not attr or attr == b".":
            return Qualifiers.empty()
        items: list[Qualifier] = []
        for token in attr.split(b";"):
            token = token.strip()
            if not token:
                continue
            key_raw, sep, val_raw = token.partition(b"=")
            key_b = _gff3_unescape(key_raw.strip())
            if not sep:
                items.append(Qualifier(key_b, b""))
                continue
            for sub_val in val_raw.split(b","):
                items.append(Qualifier(key_b, _gff3_unescape(sub_val)))
        return Qualifiers.from_list(items)

    @classmethod
    def _parse_line(cls, line: bytes) -> Feature | None:
        """Parse a single 9-column GFF3 line into a `Feature` (or `None` if malformed)."""
        cols = line.rstrip(b"\r").split(b"\t")
        if len(cols) < 8:
            return None
        _, _, type_col, start_col, end_col, score_col, strand_col, phase_col = cols[:8]
        attr_col = cols[8] if len(cols) >= 9 else b"."

        try:
            start_0 = max(0, int(start_col) - 1)
            end_0 = int(end_col)
        except ValueError:
            return None
        if end_0 < start_0:
            start_0, end_0 = end_0, start_0

        if strand_col == b"+":
            strand = Strand.FORWARD
        elif strand_col == b"-":
            strand = Strand.REVERSE
        else:
            strand = Strand.UNSTRANDED

        fkey = FeatureKey.resolve(type_col)
        quals = cls.parse_qualifiers(attr_col)

        extra_quals: list[Qualifier] = []
        if score_col and score_col != b"." and b"score" not in quals:
            extra_quals.append(Qualifier(b"score", score_col))
        default_phase = b"0" if fkey is FeatureKey.CDS else b"."
        if phase_col and phase_col not in (b".", default_phase) and b"phase" not in quals:
            extra_quals.append(Qualifier(b"phase", phase_col))
        if extra_quals:
            quals = Qualifiers.from_list([*quals, *extra_quals])

        return Feature(
            key=fkey,
            interval=Interval(start_0, end_0, strand),
            qualifiers=quals,
        )

    def _iter_records(self) -> Iterator[Feature]:
        """Stateful generator yielding `Feature` records across chunk boundaries."""
        buf = b""
        fasta_reached = False

        while not fasta_reached:
            chunk = self._handle.read(self._chunk_size)
            if not chunk:
                break
            buf = b"%s%s" % (buf, chunk) if buf else chunk
            lines = buf.split(b"\n")
            if len(lines) > 1:
                for raw_line in lines[:-1]:
                    stripped = raw_line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith(b"##FASTA"):
                        fasta_reached = True
                        break
                    if stripped[0] == 35:  # ord('#')
                        continue
                    feat = self._parse_line(raw_line)
                    if feat is not None:
                        yield feat
                buf = b"" if fasta_reached else lines[-1]

        if not fasta_reached and buf:
            stripped = buf.strip()
            if stripped and not stripped.startswith(b"#"):
                feat = self._parse_line(buf)
                if feat is not None:
                    yield feat


class GffWriter(BaseWriter[Feature, Features]):
    r"""High-performance binary GFF3 stream writer and batch serializer.

    Args:
        handle: Open writable binary stream (`IO[bytes]`).
        seqid: Default sequence/contig identifier for column 1 (`b"contig_1"`).
        source: Default annotation source for column 2 (`b"."`).
        include_header: If `True`, writes `##gff-version 3\n` before the first record.
        score: Optional explicit column-6 score override.
        phase: Optional explicit column-8 phase override.
    """

    _EXTENSIONS: ClassVar[tuple[str, ...]] = ("gff", "gff3", "gtf")

    __slots__ = (
        "_seqid",
        "_source",
        "_include_header",
        "_score",
        "_phase",
    )

    def __init__(
        self,
        handle: IO[bytes] | BufferedIOBase | RawIOBase,
        seqid: bytes | str = b"contig_1",
        source: bytes | str = b".",
        *,
        include_header: bool = True,
        score: bytes | str | float | None = None,
        phase: bytes | str | int | None = None,
    ) -> None:
        """Initialize the stateful GFF3 writer."""
        super().__init__(handle)
        self._seqid = _norm_bytes(seqid) or b"contig_1"
        self._source = _norm_bytes(source) or b"."
        self._include_header = include_header
        self._score = score
        self._phase = phase

    def _header(self) -> bytes:
        """Return the GFF3 version directive header when `include_header=True`."""
        return b"##gff-version 3\n" if self._include_header else b""

    @staticmethod
    def format_qualifiers(quals: Qualifiers, exclude_keys: frozenset[bytes] = _GFF3_META_KEYS) -> bytes:
        r"""Format a [`Qualifiers`][seroverse.core.seq.feature.Qualifiers] batch as a GFF3 column-9 byte string."""
        if len(quals.keys) == 0:
            return b"."
        parts: list[bytes] = []
        for k, vals in quals.to_multidict().items():
            if k in exclude_keys:
                continue
            esc_k = _gff3_escape(k)
            esc_vals = b",".join(_gff3_escape(v) for v in vals)
            parts.append(b"%s=%s" % (esc_k, esc_vals))
        return b";".join(parts) if parts else b"."

    def _format_scalar(self, item: Feature) -> bytes:
        """Format a single `Feature` as a 9-column GFF3 line ending with `\\n`."""
        if self._score is None:
            score_b = item.qualifiers.get(b"score") or b"."
        elif isinstance(self._score, (int, float)):
            score_b = f"{self._score:.2f}".encode("ascii")
        else:
            score_b = _norm_bytes(self._score)

        if self._phase is None:
            default_phase = b"0" if item.key is FeatureKey.CDS else b"."
            phase_b = item.qualifiers.get(b"phase") or default_phase
        elif isinstance(self._phase, int):
            phase_b = b"%d" % self._phase
        else:
            phase_b = _norm_bytes(self._phase)

        if item.strand is Strand.FORWARD:
            strand_b = b"+"
        elif item.strand is Strand.REVERSE:
            strand_b = b"-"
        else:
            strand_b = b"."

        start_1 = max(1, int(item.start) + 1)
        end_1 = max(start_1, int(item.end))
        attr_b = self.format_qualifiers(item.qualifiers)
        return b"%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n" % (
            self._seqid,
            self._source,
            _FEATURE_KEY_GFF_TYPES.get(item.key, b"region"),
            start_1,
            end_1,
            score_b,
            strand_b,
            phase_b,
            attr_b,
        )
