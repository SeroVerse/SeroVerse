"""High-performance C-level and zero-copy NumPy FASTA I/O."""

from collections.abc import Iterator
from io import BufferedIOBase, RawIOBase
from typing import IO, ClassVar

import numpy as np

from seroverse.core.io.base import BaseReader, BaseWriter
from seroverse.core.seq.seq import SeqRecord, SeqRecords


class FastaReader(BaseReader[SeqRecord, SeqRecords]):
    r"""High-performance stateful FASTA stream iterator and vectorized batch loader.

    Streams binary FASTA handles in chunks, splitting on `\n>` boundaries and applying
    C-level `bytes.translate` to yield [`SeqRecord`][seroverse.core.seq.seq.SeqRecord] instances
    with $O(1)$ memory overhead, or loads an entire stream at once into a contiguous
    [`SeqRecords`][seroverse.core.seq.seq.SeqRecords] Structure-of-Arrays container via
    [`load`][seroverse.core.io.fasta.FastaReader.load].

    Args:
        handle: Open binary stream (`IO[bytes]`) containing FASTA sequence data.
        truncate_id: If `True`, truncates header IDs at the first whitespace character.
        uppercase: If `True`, normalizes sequence bytes to uppercase ASCII.
        chunk_size: Read buffer size in bytes for streaming iteration (default 4 MiB).
    """

    _EXTENSIONS: ClassVar[tuple[str, ...]] = ("fasta", "fa", "fna", "ffn", "fas")
    _UPPER_TABLE: ClassVar[bytes] = bytes.maketrans(b"abcdefghijklmnopqrstuvwxyz", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    _WHITESPACE: ClassVar[bytes] = b"\r\n\t "

    __slots__ = (
        "_truncate_id",
        "_uppercase",
    )

    def __init__(
        self,
        handle: IO[bytes] | BufferedIOBase | RawIOBase,
        truncate_id: bool = True,
        uppercase: bool = True,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> None:
        """Initialize the stateful FASTA reader."""
        self._truncate_id = truncate_id
        self._uppercase = uppercase
        super().__init__(handle, chunk_size=chunk_size)

    def _parse_header(self, header: bytes) -> bytes:
        """Extract sequence identifier from a raw FASTA header line (without leading `>`)."""
        if self._truncate_id:
            parts = header.split(None, 1)
            return parts[0] if parts else b""
        return header.strip()

    def _iter_records(self) -> Iterator[SeqRecord]:
        """Stateful generator yielding `SeqRecord` records across chunk boundaries."""
        table = self._UPPER_TABLE if self._uppercase else None
        buf = b""
        started = False

        while True:
            chunk = self._handle.read(self._chunk_size)
            if not chunk:
                break
            buf = b"%s%s" % (buf, chunk) if buf else chunk

            if not started:
                if buf[0] == 62:  # ord('>')
                    buf = buf[1:]
                    started = True
                else:
                    nl_gt = buf.find(b"\n>")
                    if nl_gt == -1:
                        buf = buf[-1:] if buf.endswith(b"\n") else b""
                        continue
                    buf = buf[nl_gt + 2 :]
                    started = True

            parts = buf.split(b"\n>")
            if len(parts) > 1:
                for rec in parts[:-1]:
                    header, _, seq_raw = rec.partition(b"\n")
                    yield SeqRecord(
                        id=self._parse_header(header),
                        seq=seq_raw.translate(table, self._WHITESPACE),
                    )
                buf = parts[-1]

        if started and buf:
            header, _, seq_raw = buf.partition(b"\n")
            yield SeqRecord(
                id=self._parse_header(header),
                seq=seq_raw.translate(table, self._WHITESPACE),
            )

    def load(self) -> SeqRecords:
        r"""Load the FASTA stream into memory as a contiguous [`SeqRecords`][seroverse.core.seq.seq.SeqRecords] batch.

        Uses C-level `bytes.split`, `bytes.partition`, and 256-byte lookup table `bytes.translate`
        combined with a single `bytearray.join` allocation and zero-copy `np.frombuffer` view.

        Returns:
            SeqRecords: Packed Structure-of-Arrays container of all parsed sequences.
        """
        raw = self._handle.read()
        if not raw:
            return SeqRecords.empty()

        # Locate first valid '>' at start of buffer or immediately following a newline
        if raw[0] == 62:  # ord('>')
            first_gt = 0
        else:
            nl_gt = raw.find(b"\n>")
            if nl_gt == -1:
                return SeqRecords.empty()
            first_gt = nl_gt + 1

        # Split records at line-initial '>' boundaries using C memmem
        records = raw[first_gt + 1 :].split(b"\n>")
        n = len(records)

        table = self._UPPER_TABLE if self._uppercase else None
        ids_list: list[bytes] = []
        seq_parts: list[bytes] = []

        for rec in records:
            header, _, seq_raw = rec.partition(b"\n")
            ids_list.append(self._parse_header(header))
            seq_parts.append(seq_raw.translate(table, self._WHITESPACE))

        ids = np.array(ids_list, dtype="S")
        lengths = np.fromiter((len(s) for s in seq_parts), dtype=np.int32, count=n)
        offsets = SeqRecords.lengths_to_offsets(lengths)

        # Single C-level buffer allocation + zero-copy writable uint8 NumPy view
        joined_buf = bytearray().join(seq_parts)
        seqs = np.frombuffer(joined_buf, dtype=np.uint8) if len(joined_buf) > 0 else np.empty(0, dtype=np.uint8)
        return SeqRecords(ids, seqs, offsets, lengths)


class FastaWriter(BaseWriter[SeqRecord, SeqRecords]):
    r"""High-performance binary FASTA stream writer and batch serializer.

    Args:
        handle: Open writable binary stream (`IO[bytes]`).
        use_indices: If `True`, formats headers as 0-based integer indices (`>0`, `>1`, ...)
            instead of sequence `id` byte strings.
    """

    _EXTENSIONS: ClassVar[tuple[str, ...]] = ("fasta", "fa", "fna", "ffn", "fas")

    __slots__ = (
        "_use_indices",
        "_record_index",
    )

    def __init__(
        self,
        handle: IO[bytes] | BufferedIOBase | RawIOBase,
        *,
        use_indices: bool = False,
    ) -> None:
        """Initialize the stateful FASTA writer."""
        super().__init__(handle)
        self._use_indices = use_indices
        self._record_index = 0

    def _format_scalar(self, item: SeqRecord) -> bytes:
        """Format a single `SeqRecord` as a FASTA entry."""
        if self._use_indices:
            idx = self._record_index
            self._record_index += 1
            return b">%d\n%s\n" % (idx, item.seq)
        return b">%s\n%s\n" % (item.id, item.seq)

    def _format_batch(self, batch: SeqRecords) -> bytes:
        """Format a `SeqRecords` batch directly from its contiguous `uint8` buffer."""
        if len(batch.ids) == 0:
            return b""
        mv = batch.seqs.data
        if self._use_indices:
            base = self._record_index
            self._record_index += len(batch.ids)
            return b"".join(
                b">%d\n%s\n" % (base + i, bytes(mv[int(o) : int(o) + int(length_val)]))
                for i, (o, length_val) in enumerate(zip(batch.offsets, batch.lengths, strict=False))
            )
        return b"".join(
            b">%s\n%s\n" % (bytes(name), bytes(mv[int(o) : int(o) + int(length_val)]))
            for name, o, length_val in zip(batch.ids, batch.offsets, batch.lengths, strict=False)
        )
