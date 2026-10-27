r"""Genomic interval representation, strand orientations, and vectorized Structure-of-Arrays operations.

This module provides discrete coordinate intervals via [`Interval`][seroverse.core.interval.Interval],
strand representations via [`Strand`][seroverse.core.interval.Strand], and high-performance,
vectorized interval collections via [`Intervals`][seroverse.core.interval.Intervals].
"""

from dataclasses import dataclass
from enum import IntEnum

from seroverse.core.collections import BatchedContainer, Int8Array, Int32Array, ScalarView


# Enums ----------------------------------------------------------------------------------------------------------------
class Strand(IntEnum):
    r"""Integer representation of genomic strand orientation.

    Supports conversion from common string formats (`'+'`, `'-'`, `'1'`, `'-1'`).

    Attributes:
        FORWARD (int): Forward strand (`+1`).
        REVERSE (int): Reverse strand (`-1`).
        UNSTRANDED (int): Unstranded or unknown orientation (`0`).
    """

    FORWARD = 1
    REVERSE = -1
    UNSTRANDED = 0

    @classmethod
    def _missing_(cls, value: object) -> Strand:
        r"""Coerce bytes or integers to a valid [`Strand`][seroverse.core.interval.Strand] instance.

        Args:
            value (object): Value to coerce (e.g. `b'+'`, `b'-'`, `b'1'`, `b'-1'`, `1`, `-1`).

        Returns:
            Strand: Corresponding [`Strand`][seroverse.core.interval.Strand] enum member.
        """
        if value in (b"+", b"1", b"+1", "+", "1", "+1"):
            return Strand.FORWARD
        if value in (b"-", b"-1", "-", "-1"):
            return Strand.REVERSE
        return Strand.UNSTRANDED

    def __bytes__(self) -> bytes:
        r"""Return the single-byte ASCII representation of the strand (`b'+'`, `b'-'`, or `b'.'`)."""
        if self == Strand.FORWARD:
            return b"+"
        if self == Strand.REVERSE:
            return b"-"
        return b"."

    def __str__(self) -> str:
        r"""Return the string representation of the strand (`'+'`, `'-'`, or `'.'`).

        Returns:
            str: Single-character string representation.
        """
        if self == Strand.FORWARD:
            return "+"
        if self == Strand.REVERSE:
            return "-"
        return "."


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Interval(ScalarView):
    r"""A single 0-based, half-open genomic interval with strand orientation.

    Represents a discrete sequence segment defined by `start` (inclusive), `end` (exclusive),
    and `strand` orientation.

    Attributes:
        start (int): 0-based starting position (inclusive).
        end (int): 0-based ending position (exclusive).
        strand (Strand): Strand orientation [`Strand`][seroverse.core.interval.Strand].
    """

    start: int
    end: int
    strand: Strand = Strand.UNSTRANDED

    def __len__(self) -> int:
        r"""Calculate the length of the interval in base pairs (`end - start`).

        Returns:
            int: Interval length.
        """
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Intervals(BatchedContainer[Interval, "Intervals"]):
    r"""A high-performance Structure-of-Arrays (SoA) collection of genomic intervals.

    Stores starts, ends, and strands as parallel 1D NumPy arrays
    for memory-efficient vectorized operations and Numba spatial algorithms.

    Attributes:
        starts (Int32Array): 1D array of 0-based start coordinates.
        ends (Int32Array): 1D array of 0-based end coordinates.
        strands (Int8Array): 1D array of strand values (+1, -1, 0).
    """

    starts: Int32Array
    ends: Int32Array
    strands: Int8Array

    @property
    def lengths(self) -> Int32Array:
        r"""Compute the lengths of all intervals in base pairs (`ends - starts`).

        Returns:
            Int32Array: 1D array of interval lengths.
        """
        return self.ends - self.starts
