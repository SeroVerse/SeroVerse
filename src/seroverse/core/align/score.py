r"""Biological sequence substitution matrices and Numba-accelerated ungapped scoring kernels.

Provides [`SubstitutionMatrix`][seroverse.core.align.score.SubstitutionMatrix] with singleton caching,
read-only $256 \times 256$ ASCII lookup tables (`int8`), Karlin-Altschul statistical parameters
($\lambda, K$), and parallel Numba kernels for ungapped diagonal scoring and pair prefiltering.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self, overload

import numpy as np
from numba import njit, prange

from seroverse.core.seq.alphabet import Alphabet

_LN_2: float = float(np.log(2.0))


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SubstitutionMatrix:
    r"""Immutable biological substitution matrix with a precomputed $256 \times 256$ `int8` ASCII lookup table.

    Maps any pair of ASCII sequence bytes `(a, b)` (including lowercase bytes, alphabet aliases,
    stop codons `*`, and unknown characters) to an `int8` substitution score in $O(1)$ time.

    Attributes:
        alphabet (Alphabet): Underlying biological [`Alphabet`][seroverse.core.seq.alphabet.Alphabet].
        lut (np.ndarray): Read-only `(256, 256)` `int8` substitution score lookup table.
        lambda_ (float): Karlin-Altschul ungapped/gapped scale parameter $\lambda$.
        k (float): Karlin-Altschul parameter $K$.
        ln_k (float): Precomputed natural logarithm $\ln(K)$.
        name (bytes | None): Canonical byte-string identifier (e.g. `b"BLOSUM62"`).
    """

    alphabet: Alphabet
    lut: np.ndarray[tuple[int, int], np.dtype[np.int8]] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.int8), repr=False
    )
    lambda_: float = 0.267
    k: float = 0.041
    ln_k: float = float(np.log(0.041))
    name: bytes | None = None

    _CACHE: ClassVar[dict[Any, Self]] = {}

    BLOSUM62: ClassVar[SubstitutionMatrix]
    BLOSUM80: ClassVar[SubstitutionMatrix]
    BLOSUM45: ClassVar[SubstitutionMatrix]
    DNA: ClassVar[SubstitutionMatrix]

    def __new__(
        cls,
        alphabet: Alphabet,
        scores: np.ndarray[tuple[int, int], np.dtype[np.int8]] | Mapping[tuple[bytes, bytes], int],
        symbols: bytes | None = None,
        *,
        default_score: int = -128,
        lambda_: float = 0.267,
        k: float = 0.041,
        name: bytes | None = None,
    ) -> Self:
        r"""Create or return a cached singleton instance of a `SubstitutionMatrix`."""
        sym_bytes = (symbols if symbols is not None else alphabet.symbols).upper()
        if isinstance(scores, np.ndarray):
            score_arr = np.asarray(scores, dtype=np.int8)
            if score_arr.shape != (len(sym_bytes), len(sym_bytes)):
                raise ValueError(
                    f"Score matrix shape {score_arr.shape} must match symbol count ({len(sym_bytes)}, {len(sym_bytes)})."
                )
            score_bytes = score_arr.tobytes()
        else:
            n_s = len(sym_bytes)
            score_arr = np.full((n_s, n_s), default_score, dtype=np.int8)
            sym_idx = {b: idx for idx, b in enumerate(sym_bytes)}
            for (a_b, b_b), val in scores.items():
                ia = sym_idx[a_b.upper()[0]]
                ib = sym_idx[b_b.upper()[0]]
                score_arr[ia, ib] = val
            score_bytes = score_arr.tobytes()

        cache_key = (
            alphabet,
            sym_bytes,
            score_bytes,
            int(default_score),
            float(lambda_),
            float(k),
        )
        if cache_key in cls._CACHE:
            existing = cls._CACHE[cache_key]
            if name and not existing.name:
                object.__setattr__(existing, "name", name)
            return existing

        instance = object.__new__(cls)
        cls._CACHE[cache_key] = instance
        instance._init_matrix(
            alphabet=alphabet,
            sym_bytes=sym_bytes,
            score_arr=score_arr,
            default_score=int(default_score),
            lambda_=float(lambda_),
            k=float(k),
            name=name,
        )
        return instance

    def _init_matrix(
        self,
        alphabet: Alphabet,
        sym_bytes: bytes,
        score_arr: np.ndarray[tuple[int, int], np.dtype[np.int8]],
        default_score: int,
        lambda_: float,
        k: float,
        name: bytes | None,
    ) -> None:
        r"""Populate and lock the $256 \times 256$ ASCII lookup table."""
        lut = np.full((256, 256), default_score, dtype=np.int8)

        # Populate primary uppercase & lowercase pairs
        for ia, ba in enumerate(sym_bytes):
            ba_low = ba | 0x20
            for ib, bb in enumerate(sym_bytes):
                bb_low = bb | 0x20
                sc = score_arr[ia, ib]
                lut[ba, bb] = sc
                lut[ba_low, bb] = sc
                lut[ba, bb_low] = sc
                lut[ba_low, bb_low] = sc

        # Propagate alphabet aliases for any symbol not already explicitly in sym_bytes
        explicit = set(sym_bytes)
        if alphabet.aliases:
            for src, dst in alphabet.aliases.items():
                src_b = src[0]
                if src_b in explicit:
                    continue
                src_low = src_b | 0x20
                dst_b = dst[0]
                lut[src_b, :] = lut[dst_b, :]
                lut[:, src_b] = lut[:, dst_b]
                lut[src_b, src_b] = lut[dst_b, dst_b]
                lut[src_low, :] = lut[dst_b, :]
                lut[:, src_low] = lut[:, dst_b]
                lut[src_low, src_low] = lut[dst_b, dst_b]

        lut.flags.writeable = False

        object.__setattr__(self, "alphabet", alphabet)
        object.__setattr__(self, "lut", lut)
        object.__setattr__(self, "lambda_", lambda_)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "ln_k", float(np.log(k)))
        object.__setattr__(self, "name", name)

    @classmethod
    def from_match_mismatch(
        cls,
        alphabet: Alphabet = Alphabet.DNA,
        match: int = 2,
        mismatch: int = -3,
        *,
        unknown_score: int = -1,
        default_score: int = -128,
        lambda_: float = 0.625,
        k: float = 0.41,
        name: bytes | None = None,
    ) -> Self:
        r"""Construct a `SubstitutionMatrix` from uniform match and mismatch scores over `alphabet`.

        Args:
            alphabet (Alphabet): Target biological alphabet. Defaults to `Alphabet.DNA`.
            match (int): Score for identical characters (`a == b`). Defaults to `2`.
            mismatch (int): Score for distinct characters (`a != b`). Defaults to `-3`.
            unknown_score (int): Score for ambiguous/wildcard characters (`N` or `X`). Defaults to `-1`.
            default_score (int): Score for out-of-alphabet characters. Defaults to `-128`.
            lambda_ (float): Karlin-Altschul $\lambda$ parameter. Defaults to `0.625`.
            k (float): Karlin-Altschul $K$ parameter. Defaults to `0.41`.
            name (bytes | None): Optional byte-string name.

        Returns:
            Self: Cached `SubstitutionMatrix` instance.
        """
        wildcard = b"N" if alphabet.supports_canonical else b"X"
        sym_bytes = alphabet.symbols + (wildcard if wildcard[0] not in alphabet.symbols else b"")
        n = len(sym_bytes)
        mat = np.full((n, n), mismatch, dtype=np.int8)
        np.fill_diagonal(mat, match)
        if wildcard[0] in sym_bytes:
            w_idx = sym_bytes.index(wildcard[0])
            mat[w_idx, :] = unknown_score
            mat[:, w_idx] = unknown_score
        return cls(
            alphabet=alphabet,
            scores=mat,
            symbols=sym_bytes,
            default_score=default_score,
            lambda_=lambda_,
            k=k,
            name=name,
        )

    def __repr__(self) -> str:
        r"""Return a clean string representation of the SubstitutionMatrix."""
        if self.name:
            return f"SubstitutionMatrix.{self.name.decode('ascii', errors='replace')}"
        return f"SubstitutionMatrix(alphabet={self.alphabet!r}, lambda_={self.lambda_}, k={self.k})"

    def __eq__(self, other: object) -> bool:
        r"""Check equality based on matrix identity or lookup table values."""
        if self is other:
            return True
        if not isinstance(other, SubstitutionMatrix):
            return False
        return (
            self.alphabet == other.alphabet
            and self.lambda_ == other.lambda_
            and self.k == other.k
            and bool(np.array_equal(self.lut, other.lut))
        )

    def __hash__(self) -> int:
        r"""Return hash of the SubstitutionMatrix."""
        return hash((self.alphabet, self.lambda_, self.k, self.name))

    @overload
    def bitscore(self, scores: float) -> float: ...

    @overload
    def bitscore(
        self, scores: np.ndarray[tuple[int], np.dtype[Any]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float32]]: ...

    def bitscore(
        self, scores: float | np.ndarray[tuple[int], np.dtype[Any]]
    ) -> float | np.ndarray[tuple[int], np.dtype[np.float32]]:
        r"""Compute Karlin-Altschul bitscore(s) $B = (\lambda S - \ln K) / \ln 2$.

        Args:
            scores (float | np.ndarray): Raw alignment score(s).

        Returns:
            float | np.ndarray[tuple[int], np.dtype[np.float32]]: Non-negative bitscore(s).
        """
        if isinstance(scores, np.ndarray):
            raw = (self.lambda_ * scores.astype(np.float64) - self.ln_k) / _LN_2
            return np.where(scores > 0, np.maximum(0.0, raw), 0.0).astype(np.float32)
        if scores <= 0:
            return 0.0
        return max(0.0, (self.lambda_ * float(scores) - self.ln_k) / _LN_2)


# Numba Kernels --------------------------------------------------------------------------------------------------------
@njit(nogil=True, cache=True, inline="always")
def _score_ungapped_diagonal(
    q_data: np.ndarray,
    s1: int,
    l1: int,
    t_data: np.ndarray,
    s2: int,
    l2: int,
    d: int,
    lut: np.ndarray,
    min_span: int = 12,
) -> int:
    r"""Compute maximum local ungapped score along diagonal $d = i - j$ between two sequences."""
    i_start = max(0, d)
    j_start = i_start - d
    span = min(l1 - i_start, l2 - j_start)
    if span < min_span:
        return 0
    cur = 0
    max_sc = 0
    for k in range(span):
        cur += int(lut[q_data[s1 + i_start + k], t_data[s2 + j_start + k]])
        if cur < 0:
            cur = 0
        elif cur > max_sc:
            max_sc = cur
    return max_sc


@njit(parallel=True, nogil=True, cache=True)
def _ungapped_pair_prefilter(
    seqs_data: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    q_idx: np.ndarray,
    t_idx: np.ndarray,
    lut: np.ndarray,
    min_ungapped_score: int,
) -> np.ndarray:
    r"""Filter candidate pairs `(q_idx, t_idx)` using MMseqs2-style ungapped diagonal scoring."""
    n_pairs = len(q_idx)
    keep = np.zeros(n_pairs, dtype=np.bool_)
    for p in prange(n_pairs):  # type: ignore[not-iterable]
        u = q_idx[p]
        v = t_idx[p]
        l1 = lengths[u]
        l2 = lengths[v]
        s1 = offsets[u]
        s2 = offsets[v]
        d_lim = abs(l1 - l2) + 24
        max_sc = 0
        for d in range(-d_lim, d_lim + 1):
            sc = _score_ungapped_diagonal(seqs_data, s1, l1, seqs_data, s2, l2, d, lut, 12)
            if sc > max_sc:
                max_sc = sc
                if max_sc >= min_ungapped_score:
                    keep[p] = True
                    break
    return keep


# Predefined SubstitutionMatrix Singletons -----------------------------------------------------------------------------
_PROTEIN_25_SYMBOLS: bytes = b"ARNDCQEGHILKMFPSTWYVBJZX*"

_BLOSUM62_SCORES = np.array(
    [
        [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0, -2, -1, -1, -1, -4],
        [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3, -1, -2, 0, -1, -4],
        [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3, 4, -3, 0, -1, -4],
        [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3, 4, -3, 1, -1, -4],
        [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -1, -3, -1, -4],
        [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2, 0, -2, 4, -1, -4],
        [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2, 1, -3, 4, -1, -4],
        [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3, -1, -4, -2, -1, -4],
        [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3, 0, -3, 0, -1, -4],
        [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3, -3, 3, -3, -1, -4],
        [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1, -4, 3, -3, -1, -4],
        [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2, 0, -3, 1, -1, -4],
        [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1, -3, 2, -1, -1, -4],
        [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1, -3, 0, -3, -1, -4],
        [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2, -2, -3, -1, -1, -4],
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2, 0, -2, 0, -1, -4],
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0, -1, -1, -1, -1, -4],
        [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3, -4, -2, -2, -1, -4],
        [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1, -3, -1, -2, -1, -4],
        [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4, -3, 2, -2, -1, -4],
        [-2, -1, 4, 4, -3, 0, 1, -1, 0, -3, -4, 0, -3, -3, -2, 0, -1, -4, -3, -3, 4, -3, 0, -1, -4],
        [-1, -2, -3, -3, -1, -2, -3, -4, -3, 3, 3, -3, 2, 0, -3, -2, -1, -2, -1, 2, -3, 3, -3, -1, -4],
        [-1, 0, 0, 1, -3, 4, 4, -2, 0, -3, -3, 1, -1, -3, -1, 0, -1, -2, -2, -2, 0, -3, 4, -1, -4],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -4],
        [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, 1],
    ],
    dtype=np.int8,
)

_BLOSUM80_SCORES = np.array(
    [
        [5, -2, -2, -2, -1, -1, -1, 0, -2, -2, -2, -1, -1, -3, -1, 1, 0, -3, -2, 0, -2, -2, -1, -1, -6],
        [-2, 6, -1, -2, -4, 1, -1, -3, 0, -3, -3, 2, -2, -4, -2, -1, -1, -4, -3, -3, -1, -3, 0, -1, -6],
        [-2, -1, 6, 1, -3, 0, -1, -1, 0, -4, -4, 0, -3, -4, -3, 0, 0, -4, -3, -4, 5, -4, 0, -1, -6],
        [-2, -2, 1, 6, -4, -1, 1, -2, -2, -4, -5, -1, -4, -4, -2, -1, -1, -6, -4, -4, 5, -5, 1, -1, -6],
        [-1, -4, -3, -4, 9, -4, -5, -4, -4, -2, -2, -4, -2, -3, -4, -2, -1, -3, -3, -1, -4, -2, -4, -1, -6],
        [-1, 1, 0, -1, -4, 6, 2, -2, 1, -3, -3, 1, 0, -4, -2, 0, -1, -3, -2, -3, 0, -3, 4, -1, -6],
        [-1, -1, -1, 1, -5, 2, 6, -3, 0, -4, -4, 1, -2, -4, -2, 0, -1, -4, -3, -3, 1, -4, 5, -1, -6],
        [0, -3, -1, -2, -4, -2, -3, 6, -3, -5, -4, -2, -4, -4, -3, -1, -2, -4, -4, -4, -1, -5, -3, -1, -6],
        [-2, 0, 0, -2, -4, 1, 0, -3, 8, -4, -3, -1, -2, -2, -3, -1, -2, -3, 2, -4, -1, -4, 0, -1, -6],
        [-2, -3, -4, -4, -2, -3, -4, -5, -4, 5, 1, -3, 1, -1, -4, -3, -1, -3, -2, 3, -4, 3, -4, -1, -6],
        [-2, -3, -4, -5, -2, -3, -4, -4, -3, 1, 4, -3, 2, 0, -3, -3, -2, -2, -2, 1, -4, 3, -3, -1, -6],
        [-1, 2, 0, -1, -4, 1, 1, -2, -1, -3, -3, 5, -2, -4, -1, -1, -1, -4, -3, -3, -1, -3, 1, -1, -6],
        [-1, -2, -3, -4, -2, 0, -2, -4, -2, 1, 2, -2, 6, 0, -3, -2, -1, -2, -2, 1, -3, 2, -1, -1, -6],
        [-3, -4, -4, -4, -3, -4, -4, -4, -2, -1, 0, -4, 0, 6, -4, -3, -2, 0, 3, -1, -4, 0, -4, -1, -6],
        [-1, -2, -3, -2, -4, -2, -2, -3, -3, -4, -3, -1, -3, -4, 8, -1, -2, -5, -4, -3, -2, -4, -2, -1, -6],
        [1, -1, 0, -1, -2, 0, 0, -1, -1, -3, -3, -1, -2, -3, -1, 5, 1, -4, -2, -2, 0, -3, 0, -1, -6],
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -2, -1, -1, -2, -2, 1, 5, -4, -2, 0, -1, -2, -1, -1, -6],
        [-3, -4, -4, -6, -3, -3, -4, -4, -3, -3, -2, -4, -2, 0, -5, -4, -4, 11, 2, -3, -5, -3, -3, -1, -6],
        [-2, -3, -3, -4, -3, -2, -3, -4, 2, -2, -2, -3, -2, 3, -4, -2, -2, 2, 7, -2, -3, -2, -3, -1, -6],
        [0, -3, -4, -4, -1, -3, -3, -4, -4, 3, 1, -3, 1, -1, -3, -2, 0, -3, -2, 4, -4, 2, -3, -1, -6],
        [-2, -1, 5, 5, -4, 0, 1, -1, -1, -4, -4, -1, -3, -4, -2, 0, -1, -5, -3, -4, 5, -4, 0, -1, -6],
        [-2, -3, -4, -5, -2, -3, -4, -5, -4, 3, 3, -3, 2, 0, -4, -3, -2, -3, -2, 2, -4, 3, -4, -1, -6],
        [-1, 0, 0, 1, -4, 4, 5, -3, 0, -4, -3, 1, -1, -4, -2, 0, -1, -3, -3, -3, 0, -4, 5, -1, -6],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -6],
        [-6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, 1],
    ],
    dtype=np.int8,
)

_BLOSUM45_SCORES = np.array(
    [
        [5, -2, -1, -2, -1, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -2, -2, 0, -1, -1, -1, -1, -5],
        [-2, 7, 0, -1, -3, 1, 0, -2, 0, -3, -2, 3, -1, -2, -2, -1, -1, -2, -1, -2, -1, -3, 1, -1, -5],
        [-1, 0, 6, 2, -2, 0, 0, 0, 1, -2, -3, 0, -2, -2, -2, 1, 0, -4, -2, -3, 4, -3, 0, -1, -5],
        [-2, -1, 2, 7, -3, 0, 2, -1, 0, -4, -3, 0, -3, -4, -1, 0, -1, -4, -2, -3, 5, -4, 1, -1, -5],
        [-1, -3, -2, -3, 12, -3, -3, -3, -3, -3, -2, -3, -2, -2, -4, -1, -1, -5, -3, -1, -2, -2, -3, -1, -5],
        [-1, 1, 0, 0, -3, 6, 2, -2, 1, -2, -2, 1, 0, -4, -1, 0, -1, -2, -1, -3, 0, -2, 4, -1, -5],
        [-1, 0, 0, 2, -3, 2, 6, -2, 0, -3, -2, 1, -2, -3, 0, 0, -1, -3, -2, -3, 1, -3, 4, -1, -5],
        [0, -2, 0, -1, -3, -2, -2, 7, -2, -4, -3, -2, -2, -3, -2, 0, -2, -2, -3, -3, -1, -4, -2, -1, -5],
        [-2, 0, 1, 0, -3, 1, 0, -2, 10, -3, -2, -1, 0, -2, -2, -1, -2, -3, 2, -3, 0, -3, 0, -1, -5],
        [-1, -3, -2, -4, -3, -2, -3, -4, -3, 5, 2, -3, 2, 0, -2, -2, -1, -2, 0, 3, -3, 3, -3, -1, -5],
        [-1, -2, -3, -3, -2, -2, -2, -3, -2, 2, 5, -3, 2, 1, -3, -3, -1, -2, 0, 1, -3, 4, -2, -1, -5],
        [-1, 3, 0, 0, -3, 1, 1, -2, -1, -3, -3, 5, -1, -3, -1, 0, -1, -2, -1, -2, 0, -3, 1, -1, -5],
        [-1, -1, -2, -3, -2, 0, -2, -2, 0, 2, 2, -1, 6, 0, -2, -2, -1, -2, 0, 1, -2, 2, -1, -1, -5],
        [-2, -2, -2, -4, -2, -4, -3, -3, -2, 0, 1, -3, 0, 8, -3, -2, -1, 1, 3, 0, -3, 1, -3, -1, -5],
        [-1, -2, -2, -1, -4, -1, 0, -2, -2, -2, -3, -1, -2, -3, 9, -1, -1, -3, -3, -3, -2, -3, 0, -1, -5],
        [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -3, 0, -2, -2, -1, 4, 2, -4, -2, -1, 0, -2, 0, -1, -5],
        [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -1, -1, 2, 5, -3, -1, 0, 0, -1, -1, -1, -5],
        [-2, -2, -4, -4, -5, -2, -3, -2, -3, -2, -2, -2, -2, 1, -3, -4, -3, 15, 3, -3, -4, -2, -2, -1, -5],
        [-2, -1, -2, -2, -3, -1, -2, -3, 2, 0, 0, -1, 0, 3, -3, -2, -1, 3, 8, -1, -2, 0, -2, -1, -5],
        [0, -2, -3, -3, -1, -3, -3, -3, -3, 3, 1, -2, 1, 0, -3, -1, 0, -3, -1, 5, -3, 2, -3, -1, -5],
        [-1, -1, 4, 5, -2, 0, 1, -1, 0, -3, -3, 0, -2, -3, -2, 0, 0, -4, -2, -3, 5, -3, 1, -1, -5],
        [-1, -3, -3, -4, -2, -2, -3, -4, -3, 3, 4, -3, 2, 1, -3, -2, -1, -2, 0, 2, -3, 4, -2, -1, -5],
        [-1, 1, 0, 1, -3, 4, 4, -2, 0, -3, -2, 1, -1, -3, 0, 0, -1, -2, -2, -3, 1, -2, 4, -1, -5],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -5],
        [-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, 1],
    ],
    dtype=np.int8,
)

SubstitutionMatrix.BLOSUM62 = SubstitutionMatrix(
    alphabet=Alphabet.AMINO,
    scores=_BLOSUM62_SCORES,
    symbols=_PROTEIN_25_SYMBOLS,
    default_score=-128,
    lambda_=0.267,
    k=0.041,
    name=b"BLOSUM62",
)

SubstitutionMatrix.BLOSUM80 = SubstitutionMatrix(
    alphabet=Alphabet.AMINO,
    scores=_BLOSUM80_SCORES,
    symbols=_PROTEIN_25_SYMBOLS,
    default_score=-128,
    lambda_=0.299,
    k=0.071,
    name=b"BLOSUM80",
)

SubstitutionMatrix.BLOSUM45 = SubstitutionMatrix(
    alphabet=Alphabet.AMINO,
    scores=_BLOSUM45_SCORES,
    symbols=_PROTEIN_25_SYMBOLS,
    default_score=-128,
    lambda_=0.195,
    k=0.032,
    name=b"BLOSUM45",
)

SubstitutionMatrix.DNA = SubstitutionMatrix.from_match_mismatch(
    alphabet=Alphabet.DNA,
    match=2,
    mismatch=-3,
    unknown_score=-1,
    default_score=-128,
    lambda_=0.625,
    k=0.41,
    name=b"DNA",
)
