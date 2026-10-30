r"""Base classes, 1D NumPy array type aliases, and Numba kernels for Structure-of-Arrays (SoA) containers.

This module defines the concrete base classes ([`ScalarView`][seroverse.core.collections.ScalarView],
[`BatchedContainer`][seroverse.core.collections.BatchedContainer],
[`RaggedArrayContainer`][seroverse.core.collections.RaggedArrayContainer],
[`Spec`][seroverse.core.collections.Spec], [`SpecContainer`][seroverse.core.collections.SpecContainer],
and [`SpecEnum`][seroverse.core.collections.SpecEnum]) and 1D NumPy array type aliases
that govern high-performance, vectorized containers and compiled ontology registries across SeroVerse.

Upon class definition (`__init_subclass__`), [`BatchedContainer`][seroverse.core.collections.BatchedContainer]
and [`RaggedArrayContainer`][seroverse.core.collections.RaggedArrayContainer] inspect `@dataclass` field
annotations, verify that all stored container fields are NumPy arrays, register their `np.dtype`s,
and automatically provide `__len__`, `__getitem__`, `__eq__`, `filter`, `empty`, `concat`, and `from_list`.
"""

import annotationlib
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import Field, dataclass, field, fields, replace
from enum import Enum, IntEnum
from typing import Any, ClassVar, Self, TypeAliasType, get_args, get_origin, overload

import numpy as np
from numba import njit, prange

# 1D NumPy Array Type Aliases ------------------------------------------------------------------------------------------
type BoolArray = np.ndarray[tuple[int], np.dtype[np.bool_]]
type Int8Array = np.ndarray[tuple[int], np.dtype[np.int8]]
type UInt8Array = np.ndarray[tuple[int], np.dtype[np.uint8]]
type Int16Array = np.ndarray[tuple[int], np.dtype[np.int16]]
type UInt16Array = np.ndarray[tuple[int], np.dtype[np.uint16]]
type Int32Array = np.ndarray[tuple[int], np.dtype[np.int32]]
type UInt32Array = np.ndarray[tuple[int], np.dtype[np.uint32]]
type Int64Array = np.ndarray[tuple[int], np.dtype[np.int64]]
type UInt64Array = np.ndarray[tuple[int], np.dtype[np.uint64]]
type Float32Array = np.ndarray[tuple[int], np.dtype[np.float32]]
type Float64Array = np.ndarray[tuple[int], np.dtype[np.float64]]
type BytesArray = np.ndarray[tuple[int], np.dtype[np.bytes_]]

_STRING_DTYPE_MAP: dict[str, np.dtype[Any]] = {
    "BoolArray": np.dtype(np.bool_),
    "Int8Array": np.dtype(np.int8),
    "UInt8Array": np.dtype(np.uint8),
    "Int16Array": np.dtype(np.int16),
    "UInt16Array": np.dtype(np.uint16),
    "Int32Array": np.dtype(np.int32),
    "UInt32Array": np.dtype(np.uint32),
    "Int64Array": np.dtype(np.int64),
    "UInt64Array": np.dtype(np.uint64),
    "Float32Array": np.dtype(np.float32),
    "Float64Array": np.dtype(np.float64),
    "BytesArray": np.dtype("S"),
    "np.bool_": np.dtype(np.bool_),
    "bool_": np.dtype(np.bool_),
    "np.int8": np.dtype(np.int8),
    "int8": np.dtype(np.int8),
    "np.uint8": np.dtype(np.uint8),
    "uint8": np.dtype(np.uint8),
    "np.int16": np.dtype(np.int16),
    "int16": np.dtype(np.int16),
    "np.uint16": np.dtype(np.uint16),
    "uint16": np.dtype(np.uint16),
    "np.int32": np.dtype(np.int32),
    "int32": np.dtype(np.int32),
    "np.uint32": np.dtype(np.uint32),
    "uint32": np.dtype(np.uint32),
    "np.int64": np.dtype(np.int64),
    "int64": np.dtype(np.int64),
    "np.uint64": np.dtype(np.uint64),
    "uint64": np.dtype(np.uint64),
    "np.float32": np.dtype(np.float32),
    "float32": np.dtype(np.float32),
    "np.float64": np.dtype(np.float64),
    "float64": np.dtype(np.float64),
    "np.bytes_": np.dtype("S"),
    "bytes_": np.dtype("S"),
}


# Classes --------------------------------------------------------------------------------------------------------------
class ScalarView:
    r"""Concrete base class for immutable scalar views extracted from a [`BatchedContainer`][seroverse.core.collections.BatchedContainer].

    All scalar views are `@dataclass(frozen=True, slots=True)` records representing a single
    element of a Structure-of-Arrays (SoA) batch container. Inheriting from `ScalarView`
    automatically provides:
    - NumPy-safe structural equality (`__eq__`) using `np.array_equal` for any `np.ndarray` fields
      while preserving `@dataclass`-generated `__hash__` on scalar-only views.
    - Functional copy-update via [`.replace(**changes)`][seroverse.core.collections.ScalarView.replace].
    - Shallow dictionary serialization via [`.to_dict()`][seroverse.core.collections.ScalarView.to_dict].
    """

    __slots__ = ()
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__dataclass_fields__" in cls.__dict__:
            cls.__eq__ = ScalarView._dataclass_eq  # type: ignore[method-assign]

    def _dataclass_eq(self, other: object) -> bool:
        r"""Compare two ScalarView instances, handling both scalar and `np.ndarray` fields safely."""
        if type(self) is not type(other):
            return False
        for f in fields(self):
            if not f.compare:
                continue
            a = getattr(self, f.name)
            b = getattr(other, f.name)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not np.array_equal(a, b):
                    return False
            elif a != b:
                return False
        return True

    def replace(self, **changes: Any) -> Self:
        r"""Return a new scalar view instance with specified fields replaced."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        r"""Return a shallow dictionary mapping field names to their values."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


class BatchedContainer[T: ScalarView = ScalarView]:
    r"""Concrete generic base class for Structure-of-Arrays (SoA) batch containers.

    Subclasses must be `@dataclass(frozen=True, slots=True)` classes whose stored fields are all
    annotated as 1D NumPy arrays (e.g. `Int32Array`, `UInt32Array`, `Float64Array`). Upon class
    creation, `BatchedContainer.__init_subclass__` validates and registers each field's `np.dtype`
    into `_field_dtypes` and automatically provides:
    - `__len__`, `__iter__`, and NumPy-safe `__eq__`
    - `__getitem__` with `@overload` typing, negative-index bounds checking, and automatic dispatch
      to `_get_scalar(idx)` (for integer indices) and `_get_slice(item)` (for slices/masks/index arrays)
    - `filter(mask)`, `replace(**changes)`, `empty()`, `concat(batches)`, and `from_list(items)`

    Type Parameters:
        T: The [`ScalarView`][seroverse.core.collections.ScalarView] dataclass returned when indexing a single element.
    """

    __slots__ = ()
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
    _field_dtypes: ClassVar[dict[str, np.dtype[Any]]] = {}
    _nested_fields: ClassVar[dict[str, type[BatchedContainer[Any]]]] = {}
    _all_fields: ClassVar[tuple[str, ...]] = ()
    _primary_field: ClassVar[str] = ""
    _scalar_type: ClassVar[type[Any] | None] = None
    _scalar_source_fields: ClassVar[tuple[str, ...]] = ()
    _scalar_converters: ClassVar[tuple[Callable[[Any], Any] | None, ...]] = ()

    @classmethod
    def _resolve_field_spec(
        cls,
        annotation: Any,
        field_name: str,
    ) -> np.dtype[Any] | type[BatchedContainer[Any]]:
        r"""Resolve a `BatchedContainer` field annotation to either a 1D `np.dtype` or a nested `BatchedContainer` subclass.

        Args:
            annotation: Field type annotation (e.g. `Int32Array` or `Intervals`).
            cls: Container class being inspected.
            field_name: Name of the field being inspected.

        Returns:
            np.dtype[Any] | type[BatchedContainer[Any]]: Resolved NumPy dtype or nested `BatchedContainer` class.

        Raises:
            TypeError: If the field is neither a 1D NumPy array nor a `BatchedContainer` subclass.
        """
        ann = annotation
        if isinstance(ann, annotationlib.ForwardRef):
            mod = sys.modules.get(cls.__module__)
            ann = getattr(mod, ann.__forward_arg__, ann.__forward_arg__) if mod is not None else ann.__forward_arg__
        elif isinstance(ann, str) and ann not in _STRING_DTYPE_MAP and "ndarray" not in ann:
            mod = sys.modules.get(cls.__module__)
            if mod is not None and hasattr(mod, ann):
                ann = getattr(mod, ann)

        while isinstance(ann, TypeAliasType):
            ann = ann.__value__

        if isinstance(ann, type) and issubclass(ann, BatchedContainer):
            return ann

        origin = get_origin(ann)
        if origin is not None and isinstance(origin, type) and issubclass(origin, BatchedContainer):
            return origin

        if isinstance(ann, str):
            if ann in _STRING_DTYPE_MAP:
                return _STRING_DTYPE_MAP[ann]
            if "ndarray" in ann:
                for key, dt in _STRING_DTYPE_MAP.items():
                    if f"dtype[{key}]" in ann:
                        return dt
            raise TypeError(
                f"{cls.__name__}.{field_name} must be annotated as a 1D NumPy array or BatchedContainer, got {annotation!r}"
            )

        if origin is np.ndarray:
            args = get_args(ann)
            if len(args) >= 2:
                dtype_arg = args[1]
                dtype_inner = get_args(dtype_arg)
                if dtype_inner:
                    scalar_t = dtype_inner[0]
                    if scalar_t is np.bytes_:
                        return np.dtype("S")
                    if isinstance(scalar_t, type) and issubclass(scalar_t, np.generic):
                        return np.dtype(scalar_t)
            return np.dtype(np.float64)

        if ann is np.ndarray:
            return np.dtype(np.float64)

        raise TypeError(
            f"{cls.__name__}.{field_name} must be annotated as a 1D NumPy array or BatchedContainer, got {annotation!r}"
        )

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Extract scalar view type parameter T from generic bases if concrete
        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            if origin is not None and isinstance(origin, type) and issubclass(origin, BatchedContainer):
                args = get_args(base)
                if args and isinstance(args[0], type) and issubclass(args[0], ScalarView):
                    cls._scalar_type = args[0]
                    break

        if "__dataclass_fields__" not in cls.__dict__:
            return

        base_dtypes: dict[str, np.dtype[Any]] = {}
        base_nested: dict[str, type[BatchedContainer[Any]]] = {}
        for mro_cls in reversed(cls.__mro__[1:]):
            if issubclass(mro_cls, BatchedContainer):
                if "_field_dtypes" in mro_cls.__dict__:
                    base_dtypes.update(mro_cls._field_dtypes)
                if "_nested_fields" in mro_cls.__dict__:
                    base_nested.update(mro_cls._nested_fields)

        try:
            hints = annotationlib.get_annotations(cls, format=annotationlib.Format.VALUE)
        except Exception:
            hints = annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)

        dtypes: dict[str, np.dtype[Any]] = {}
        nested: dict[str, type[BatchedContainer[Any]]] = {}
        all_fields: list[str] = []
        for f in fields(cls):
            if not f.init:
                continue
            all_fields.append(f.name)
            if f.name not in hints:
                if f.name in base_dtypes:
                    dtypes[f.name] = base_dtypes[f.name]
                    continue
                if f.name in base_nested:
                    nested[f.name] = base_nested[f.name]
                    continue
            ann = hints.get(f.name, f.type)
            resolved = cls._resolve_field_spec(ann, f.name)
            if isinstance(resolved, np.dtype):
                dtypes[f.name] = resolved
            else:
                nested[f.name] = resolved

        cls._field_dtypes = dtypes
        cls._nested_fields = nested
        cls._all_fields = tuple(all_fields)
        cls._primary_field = all_fields[0] if all_fields else ""

        scalar_cls = cls._scalar_type
        if scalar_cls is not None and "__dataclass_fields__" in scalar_cls.__dict__:
            s_fields = fields(scalar_cls)
            cls._scalar_source_fields = cls._all_fields[: len(s_fields)]
            s_hints: dict[str, Any] = {}
            for s_mro in reversed(scalar_cls.__mro__):
                if "__dataclass_fields__" in s_mro.__dict__:
                    try:
                        s_hints.update(annotationlib.get_annotations(s_mro, format=annotationlib.Format.VALUE))
                    except Exception:
                        pass
            convs: list[Callable[[Any], Any] | None] = []
            has_conv = False
            for sf in s_fields:
                sf_ann = s_hints.get(sf.name, sf.type)
                if isinstance(sf_ann, type) and issubclass(sf_ann, Enum):
                    convs.append(sf_ann)
                    has_conv = True
                else:
                    convs.append(None)
            cls._scalar_converters = tuple(convs) if has_conv else ()
        else:
            cls._scalar_source_fields = cls._all_fields
            cls._scalar_converters = ()

        cls.__eq__ = BatchedContainer._container_eq  # type: ignore[method-assign]

    def _container_eq(self, other: object) -> bool:
        r"""Compare two containers for exact array and nested-container equality across all registered SoA fields."""
        if type(self) is not type(other):
            return False
        return all(np.array_equal(getattr(self, name), getattr(other, name)) for name in self._field_dtypes) and all(
            getattr(self, name) == getattr(other, name) for name in self._nested_fields
        )

    def __len__(self) -> int:
        r"""Return the number of records in the batch."""
        return len(getattr(self, self._primary_field))

    def _resolve_index(self, item: int | np.integer[Any]) -> int:
        r"""Normalize a potentially negative integer index and validate bounds."""
        idx = int(item)
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"{type(self).__name__} index {item} out of range for batch of length {n}")
        return idx

    def _get_scalar(self, idx: int) -> T:
        r"""Materialize the scalar view `T` at normalized 0-based index `idx`."""
        scalar_cls = self._scalar_type
        if scalar_cls is None:
            raise NotImplementedError(f"{type(self).__name__} must implement _get_scalar(idx)")
        convs = self._scalar_converters
        nested = self._nested_fields
        if convs:
            return scalar_cls(
                *(
                    getattr(self, f)[idx]
                    if f in nested
                    else (conv(getattr(self, f)[idx].item()) if conv is not None else getattr(self, f)[idx].item())
                    for f, conv in zip(self._scalar_source_fields, convs, strict=True)
                )
            )
        return scalar_cls(
            *(
                getattr(self, f)[idx] if f in nested else getattr(self, f)[idx].item()
                for f in self._scalar_source_fields
            )
        )

    def _get_slice(self, item: slice | np.ndarray[Any, Any] | list[Any]) -> Self:
        r"""Slice all 1D SoA arrays and 1-to-1 nested containers by a slice, boolean mask, or integer index sequence."""
        indices: Any = item if isinstance(item, slice) else self._normalize_indices(item)
        return type(self)(**{f: getattr(self, f)[indices] for f in self._all_fields})

    @overload
    def __getitem__(self, item: int | np.integer[Any]) -> T: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]
    ) -> T | Self: ...

    def __getitem__(self, item: int | np.integer[Any] | slice | np.ndarray[tuple[int], Any] | list[Any]) -> T | Self:
        r"""Access a single scalar record `T` by integer index, or a sliced container `Self` by slice/mask."""
        if isinstance(item, (int, np.integer)):
            return self._get_scalar(self._resolve_index(item))
        return self._get_slice(item)

    def __iter__(self) -> Iterator[T]:
        r"""Iterate over the container, yielding each scalar view `T` in order."""
        for i in range(len(self)):
            yield self._get_scalar(i)

    def _normalize_indices(self, item: slice | np.ndarray[Any, Any] | list[Any]) -> Int32Array:
        r"""Normalize a slice, boolean mask, or integer index sequence into a 1D `int32` index array.

        Args:
            item: Slice, boolean mask array, or integer index sequence/array.
            n: Total number of records in the target container.

        Returns:
            Int32Array: Resolved 0-based `int32` indices.
        """
        n = len(self)
        if isinstance(item, slice):
            return np.arange(n, dtype=np.int32)[item]
        if not isinstance(item, (np.ndarray, Sequence)) or isinstance(item, (str, bytes, bytearray)):
            raise TypeError(f"Unsupported index type: {type(item).__name__}")
        arr = np.asarray(item)
        if arr.dtype.kind == "b":
            return np.nonzero(arr)[0].astype(np.int32)
        if len(arr) == 0:
            return np.empty(0, dtype=np.int32)
        indices = arr.astype(np.int32, copy=False)
        if np.any(indices < 0):
            indices = np.where(indices < 0, indices + n, indices)
        return indices

    def filter(self, mask: BoolArray | list[bool]) -> Self:
        r"""Return a filtered container containing only records where `mask` is True."""
        return self._get_slice(mask)

    def replace(self, **changes: Any) -> Self:
        r"""Return a new container instance with specified array fields replaced."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        r"""Return a shallow dictionary mapping container field names to their underlying arrays or nested containers."""
        return {f: getattr(self, f) for f in self._all_fields}

    def chunks(self, size: int) -> Iterator[Self]:
        r"""Yield contiguous $O(1)$ zero-copy mini-batch slices of at most `size` records.

        Args:
            size: Maximum number of records per chunk (must be `> 0`).

        Yields:
            Self: Contiguous mini-batch container slice.

        Raises:
            ValueError: If `size <= 0`.
        """
        if size <= 0:
            raise ValueError(f"Chunk size must be positive, got {size}")
        n = len(self)
        for start in range(0, n, size):
            yield self._get_slice(slice(start, min(start + size, n)))

    def argsort(
        self,
        *by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Int32Array:
        r"""Compute the stable 0-based `int32` permutation indices that sort this container by `*by`.

        Supports sorting by one or more registered 1D array columns or computed 1D array properties
        (in primary-to-secondary priority order), with either a single `descending` boolean or a
        per-key sequence of booleans.

        Args:
            *by: Field or property names to sort by (defaults to the primary container field).
            descending: Sort direction (`bool` applied to all keys, or a `Sequence[bool]` per key).

        Returns:
            Int32Array: 1D `int32` index permutation of length `len(self)`.

        Raises:
            ValueError: If a key does not resolve to a 1D NumPy array of length `len(self)` or
                if `descending` length does not match `len(by)`.
        """
        n = len(self)
        keys = by if by else (self._primary_field,)
        if isinstance(descending, bool):
            desc_flags = (descending,) * len(keys)
        else:
            desc_flags = tuple(bool(d) for d in descending)
            if len(desc_flags) != len(keys):
                raise ValueError(
                    f"Length of descending ({len(desc_flags)}) must match number of sort keys ({len(keys)})"
                )

        arrays: list[np.ndarray[Any, Any]] = []
        for k in keys:
            if not hasattr(self, k):
                raise ValueError(f"{type(self).__name__} has no field or property {k!r}")
            arr = getattr(self, k)
            if not isinstance(arr, np.ndarray) or arr.ndim != 1 or len(arr) != n:
                raise ValueError(f"Sort key {k!r} on {type(self).__name__} must be a 1D NumPy array of length {n}")
            arrays.append(arr)

        if n <= 1:
            return np.arange(n, dtype=np.int32)

        order = np.arange(n, dtype=np.int32)
        for arr, desc in zip(reversed(arrays), reversed(desc_flags), strict=True):
            sub = arr[order]
            if desc:
                step = (n - 1) - np.argsort(sub[::-1], kind="stable")[::-1]
            else:
                step = np.argsort(sub, kind="stable")
            order = order[step]
        return order.astype(np.int32, copy=False)

    def sort_by(
        self,
        *by: str,
        descending: bool | Sequence[bool] = False,
    ) -> Self:
        r"""Return a new container sorted stably by one or more 1D array fields or properties."""
        if len(self) <= 1 and not by:
            return self
        return self._get_slice(self.argsort(*by, descending=descending))

    @classmethod
    def empty(cls) -> Self:
        r"""Create an empty 0-length container with registered `np.dtype` arrays and empty nested containers."""
        kwargs: dict[str, Any] = {f: np.empty(0, dtype=dt) for f, dt in cls._field_dtypes.items()}
        for f, child_cls in cls._nested_fields.items():
            kwargs[f] = child_cls.empty()
        return cls(**kwargs)

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple containers of the same type into a single batch."""
        batches_list = [b for b in batches if len(b) > 0]
        if not batches_list:
            return cls.empty()
        if len(batches_list) == 1:
            return batches_list[0]
        kwargs: dict[str, Any] = {f: np.concatenate([getattr(b, f) for b in batches_list]) for f in cls._field_dtypes}
        for f, child_cls in cls._nested_fields.items():
            kwargs[f] = child_cls.concat([getattr(b, f) for b in batches_list])
        return cls(**kwargs)

    @classmethod
    def from_list(cls, items: Sequence[T]) -> Self:
        r"""Pack a sequence of [`ScalarView`][seroverse.core.collections.ScalarView] instances into a batch container."""
        if not items:
            return cls.empty()
        scalar_cls = cls._scalar_type or type(items[0])
        s_fields = tuple(f.name for f in fields(scalar_cls))
        kwargs: dict[str, Any] = {}
        for b_f, s_f in zip(cls._all_fields, s_fields, strict=True):
            if b_f in cls._nested_fields:
                kwargs[b_f] = cls._nested_fields[b_f].from_list([getattr(it, s_f) for it in items])
            else:
                kwargs[b_f] = np.array([getattr(it, s_f) for it in items], dtype=cls._field_dtypes[b_f])
        return cls(**kwargs)


class RaggedArrayContainer[T: ScalarView = ScalarView](BatchedContainer[T]):
    r"""Concrete generic base class for ragged Structure-of-Arrays (SoA) containers.

    Ragged containers store variable-length per-record arrays (declared in `_ragged_fields`)
    in flat, contiguous 1D NumPy buffers indexed by `offsets` and `lengths` (`Int32Array`),
    alongside any fixed-length per-record 1D arrays (`_record_fields`).

    Subclasses only need to declare `_ragged_fields: ClassVar[tuple[str, ...]]` (e.g. `("seqs",)`
    or `("members",)`), and inherit `__len__`, `__getitem__`, `_get_scalar`, `_get_slice`,
    `empty`, `concat`, `from_list`, `filter`, and `indptr` automatically.
    """

    __slots__ = ()
    offsets: Int32Array
    lengths: Int32Array
    _ragged_fields: ClassVar[tuple[str, ...]] = ()
    _ragged_fields_set: ClassVar[frozenset[str]] = frozenset()
    _record_fields: ClassVar[tuple[str, ...]] = ()
    _data_fields: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__dataclass_fields__" not in cls.__dict__:
            return
        ragged_set = frozenset(cls._ragged_fields)
        cls._ragged_fields_set = ragged_set
        cls._data_fields = tuple(f for f in cls._all_fields if f not in ("offsets", "lengths"))
        cls._record_fields = tuple(
            f for f in cls._field_dtypes if f not in ("offsets", "lengths") and f not in ragged_set
        )

    def __len__(self) -> int:
        r"""Return the number of records in the ragged container."""
        return len(self.offsets)

    @property
    def indptr(self) -> Int32Array:
        r"""CSR-compatible $(N + 1)$-length index pointer array (`[0, *cumsum(lengths)]`)."""
        n = len(self.offsets)
        if n == 0:
            return np.zeros(1, dtype=np.int32)
        out = np.empty(n + 1, dtype=np.int32)
        out[:n] = self.offsets
        out[n] = self.offsets[-1] + self.lengths[-1]
        return out

    @staticmethod
    def lengths_to_offsets(
        lengths: Int32Array,
    ) -> Int32Array:
        r"""Compute 0-based exclusive prefix-sum offsets from a 1D array of element lengths.

        Args:
            lengths: 1D array of non-negative record lengths (`int32`).

        Returns:
            Int32Array: 1D `int32` array of starting offsets.
        """
        n = len(lengths)
        offsets = np.zeros(n, dtype=np.int32)
        if n > 1:
            np.cumsum(lengths[:-1], out=offsets[1:])
        return offsets

    def _get_scalar(self, idx: int) -> T:
        r"""Materialize the scalar view `T` at normalized 0-based index `idx`."""
        scalar_cls = self._scalar_type
        if scalar_cls is None:
            raise NotImplementedError(f"{type(self).__name__} must implement _get_scalar(idx)")
        off = int(self.offsets[idx])
        end = off + int(self.lengths[idx])
        ragged = self._ragged_fields_set
        nested = self._nested_fields
        convs = self._scalar_converters
        if convs:
            return scalar_cls(
                *(
                    getattr(self, f)[off:end]
                    if f in ragged
                    else (
                        getattr(self, f)[idx]
                        if f in nested
                        else (conv(getattr(self, f)[idx].item()) if conv is not None else getattr(self, f)[idx].item())
                    )
                    for f, conv in zip(self._data_fields, convs, strict=True)
                )
            )
        return scalar_cls(
            *(
                getattr(self, f)[off:end]
                if f in ragged
                else (getattr(self, f)[idx] if f in nested else getattr(self, f)[idx].item())
                for f in self._data_fields
            )
        )

    def _get_slice(self, item: slice | np.ndarray[Any, Any] | list[Any]) -> Self:
        r"""Slice both per-record arrays and variable-length ragged arrays in parallel."""
        ragged_arrays = tuple(getattr(self, f) for f in self._ragged_fields)
        indices, new_offsets, new_lengths, sliced_ragged = self._slice_ragged(item, *ragged_arrays)
        if len(indices) == 0:
            return self.empty()
        kwargs: dict[str, Any] = {"offsets": new_offsets, "lengths": new_lengths}
        for f, arr in zip(self._ragged_fields, sliced_ragged, strict=True):
            kwargs[f] = arr
        for f in self._record_fields:
            kwargs[f] = getattr(self, f)[indices]
        for f in self._nested_fields:
            kwargs[f] = getattr(self, f)[indices]
        return type(self)(**kwargs)

    @classmethod
    def concat(cls, batches: Iterable[Self]) -> Self:
        r"""Concatenate multiple ragged containers into a single contiguous batch."""
        batches_list = [b for b in batches if len(b) > 0]
        if not batches_list:
            return cls.empty()
        if len(batches_list) == 1:
            return batches_list[0]
        all_lengths = np.concatenate([b.lengths for b in batches_list]).astype(np.int32, copy=False)
        kwargs: dict[str, Any] = {
            "offsets": cls.lengths_to_offsets(all_lengths),
            "lengths": all_lengths,
        }
        for f in cls._ragged_fields:
            parts = [getattr(b, f) for b in batches_list if len(getattr(b, f)) > 0]
            kwargs[f] = np.concatenate(parts) if parts else np.empty(0, dtype=cls._field_dtypes[f])
        for f in cls._record_fields:
            kwargs[f] = np.concatenate([getattr(b, f) for b in batches_list])
        for f, child_cls in cls._nested_fields.items():
            kwargs[f] = child_cls.concat([getattr(b, f) for b in batches_list])
        return cls(**kwargs)

    @classmethod
    def from_list(cls, items: Sequence[T]) -> Self:
        r"""Pack a sequence of [`ScalarView`][seroverse.core.collections.ScalarView] instances into a ragged container."""
        if not items:
            return cls.empty()
        scalar_cls = cls._scalar_type or type(items[0])
        s_fields = tuple(f.name for f in fields(scalar_cls))
        field_map = dict(zip(cls._data_fields, s_fields, strict=True))
        first_ragged_s = field_map[cls._ragged_fields[0]]
        lengths = np.array([len(getattr(it, first_ragged_s)) for it in items], dtype=np.int32)
        has_elems = bool(np.any(lengths > 0))
        kwargs: dict[str, Any] = {
            "offsets": cls.lengths_to_offsets(lengths),
            "lengths": lengths,
        }
        for b_f in cls._ragged_fields:
            s_f = field_map[b_f]
            dt = cls._field_dtypes[b_f]
            parts = [
                np.frombuffer(v, dtype=dt) if isinstance((v := getattr(it, s_f)), bytes) else np.asarray(v, dtype=dt)
                for it in items
                if len(getattr(it, s_f)) > 0
            ]
            kwargs[b_f] = np.concatenate(parts) if (has_elems and parts) else np.empty(0, dtype=dt)
        for b_f in cls._record_fields:
            s_f = field_map[b_f]
            dt = cls._field_dtypes[b_f]
            kwargs[b_f] = np.array([getattr(it, s_f) for it in items], dtype=dt)
        for b_f, child_cls in cls._nested_fields.items():
            s_f = field_map[b_f]
            kwargs[b_f] = child_cls.from_list([getattr(it, s_f) for it in items])
        return cls(**kwargs)

    def _slice_ragged(
        self,
        item: slice | np.ndarray[Any, Any] | list[Any],
        *arrays: np.ndarray[tuple[int], Any],
    ) -> tuple[
        Int32Array,
        Int32Array,
        Int32Array,
        tuple[np.ndarray[tuple[int], Any], ...],
    ]:
        r"""Slice or gather one or more parallel ragged data arrays while maintaining contiguous packing.

        Uses an $O(1)$ zero-copy view slice on `arrays` when `item` is a contiguous `slice(start, stop, 1)`,
        and falls back to the parallel Numba `_gather_ragged_kernel` for stepped slices, boolean masks,
        or arbitrary index arrays.

        Args:
            item: Slice, boolean mask, or integer index list/array.
            *arrays: Parallel 1D flat data arrays indexed by `self.offsets` and `self.lengths`.

        Returns:
            tuple: `(indices, new_offsets, new_lengths, sliced_arrays)`.
        """
        n = len(self)
        if isinstance(item, slice) and item.step in (None, 1):
            start, stop, _ = item.indices(n)
            if stop <= start:
                empty_i32 = np.empty(0, dtype=np.int32)
                return (
                    empty_i32,
                    empty_i32,
                    empty_i32,
                    tuple(np.empty(0, dtype=a.dtype) for a in arrays),
                )
            indices = np.arange(start, stop, dtype=np.int32)
            new_lengths = self.lengths[start:stop]
            base_offset = int(self.offsets[start])
            end_offset = int(self.offsets[stop - 1]) + int(self.lengths[stop - 1])
            new_offsets = self.offsets[start:stop] - base_offset
            return (
                indices,
                new_offsets,
                new_lengths,
                tuple(a[base_offset:end_offset] for a in arrays),
            )

        indices = self._normalize_indices(item)
        if len(indices) == 0:
            empty_i32 = np.empty(0, dtype=np.int32)
            return (
                indices,
                empty_i32,
                empty_i32,
                tuple(np.empty(0, dtype=a.dtype) for a in arrays),
            )

        new_lengths = self.lengths[indices].astype(np.int32, copy=False)
        new_offsets = self.lengths_to_offsets(new_lengths)
        total_len = int(new_offsets[-1] + new_lengths[-1])
        if total_len == 0:
            return (
                indices,
                new_offsets,
                new_lengths,
                tuple(np.empty(0, dtype=a.dtype) for a in arrays),
            )

        sliced = tuple(
            _gather_ragged_kernel(a, self.offsets, new_offsets, new_lengths, indices, total_len) for a in arrays
        )
        return indices, new_offsets, new_lengths, sliced

    def pad(
        self,
        field_name: str | None = None,
        *,
        max_len: int | None = None,
        fill_value: Any = 0,
    ) -> np.ndarray[tuple[int, int], Any]:
        r"""Pack a ragged 1D field into a dense 2D `(N, max_len)` NumPy matrix.

        Args:
            field_name: Name of the ragged field in `_ragged_fields` to pad. Defaults to the sole
                ragged field when `len(_ragged_fields) == 1`.
            max_len: Target column width $L$ of the output 2D matrix. Defaults to `max(self.lengths)`
                (or `0` when the container is empty); rows longer than `max_len` are truncated.
            fill_value: Padding value used to fill positions `[lengths[i] : max_len]`.

        Returns:
            np.ndarray[tuple[int, int], Any]: Dense 2D array of shape `(len(self), max_len)`.

        Raises:
            ValueError: If `field_name` is omitted on a multi-field ragged container, is not in
                `_ragged_fields`, or `max_len < 0`.
        """
        if field_name is None:
            if len(self._ragged_fields) != 1:
                raise ValueError(
                    f"{type(self).__name__} has {len(self._ragged_fields)} ragged fields {self._ragged_fields}; "
                    "specify field_name explicitly"
                )
            target_field = self._ragged_fields[0]
        else:
            if field_name not in self._ragged_fields_set:
                raise ValueError(
                    f"{field_name!r} is not a ragged field of {type(self).__name__} ({self._ragged_fields})"
                )
            target_field = field_name

        data = getattr(self, target_field)
        n = len(self)
        target_len = (int(np.max(self.lengths)) if n > 0 else 0) if max_len is None else int(max_len)
        if target_len < 0:
            raise ValueError(f"max_len must be non-negative, got {target_len}")
        if n == 0 or target_len == 0:
            return np.empty((n, max(target_len, 0)), dtype=data.dtype)
        if data.dtype.kind in ("S", "U", "O"):
            out = np.full((n, target_len), fill_value, dtype=data.dtype)
            for i in range(n):
                row_len = min(int(self.lengths[i]), target_len)
                if row_len > 0:
                    off = int(self.offsets[i])
                    out[i, :row_len] = data[off : off + row_len]
            return out
        typed_fill = data.dtype.type(fill_value)
        return _pad_ragged_kernel(data, self.offsets, self.lengths, target_len, typed_fill)


@dataclass(frozen=True, slots=True)
class Spec(ScalarView):
    r"""Immutable scalar specification for an ontology or reference database record.

    Serves as the scalar element view `T` for [`SpecContainer`][seroverse.core.collections.SpecContainer]
    and [`SpecEnum`][seroverse.core.collections.SpecEnum]. Every `Spec` defines a canonical ASCII/UTF-8
    `name: bytes` attribute. Subclasses may add domain-specific fields (e.g. abbreviations, category
    enums, bitmasks, or numerical priors) and optionally override
    [`lookup_keys`][seroverse.core.collections.Spec.lookup_keys] to register secondary lookup keys.
    """

    name: bytes

    @property
    def name_str(self) -> str:
        r"""Canonical specification name decoded as a UTF-8 `str`."""
        return self.name.decode("utf-8", errors="replace")

    def lookup_keys(self) -> tuple[bytes, ...]:
        r"""Return normalized lowercase `bytes` keys used to index this spec in a [`SpecContainer`][seroverse.core.collections.SpecContainer]."""
        return (self.name.strip().lower(),)


@dataclass(frozen=True, slots=True)
class SpecContainer[T: Spec = Spec](BatchedContainer[T]):
    r"""Immutable, read-only Structure-of-Arrays (SoA) database container for [`Spec`][seroverse.core.collections.Spec] records.

    Extends [`BatchedContainer`][seroverse.core.collections.BatchedContainer] with:
    - A canonical `names: BytesArray` column.
    - Automatic read-only array freezing (`arr.setflags(write=False)`) across all registered NumPy
      arrays upon construction (`__post_init__`).
    - Lazy $O(1)$ case-insensitive name/key lookup via [`.index(key)`][seroverse.core.collections.SpecContainer.index],
      [`.resolve_index(key)`][seroverse.core.collections.SpecContainer.resolve_index], `key in container`,
      and `container[key]` (`str | bytes` indexing).
    """

    names: BytesArray
    _lookup_cache: dict[bytes, int] | None = field(default=None, init=False, repr=False, compare=False, hash=False)
    _scalar_type: ClassVar[type[Any] | None] = Spec

    def __post_init__(self) -> None:
        r"""Freeze all NumPy arrays in the container as read-only (`write=False`)."""
        for f in self._field_dtypes:
            arr = getattr(self, f)
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)

    def _get_lookup(self) -> dict[bytes, int]:
        r"""Build and cache the case-insensitive `bytes -> int` lookup dictionary."""
        cache = self._lookup_cache
        if cache is not None:
            return cache
        lookup: dict[bytes, int] = {}
        scalar_cls = self._scalar_type or Spec
        has_custom_keys = scalar_cls.lookup_keys is not Spec.lookup_keys
        n = len(self)
        for i in range(n):
            raw_name = bytes(self.names[i]).strip().lower()
            if raw_name:
                lookup[raw_name] = i
            if has_custom_keys:
                for k in self._get_scalar(i).lookup_keys():
                    norm_k = k.strip().lower()
                    if norm_k:
                        lookup[norm_k] = i
        object.__setattr__(self, "_lookup_cache", lookup)
        return lookup

    def index(self, key: str | bytes | int | np.integer[Any]) -> int:
        r"""Resolve a `Spec` name, lookup key, or integer index to its 0-based integer index.

        Args:
            key: Canonical name (`str` or `bytes`, case-insensitive), secondary lookup key, or integer index.

        Returns:
            int: Normalized 0-based index in `0..len(self) - 1`.

        Raises:
            IndexError: If `key` is an out-of-range integer.
            KeyError: If `key` is a `str` or `bytes` not found in the container.
        """
        if isinstance(key, (int, np.integer)):
            return self._resolve_index(key)
        b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
        norm = b.strip().lower()
        lookup = self._get_lookup()
        if norm in lookup:
            return lookup[norm]
        raise KeyError(f"Unknown {type(self).__name__} key: {key!r}")

    def resolve_index(self, key: str | bytes | int | np.integer[Any], default: int = -1) -> int:
        r"""Resolve a `Spec` name, lookup key, or integer index, returning `default` if not found."""
        try:
            return self.index(key)
        except KeyError, IndexError:
            return default

    def __contains__(self, key: object) -> bool:
        r"""Return `True` if `key` (`str`, `bytes`, or `Spec`) exists in this container."""
        if isinstance(key, Spec):
            return key.name.strip().lower() in self._get_lookup()
        if isinstance(key, (str, bytes)):
            b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            return b.strip().lower() in self._get_lookup()
        return False

    @overload
    def __getitem__(self, item: int | np.integer[Any] | str | bytes) -> T: ...

    @overload
    def __getitem__(self, item: slice | np.ndarray[tuple[int], Any] | list[Any]) -> Self: ...

    @overload
    def __getitem__(
        self,
        item: int | np.integer[Any] | str | bytes | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> T | Self: ...

    def __getitem__(
        self,
        item: int | np.integer[Any] | str | bytes | slice | np.ndarray[tuple[int], Any] | list[Any],
    ) -> T | Self:
        r"""Access a single `Spec` view `T` by integer index or name (`str | bytes`), or slice `Self`."""
        if isinstance(item, (int, np.integer)):
            return self._get_scalar(self._resolve_index(item))
        if isinstance(item, (str, bytes)):
            return self._get_scalar(self.index(item))
        return self._get_slice(item)


_ENUM_SPECS_CACHE: dict[type[Any], SpecContainer[Any]] = {}
_ENUM_LOOKUP_CACHE: dict[type[Any], dict[bytes, Any]] = {}


class SpecEnum(IntEnum):
    r"""Base `IntEnum` for 0-based contiguous ontology registries backed by a singleton [`SpecContainer`][seroverse.core.collections.SpecContainer].

    Subclasses declare standard 0-based integer enum members (`MEMBER = 0..N-1`) and optionally override
    `_compile_specs` to return a domain-specific [`SpecContainer`][seroverse.core.collections.SpecContainer]
    (if not overridden, a default `SpecContainer[Spec]` is compiled from the lowercase
    enum member names).

    Provides out-of-the-box:
    - [`cls.specs()`][seroverse.core.collections.SpecEnum.specs]: Cached singleton `SpecContainer` validated
      to have 1-to-1 contiguous correspondence (`0..N-1`) with `cls`.
    - [`self.spec`][seroverse.core.collections.SpecEnum.spec]: O(1) scalar `Spec` view for the enum member.
    - [`self.name_bytes`][seroverse.core.collections.SpecEnum.name_bytes] and
      [`self.name_str`][seroverse.core.collections.SpecEnum.name_str]: Canonical `Spec` name accessors.
    - [`cls.from_name(key)`][seroverse.core.collections.SpecEnum.from_name] and
      [`cls.resolve(key, default=...)`][seroverse.core.collections.SpecEnum.resolve]: O(1) case-insensitive
      resolution across enum instances, integer IDs, enum member names (`_`, `-`, or space-separated),
      canonical `Spec.name`, and any secondary [`Spec.lookup_keys`][seroverse.core.collections.Spec.lookup_keys].
    """

    @classmethod
    def _compile_specs(cls) -> SpecContainer[Any]:
        r"""Compile the singleton [`SpecContainer`][seroverse.core.collections.SpecContainer] for this enum class."""
        return SpecContainer.from_list([Spec(name=m.name.lower().encode("ascii")) for m in cls])

    @classmethod
    def specs(cls) -> SpecContainer[Any]:
        r"""Return the cached, validated singleton [`SpecContainer`][seroverse.core.collections.SpecContainer] for this enum."""
        cached = _ENUM_SPECS_CACHE.get(cls)
        if cached is not None:
            return cached
        members = tuple(cls)
        expected_ids = tuple(range(len(members)))
        actual_ids = tuple(int(m) for m in members)
        if actual_ids != expected_ids:
            raise ValueError(
                f"{cls.__name__} members must be contiguous 0-based integers 0..{len(members) - 1}, got {actual_ids}"
            )
        container = cls._compile_specs()
        if len(container) != len(members):
            raise ValueError(
                f"{cls.__name__}._compile_specs() returned {len(container)} specs for {len(members)} enum members"
            )
        _ENUM_SPECS_CACHE[cls] = container
        return container

    @property
    def spec(self) -> Any:
        r"""Return the scalar [`Spec`][seroverse.core.collections.Spec] view for this enum member."""
        return self.specs()[int(self)]

    @property
    def name_bytes(self) -> bytes:
        r"""Canonical `Spec.name` as raw ASCII/UTF-8 `bytes`."""
        return bytes(self.specs().names[int(self)])

    @property
    def name_str(self) -> str:
        r"""Canonical `Spec.name` decoded as a UTF-8 `str`."""
        return self.name_bytes.decode("utf-8", errors="replace")

    @classmethod
    def _get_lookup(cls) -> dict[bytes, Self]:
        r"""Build and cache the case-insensitive `bytes -> Self` lookup table for this enum."""
        cached = _ENUM_LOOKUP_CACHE.get(cls)
        if cached is not None:
            return cached
        specs = cls.specs()
        lookup: dict[bytes, Self] = {}
        for m in cls:
            low_member = m.name.lower()
            lookup[low_member.encode("ascii")] = m
            lookup[low_member.replace("_", " ").encode("ascii")] = m
            lookup[low_member.replace("_", "-").encode("ascii")] = m
        for key_bytes, idx in specs._get_lookup().items():
            lookup[key_bytes] = cls(idx)
        _ENUM_LOOKUP_CACHE[cls] = lookup
        return lookup

    @classmethod
    def from_name(cls, key: str | bytes | int | np.integer[Any] | Self) -> Self:
        r"""Resolve an enum member by instance, integer value, member name, or `Spec` name/key.

        Args:
            key: Enum member, 0-based integer value, or string/bytes identifier (case-insensitive).

        Returns:
            Self: Resolved enum member.

        Raises:
            ValueError: If `key` is an integer not corresponding to a valid enum member.
            KeyError: If `key` is a `str` or `bytes` not found in the lookup table.
        """
        if isinstance(key, cls):
            return key
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            return cls(int(key))
        if isinstance(key, (str, bytes)):
            b = key.encode("utf-8") if isinstance(key, str) else bytes(key)
            norm = b.strip().lower()
            lookup = cls._get_lookup()
            if norm in lookup:
                return lookup[norm]
        raise KeyError(f"Unknown {cls.__name__} identifier: {key!r}")

    @classmethod
    def resolve(
        cls,
        key: str | bytes | int | np.integer[Any] | Self,
        default: Self | None = None,
    ) -> Self:
        r"""Resolve an enum identifier, falling back to `default` (or `cls.UNKNOWN` if defined).

        Args:
            key: Enum member, 0-based integer value, or string/bytes identifier.
            default: Explicit fallback member if `key` cannot be resolved. If `None`, falls back
                to `cls.UNKNOWN` when defined on `cls`.

        Returns:
            Self: Resolved enum member or fallback.

        Raises:
            KeyError: If `key` cannot be resolved and neither `default` nor `cls.UNKNOWN` is available.
        """
        try:
            return cls.from_name(key)
        except KeyError, ValueError:
            if default is not None:
                return default
            unknown = cls.__members__.get("UNKNOWN")
            if unknown is not None:
                return unknown
            raise


# Kernels --------------------------------------------------------------------------------------------------------------
@njit(cache=True, nogil=True, parallel=True)
def _gather_ragged_kernel(
    data: np.ndarray[tuple[int], Any],
    offsets: Int32Array,
    new_offsets: Int32Array,
    new_lengths: Int32Array,
    indices: Int32Array,
    total_len: int,
) -> np.ndarray[tuple[int], Any]:
    r"""Parallel Numba kernel to gather and compact ragged slices for any 1D NumPy dtype."""
    out = np.empty(total_len, dtype=data.dtype)
    n = len(indices)
    for i in prange(n):
        length_val = new_lengths[i]
        if length_val == 0:
            continue
        src_off = offsets[indices[i]]
        dst_off = new_offsets[i]
        for c in range(length_val):
            out[dst_off + c] = data[src_off + c]
    return out


@njit(cache=True, nogil=True, parallel=True)
def _pad_ragged_kernel(
    data: np.ndarray[tuple[int], Any],
    offsets: Int32Array,
    lengths: Int32Array,
    max_len: int,
    fill_value: Any,
) -> np.ndarray[tuple[int, int], Any]:
    r"""Parallel Numba kernel to pack a ragged 1D buffer into a dense 2D `(N, max_len)` array."""
    n = len(lengths)
    out = np.full((n, max_len), fill_value, dtype=data.dtype)
    for i in prange(n):
        row_len = lengths[i]
        if row_len > max_len:
            row_len = max_len
        src_off = offsets[i]
        for c in range(row_len):
            out[i, c] = data[src_off + c]
    return out
