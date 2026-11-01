r"""Core NumPy array type aliases, runtime `np.dtype` resolution, and Numba JIT signature derivation via [`Types`][seroverse.core.types.Types]."""

import annotationlib
import ast
import inspect
import itertools
import textwrap
from collections.abc import Callable, Mapping
from types import NoneType, UnionType
from typing import Any, ClassVar, Literal, Never, TypeAliasType, TypeVar, Union, final, get_args, get_origin, overload

import numba as nb
import numpy as np
from numba.core.typing.templates import Signature

_NUMERIC_SCALAR_TYPES: tuple[type[np.generic], ...] = (
    np.bool_,
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
    np.float32,
    np.float64,
)

_ABSTRACT_NP_GENERICS: frozenset[type[Any]] = frozenset(
    {
        np.generic,
        np.number,
        np.integer,
        np.signedinteger,
        np.unsignedinteger,
        np.inexact,
        np.floating,
        np.complexfloating,
        np.flexible,
        np.character,
    }
)


@final
class Types:
    r"""Static namespace of 1D/2D/3D NumPy array type aliases, `np.dtype` resolution, and `@Types.njit` compilation."""

    # 1D NumPy Array Type Aliases
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

    # 2D NumPy Array Type Aliases
    type BoolArray2D = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    type Int8Array2D = np.ndarray[tuple[int, int], np.dtype[np.int8]]
    type UInt8Array2D = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
    type Int16Array2D = np.ndarray[tuple[int, int], np.dtype[np.int16]]
    type UInt16Array2D = np.ndarray[tuple[int, int], np.dtype[np.uint16]]
    type Int32Array2D = np.ndarray[tuple[int, int], np.dtype[np.int32]]
    type UInt32Array2D = np.ndarray[tuple[int, int], np.dtype[np.uint32]]
    type Int64Array2D = np.ndarray[tuple[int, int], np.dtype[np.int64]]
    type UInt64Array2D = np.ndarray[tuple[int, int], np.dtype[np.uint64]]
    type Float32Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]
    type Float64Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]

    # 3D NumPy Array Type Aliases
    type Int32Array3D = np.ndarray[tuple[int, int, int], np.dtype[np.int32]]
    type Float32Array3D = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
    type Float64Array3D = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

    _DTYPE_MAP: ClassVar[dict[str, np.dtype[Any]]] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Never:
        raise TypeError("Types is a static namespace class and cannot be instantiated.")

    @classmethod
    def _extract_ndarray_spec(
        cls,
        tp: Any,
        type_env: Mapping[TypeVar, Any] | None = None,
    ) -> tuple[np.dtype[Any], int] | None:
        r"""Extract `(np.dtype, ndim)` from a `np.ndarray[Shape, np.dtype[Scalar]]` generic alias."""
        if tp is np.ndarray:
            return np.dtype(np.float64), 1
        if get_origin(tp) is np.ndarray:
            args = get_args(tp)
            ndim = 1
            if len(args) >= 1 and get_origin(args[0]) is tuple:
                shape_args = get_args(args[0])
                if shape_args and Ellipsis not in shape_args:
                    ndim = len(shape_args)
            if len(args) >= 2 and (dtype_inner := get_args(args[1])):
                scalar_t = dtype_inner[0]
                if isinstance(scalar_t, TypeVar) and type_env is not None and scalar_t in type_env:
                    scalar_t = type_env[scalar_t]
                if scalar_t is np.bytes_:
                    return np.dtype("S"), ndim
                if (
                    isinstance(scalar_t, type)
                    and issubclass(scalar_t, np.generic)
                    and scalar_t not in _ABSTRACT_NP_GENERICS
                ):
                    return np.dtype(scalar_t), ndim
                return None
            return np.dtype(np.float64), ndim
        return None

    @classmethod
    def _extract_ndarray_dtype(cls, tp: Any) -> np.dtype[Any] | None:
        r"""Extract `np.dtype` from a 1D `np.ndarray[tuple[int], np.dtype[Scalar]]` generic alias."""
        spec = cls._extract_ndarray_spec(tp)
        if spec is not None and spec[1] == 1:
            return spec[0]
        return None

    @classmethod
    def _get_dtype_map(cls) -> dict[str, np.dtype[Any]]:
        r"""Lazily build and cache the string-to-dtype map from declared 1D `TypeAliasType` attributes."""
        if not cls._DTYPE_MAP:
            mapping: dict[str, np.dtype[Any]] = {}
            for name, val in cls.__dict__.items():
                if isinstance(val, TypeAliasType):
                    dt = cls._extract_ndarray_dtype(val.__value__)
                    if dt is not None:
                        scalar_name = dt.type.__name__
                        mapping[name] = dt
                        mapping[f"Types.{name}"] = dt
                        mapping[scalar_name] = dt
                        mapping[f"np.{scalar_name}"] = dt
                        if scalar_name == "bool":
                            mapping["bool_"] = dt
                            mapping["np.bool_"] = dt
            cls._DTYPE_MAP = mapping
        return cls._DTYPE_MAP

    @classmethod
    def resolve_dtype(cls, annotation: Any) -> np.dtype[Any] | None:
        r"""Resolve a 1D `Types.*Array` alias, `np.ndarray` generic, or string annotation to an `np.dtype`."""
        ann = annotation
        while isinstance(ann, TypeAliasType):
            ann = ann.__value__

        if (dt := cls._extract_ndarray_dtype(ann)) is not None:
            return dt

        if isinstance(ann, str):
            dtype_map = cls._get_dtype_map()
            if ann in dtype_map:
                return dtype_map[ann]
            if "ndarray" in ann and "tuple[int, int" not in ann:
                for key, dt in dtype_map.items():
                    if f"dtype[{key}]" in ann:
                        return dt
        return None

    @classmethod
    def to_numba_type(
        cls,
        annotation: Any,
        *,
        is_return: bool = False,
        readonly: bool = False,
        type_env: Mapping[TypeVar, Any] | None = None,
    ) -> tuple[nb.types.Type, ...] | None:
        r"""Resolve a Python/NumPy/`Types` annotation into a tuple of Numba types (expanding unions)."""
        ann = annotation
        if isinstance(ann, TypeVar):
            if type_env is not None and ann in type_env:
                ann = type_env[ann]
            else:
                return None

        while isinstance(ann, TypeAliasType):
            ann = ann.__value__

        if ann is None or ann is NoneType:
            return (nb.void,) if is_return else (nb.types.none,)

        origin = get_origin(ann)
        if origin is UnionType or origin is Union:
            branches: list[nb.types.Type] = []
            for branch in get_args(ann):
                resolved = cls.to_numba_type(branch, is_return=is_return, readonly=readonly, type_env=type_env)
                if resolved is None:
                    return None
                for t in resolved:
                    if t not in branches:
                        branches.append(t)
            return tuple(branches)

        if origin is tuple:
            args = get_args(ann)
            if not args:
                return (nb.types.Tuple(()),)
            if Ellipsis in args:
                return None
            elem_options: list[tuple[nb.types.Type, ...]] = []
            for a in args:
                resolved = cls.to_numba_type(a, is_return=False, readonly=readonly, type_env=type_env)
                if resolved is None:
                    return None
                elem_options.append(resolved)
            return tuple(nb.types.Tuple(combo) for combo in itertools.product(*elem_options))

        if (spec := cls._extract_ndarray_spec(ann, type_env=type_env)) is not None:
            dt, ndim = spec
            if dt.kind in ("S", "U", "O", "V"):
                return None
            return (nb.types.Array(nb.from_dtype(dt), ndim, "A", readonly=readonly and not is_return),)

        if ann is int:
            return (nb.int64,)
        if ann is float:
            return (nb.float64,)
        if ann is bool:
            return (nb.boolean,)
        if isinstance(ann, type) and issubclass(ann, np.generic) and ann not in _ABSTRACT_NP_GENERICS:
            return (nb.from_dtype(np.dtype(ann)),)

        return None

    @classmethod
    def _writable_params(cls, fn: Callable[..., Any], param_names: set[str]) -> set[str]:
        r"""Identify array parameter names that are mutated in-place or returned directly by `fn`."""
        try:
            src = textwrap.dedent(inspect.getsource(fn))
            tree = ast.parse(src)
        except Exception:
            return set()

        writable: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                curr: ast.AST = node.value
                while isinstance(curr, ast.Subscript):
                    curr = curr.value
                if isinstance(curr, ast.Name) and curr.id in param_names:
                    writable.add(curr.id)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("fill", "sort", "ravel")
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in param_names
            ):
                writable.add(node.func.value.id)
            elif isinstance(node, ast.Return) and node.value is not None:
                ret_elts = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
                for elt in ret_elts:
                    if isinstance(elt, ast.Name) and elt.id in param_names:
                        writable.add(elt.id)
        return writable

    @classmethod
    def numba_signature(cls, fn: Callable[..., Any]) -> Signature | list[Signature] | None:
        r"""Derive an eager Numba `Signature` (or `list[Signature]`) from a function's `Types` annotations."""
        try:
            hints = annotationlib.get_annotations(fn, format=annotationlib.Format.VALUE)
        except Exception:
            return None

        if "return" not in hints:
            return None

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if any(p.name not in hints for p in params):
            return None

        writable_set = cls._writable_params(fn, {p.name for p in params})

        type_params: tuple[Any, ...] = getattr(fn, "__type_params__", ())
        envs: list[dict[TypeVar, Any]] = [{}]
        for tp in type_params:
            if not isinstance(tp, TypeVar):
                return None
            if tp.__constraints__:
                candidates = tp.__constraints__
            elif tp.__bound__ is np.generic:
                candidates = _NUMERIC_SCALAR_TYPES
            else:
                return None
            envs = [{**env, tp: c} for env in envs for c in candidates]

        signatures: list[Signature] = []
        for env in envs:
            ret_types = cls.to_numba_type(hints["return"], is_return=True, readonly=False, type_env=env)
            if not ret_types:
                return None

            param_options: list[tuple[nb.types.Type, ...]] = []
            for p in params:
                is_ro = p.name not in writable_set
                p_types = cls.to_numba_type(hints[p.name], is_return=False, readonly=is_ro, type_env=env)
                if not p_types:
                    return None
                if p.default is not inspect.Parameter.empty:
                    p_types = (*p_types, nb.types.Omitted(p.default))
                param_options.append(p_types)

            for ret_t in ret_types:
                for arg_combo in itertools.product(*param_options):
                    compiled_sig = ret_t(*arg_combo)
                    if compiled_sig not in signatures:
                        signatures.append(compiled_sig)

        if not signatures:
            return None
        return signatures[0] if len(signatures) == 1 else signatures

    @overload
    @classmethod
    def njit[**P, R](cls, fn: Callable[P, R], /) -> Callable[P, R]: ...

    @overload
    @classmethod
    def njit[**P, R](
        cls,
        *,
        cache: bool = True,
        nogil: bool = True,
        parallel: bool = False,
        fastmath: bool = False,
        inline: Literal["never", "always"] = "never",
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    @classmethod
    def njit(
        cls,
        fn: Callable[..., Any] | None = None,
        /,
        *,
        cache: bool = True,
        nogil: bool = True,
        parallel: bool = False,
        fastmath: bool = False,
        inline: Literal["never", "always"] = "never",
    ) -> Any:
        r"""Compile a Numba `@njit` kernel with `nogil=True, cache=True` and eager `Signature` from annotations."""

        def _decorate(func: Callable[..., Any]) -> Any:
            sig = cls.numba_signature(func)
            if sig is None:
                qualname = getattr(func, "__qualname__", repr(func))
                raise TypeError(
                    f"Cannot derive Numba signature for {qualname}: "
                    "ensure all parameters and the return type have concrete Types annotations."
                )
            return nb.njit(
                sig,
                cache=cache,
                nogil=nogil,
                parallel=parallel,
                fastmath=fastmath,
                inline=inline,
            )(func)

        if fn is not None:
            return _decorate(fn)
        return _decorate
