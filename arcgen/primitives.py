"""Primitive registry for ARC transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


PrimitiveFn = Callable[..., Any]


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata describing a primitive parameter."""

    name: str
    type: str
    description: str = ""
    default: Any = field(default_factory=lambda: _NoDefault)
    choices: Optional[Tuple[Any, ...]] = None

    def has_default(self) -> bool:
        return self.default is not _NoDefault


@dataclass(frozen=True)
class PrimitiveSpec:
    """Registered primitive entry."""

    name: str
    func: PrimitiveFn
    category: str
    description: str = ""
    parameters: Tuple[ParameterSpec, ...] = ()

    def apply(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


class PrimitiveRegistry:
    """Registry managing ARC transformation primitives."""

    def __init__(self) -> None:
        self._entries: Dict[str, PrimitiveSpec] = {}

    # -------------------------------------------------------------- registration
    def register(
        self,
        name: str,
        *,
        category: str,
        description: str = "",
        parameters: Iterable[ParameterSpec] | None = None,
    ) -> Callable[[PrimitiveFn], PrimitiveFn]:
        """Decorator to register a primitive function."""

        if name in self._entries:
            raise ValueError(f"primitive '{name}' is already registered")

        params_tuple = tuple(parameters or ())

        def decorator(func: PrimitiveFn) -> PrimitiveFn:
            spec = PrimitiveSpec(
                name=name,
                func=func,
                category=category,
                description=description,
                parameters=params_tuple,
            )
            self._entries[name] = spec
            return func

        return decorator

    # ------------------------------------------------------------------- queries
    def get(self, name: str) -> PrimitiveSpec:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown primitive '{name}'") from exc

    def list(self, *, category: str | None = None) -> List[PrimitiveSpec]:
        if category is None:
            return list(self._entries.values())
        return [spec for spec in self._entries.values() if spec.category == category]

    def categories(self) -> List[str]:
        return sorted({spec.category for spec in self._entries.values()})


class _NoDefaultType:
    pass


_NoDefault = _NoDefaultType()


# Global registry --------------------------------------------------------------
REGISTRY = PrimitiveRegistry()


def register_primitive(
    name: str,
    *,
    category: str,
    description: str = "",
    parameters: Iterable[ParameterSpec] | None = None,
) -> Callable[[PrimitiveFn], PrimitiveFn]:
    """Public helper to register primitives into the global registry."""

    return REGISTRY.register(
        name,
        category=category,
        description=description,
        parameters=parameters,
    )


__all__ = [
    "ParameterSpec",
    "PrimitiveSpec",
    "PrimitiveRegistry",
    "register_primitive",
    "REGISTRY",
]
