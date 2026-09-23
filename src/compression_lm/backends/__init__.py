"""Registry of compressor backends.

Backends are constructed lazily and probed for availability so that missing
optional dependencies degrade gracefully: they remain listed, marked
unavailable, and are skipped by comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .base import Backend

#: name -> zero-argument factory returning a Backend instance.
#: Factories are lazy so importing this module never imports optional deps.
_REGISTRY: dict[str, Callable[[], Backend]] = {}


def register(name: str, factory: Callable[[], Backend]) -> None:
    """Register a backend under ``name``. Intended for module setup/tests."""
    _REGISTRY[name] = factory


def _register_builtins() -> None:
    from . import zlib_backend

    zlib_backend.register_into(register)


@dataclass(frozen=True)
class BackendInfo:
    """Static description of a registered backend."""

    name: str
    family: str
    package: str
    context_limit: int
    speed: str
    available: bool
    unavailable_reason: str | None


def names() -> list[str]:
    """All registered backend names, sorted."""
    return sorted(_REGISTRY)


def get(name: str) -> Backend:
    """Instantiate the backend registered under ``name``.

    Raises ``KeyError`` for unknown names and ``RuntimeError`` if the
    backend's optional dependency is missing.
    """
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; registered: {', '.join(names())}") from None
    backend = factory()
    if not backend.available():
        raise RuntimeError(f"backend {name!r} unavailable: {backend.unavailable_reason()}")
    return backend


def info(name: str) -> BackendInfo:
    """Describe a backend without requiring its dependency to be installed."""
    factory = _REGISTRY[name]  # KeyError propagates; use names() to enumerate
    backend = factory()
    cls = type(backend)
    return BackendInfo(
        name=name,
        family=backend.family or name,
        package=backend.package,
        context_limit=backend.context_limit,
        speed=backend.speed,
        available=cls.available(),
        unavailable_reason=cls.unavailable_reason(),
    )


def all_info() -> Iterator[BackendInfo]:
    """Describe every registered backend."""
    for name in names():
        yield info(name)


_register_builtins()
