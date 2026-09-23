"""Backend abstraction for compressor-based scoring.

A *backend* wraps a general-purpose compression algorithm. Given a context
(corpus excerpt plus the recent output the compressor is primed with), a
backend produces a :class:`Scorer` that estimates the *conditional code
length* of a candidate continuation: roughly, how many extra bytes the
compressor needs to encode the candidate after having consumed the context.

``measure(candidate) ≈ len(compress(context + candidate)) - len(compress(context))``

Shorter conditional code length means the candidate is more predictable
under everything the compressor has seen — which is what lets a compressor
act like a language model. Backends differ only in *how* they compute this:
some reuse encoder state (zlib), some compress against a dictionary (zstd),
and the rest fall back to a full recompression per candidate.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import ClassVar

#: Speed classes used to scale default search budgets per backend.
SPEED_FAST = "fast"
SPEED_MEDIUM = "medium"
SPEED_SLOW = "slow"


class Scorer(ABC):
    """Conditional code-length estimator bound to a fixed context.

    Implementations must make :meth:`measure` safe to call from multiple
    threads concurrently (each call must use independent encoder state) so
    that callers may fan out batch scoring over a thread pool — most
    compression libraries release the GIL while compressing.
    """

    #: The context this scorer was prepared with.
    context: bytes

    @abstractmethod
    def measure(self, candidate: bytes) -> int:
        """Extra compressed bytes attributable to ``candidate``.

        Equal to ``len(compress(context + candidate)) - len(compress(context))``
        up to a small constant for per-stream framing overhead. May be
        slightly negative for empty/degenerate inputs; callers only compare
        values produced by the same scorer.
        """
        ...

    def measure_many(self, candidates: Sequence[bytes]) -> list[int]:
        """Measure every candidate; default is sequential."""
        return [self.measure(c) for c in candidates]


class OneShotScorer(Scorer):
    """Scorer for compressors that expose only a one-shot compress function.

    Recompresses ``context + candidate`` in full for every candidate.
    Universally applicable but costs O(context) per candidate, so backends
    using it set a conservative ``context_limit``.
    """

    def __init__(self, compress: Callable[[bytes], bytes], context: bytes) -> None:
        self._compress = compress
        self.context = context
        self._base_len = len(compress(context))

    def measure(self, candidate: bytes) -> int:
        return len(self._compress(self.context + candidate)) - self._base_len


class Backend(ABC):
    """A named compression algorithm that can score continuations."""

    #: Unique lowercase identifier used on the CLI and in reports.
    name: ClassVar[str]
    #: Human-facing compression family label for reports.
    family: ClassVar[str] = ""
    #: pip extra/package that provides the implementation ("stdlib" if none).
    package: ClassVar[str] = "stdlib"
    #: Recommended upper bound on context bytes for reasonable latency.
    context_limit: ClassVar[int] = 32768
    #: Throughput hint: "fast" | "medium" | "slow".
    speed: ClassVar[str] = SPEED_MEDIUM

    @classmethod
    def available(cls) -> bool:
        """Whether the underlying implementation can be used right now."""
        return True

    @classmethod
    def unavailable_reason(cls) -> str | None:
        """Human-readable reason the backend is unusable, or None."""
        return None if cls.available() else "unavailable"

    @abstractmethod
    def prepare(self, context: bytes, **options) -> Scorer:
        """Bind a scorer to ``context``. Options are backend-specific."""
        ...


class OptionalBackend(Backend):
    """Backend whose implementation lives in an optional third-party package."""

    #: Importable module name, e.g. "zstandard".
    module: ClassVar[str]

    @classmethod
    def _import(cls):
        return importlib.import_module(cls.module)

    @classmethod
    def available(cls) -> bool:
        try:
            cls._import()
        except ImportError:
            return False
        return True

    @classmethod
    def unavailable_reason(cls) -> str | None:
        if cls.available():
            return None
        return f"optional dependency '{cls.package}' is not installed"
