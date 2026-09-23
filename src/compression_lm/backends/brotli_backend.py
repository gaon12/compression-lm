"""brotli backend — google's brotli bindings, dictionary-primed when possible.

Recent brotli releases accept a ``dictionary`` argument to ``compress()``,
giving the same context-as-match-history scoring as zstd. Older bindings
lack it; in that case the backend falls back to one-shot recompression of
``context + candidate``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .base import (
    SPEED_MEDIUM,
    Backend,
    DictScorer,
    OneShotScorer,
    OptionalBackend,
    Scorer,
)


class BrotliBackend(OptionalBackend):
    name = "brotli"
    family = "lz77+static-dict"
    package = "brotli"
    module: ClassVar[str] = "brotli"
    speed = SPEED_MEDIUM
    context_limit = 32768

    def prepare(self, context: bytes, *, quality: int = 5, **_: object) -> Scorer:
        brotli = self._import()
        try:
            probe = brotli.compress(b"", quality=quality, dictionary=context)
        except TypeError:
            probe = None
        if probe is not None:

            def with_dict(candidate: bytes) -> bytes:
                return brotli.compress(candidate, quality=quality, dictionary=context)

            return DictScorer(with_dict, context)

        def full(data: bytes) -> bytes:
            return brotli.compress(data, quality=quality)

        return OneShotScorer(full, context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("brotli", BrotliBackend)
