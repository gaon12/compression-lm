"""snappy backend — python-snappy, one-shot scoring.

The snappy format has no dictionary support, so every candidate is scored
by recompressing ``context + candidate`` in full. Snappy is extremely fast
(minimal parsing, byte-oriented matches), which partially offsets the
one-shot cost and makes it a worthwhile speed baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .base import SPEED_FAST, Backend, OneShotScorer, OptionalBackend, Scorer


class SnappyBackend(OptionalBackend):
    name = "snappy"
    family = "lz77-fast"
    package = "python-snappy"
    module: ClassVar[str] = "snappy"
    speed = SPEED_FAST
    #: Full recompression per candidate; 16 KiB keeps it cheap.
    context_limit = 16384

    def prepare(self, context: bytes, **_: object) -> Scorer:
        snappy = self._import()
        return OneShotScorer(snappy.compress, context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("snappy", SnappyBackend)
