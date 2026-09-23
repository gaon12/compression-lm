"""bzip2 backend — stdlib bz2 via one-shot scoring.

``bz2.BZ2Compressor`` compresses incrementally but its state cannot be
cloned, so every candidate costs a full recompression of the context.
bzip2's Burrows-Wheeler transform has no match window at all — it exploits
repetition by sorting contexts together — which makes it an interesting
contrast to LZ77-family backends in comparisons.
"""

from __future__ import annotations

import bz2
from collections.abc import Callable

from .base import SPEED_SLOW, Backend, OneShotScorer, Scorer


class Bzip2Backend(Backend):
    name = "bzip2"
    family = "bwt"
    speed = SPEED_SLOW
    #: BWT blocks are large; small contexts keep per-candidate cost sane.
    context_limit = 8192

    def prepare(self, context: bytes, *, level: int = 9, **_: object) -> Scorer:
        return OneShotScorer(lambda d: bz2.compress(d, compresslevel=level), context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("bzip2", Bzip2Backend)
