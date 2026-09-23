"""lzo backend — python-lzo, one-shot scoring.

LZO is a dictionary-less LZ77 variant tuned for decompression speed. The
python bindings expose only one-shot ``lzo.compress``, so scoring
recompresses ``context + candidate`` in full. The package has no Windows
wheels in practice, so this backend will commonly report itself
unavailable — the registry handles that gracefully.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .base import SPEED_FAST, Backend, OneShotScorer, OptionalBackend, Scorer


class LzoBackend(OptionalBackend):
    name = "lzo"
    family = "lz77-fast"
    package = "python-lzo"
    module: ClassVar[str] = "lzo"
    speed = SPEED_FAST
    context_limit = 16384

    def prepare(self, context: bytes, *, level: int = 9, **_: object) -> Scorer:
        lzo = self._import()
        return OneShotScorer(lambda d: lzo.compress(d, level), context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("lzo", LzoBackend)
