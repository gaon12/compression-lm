"""LZMA-family backends — stdlib lzma in three stream shapes.

``xz``    : .xz container around LZMA2 (the format xz files use)
``lzma``  : legacy .lzma "alone" format (LZMA1)
``lzma2`` : bare LZMA2 stream with no container framing

``lzma.LZMACompressor`` state cannot be cloned, so scoring uses one-shot
recompression. LZMA's range coder and large match finder make it the
strongest — and slowest — stdlib scorer here; the context limit is kept
deliberately small.
"""

from __future__ import annotations

import lzma
from collections.abc import Callable

from .base import SPEED_SLOW, Backend, OneShotScorer, Scorer


class LzmaBackend(Backend):
    family = "lzma"
    speed = SPEED_SLOW
    #: Range coding over a big context per candidate is expensive.
    context_limit = 4096

    def __init__(self, name: str, fmt: int, use_filters: bool = False) -> None:
        self.name = name
        self._format = fmt
        self._use_filters = use_filters

    def prepare(self, context: bytes, *, preset: int = 6, **_: object) -> Scorer:
        fmt = self._format

        def compress(data: bytes) -> bytes:
            if self._use_filters:
                filters = [{"id": lzma.FILTER_LZMA2, "preset": preset}]
                return lzma.compress(data, format=fmt, filters=filters)
            return lzma.compress(data, format=fmt, preset=preset)

        return OneShotScorer(compress, context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("xz", lambda: LzmaBackend("xz", lzma.FORMAT_XZ))
    register("lzma", lambda: LzmaBackend("lzma", lzma.FORMAT_ALONE))
    register("lzma2", lambda: LzmaBackend("lzma2", lzma.FORMAT_RAW, use_filters=True))
