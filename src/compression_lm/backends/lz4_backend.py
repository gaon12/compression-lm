"""lz4 backend — lz4 package, block format with dictionary scoring.

``lz4.block.compress`` accepts a ``dict`` argument (up to 64 KiB) that acts
as match history, so candidates are scored exactly like zstd's dictionary
mode. LZ4 trades ratio for speed, which makes it a useful baseline: how
much does generation quality drop for the fastest scorer?
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .base import SPEED_FAST, Backend, DictScorer, OptionalBackend, Scorer


class Lz4Backend(OptionalBackend):
    name = "lz4"
    family = "lz77-fast"
    package = "lz4"
    module: ClassVar[str] = "lz4.block"
    speed = SPEED_FAST
    context_limit = 65536  # LZ4 dictionaries are capped at 64 KiB

    def prepare(self, context: bytes, *, acceleration: int = 1, **_: object) -> Scorer:
        lz4_block = self._import()

        def with_dict(candidate: bytes) -> bytes:
            return lz4_block.compress(
                candidate,
                store_size=False,
                dict=context or None,
                acceleration=acceleration,
            )

        return DictScorer(with_dict, context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("lz4", Lz4Backend)
