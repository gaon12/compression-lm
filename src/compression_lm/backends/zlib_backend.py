"""zlib-family backends: raw deflate, zlib stream, and gzip.

zlib exposes a persistent, clonable encoder state. The context is fed into a
``compressobj`` once; each candidate is then scored on a cheap copy of that
live encoder, so scoring costs O(candidate) rather than O(context). This is
the fast path every other backend family is measured against.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from typing import ClassVar

from .base import SPEED_FAST, Backend, Scorer

_WBITS: dict[str, int] = {
    "deflate": -15,  # raw LZ77+Huffman stream, no container
    "zlib": 15,  # RFC 1950 wrapper (2-byte header + adler32)
    "gzip": 31,  # RFC 1952 wrapper (10-byte header + crc32)
}


class _ZlibScorer(Scorer):
    """Scores candidates on clones of a context-primed encoder."""

    def __init__(self, context: bytes, level: int, wbits: int) -> None:
        self.context = context
        self._encoder = zlib.compressobj(level, zlib.DEFLATED, wbits)
        self._encoder.compress(context)
        # Cost of terminating a copy with no further input: final-block and
        # trailer bytes. Subtracted so measure() reports the marginal cost
        # of the candidate itself.
        probe = self._encoder.copy()
        self._overhead = len(probe.flush(zlib.Z_FINISH))

    def measure(self, candidate: bytes) -> int:
        clone = self._encoder.copy()
        emitted = clone.compress(candidate) + clone.flush(zlib.Z_FINISH)
        return len(emitted) - self._overhead


class ZlibBackend(Backend):
    """deflate/zlib/gzip via the standard library's incremental encoder."""

    family: ClassVar[str] = "deflate"
    speed: ClassVar[str] = SPEED_FAST
    context_limit: ClassVar[int] = 32768  # deflate's 32 KiB match window

    def __init__(self, name: str, wbits: int) -> None:
        self.name = name
        self._wbits = wbits

    def prepare(self, context: bytes, *, level: int = 9, **_: object) -> Scorer:
        return _ZlibScorer(context, level=level, wbits=self._wbits)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    for name, wbits in _WBITS.items():
        register(name, lambda n=name, w=wbits: ZlibBackend(n, w))
