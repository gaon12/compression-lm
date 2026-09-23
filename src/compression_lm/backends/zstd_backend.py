"""zstd backend — zstandard package, dictionary-primed scoring.

zstd supports *raw content* dictionaries: arbitrary bytes usable as match
history without any training step. Scoring a candidate means compressing it
against a dictionary built from the context — the dictionary plays the role
of the trained model, and the emitted size is the candidate's code length
under it. Candidates compress alone, so each call is O(candidate).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

from .base import SPEED_FAST, Backend, DictScorer, OptionalBackend, Scorer


class ZstdBackend(OptionalBackend):
    name = "zstd"
    family = "lz77+entropy"
    package = "zstandard"
    module: ClassVar[str] = "zstandard"
    speed = SPEED_FAST
    context_limit = 32768  # keep parity with deflate's window for comparability

    def prepare(self, context: bytes, *, level: int = 3, **_: object) -> Scorer:
        zstd = self._import()
        dict_data = None
        if context:
            dict_data = zstd.ZstdCompressionDict(context, dict_type=zstd.DICT_TYPE_RAWCONTENT)
        compressor = zstd.ZstdCompressor(level=level, dict_data=dict_data)
        return DictScorer(compressor.compress, context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("zstd", ZstdBackend)
