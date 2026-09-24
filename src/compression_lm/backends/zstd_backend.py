"""zstd backend — zstandard package, dictionary-primed scoring.

zstd supports *raw content* dictionaries: arbitrary bytes usable as match
history without any training step. Scoring a candidate means compressing it
against a dictionary built from the context — the dictionary plays the role
of the trained model, and the emitted size is the candidate's code length
under it. Candidates compress alone, so each call is O(candidate).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import ClassVar

from .base import (
    SPEED_FAST,
    SPEED_SLOW,
    Backend,
    DictScorer,
    OneShotScorer,
    OptionalBackend,
    Scorer,
)


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
        # ZstdCompressor.compress() is not reentrant across threads — the
        # shared CCtx errors (or crashes) under concurrent calls. Keep one
        # compressor per thread instead.
        local = threading.local()

        def compress_with_dict(candidate: bytes) -> bytes:
            compressor = getattr(local, "compressor", None)
            if compressor is None:
                compressor = local.compressor = zstd.ZstdCompressor(
                    level=level, dict_data=dict_data
                )
            return compressor.compress(candidate)

        return DictScorer(compress_with_dict, context)


class ZstdDictBackend(OptionalBackend):
    """zstd with a *trained* dictionary — the dictionary-as-model experiment.

    Unlike :class:`ZstdBackend` (raw context as dict), this backend is
    constructed with a dictionary trained from the TRAIN split only —
    validation/test bytes never reach dictionary training. Scoring then
    compresses ``context + candidate`` as payload against that dictionary:
    the dict supplies corpus-level knowledge while the payload supplies
    local context. Kept as a separate backend name so its results are never
    mixed with plain zstd.
    """

    name = "zstd_dict"
    family = "lz77+entropy"
    package = "zstandard"
    module: ClassVar[str] = "zstandard"
    speed = SPEED_SLOW  # payload includes full context per candidate
    context_limit = 32768  # compute cap for the one-shot payload

    def __init__(self, dict_bytes: bytes, dict_kind: str = "trained") -> None:
        self.dict_bytes = dict_bytes
        self.dict_kind = dict_kind

    def prepare(self, context: bytes, *, level: int = 3, **_: object) -> Scorer:
        zstd = self._import()
        dict_data = zstd.ZstdCompressionDict(self.dict_bytes)
        local = threading.local()

        def compress(data: bytes) -> bytes:
            c = getattr(local, "compressor", None)
            if c is None:
                c = local.compressor = zstd.ZstdCompressor(level=level, dict_data=dict_data)
            return c.compress(data)

        return OneShotScorer(compress, context)


def register_into(register: Callable[[str, Callable[[], Backend]], None]) -> None:
    register("zstd", ZstdBackend)
