"""v3 backend set — explicit construction, explicit scoring modes.

Every backend here is instantiated with its full option set visible (see
``options.py``); nothing relies on implicit defaults. The zstd experiment
gets three *separate* backend labels — ``zstd_full``, ``zstd_stream``,
``zstd_dict`` — plus ``zstd_dict_trained`` (the v2 trained-dictionary
variant), so the four scoring semantics are never merged into one row.

zstd streaming (``zstd_stream``): zstandard cannot clone a live compressor
state, so true "prepare once, score 256 times on the same stream" is not
expressible. The implemented marginal-cost semantic feeds ``context`` into
a fresh ``compressobj``, ends that block with ``FLUSH_BLOCK``, then feeds
the candidate and ends the frame (``FLUSH_FRAME``). zstd's window survives
block boundaries inside a frame, so candidates can still match back into
the context — this measures the encoded cost of the candidate *block*
following a completed context block. Because state cannot be reused, each
candidate costs O(context); the throughput note is recorded in the row.
"""

from __future__ import annotations

import threading
from typing import ClassVar

from ..backends.base import (
    SPEED_MEDIUM,
    OneShotScorer,
    OptionalBackend,
    Scorer,
)
from ..backends.brotli_backend import BrotliBackend
from ..backends.bzip2_backend import Bzip2Backend
from ..backends.lz4_backend import Lz4Backend
from ..backends.lzma_backend import LzmaBackend
from ..backends.lzo_backend import LzoBackend
from ..backends.snappy_backend import SnappyBackend
from ..backends.zlib_backend import ZlibBackend
from ..backends.zstd_backend import ZstdBackend, ZstdDictBackend


class ZstdFullBackend(OptionalBackend):
    """ZSTD-A: plain ``len(compress(ctx + cand)) - len(compress(ctx))``."""

    name = "zstd_full"
    family = "lz77+entropy"
    package = "zstandard"
    module: ClassVar[str] = "zstandard"
    speed = SPEED_MEDIUM  # one-shot O(context) per candidate
    context_limit = 32768

    def prepare(self, context: bytes, *, level: int = 3, **_: object) -> Scorer:
        zstd = self._import()
        local = threading.local()

        def compress(data: bytes) -> bytes:
            c = getattr(local, "compressor", None)
            if c is None:
                c = local.compressor = zstd.ZstdCompressor(level=level)
            return c.compress(data)

        return OneShotScorer(compress, context)


class ZstdRawBackend(OptionalBackend):
    """zstd without the frame magic — the rawest stream zstandard emits.

    Paired against ``zstd_full`` to isolate the frame header's contribution
    to score deltas (spec §10's "raw zstd vs zstd frame" pair).
    """

    name = "zstd_raw"
    family = "lz77+entropy"
    package = "zstandard"
    module: ClassVar[str] = "zstandard"
    speed = SPEED_MEDIUM
    context_limit = 32768

    def prepare(self, context: bytes, *, level: int = 3, **_: object) -> Scorer:
        zstd = self._import()
        params = zstd.ZstdCompressionParameters.from_level(
            level, format=zstd.FORMAT_ZSTD1_MAGICLESS
        )
        local = threading.local()

        def compress(data: bytes) -> bytes:
            c = getattr(local, "compressor", None)
            if c is None:
                c = local.compressor = zstd.ZstdCompressor(compression_params=params)
            return c.compress(data)

        return OneShotScorer(compress, context)


class _ZstdStreamScorer(Scorer):
    """Marginal encoded cost of a candidate *block* after a context block.

    Uses ``stream_writer`` (the only zstandard object that survives a
    ``FLUSH_BLOCK`` — ``compressobj().flush`` finishes the stream in
    zstandard 0.25). A fresh writer is created per candidate because stream
    state cannot be cloned; each measure therefore costs O(context).
    """

    def __init__(self, context: bytes, level: int) -> None:
        import zstandard

        self._zstd = zstandard
        self._level = level
        self.context = context
        self._local = threading.local()
        self._base_len = len(self._stream(context, None))

    def _writer(self):
        import io

        buf = io.BytesIO()
        c = getattr(self._local, "compressor", None)
        if c is None:
            c = self._local.compressor = self._zstd.ZstdCompressor(level=self._level)
        return c.stream_writer(buf, write_return_read=False, closefd=False), buf

    def _stream(self, context: bytes, candidate: bytes | None) -> bytes:
        w, buf = self._writer()
        w.write(context)
        if candidate is None:
            w.flush(self._zstd.FLUSH_FRAME)
            return buf.getvalue()
        w.flush(self._zstd.FLUSH_BLOCK)
        w.write(candidate)
        w.flush(self._zstd.FLUSH_FRAME)
        return buf.getvalue()

    def measure(self, candidate: bytes) -> int:
        return len(self._stream(self.context, candidate)) - self._base_len


class ZstdStreamBackend(OptionalBackend):
    """ZSTD-B: streaming marginal cost (FLUSH_BLOCK boundary after ctx)."""

    name = "zstd_stream"
    family = "lz77+entropy"
    package = "zstandard"
    module: ClassVar[str] = "zstandard"
    speed = SPEED_MEDIUM
    context_limit = 32768

    def prepare(self, context: bytes, *, level: int = 3, **_: object) -> Scorer:
        return _ZstdStreamScorer(context, level)


class ZstdDictRawBackend(ZstdBackend):
    """ZSTD-C: raw context used as a content dictionary (v2's ``zstd``)."""

    name = "zstd_dict"


class ZstdDictTrainedBackend(ZstdDictBackend):
    """v2 trained-dictionary variant — kept distinct from raw-dict mode."""

    name = "zstd_dict_trained"


def v3_factories() -> dict[str, object]:
    """Label -> zero-arg factory for the full v3 backend set.

    ``zstd_dict`` is the raw-context-as-dictionary mode (v2's ``zstd``
    backend); ``zstd_dict_trained`` is the v2 trained-dictionary variant.
    ``zstd_dict_trained`` is created lazily by the runner because it needs
    the trained dictionary bytes.
    """
    import lzma as _lzma

    return {
        "deflate": lambda: ZlibBackend("deflate", -15),
        "zlib": lambda: ZlibBackend("zlib", 15),
        "gzip": lambda: ZlibBackend("gzip", 31),
        "bzip2": Bzip2Backend,
        "lzma": lambda: LzmaBackend("lzma", _lzma.FORMAT_ALONE),
        "lzma2": lambda: LzmaBackend("lzma2", _lzma.FORMAT_RAW, use_filters=True),
        "xz": lambda: LzmaBackend("xz", _lzma.FORMAT_XZ),
        "brotli": BrotliBackend,
        "lz4": Lz4Backend,
        "snappy": SnappyBackend,
        "zstd_full": ZstdFullBackend,
        "zstd_raw": ZstdRawBackend,  # magicless frame — container-pair partner
        "zstd_stream": ZstdStreamBackend,
        "zstd_dict": ZstdDictRawBackend,  # raw context as dictionary (mode C)
        "lzo": LzoBackend,
    }


def check_backend(name: str):
    """Instantiate a v3 backend or return (None, reason) if unavailable."""
    factory = v3_factories()[name]
    try:
        be = factory()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not be.available():
        return None, be.unavailable_reason() or "unavailable"
    return be, None
