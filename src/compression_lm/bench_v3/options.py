"""Explicit backend options for the v3 benchmark.

Every backend runs with a fully-specified option dict — the same dict is
passed to ``Backend.prepare`` and recorded verbatim in
``backend_options.csv`` / ``config.json``, so "what settings were used" is
never implicit. Field meanings follow each library's own documentation.

``scoring_mode`` records which conditional-code-length semantics the row
used: incremental encoder-state cloning, streaming marginal cost, raw or
trained dictionary, or full one-shot recompression.
"""

from __future__ import annotations

from ..bench.instrument import package_versions

#: label -> prepare() kwargs + recorded metadata. Keep in sync with
#: backends_v3.v3_factories().
BACKEND_OPTIONS: dict[str, dict] = {
    "deflate": {
        "level": 9,
        "window_bits": -15,
        "container": "raw deflate stream (no wrapper)",
        "checksum": "none",
        "streaming": "incremental compressobj, state cloned per candidate",
        "dictionary": "none",
        "scoring_mode": "incremental_state",
    },
    "zlib": {
        "level": 9,
        "window_bits": 15,
        "container": "zlib wrapper (2-byte header + adler32 trailer)",
        "checksum": "adler32",
        "streaming": "incremental compressobj, state cloned per candidate",
        "dictionary": "none",
        "scoring_mode": "incremental_state",
    },
    "gzip": {
        "level": 9,
        "window_bits": 31,
        "container": "gzip wrapper (10-byte header + crc32 + isize)",
        "checksum": "crc32",
        "streaming": "incremental compressobj, state cloned per candidate",
        "dictionary": "none",
        "scoring_mode": "incremental_state",
    },
    "bzip2": {
        "level": 9,
        "block_size": "9 * 100k",
        "container": "bzip2 stream (BZh header + CRC)",
        "checksum": "block+stream CRC32",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
    "lzma": {
        "preset": 6,
        "dict_size": "8 MiB (preset default)",
        "container": ".lzma alone (13-byte header: props+dict+uncomp size)",
        "checksum": "none",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
    "lzma2": {
        "preset": 6,
        "dict_size": "8 MiB (preset default)",
        "container": "raw LZMA2 chunk stream (no header)",
        "checksum": "none",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
    "xz": {
        "preset": 6,
        "dict_size": "8 MiB (preset default)",
        "container": "xz stream (12B header + block + index + footer)",
        "checksum": "crc32 (xz default)",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
    "brotli": {
        "quality": 5,
        "window": "lgwin=22 (library default)",
        "container": "raw brotli stream",
        "checksum": "none",
        "streaming": "one-shot per candidate",
        "dictionary": "raw context attached per call",
        "scoring_mode": "dictionary",
    },
    "lz4": {
        "acceleration": 1,
        "container": "LZ4 block format (no frame)",
        "checksum": "none",
        "streaming": "one-shot per candidate",
        "dictionary": "raw context as block dictionary (<=64 KiB)",
        "scoring_mode": "dictionary",
    },
    "snappy": {
        "level": "n/a (no levels)",
        "container": "snappy block format",
        "checksum": "none",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
    "zstd_full": {
        "level": 3,
        "container": "zstd frame (4-byte magic + header)",
        "checksum": "none (default, no checksum flag)",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
    "zstd_raw": {
        "level": 3,
        "container": "zstd frame without magic number (raw stream)",
        "checksum": "none (default)",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
    "zstd_stream": {
        "level": 3,
        "container": "zstd frame; FLUSH_BLOCK after context, FLUSH_FRAME at end",
        "checksum": "none (default)",
        "streaming": "compressobj; candidate encoded as block after ctx block",
        "dictionary": "none (window persists across the block boundary)",
        "scoring_mode": "stream_incremental",
    },
    "zstd_dict": {
        "level": 3,
        "container": "zstd frame",
        "checksum": "none (default)",
        "streaming": "one-shot payload per candidate",
        "dictionary": "raw context as DICT_TYPE_RAWCONTENT",
        "scoring_mode": "dictionary_raw",
    },
    "zstd_dict_trained": {
        "level": 3,
        "container": "zstd frame",
        "checksum": "none (default)",
        "streaming": "one-shot payload per candidate",
        "dictionary": "dictionary trained on TRAIN split only (~110 KiB)",
        "scoring_mode": "dictionary_trained",
    },
    "lzo": {
        "level": 9,
        "container": "LZO1X stream",
        "checksum": "adler32",
        "streaming": "one-shot per candidate",
        "dictionary": "none",
        "scoring_mode": "full_recompress",
    },
}


def prepare_options(name: str) -> dict:
    """Kwargs actually passed to ``Backend.prepare`` for ``name``."""
    src = BACKEND_OPTIONS[name]
    out: dict = {}
    for key, kwarg in (
        ("level", "level"),
        ("preset", "preset"),
        ("quality", "quality"),
        ("acceleration", "acceleration"),
    ):
        v = src.get(key)
        if isinstance(v, int):
            out[kwarg] = v
    return out


def options_rows() -> list[dict]:
    """backend_options.csv rows: every recorded setting plus versions."""
    versions = package_versions()
    rows = []
    for name, opts in sorted(BACKEND_OPTIONS.items()):
        row = {"backend": name}
        row.update(opts)
        row["python_package"] = {
            "zstd_full": "zstandard",
            "zstd_raw": "zstandard",
            "zstd_stream": "zstandard",
            "zstd_dict": "zstandard",
            "zstd_dict_trained": "zstandard",
            "brotli": "brotli",
            "lz4": "lz4",
            "snappy": "python-snappy",
            "lzo": "python-lzo",
        }.get(name, "stdlib")
        pkg = row["python_package"]
        row["package_version"] = versions.get(pkg, "stdlib") if pkg != "stdlib" else "stdlib"
        rows.append(row)
    return rows
