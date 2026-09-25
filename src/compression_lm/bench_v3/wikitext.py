"""WikiText-2 dataset — replication corpus for the v3 experiments.

The canonical release ships already-split ``wiki.{train,valid,test}.tokens``
files, so no contiguous cut is needed: the files are concatenated with the
same 4 KiB discarded gaps used for tiny_shakespeare, giving an identical
``Dataset`` layout — train primes compressors, validation tunes, test
positions are scored, and a test context may legitimately reach back into
earlier bytes (observable history, never the answer).

Download: the torchtext canonical archive (~4.4 MB zip). Offline failure
leaves the dataset absent; the runner records ``skipped`` rather than
aborting the benchmark.
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

from ..bench.datasets import DEFAULT_SEED, Dataset


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
# canonical torchtext MD5 — verified after download
MD5 = "542ccefacc6c27f945fb54453812b3cd"
# the metamind S3 bucket now returns a bare 301 (path-style deprecation);
# the Wayback Machine holds the identical bytes.
MIRRORS = (
    URL,
    "https://web.archive.org/web/20230301182406id_/" + URL,
)
FILES = ("wiki.train.tokens", "wiki.valid.tokens", "wiki.test.tokens")


def fetch_wikitext2(dest_dir: Path) -> dict[str, Path]:
    """Download + extract the three token files; returns their paths."""
    dest_dir = dest_dir / "wikitext2"
    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = {f: dest_dir / f for f in FILES}
    if all(p.exists() for p in paths.values()):
        return paths
    zpath = dest_dir / "wikitext-2-v1.zip"
    if not zpath.exists() or _md5(zpath) != MD5:
        last: Exception | None = None
        for url in MIRRORS:
            try:
                print(f"downloading {url}")
                urllib.request.urlretrieve(url, zpath)
                if _md5(zpath) == MD5:
                    break
            except Exception as exc:
                last = exc
        else:
            raise RuntimeError(f"wikitext-2 download failed from all mirrors: {last}")
        if _md5(zpath) != MD5:
            raise RuntimeError("wikitext-2 zip md5 mismatch")
    with zipfile.ZipFile(zpath) as z:
        for f in FILES:
            with z.open(f"wikitext-2/{f}") as src, paths[f].open("wb") as dst:
                dst.write(src.read())
    return paths


def load_wikitext2(data_dir: Path, *, gap: int = 4096, seed: int = DEFAULT_SEED) -> Dataset:
    """Build a ``Dataset`` from the natural WT2 splits + gap separators."""
    paths = fetch_wikitext2(data_dir)
    train = paths["wiki.train.tokens"].read_bytes()
    val = paths["wiki.valid.tokens"].read_bytes()
    test = paths["wiki.test.tokens"].read_bytes()
    fill = b"\x00" * gap
    source = train + fill + val + fill + test
    t_end = len(train)
    v_start = t_end + gap
    v_end = v_start + len(val)
    s_start = v_end + gap
    return Dataset(
        name="wikitext2",
        source=source,
        train=train,
        val=val,
        test=test,
        train_span=(0, t_end),
        val_span=(v_start, v_end),
        test_span=(s_start, len(source)),
        gap=gap,
        seed=seed,
    )
