"""Dataset loading and leakage-safe splitting.

tiny_shakespeare is a single concatenated stream (multiple plays stitched
together), so a *contiguous* split is used rather than a document-level one:
the source is cut into train / validation / test ranges and a ``gap`` of
bytes at each boundary is discarded, so no byte adjacent to a test string
is ever also adjacent to a train string.

- ``train`` is the only corpus a compressor may be primed with.
- ``validation`` is for hyperparameter / benchmark-config tuning only.
- ``test`` positions are the only places where prediction is scored; the
  scored byte is always inside the test range and therefore never appears
  in the train bytes. A position's *context* may legitimately reach back
  into earlier ranges (that's the model's observable history, not the
  answer).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEED = 42


@dataclass(frozen=True)
class Dataset:
    """A byte corpus split into non-overlapping ranges."""

    name: str
    source: bytes
    train: bytes
    val: bytes
    test: bytes
    #: (start, end) byte offsets of each split inside ``source``.
    train_span: tuple[int, int]
    val_span: tuple[int, int]
    test_span: tuple[int, int]
    #: Bytes discarded at each split boundary (leakage margin).
    gap: int
    seed: int

    def sha256(self) -> str:
        return hashlib.sha256(self.source).hexdigest()

    def context_at(self, position: int, context_bytes: int) -> bytes:
        """Up to ``context_bytes`` of ``source`` preceding ``position``.

        The context may cross split boundaries — it is the observable
        history, while the scored byte always lies inside the test range.
        """
        start = max(0, position - context_bytes)
        return self.source[start:position]

    def split_meta(self) -> dict:
        return {
            "name": self.name,
            "sha256": self.sha256(),
            "source_bytes": len(self.source),
            "train_bytes": len(self.train),
            "val_bytes": len(self.val),
            "test_bytes": len(self.test),
            "train_span": list(self.train_span),
            "val_span": list(self.val_span),
            "test_span": list(self.test_span),
            "gap": self.gap,
            "seed": self.seed,
            "method": "contiguous ranges with discarded boundary gaps",
        }


def split_contiguous(
    data: bytes,
    *,
    name: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    gap: int = 4096,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Split ``data`` into train/val/test contiguous ranges with gaps.

    Layout: [train][gap][val][gap][test]. The tail becomes test so the
    held-out region is never adjacent to train.
    """
    n = len(data)
    t_end = int(n * train_frac)
    v_start = t_end + gap
    v_end = v_start + int(n * val_frac)
    s_start = v_end + gap
    if not (0 < t_end < v_start < v_end < s_start < n):
        raise ValueError(f"dataset too small for split (n={n}, gap={gap})")
    return Dataset(
        name=name,
        source=data,
        train=data[:t_end],
        val=data[v_start:v_end],
        test=data[s_start:],
        train_span=(0, t_end),
        val_span=(v_start, v_end),
        test_span=(s_start, n),
        gap=gap,
        seed=seed,
    )


def load_tiny_shakespeare(
    path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    gap: int = 4096,
) -> Dataset:
    data = Path(path).read_bytes()
    return split_contiguous(data, name="tiny_shakespeare", gap=gap, seed=seed)


def sample_positions(
    span: tuple[int, int],
    n: int,
    *,
    seed: int = DEFAULT_SEED,
    exclude_first: int = 0,
) -> list[int]:
    """Sample up to ``n`` absolute byte offsets inside ``span``.

    The same seed yields the same position list, so every backend is scored
    on identical positions. Slow backends should take a prefix of this list
    so their subset is comparable point-for-point.
    """
    lo, hi = span
    lo += exclude_first
    population = hi - lo
    if n >= population:
        return list(range(lo, hi))
    rng = random.Random(seed)
    return sorted(rng.sample(range(lo, hi), n))
