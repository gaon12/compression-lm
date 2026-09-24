"""Bootstrap confidence intervals (spec §25).

Every headline metric is a *mean over per-position contributions*, so a
percentile bootstrap over positions gives a distribution-free CI without
re-running any compressor. Fixed seed -> reproducible intervals.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence

BOOT_ITERS = 1000
BOOT_SEED = 42


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = BOOT_ITERS,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_lo, ci_hi)`` — 95% percentile bootstrap."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    if n == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    vals = list(values)
    boot = sorted(
        statistics.fmean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)
    )
    lo = boot[int(0.025 * n_boot)]
    hi = boot[min(n_boot - 1, int(0.975 * n_boot))]
    return mean, lo, hi


def ci_text(ci: tuple[float, float, float], digits: int = 3) -> str:
    """Render ``(mean, lo, hi)`` as ``0.277 [0.255, 0.299]``."""
    mean, lo, hi = ci
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
