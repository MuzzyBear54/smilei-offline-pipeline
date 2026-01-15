from __future__ import annotations
import numpy as np
from typing import Tuple

def median_dt(t: np.ndarray) -> float:
    if t.size < 2:
        return float("nan")
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return float("nan")
    return float(np.median(dt))

def window_seconds_to_samples(t: np.ndarray, window_s: float) -> int:
    """Convert a window size in seconds to number of samples using median dt."""
    dt = median_dt(t)
    if not np.isfinite(dt) or dt <= 0:
        return 1
    n = int(round(window_s / dt))
    return max(1, n)
