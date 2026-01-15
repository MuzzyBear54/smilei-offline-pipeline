from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, List
from utils.time_utils import window_seconds_to_samples
from utils.rolling import rolling_mean

def smooth_and_residual(t: np.ndarray, y: np.ndarray, windows_s: Tuple[float, ...], mode: str) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for w in windows_s:
        n = window_seconds_to_samples(t, float(w))
        ys = rolling_mean(y, n, mode=mode)
        out[f"smooth_{w:g}s"] = ys
        out[f"residual_{w:g}s"] = y - ys
        out[f"window_samples_{w:g}s"] = np.array([n], dtype=int)
    return out
