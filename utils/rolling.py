from __future__ import annotations
import numpy as np

def rolling_mean(y: np.ndarray, window: int, mode: str = "trailing") -> np.ndarray:
    """
    O(N) rolling mean using cumulative sum.
    mode:
      - "trailing": y_smooth[i] = mean(y[i-window+1 : i+1])
      - "centered": centered window (symmetric). Implemented via trailing on padded series.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if window <= 1 or n == 0:
        return y.copy()

    if mode not in ("trailing", "centered"):
        raise ValueError(f"Unknown rolling mode: {mode}")

    if mode == "centered":
        # For centered mean: pad at both ends by edge values, then trailing mean, then slice.
        half = window // 2
        pad_left = half
        pad_right = window - 1 - half
        ypad = np.pad(y, (pad_left, pad_right), mode="edge")
        out = _rolling_trailing(ypad, window)
        return out[pad_left:pad_left+n]

    return _rolling_trailing(y, window)

def _rolling_trailing(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    c = np.cumsum(y, dtype=float)
    out = np.empty(n, dtype=float)
    for i in range(n):
        j0 = i - window + 1
        if j0 <= 0:
            out[i] = c[i] / (i + 1)
        else:
            out[i] = (c[i] - c[j0 - 1]) / window
    return out
