from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from scipy.signal import spectrogram
from utils.physics import omega_pe

def compute_spectrogram(
    t: np.ndarray,
    y: np.ndarray,
    *,
    window_seconds: float,
    overlap_frac: float = 0.5,
    detrend: str = "constant",
    scaling: str = "density",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns f_Hz, t_mid_s, Sxx
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size < 2:
        return np.array([]), np.array([]), np.array([[]])

    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.isfinite(dt):
        return np.array([]), np.array([]), np.array([[]])

    fs = 1.0 / dt
    nperseg = max(8, int(round(window_seconds * fs)))
    noverlap = int(round(overlap_frac * nperseg))
    f, tt, Sxx = spectrogram(
        y, fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        detrend=detrend, scaling=scaling, mode="psd"
    )
    return f, tt, Sxx

def f_to_omega_over_omegape(f_hz: np.ndarray, n_ref_m3: float) -> np.ndarray:
    w = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    return w / omega_pe(n_ref_m3)
