from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

from utils.processing import smooth_and_residual
from utils.spectral import compute_spectrogram, f_to_omega_over_omegape


def _downsample_even(t: np.ndarray, *ys: np.ndarray, max_points: int) -> Tuple[np.ndarray, ...]:
    """Evenly downsample for plotting (keeps endpoints).

    This is ONLY for plotting readability/performance. All smoothing + spectra
    are computed on the full-resolution arrays.
    """
    if max_points is None or max_points <= 0:
        return (t, *ys)
    n = len(t)
    if n <= max_points:
        return (t, *ys)
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    # Ensure monotonic unique indices
    idx = np.unique(idx)
    out = [t[idx]]
    for y in ys:
        out.append(np.asarray(y)[idx])
    return tuple(out)

def plot_time_domain(
    out_dir: Path,
    *,
    signal_name: str,
    t: np.ndarray,
    y: np.ndarray,
    y_label: str = "value",
    windows_s: Tuple[float, ...],
    mode: str,
    figsize: Tuple[float, float] = (12.0, 4.0),
    dpi: int = 150,
    max_points: int = 20000,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = smooth_and_residual(t, y, windows_s, mode=mode)

    # Downsample ONLY for plots (reuse same indices for all curves)
    if max_points is None or max_points <= 0 or len(t) <= max_points:
        idx = None
        t_plot = t
        y_plot = y
    else:
        idx = np.unique(np.linspace(0, len(t) - 1, num=max_points, dtype=int))
        t_plot = t[idx]
        y_plot = np.asarray(y)[idx]

    plt.figure(figsize=figsize)
    plt.plot(t_plot, y_plot, label="raw", linewidth=1)
    for w in windows_s:
        key = f"smooth_{w:g}s"
        sm_y = sm[key]
        sm_plot = np.asarray(sm_y)[idx] if idx is not None else sm_y
        plt.plot(t_plot, sm_plot, label=f"MA {w:g}s")
    plt.xlabel("t [s]")
    plt.ylabel(y_label)
    plt.title(signal_name)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fname = out_dir / (signal_name.replace("/", "__") + "_time.png")
    plt.savefig(fname, dpi=dpi)
    plt.close()

    # residuals
    res_dir = out_dir / "residuals"
    res_dir.mkdir(parents=True, exist_ok=True)
    for w in windows_s:
        res = sm[f"residual_{w:g}s"]
        t_res = t_plot
        res_plot = np.asarray(res)[idx] if idx is not None else res
        plt.figure(figsize=figsize)
        plt.plot(t_res, res_plot, linewidth=1)
        plt.xlabel("t [s]")
        plt.ylabel(f"raw - smooth ({y_label})")
        plt.title(f"{signal_name} residual (MA {w:g}s)")
        plt.tight_layout()
        fname = res_dir / (signal_name.replace("/", "__") + f"_res_{w:g}s.png")
        plt.savefig(fname, dpi=dpi)
        plt.close()

def plot_spectrogram(
    out_dir: Path,
    *,
    signal_name: str,
    t: np.ndarray,
    y: np.ndarray,
    window_seconds: float,
    overlap_frac: float,
    n_ref_m3: float,
    plot_omega_over_omegape: bool = True,
    figsize: Tuple[float, float] = (12.0, 4.0),
    dpi: int = 150,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    f, tt, Sxx = compute_spectrogram(t, y, window_seconds=window_seconds, overlap_frac=overlap_frac)
    if f.size == 0 or tt.size == 0 or Sxx.size == 0:
        return

    # Convert frequency axis
    if plot_omega_over_omegape:
        fy = f_to_omega_over_omegape(f, n_ref_m3)
        ylab = "ω/ω_pe"
    else:
        fy = f
        ylab = "f [Hz]"

    # Avoid log(0)
    P = 10.0 * np.log10(Sxx + 1e-300)

    plt.figure(figsize=figsize)
    plt.pcolormesh(tt, fy, P, shading="auto")
    plt.xlabel("t [s]")
    plt.ylabel(ylab)
    plt.title(f"{signal_name} spectrogram (win={window_seconds:g}s)")
    plt.tight_layout()
    fname = out_dir / (signal_name.replace("/", "__") + "_spec.png")
    plt.savefig(fname, dpi=dpi)
    plt.close()
