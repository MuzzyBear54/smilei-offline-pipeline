
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Sequence, Callable
import numpy as np

import happi

# -----------------------------------------------------------------------------
# ParticleBinning extractor (Step 2.C)
#
# Goal: stream ParticleBinning dumps and reduce each dump to compact scalar
# time-series metrics suitable for smoothing and spectral analysis.
#
# Supported:
#  - 1D momentum histograms: px histogram -> moments, asymmetry, tail fraction
#  - 2D phase space (x,px): reduce to global momentum distribution (sum over x)
#    and compute same metrics.
#
# Robustness:
#  - happi versions disagree about whether keys passed to getData are code-time
#    or iteration numbers. We probe once and decide.
#  - Data arrays may come back with extra singleton dimensions; we squeeze.
# -----------------------------------------------------------------------------

def _as_float_array(a) -> Optional[np.ndarray]:
    if a is None:
        return None
    try:
        arr = np.asarray(a, dtype=float)
    except Exception:
        return None
    arr = np.squeeze(arr)
    return arr

def _pb_get_times_like(diag) -> np.ndarray:
    # Prefer lightweight accessor
    for meth in ("getTimes", "getTime"):
        if hasattr(diag, meth):
            try:
                t = np.asarray(getattr(diag, meth)(), dtype=float)
                if t.size:
                    return t
            except Exception:
                pass
    # Fallback: try diag.get() (can be heavy, but should still be manageable for times)
    try:
        res = diag.get()
        t = np.asarray(res.get("times", []), dtype=float)
        return t
    except Exception:
        return np.asarray([], dtype=float)

def _pb_get_axis(diag, name: str) -> Optional[np.ndarray]:
    # Try common light APIs first
    if hasattr(diag, "getAxis"):
        try:
            ax = diag.getAxis(name)
            ax = _as_float_array(ax)
            if ax is not None and ax.size:
                return ax
        except Exception:
            pass
    if hasattr(diag, "getAxes"):
        try:
            axes = diag.getAxes()
            if isinstance(axes, dict) and name in axes:
                ax = _as_float_array(axes[name])
                if ax is not None and ax.size:
                    return ax
        except Exception:
            pass
    # Try attributes
    for attr in ("axes", "_axes"):
        if hasattr(diag, attr):
            try:
                axes = getattr(diag, attr)
                if isinstance(axes, dict) and name in axes:
                    ax = _as_float_array(axes[name])
                    if ax is not None and ax.size:
                        return ax
            except Exception:
                pass
    return None

def _pb_fetch(diag, key: float):
    """Attempt to fetch PB data for a single key using common happi APIs."""
    key = float(key)
    attempts = [
        ("getData", {"time": key}),
        ("getData", {"timesteps": key}),
        ("getData", {"timestep": key}),
        ("get", {"time": key}),
        ("get", {"timesteps": key}),
        ("get", {"timestep": key}),
    ]
    for meth, kwargs in attempts:
        if not hasattr(diag, meth):
            continue
        try:
            out = getattr(diag, meth)(**kwargs)
            if out is None:
                continue
            # happi sometimes returns dict-like
            if isinstance(out, dict):
                # common key for data: "data" or first ndarray
                if "data" in out:
                    out = out["data"]
                else:
                    for v in out.values():
                        if isinstance(v, (list, tuple, np.ndarray)):
                            out = v
                            break
            arr = np.asarray(out)
            if arr.size == 0:
                continue
            return out
        except Exception:
            continue
    return None

def _pb_resolve_query_keys(diag, dt_code: float) -> Tuple[np.ndarray, np.ndarray, Callable[[float], Optional[np.ndarray]]]:
    """
    Decide what key values to pass to getData/get in order to retrieve a dump.
    Returns:
      dump_keys: array of keys to query with
      times_code: code-time values (T_ref units) for labeling
      fetch_fn: function(key)->ndarray|None using already-chosen convention
    """
    keys_like = _pb_get_times_like(diag)
    keys_like = np.asarray(keys_like, dtype=float)
    keys_like = keys_like[np.isfinite(keys_like)]
    if keys_like.size == 0:
        return np.asarray([], float), np.asarray([], float), lambda k: None

    # Infer code-time labels
    # If near integer and large -> likely iterations
    near_int = np.mean(np.abs(keys_like - np.round(keys_like)) < 1e-9)
    if near_int > 0.95 and np.nanmax(keys_like) > 1000:
        times_code = keys_like * float(dt_code)
    else:
        times_code = keys_like.copy()

    # Probe mid key: does PB expect raw key (code-time) or iteration?
    test = float(keys_like[keys_like.size // 2])
    raw_key = test
    iter_key = float(np.round(test / dt_code)) if dt_code else test

    ok_raw = _pb_fetch(diag, raw_key) is not None
    ok_iter = _pb_fetch(diag, iter_key) is not None

    if ok_raw and not ok_iter:
        dump_keys = keys_like.copy()
        def fetch_fn(k: float):
            out = _pb_fetch(diag, float(k))
            return None if out is None else np.asarray(out)
        return dump_keys, times_code, fetch_fn
    if ok_iter and not ok_raw:
        dump_keys = np.asarray(np.round(keys_like / dt_code), dtype=float)
        def fetch_fn(k: float):
            out = _pb_fetch(diag, float(k))
            return None if out is None else np.asarray(out)
        return dump_keys, times_code, fetch_fn

    # If both work (or both fail), prefer raw_key convention.
    dump_keys = keys_like.copy()
    def fetch_fn(k: float):
        out = _pb_fetch(diag, float(k))
        return None if out is None else np.asarray(out)
    return dump_keys, times_code, fetch_fn

def _coerce_hist_1d(arr: np.ndarray, n: int) -> Optional[np.ndarray]:
    A = np.asarray(arr)
    A = np.squeeze(A)
    if A.ndim == 1 and A.size == n:
        return A.astype(float)
    # Some happi returns (1, n) or (n, 1)
    if A.ndim == 2:
        if A.shape[0] == 1 and A.shape[1] == n:
            return A[0, :].astype(float)
        if A.shape[1] == 1 and A.shape[0] == n:
            return A[:, 0].astype(float)
    # Last resort: flatten if size matches
    B = A.reshape(-1)
    if B.size == n:
        return B.astype(float)
    return None

def _coerce_phase_2d(arr: np.ndarray, ny: int, nx: int) -> Optional[np.ndarray]:
    A = np.asarray(arr)
    A = np.squeeze(A)
    # Common: (ny, nx) or (nx, ny)
    if A.ndim == 2:
        if A.shape == (ny, nx):
            return A.astype(float)
        if A.shape == (nx, ny):
            return A.T.astype(float)
    # Sometimes extra leading singleton dims: (1, ny, nx)
    if A.ndim == 3:
        # pick the last two
        B = A.reshape(-1, A.shape[-2], A.shape[-1])
        B = B[0]
        if B.shape == (ny, nx):
            return B.astype(float)
        if B.shape == (nx, ny):
            return B.T.astype(float)
    # Flatten and reshape if size matches
    B = A.reshape(-1)
    if B.size == ny * nx:
        return B.reshape(ny, nx).astype(float)
    return None

def _compute_moments_from_w(p: np.ndarray, w: np.ndarray) -> Tuple[float, float, float]:
    """Return (mean, var, asym) for distribution w(p)."""
    W = float(np.nansum(w))
    if not np.isfinite(W) or W <= 0:
        return np.nan, np.nan, np.nan
    mean = float(np.nansum(p * w) / W)
    var = float(np.nansum((p - mean) ** 2 * w) / W)
    wp = float(np.nansum(w[p > 0]))
    wm = float(np.nansum(w[p < 0]))
    asym = float((wp - wm) / W)
    return mean, var, asym

def extract_particle_binning_signals(
    *,
    run_dir: Path,
    diag_numbers: Sequence[int],
    dt_code: float,
    tail_multiple: float = 3.0,
    progress_every: int = 500,
    max_dumps: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract PB-derived signals from specified diag_numbers.

    Returns dict mapping signal_name -> {"t_code": times_code, "y": series}
    (time in code units; caller may convert to SI via T_ref)
    """
    run_dir = Path(run_dir)
    S = happi.Open(str(run_dir), verbose=False)

    out: Dict[str, Dict[str, np.ndarray]] = {}

    for diag_number in diag_numbers:
        diag = S.ParticleBinning(diag_number)

        dump_keys, times_code, fetch_fn = _pb_resolve_query_keys(diag, float(dt_code))
        if dump_keys.size == 0:
            print(f"[WARN] ParticleBinning({diag_number}): no dumps found.", flush=True)
            continue

        # Apply max_dumps if requested
        if max_dumps is not None:
            dump_keys = dump_keys[: int(max_dumps)]
            times_code = times_code[: int(max_dumps)]

        # Axes
        p_axis = _pb_get_axis(diag, "px")
        if p_axis is None:
            p_axis = _pb_get_axis(diag, "vx")
        x_axis = _pb_get_axis(diag, "x")

        is_2d = (x_axis is not None and x_axis.size > 1)

        nT = int(times_code.size)
        mean_s = np.full(nT, np.nan)
        var_s = np.full(nT, np.nan)
        asym_s = np.full(nT, np.nan)
        tail_frac_s = np.full(nT, np.nan)

        p_cut: Optional[float] = None

        print(f"[INFO] PB({diag_number}): {nT} dumps; 2D={is_2d}", flush=True)

        for i in range(nT):
            if progress_every and (i % int(progress_every) == 0):
                print(f"[INFO]   PB({diag_number}): {i}/{nT} dumps processed...", flush=True)

            data = fetch_fn(float(dump_keys[i]))
            if data is None:
                continue

            if p_axis is not None:
                p = np.asarray(p_axis, dtype=float)
            else:
                # Fallback: use bin index as a proxy axis in [-1,1]
                # This keeps moments dimensionless but still tracks broadening/asymmetry.
                n = int(np.size(np.squeeze(data)))
                p = np.linspace(-1.0, 1.0, n)

            if is_2d and x_axis is not None:
                ny = int(np.size(p))
                nx = int(np.size(x_axis))
                C = _coerce_phase_2d(data, ny, nx)
                if C is None:
                    continue
                w = np.nansum(C, axis=1)  # sum over x -> w(p)
            else:
                w = _coerce_hist_1d(data, int(np.size(p)))
                if w is None:
                    # sometimes 2D but x_axis missing; try treat as 2D by squashing last axis
                    A = np.asarray(data)
                    A = np.squeeze(A)
                    if A.ndim == 2:
                        w = np.nansum(A, axis=-1)
                    else:
                        continue

            m, v, a = _compute_moments_from_w(p, w)
            mean_s[i] = m
            var_s[i] = v
            asym_s[i] = a

            if p_cut is None and np.isfinite(v) and v > 0:
                p_cut = float(tail_multiple * np.sqrt(v))

            if p_cut is not None:
                W = float(np.nansum(w))
                if W > 0:
                    tail = float(np.nansum(w[np.abs(p) > p_cut]) / W)
                    tail_frac_s[i] = tail

        # Register signals
        base = f"PB{diag_number}"
        out[f"{base}/p_mean"] = {"t_code": times_code, "y": mean_s}
        out[f"{base}/p_var"] = {"t_code": times_code, "y": var_s}
        out[f"{base}/asym"] = {"t_code": times_code, "y": asym_s}
        out[f"{base}/tail_frac"] = {"t_code": times_code, "y": tail_frac_s}

    return out
