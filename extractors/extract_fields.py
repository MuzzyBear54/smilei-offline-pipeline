from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from utils.physics import T_ref_from_n

# ---------- Helpers ----------

def _try_open_happi(run_dir: Path):
    import happi  # type: ignore
    # Some happi versions support verbose kw, some don't.
    try:
        return happi.Open(str(run_dir), verbose=False)
    except TypeError:
        return happi.Open(str(run_dir))

def _field_name_candidates(field: str) -> List[str]:
    # Allow automatic fallbacks for magnetics in some Smilei outputs (Bx_m, etc.)
    cands = [field]
    if field in ("Bx", "By", "Bz"):
        cands.append(field + "_m")
    return cands

def _get_field_object(S, diag_number: int, field: str):
    last_err = None
    for cand in _field_name_candidates(field):
        try:
            return S.Field(diagNumber=diag_number, field=cand), cand
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not open Field '{field}' (tried { _field_name_candidates(field) }) for diagNumber={diag_number}. Last error: {last_err}")

def _infer_access_mode(F, t0) -> str:
    """
    Decide whether to access dumps via getData(time=...) or getData(timestep=...).
    Prefer time= if supported.
    """
    try:
        _ = F.getData(time=t0)
        return "time"
    except TypeError:
        return "timestep"
    except Exception:
        # If time= exists but fails for other reasons, still try timestep mode
        return "timestep"

def _to_1d_space_array(data: Any) -> np.ndarray:
    a = np.asarray(data)
    a = np.squeeze(a)
    # Common cases:
    #   (nx,) ok
    #   (1,nx) -> squeezed to (nx,)
    #   (nx,1) -> squeezed to (nx,)
    if a.ndim == 0:
        return a.reshape(1)
    if a.ndim > 1:
        # In 1D cartesian, last axis is usually space
        a = np.ravel(a)
    return a.astype(float, copy=False)

def _interior_slice(n: int, exclude_frac: float) -> slice:
    cut = int(round(exclude_frac * n))
    if cut <= 0:
        return slice(0, n)
    if 2 * cut >= n:
        # Safety: don't create empty interior
        return slice(0, n)
    return slice(cut, n - cut)

def _compute_metrics(a: np.ndarray, metrics: Tuple[str, ...]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if a.size == 0:
        # Defensive, should not happen after interior slice safety
        for m in metrics:
            out[m] = float("nan")
        return out
    for m in metrics:
        if m == "rms":
            out[m] = float(np.sqrt(np.mean(a * a)))
        elif m == "maxabs":
            out[m] = float(np.max(np.abs(a)))
        elif m == "mean":
            out[m] = float(np.mean(a))
        elif m == "std":
            out[m] = float(np.std(a))
        else:
            raise ValueError(f"Unknown metric '{m}'")
    return out

# ---------- Main extraction ----------

def extract_fields_signals(
    *,
    run_dir: Path,
    out_dir: Path,
    diag_number: int,
    field_names: Tuple[str, ...],
    metrics: Tuple[str, ...],
    virtual_probe_fracs: Tuple[float, ...],
    exclude_frac: float,
    n_ref_m3: float,
    dt_code: float,
    T_ref_s: Optional[float],
    progress_every: int = 2000,
    max_dumps: Optional[int] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Streams each Field dump to compute metrics over x, discarding snapshots immediately.

    Returns:
      signals dict: signal_name -> {'t_si': ..., 'y': ..., 'units': str, 'notes': str}
    """
    # Compute time scale if needed
    if T_ref_s is None:
        T_ref_s = T_ref_from_n(n_ref_m3)

    print("[INFO] Fields: opening happi...", flush=True)
    S = _try_open_happi(run_dir)
    print("[INFO] Fields: happi opened.", flush=True)

    # Use first requested field that exists to define time base
    first_field_obj = None
    first_field_used = None
    for f in field_names:
        try:
            first_field_obj, first_field_used = _get_field_object(S, diag_number, f)
            break
        except Exception:
            continue
    if first_field_obj is None:
        raise RuntimeError("None of the requested field_names could be opened by happi.")

    times_raw = np.asarray(first_field_obj.getTimes(), dtype=float)
    if times_raw.size == 0:
        raise RuntimeError("Field.getTimes() returned empty array.")
    # Convert to SI seconds; happi times are typically code-time (t_code), so SI = t_code * T_ref
    t_si = times_raw * T_ref_s

    # Determine access mode (time vs timestep) and conversion if timestep required
    access_mode = _infer_access_mode(first_field_obj, float(times_raw[0]))
    # If timestep mode, times_raw are likely code-time; convert to iterations by /dt_code
    if access_mode == "timestep":
        iters = np.round(times_raw / dt_code).astype(int)
    else:
        iters = None

    n_total = times_raw.size
    if max_dumps is not None:
        n_total = min(n_total, int(max_dumps))
        times_raw = times_raw[:n_total]
        t_si = t_si[:n_total]
        if iters is not None:
            iters = iters[:n_total]

    # Prepare output signals
    signals: Dict[str, Dict[str, np.ndarray]] = {}

    for field in field_names:
        try:
            F, actual_name = _get_field_object(S, diag_number, field)
        except Exception as e:
            print(f"[WARN] Skipping field '{field}': {e}", flush=True)
            continue

        # Use the same access mode decision per field (time= might exist for some; still keep stable)
        mode = _infer_access_mode(F, float(times_raw[0]))
        use_iters = (mode == "timestep")

        # Preallocate arrays for metrics
        metric_arrays = {m: np.empty(n_total, dtype=float) for m in metrics}

        # Virtual probe arrays (one per frac)
        probe_arrays = {frac: np.empty(n_total, dtype=float) for frac in virtual_probe_fracs}

        interior = None
        nx_interior = None
        probe_indices = None

        print(f"[INFO] Fields: extracting '{field}' (using '{actual_name}') for {n_total} dumps...", flush=True)

        for k in range(n_total):
            traw = float(times_raw[k])
            if mode == "time":
                data = F.getData(time=traw)
            else:
                # timestep mode
                it = int(iters[k]) if iters is not None else int(round(traw))
                data = F.getData(timestep=it)

            a = _to_1d_space_array(data)

            # Setup interior slice and probe indices once, after first dump tells us nx
            if interior is None:
                nx = int(a.size)
                interior = _interior_slice(nx, exclude_frac)
                nx_interior = int((interior.stop - interior.start) if (interior.stop is not None) else nx)
                # map fracs -> indices within interior
                idxs = []
                for frac in virtual_probe_fracs:
                    frac_clamped = min(max(float(frac), 0.0), 1.0)
                    # pick index inside interior [0, nx_interior-1]
                    ii = int(round(frac_clamped * (nx_interior - 1)))
                    idxs.append(interior.start + ii)
                probe_indices = idxs

            a_in = a[interior]
            mvals = _compute_metrics(a_in, metrics)
            for m, v in mvals.items():
                metric_arrays[m][k] = v

            # probes
            for frac, idx in zip(virtual_probe_fracs, probe_indices):
                probe_arrays[frac][k] = float(a[idx])

            if progress_every and (k % progress_every == 0):
                print(f"[INFO]   {field}: {k}/{n_total} dumps", flush=True)

        # Register signals (metrics)
        for m in metrics:
            name = f"Fields/{field}/{m}"
            signals[name] = {
                "t_si": t_si,
                "y": metric_arrays[m],
                "units": "code_units",  # happi already applies any unit? Without Pint, treat as code units
                "notes": f"diag={diag_number}, field={actual_name}, metric={m}, exclude_frac={exclude_frac}",
            }

        # Register probe signals
        for frac in virtual_probe_fracs:
            name = f"Fields/{field}/probe@{frac:.3f}"
            signals[name] = {
                "t_si": t_si,
                "y": probe_arrays[frac],
                "units": "code_units",
                "notes": f"diag={diag_number}, field={actual_name}, probe_frac={frac}, exclude_frac={exclude_frac}",
            }

    return signals
