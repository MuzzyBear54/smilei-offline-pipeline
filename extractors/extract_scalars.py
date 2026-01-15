from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import numpy as np
import re

def _parse_scalars_header(lines: List[str]) -> List[str]:
    """
    Supports:
      1) single header: "# time Utot Ukin ..."
      2) enumerated header: "# 1 time", "# 2 Utot", ...
    Returns list of column names in order.
    """
    # case 1: header with names on one line
    for ln in lines[:50]:
        s = ln.strip()
        if s.startswith("#") and "time" in s and len(s.split()) > 3 and re.search(r"\bUtot\b|\bUkin\b|\bUelm\b", s):
            parts = s.lstrip("#").strip().split()
            return parts

    # case 2: enumerated
    cols = []
    enum_re = re.compile(r"^#\s*(\d+)\s+(\S+)")
    for ln in lines[:200]:
        m = enum_re.match(ln.strip())
        if m:
            cols.append(m.group(2))
    return cols

def load_scalars_txt(scalars_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      t_raw: time column as in file (usually code-time)
      data: dict column_name -> values
    """
    lines = scalars_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    cols = _parse_scalars_header(lines)
    if not cols:
        raise RuntimeError("Could not parse scalars header. Please paste first ~20 lines of scalars.txt.")

    # Load numeric rows (skip comment lines)
    numeric = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        numeric.append(s)

    arr = np.loadtxt(numeric, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.shape[1] < len(cols):
        # Some Smilei outputs omit last columns; trim col list
        cols = cols[:arr.shape[1]]

    data = {cols[i]: arr[:, i] for i in range(len(cols))}
    if "time" not in data:
        raise RuntimeError("scalars.txt parsed but no 'time' column found.")
    return data["time"], {k: v for k, v in data.items() if k != "time"}

def extract_scalars_signals(run_dir: Path, preferred: Tuple[str, ...]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns dict: signal_name -> {'t': t_code, 'y': y, 'units': str}
    Signal names for scalars: "Scalars/<col>"
    """
    scalars_path = run_dir / "scalars.txt"
    if not scalars_path.exists():
        raise FileNotFoundError(f"Missing scalars.txt at {scalars_path}")

    t_code, cols = load_scalars_txt(scalars_path)

    signals: Dict[str, Dict[str, np.ndarray]] = {}
    # Always export all preferred columns that exist
    for name in preferred:
        if name in cols:
            signals[f"Scalars/{name}"] = {"t_code": t_code, "y": cols[name], "units": "arb"}
    # If none matched, export everything (small cost)
    if not signals:
        for name, y in cols.items():
            signals[f"Scalars/{name}"] = {"t_code": t_code, "y": y, "units": "arb"}
    return signals
