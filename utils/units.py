from __future__ import annotations

"""Unit conversion + labeling for Stage-2 plotting.

Stage 1 caches raw arrays in Smilei code units. Stage 2 can optionally convert
selected signals to SI for plotting, without re-extraction.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from utils.physics import normalization_scales


@dataclass(frozen=True)
class UnitContext:
    mode: str  # "si" or "code"
    n_ref_m3: float
    T_ref_s: float
    fields_to_si: bool = True
    pb_to_si: bool = True
    pb_axis_mode: str = "auto"  # auto|px|vx


def _pb_axis_kind_auto(y: np.ndarray, sig_name: str) -> str:
    """Heuristic: decide whether PB moments look like vx (v/c) or px (p/(m_e c)).

    This is intentionally conservative; you can override with pb_axis_mode.
    """
    y = np.asarray(y)
    finite = np.isfinite(y)
    if not np.any(finite):
        return "px"
    ymax = float(np.nanmax(np.abs(y[finite])))

    # If moments are very small and bounded ~O(1), vx is plausible.
    # Momentum can also be <1 for cold plasmas, so default stays px unless
    # the signal explicitly looks velocity-like.
    if "vx" in sig_name.lower():
        return "vx"
    if ymax <= 0.3:
        return "vx"
    return "px"


def convert_for_plot(sig_name: str, t: np.ndarray, y: np.ndarray, ctx: UnitContext) -> Tuple[np.ndarray, str]:
    """Return (y_converted, y_label) for plotting."""
    # Default labels
    if ctx.mode not in ("si", "code"):
        return y, "value"

    # Scalars: keep in code units (volume-dependent SI is ambiguous)
    if sig_name.startswith("Scalars/"):
        return y, "code units"

    # PB: only some signals have dimensions
    if sig_name.startswith("PB"):
        if ctx.mode == "code" or not ctx.pb_to_si:
            return y, "code units"

        scales = normalization_scales(ctx.n_ref_m3, ctx.T_ref_s)
        me_c = scales["me_c"]
        c = scales["c"]

        if sig_name.endswith("/tail_frac") or sig_name.endswith("/asym"):
            return y, "fraction" if sig_name.endswith("/tail_frac") else "dimensionless"

        # p_mean and p_var are moments of the PB x-axis (either px or vx)
        kind = ctx.pb_axis_mode
        if kind == "auto":
            kind = _pb_axis_kind_auto(y, sig_name)

        if sig_name.endswith("/p_mean"):
            if kind == "vx":
                return y * c, "v [m/s]"
            return y * me_c, "p [kg·m/s]"

        if sig_name.endswith("/p_var"):
            if kind == "vx":
                return y * (c ** 2), "v² [(m/s)²]"
            return y * (me_c ** 2), "p² [(kg·m/s)²]"

        return y, "code units"

    # Fields
    if sig_name.startswith("Fields/"):
        if ctx.mode == "code" or not ctx.fields_to_si:
            return y, "code units"

        scales = normalization_scales(ctx.n_ref_m3, ctx.T_ref_s)
        parts = sig_name.split("/")
        field = parts[1] if len(parts) >= 2 else ""

        if field in ("Ex", "Ey", "Ez"):
            return y * scales["E_ref"], "E [V/m]"
        if field in ("Bx", "By", "Bz", "Bx_m", "By_m", "Bz_m"):
            return y * scales["B_ref"], "B [T]"
        if field in ("Jx", "Jy", "Jz"):
            return y * scales["J_ref"], "J [A/m²]"
        if field in ("Rho",):
            return y * scales["rho_ref"], "ρ [C/m³]"

        return y, "code units"

    # Fallback
    return y, "value"
