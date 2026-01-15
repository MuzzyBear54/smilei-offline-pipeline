from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

from utils.plots import plot_time_domain, plot_spectrogram
from utils.units import UnitContext, convert_for_plot

def analyze_signals(
    run_dir: Path,
    arrays: Dict[str, np.ndarray],
    registry: Dict[str, Any],
    *,
    windows_s: Tuple[float, ...],
    rolling_mode: str,
    spec_cfg: Dict[str, Any],
    n_ref_m3: float,
    T_ref_s: float,
    units_cfg: Dict[str, Any],
    plot_cfg: Dict[str, Any],
):
    out_root = run_dir / "diagnostics_output_offline"
    out_root.mkdir(parents=True, exist_ok=True)

    unit_ctx = UnitContext(
        mode=str(units_cfg.get("mode", "si")),
        n_ref_m3=float(n_ref_m3),
        T_ref_s=float(T_ref_s),
        fields_to_si=bool(units_cfg.get("fields_to_si", True)),
        pb_to_si=bool(units_cfg.get("pb_to_si", True)),
        pb_axis_mode=str(units_cfg.get("pb_axis_mode", "auto")),
    )

    for sig_name, info in registry.items():
        t_key = info["t_key"]
        y_key = info["y_key"]
        t = arrays[t_key]
        y = arrays[y_key]

        y_plot, ylab = convert_for_plot(sig_name, t, y, unit_ctx)

        # Time-domain plot
        td_dir = out_root / "time_domain"
        # Time-domain plots: allow wide aspect + point capping.
        td_figsize = tuple(plot_cfg.get(
            "figsize_time_domain",
            plot_cfg.get("figsize", (12.0, 4.0))
        ))
        plot_time_domain(
            td_dir,
            signal_name=sig_name,
            t=t,
            y=y_plot,
            y_label=ylab,
            windows_s=windows_s,
            mode=rolling_mode,
            figsize=td_figsize,
            dpi=int(plot_cfg.get("dpi", 150)),
            max_points=int(plot_cfg.get("max_points_time_domain", 20000)),
        )

        # Spectrogram
        if spec_cfg.get("enabled", True):
            spec_dir = out_root / "spectrograms"
            # Spectrograms: DO NOT inherit the wide time-domain aspect ratio.
            spec_figsize = tuple(plot_cfg.get("figsize_spectrogram", (10.0, 6.0)))
            plot_spectrogram(
                spec_dir,
                signal_name=sig_name,
                t=t,
                y=y_plot,
                window_seconds=float(spec_cfg["window_seconds"]),
                overlap_frac=float(spec_cfg["overlap_frac"]),
                n_ref_m3=n_ref_m3,
                plot_omega_over_omegape=bool(spec_cfg.get("plot_omega_over_omegape", True)),
                figsize=spec_figsize,
                dpi=int(plot_cfg.get("dpi", 150)),
            )
