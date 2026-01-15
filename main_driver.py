from __future__ import annotations

import argparse
import json
import faulthandler
import signal
from pathlib import Path
from typing import Dict, Any
import numpy as np

from pipeline_config import PipelineConfig, config_to_dict
from reduced_io import (
    reduced_paths, cache_valid, initialize_cache, load_npz, save_npz, load_metadata, save_metadata
)
from utils.physics import T_ref_from_n

from extractors.extract_scalars import extract_scalars_signals
from extractors.extract_fields import extract_fields_signals
from extractors.extract_particle_binning import extract_particle_binning_signals
from analyzer import analyze_signals

def _register_sigusr1_trace():
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass

def _add_signals_to_npz(arrays: Dict[str, np.ndarray], registry: Dict[str, Any], signals: Dict[str, Dict[str, np.ndarray]]) -> None:
    """
    Store each signal's t and y in arrays dict using safe keys, and registry entries.
    """
    for sig_name, s in signals.items():
        safe = sig_name.replace("/", "__")
        t_key = f"t__{safe}"
        y_key = f"y__{safe}"
        arrays[t_key] = np.asarray(s["t_si"], dtype=float)
        arrays[y_key] = np.asarray(s["y"], dtype=float)
        registry[sig_name] = {
            "t_key": t_key,
            "y_key": y_key,
            "units": s.get("units", "arb"),
            "notes": s.get("notes", ""),
        }

def run_pipeline(
    cfg: PipelineConfig,
    run_dir: Path,
    *,
    only_scalars: bool = False,
    only_fields: bool = False,
    only_pb: bool = False,
    analyze_only: bool = False,
) -> None:
    run_dir = run_dir.resolve()
    cfg_dict = config_to_dict(cfg)

    reduced_dir, npz_path, meta_path = reduced_paths(run_dir)

    # --- Analyze-only mode: never touch heavy data ---
    if analyze_only:
        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                "Analyze-only requested but reduced cache not found. "
                "Run once without --analyze-only to generate reduced_offline/."
            )
        print("[INFO] Analyze-only: loading reduced cache and generating plots...", flush=True)
        arrays = load_npz(npz_path)
        meta = load_metadata(meta_path)
        registry = meta.get("registry", {})
        # Record latest analysis config snapshot (informational only)
        meta["last_analysis_config_snapshot"] = cfg_dict
        save_metadata(meta_path, meta)
        analyze_signals(
            run_dir,
            arrays,
            registry,
            windows_s=cfg.windowing.windows_seconds,
            rolling_mode=cfg.windowing.mode,
            spec_cfg=config_to_dict(cfg)["spectrogram"],
            n_ref_m3=cfg.physics.n_ref_m3,
            T_ref_s=float(meta.get("resolved_T_ref_s", T_ref_from_n(cfg.physics.n_ref_m3))),
            units_cfg=config_to_dict(cfg)["units"],
            plot_cfg=config_to_dict(cfg)["plot"],
        )
        print("[INFO] Done.", flush=True)
        return

    # If Stage-1 cache is valid, skip extraction even if windowing/spectrogram config changed.
    if cache_valid(run_dir, cfg_dict):
        print("[INFO] Stage-1 cache valid: skipping extraction (Stage 1).", flush=True)
        arrays = load_npz(npz_path)
        meta = load_metadata(meta_path)
        registry = meta.get("registry", {})
        # Record latest analysis config snapshot (informational only)
        meta["last_analysis_config_snapshot"] = cfg_dict
        save_metadata(meta_path, meta)
        analyze_signals(
            run_dir,
            arrays,
            registry,
            windows_s=cfg.windowing.windows_seconds,
            rolling_mode=cfg.windowing.mode,
            spec_cfg=config_to_dict(cfg)["spectrogram"],
            n_ref_m3=cfg.physics.n_ref_m3,
            T_ref_s=float(meta.get("resolved_T_ref_s", T_ref_from_n(cfg.physics.n_ref_m3))),
            units_cfg=config_to_dict(cfg)["units"],
            plot_cfg=config_to_dict(cfg)["plot"],
        )
        print("[INFO] Done.", flush=True)
        return

    print("[INFO] Cache missing/invalid: extracting requested diagnostics (Stage 1)...", flush=True)
    meta = initialize_cache(run_dir, cfg_dict)
    arrays = load_npz(npz_path)
    registry = meta.get("registry", {})

    # Ensure T_ref_s is computed once and recorded for reproducibility
    T_ref_s = cfg.physics.T_ref_s
    if T_ref_s is None:
        T_ref_s = T_ref_from_n(cfg.physics.n_ref_m3)
    meta["resolved_T_ref_s"] = float(T_ref_s)
    save_metadata(meta_path, meta)

    # ---- Scalars (fast) ----
    if cfg.scalars.enabled and not (only_fields or only_pb):
        print("[INFO] Scalars: extracting...", flush=True)
        ssignals = extract_scalars_signals(run_dir, cfg.scalars.preferred_columns)

        # convert scalars time from code-time to SI using T_ref
        for k in list(ssignals.keys()):
            t_code = np.asarray(ssignals[k]["t_code"], dtype=float)
            ssignals[k]["t_si"] = t_code * T_ref_s
            del ssignals[k]["t_code"]

        _add_signals_to_npz(arrays, registry, ssignals)
        save_npz(npz_path, arrays)
        meta["registry"] = registry
        save_metadata(meta_path, meta)
        print(f"[INFO] Scalars: saved to {npz_path}", flush=True)

    
    # ---- ParticleBinning (heavy-ish) ----
    if cfg.particle_binning.enabled and not (only_scalars or only_fields):
        print("[INFO] ParticleBinning: extracting...", flush=True)
        pbsignals = extract_particle_binning_signals(
            run_dir=run_dir,
            diag_numbers=cfg.particle_binning.diag_numbers,
            dt_code=cfg.physics.dt_code,
            tail_multiple=cfg.particle_binning.tail_multiple,
            progress_every=cfg.particle_binning.progress_every,
            max_dumps=cfg.particle_binning.max_dumps,
        )

        # convert PB time from code-time to SI
        for k in list(pbsignals.keys()):
            t_code = np.asarray(pbsignals[k]["t_code"], dtype=float)
            pbsignals[k]["t_si"] = t_code * T_ref_s
            del pbsignals[k]["t_code"]

        _add_signals_to_npz(arrays, registry, pbsignals)
        save_npz(npz_path, arrays)
        meta["registry"] = registry
        save_metadata(meta_path, meta)
        print(f"[INFO] ParticleBinning: saved to {npz_path}", flush=True)

# ---- Fields (heavy) ----
    if cfg.fields.enabled and not (only_scalars or only_pb):
        fsignals = extract_fields_signals(
            run_dir=run_dir,
            out_dir=run_dir / "diagnostics_output_offline",
            diag_number=cfg.fields.diag_number,
            field_names=cfg.fields.field_names,
            metrics=cfg.fields.metrics,
            virtual_probe_fracs=cfg.fields.virtual_probe_fracs,
            exclude_frac=cfg.fields.exclude_frac,
            n_ref_m3=cfg.physics.n_ref_m3,
            dt_code=cfg.physics.dt_code,
            T_ref_s=T_ref_s,
            progress_every=cfg.fields.progress_every,
            max_dumps=cfg.fields.max_dumps,
        )
        _add_signals_to_npz(arrays, registry, fsignals)
        save_npz(npz_path, arrays)
        meta["registry"] = registry
        save_metadata(meta_path, meta)
        print(f"[INFO] Fields: saved to {npz_path}", flush=True)

    # Stage 2 analysis
    print("[INFO] Stage 2: analysis + plots...", flush=True)
    analyze_signals(
        run_dir, arrays, registry,
        windows_s=cfg.windowing.windows_seconds,
        rolling_mode=cfg.windowing.mode,
        spec_cfg=config_to_dict(cfg)["spectrogram"],
        n_ref_m3=cfg.physics.n_ref_m3,
        T_ref_s=T_ref_s,
        units_cfg=config_to_dict(cfg)["units"],
        plot_cfg=config_to_dict(cfg)["plot"],
    )
    print("[INFO] Done.", flush=True)

def main():
    _register_sigusr1_trace()

    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to Smilei run directory (contains scalars.txt, Fields, etc.)")
    p.add_argument("--only-scalars", action="store_true", help="Extract/analyze scalars only (skip fields).")
    p.add_argument("--only-fields", action="store_true", help="Extract/analyze fields only (skip scalars + PB).")
    p.add_argument("--only-pb", action="store_true", help="Extract/analyze ParticleBinning only (skip scalars + fields).")
    p.add_argument("--analyze-only", action="store_true", help="Skip extraction and only run analysis/plotting from reduced_offline cache.")
    args = p.parse_args()

    cfg = PipelineConfig()
    run_pipeline(
        cfg,
        Path(args.run_dir),
        only_scalars=args.only_scalars,
        only_fields=args.only_fields,
        only_pb=args.only_pb,
        analyze_only=args.analyze_only,
    )

if __name__ == "__main__":
    main()
