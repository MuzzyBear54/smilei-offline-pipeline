from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np

from utils.hash_utils import sha256_file

REDUCED_DIRNAME = "reduced_offline"
NPZ_NAME = "reduced_data.npz"
META_NAME = "metadata.json"

def reduced_paths(run_dir: Path) -> Tuple[Path, Path, Path]:
    reduced_dir = run_dir / REDUCED_DIRNAME
    return reduced_dir, reduced_dir / NPZ_NAME, reduced_dir / META_NAME

def load_metadata(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))

def save_metadata(meta_path: Path, meta: Dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

def load_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        return {}
    with np.load(npz_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def save_npz(npz_path: Path, arrays: Dict[str, np.ndarray]) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)


def _extract_cfg_key(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the config parts that affect Stage-1 *extraction* outputs.

    Windowing/spectrogram settings affect only Stage-2 analysis/plots and must
    NOT invalidate the heavy reduced cache.

    We keep this as a pure-dict transform (no dataclass imports) so it stays
    robust across versions.
    """
    if not isinstance(cfg_dict, dict):
        return {}
    # Deep-ish copy of relevant top-level keys.
    keep = {}
    for k in ("physics", "scalars", "fields", "particle_binning"):
        if k in cfg_dict:
            keep[k] = cfg_dict[k]
    return keep

def cache_valid(run_dir: Path, cfg_dict: Dict[str, Any]) -> bool:
    reduced_dir, npz_path, meta_path = reduced_paths(run_dir)
    if not npz_path.exists() or not meta_path.exists():
        return False
    meta = load_metadata(meta_path)
    # Back-compat: older metadata stored a single full snapshot.
    stored = meta.get("config_extract_snapshot")
    if stored is None:
        stored = _extract_cfg_key(meta.get("config_snapshot", {}))
    if stored != _extract_cfg_key(cfg_dict):
        return False

    scalars_path = run_dir / "scalars.txt"
    if scalars_path.exists():
        if meta.get("scalars_sha256") != sha256_file(scalars_path):
            return False

    # If smilei.py exists, include its hash in validity
    smilei_py = run_dir / "smilei.py"
    if smilei_py.exists():
        if meta.get("smilei_sha256") != sha256_file(smilei_py):
            return False
    return True

def initialize_cache(run_dir: Path, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Create reduced_offline dir and write initial metadata."""
    reduced_dir, npz_path, meta_path = reduced_paths(run_dir)
    reduced_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "created_utc": None,
        # Full snapshot (for reproducibility/debugging)
        "config_snapshot": cfg_dict,
        # Extraction-only snapshot (for cache validity)
        "config_extract_snapshot": _extract_cfg_key(cfg_dict),
        # Last analysis snapshot (purely informational; does not gate cache validity)
        "last_analysis_config_snapshot": cfg_dict,
        "scalars_sha256": None,
        "smilei_sha256": None,
    }

    scalars_path = run_dir / "scalars.txt"
    if scalars_path.exists():
        meta["scalars_sha256"] = sha256_file(scalars_path)

    smilei_py = run_dir / "smilei.py"
    if smilei_py.exists():
        meta["smilei_sha256"] = sha256_file(smilei_py)

    save_metadata(meta_path, meta)
    # Initialize empty npz (so folder isn't "empty" while fields run)
    if not npz_path.exists():
        save_npz(npz_path, {"__init__": np.array([1], dtype=np.int8)})
    return meta
