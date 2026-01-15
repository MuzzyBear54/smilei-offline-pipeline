from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Sequence, Tuple, Dict, Any

# ---------- User-editable configuration (Step 2.A + 2.B) ----------
# You can safely edit values in this file.
# Do NOT delete fields; set them to None / False instead.

@dataclass(frozen=True)
class PhysicsConfig:
    # Reference density used for omega_pe and for converting code-time -> seconds when T_ref_s=None
    n_ref_m3: float = 1e15

    # Smilei code timestep (Main.timestep) in "code time units" per iteration
    dt_code: float = 0.00056

    # Reference time scale in seconds (T_ref = 1/omega_ref). If None, computed as 1/omega_pe(n_ref_m3).
    T_ref_s: Optional[float] = None


@dataclass(frozen=True)
class ParticleBinningConfig:
    enabled: bool = True

    # Which ParticleBinning diagnostics to reduce.
    # Your common convention:
    #   0: electron px histogram
    #   1: proton  px histogram
    #   2: electron x-px phase space
    #   3: proton  x-px phase space
    #
    # Default: protons only (1D hist + 2D phase space)
    diag_numbers: Tuple[int, ...] = (1, 3)

    # Tail threshold is defined as: p_cut = tail_multiple * sqrt(var_p) from the first
    # non-empty dump. Tail metric reports fraction of weight with |p| > p_cut.
    tail_multiple: float = 3.0

    # Print progress every N dumps (PB cadence is usually lower than Fields)
    progress_every: int = 500

    # Safety: cap dumps processed (None means full run)
    max_dumps: Optional[int] = None

@dataclass(frozen=True)
class WindowingConfig:
    # Moving-average windows in *seconds* (SI). Trailing by default.
    windows_seconds: Tuple[float, ...] = (
        4.58e-10,  # ~0.13 * (2π/ωpe)
        1.76e-9,   # ~0.5  * (2π/ωpe)
        7.04e-9,   # ~2    * (2π/ωpe)
        3.52e-8,   # ~10   * (2π/ωpe)  ≈ 0.23 * (2π/ωpi)
        1.51e-7,   # ~1    * (2π/ωpi) (one ion/proton plasma period)
        7.55e-7,   # ~5    * (2π/ωpi)
    )
    mode: str = "trailing"  # "trailing" or "centered"

@dataclass(frozen=True)
class SpectrogramConfig:
    enabled: bool = True
    # Spectrogram window (SI seconds), independent of smoothing.
    window_seconds: float = 7.04e-8
    overlap_frac: float = 0.5
    detrend: str = "constant"  # scipy.signal.spectrogram detrend
    scaling: str = "density"   # "density" or "spectrum"
    # Plot frequency axis as omega/omega_pe if physics info available
    plot_omega_over_omegape: bool = True


@dataclass(frozen=True)
class UnitsConfig:
    """How to label/convert Y-axis values during Stage 2 plotting.

    We keep the Stage-1 cache in *code units* so you can regenerate plots with
    different unit systems (or no conversion) without re-extracting heavy data.

    Notes
    -----
    - Scalars are typically plotted in Smilei's code/normalized units (SI energy
      conversion would require defining an effective volume), so we keep them as
      "code" even when `mode="si"`.
    - Fields (E/B/J/Rho) and ParticleBinning (p/v moments) can be converted.
    """

    mode: str = "si"  # "si" or "code"

    # Convert Field-like signals (E/B/J/Rho) to SI if mode=="si"
    fields_to_si: bool = True

    # Convert ParticleBinning moment-like signals to SI if mode=="si"
    pb_to_si: bool = True

    # ParticleBinning axis interpretation for moment conversion.
    #  - "auto": heuristic per-signal
    #  - "px"  : treat p as px normalized to (m_e c)
    #  - "vx"  : treat p as vx normalized to c
    pb_axis_mode: str = "auto"


@dataclass(frozen=True)
class PlotConfig:
    """Global plot defaults (Stage 2)."""

    # Wider plots make long runs readable. (width, height) in inches.
    # Applies to *time-domain* plots only (raw/smoothed/residual).
    figsize_time_domain: Tuple[float, float] = (72.0, 4.0)

    # Spectrograms benefit from a more square-ish aspect ratio.
    # This is intentionally independent from `figsize_time_domain`.
    figsize_spectrogram: Tuple[float, float] = (10.0, 6.0)

    # Backwards-compat alias: if you already edited `figsize`, we still honor it
    # as the time-domain size.
    figsize: Tuple[float, float] = (12.0, 4.0)
    dpi: int = 300

    # Cap number of points drawn for *time-domain* plots (raw/smoothed/residual).
    # Spectrogram/STFT is computed from full-resolution data.
    max_points_time_domain: int = 3000

@dataclass(frozen=True)
class ScalarsConfig:
    enabled: bool = True
    # If a name doesn't exist in scalars.txt, it will be skipped.
    preferred_columns: Tuple[str, ...] = (
        "Utot", "Ukin", "Uelm",
        "Ukin_ion1", "Ukin_eon1",
        "ExMax", "JxMax",
    )

@dataclass(frozen=True)
class FieldsConfig:
    enabled: bool = True
    diag_number: int = 0

    # Fields to reduce to time-series metrics.
    # If Bx/By/Bz are stored as Bx_m/By_m/Bz_m, the extractor will try those automatically.
    field_names: Tuple[str, ...] = (
        "Ex", "Ey", "Ez",
        "Bx", "By", "Bz",
        "Jx", "Jy", "Jz",
        "Rho",
    )

    # Metrics computed over x per dump.
    metrics: Tuple[str, ...] = ("rms", "maxabs", "mean", "std")

    # Virtual probe locations as fractions of the *interior* (after excluding boundaries).
    virtual_probe_fracs: Tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 0.875)

    # Exclude boundaries when computing global metrics (fraction of cells on each side).
    exclude_frac: float = 0.05

    # Print progress every N dumps per field
    progress_every: int = 2000

    # Safety: for debugging you can cap dumps processed (None means full run)
    max_dumps: Optional[int] = None

@dataclass(frozen=True)
class PipelineConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    scalars: ScalarsConfig = field(default_factory=ScalarsConfig)
    fields: FieldsConfig = field(default_factory=FieldsConfig)
    particle_binning: ParticleBinningConfig = field(default_factory=ParticleBinningConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    units: UnitsConfig = field(default_factory=UnitsConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

def config_to_dict(cfg: PipelineConfig) -> Dict[str, Any]:
    return asdict(cfg)
