# Smilei Offline Multi-Scale Analysis Pipeline (Scalars / Fields / ParticleBinning)

An **offline**, **reproducible**, **multi-scale** post-processing pipeline for Smilei PIC simulations (currently optimized for **1D** runs) that:

- reduces heavy diagnostics on disk into lightweight time-series caches,
- applies **multiple independent smoothing windows** (electron → ion scales),
- computes **spectrogram/STFT** time–frequency maps,
- produces consistent plots to compare physics across scales,
- avoids rerunning expensive simulations.

This is designed for workflows where Smilei output is large (10s of GB) and you want to iterate on analysis parameters quickly without touching the heavy data repeatedly.

## TL;DR (30 seconds)

From your project root (where your Smilei run directory lives):

```bash
# one command: extracts once (if cache missing) + makes plots
python -u smilei_offline_pipeline/main_driver.py --run-dir full_sim_output

# plotting only (no heavy HDF5 reads)
python -u smilei_offline_pipeline/main_driver.py --run-dir full_sim_output --analyze-only
```

Outputs:
- cache: `full_sim_output/reduced_offline/` (MB-scale)
- figures: `full_sim_output/diagnostics_output_offline/`

**Important:** do *not* commit `full_sim_output/`, `reduced_offline/`, or `diagnostics_output_offline/` to git.


## What this project does (high level)

### Stage 1 — Extraction (heavy, runs rarely)
Reads Smilei data from disk via `happi` and turns it into compact time-series arrays:

- **Scalars**: selected columns from `scalars.txt`
- **Fields**: for each dump (spatial snapshot), compute compact domain metrics such as:
  - RMS, max|field|, mean, std vs time for Ex/Ey/Ez, Bx/By/Bz, Jx/Jy/Jz, Rho
  - “Virtual probes” (sample a field at several x-locations) without rerunning with Probe diagnostics
- **ParticleBinning**: reduce histogram/phase-space diagnostics into time-series metrics (e.g. momentum variance, asymmetry, tail measures)

Outputs are written to a **lightweight cache**:
- `reduced_offline/reduced_data.npz` (compact arrays)
- `reduced_offline/metadata.json` (config + timing info for reproducibility)

### Stage 2 — Analysis (fast, runs often)
Loads the compact `.npz` cache and generates plots:

- moving averages with **multiple window sizes** in SI seconds
- raw vs smoothed comparisons (same dataset, different windows)
- residuals: `raw - smoothed` (isolates fast components)
- spectrogram/STFT maps (time–frequency evolution), **independent** from smoothing window sizes
- figures saved to `diagnostics_output_offline/`
- optionally converts cached signals to **SI units for plotting** (without re-extraction) via a `units_mode` setting
- **time-domain plots** may be downsampled for readability (screen point cap), but STFT/spectrogram calculations are not downsampled

---

## Why multi-window smoothing matters (physics motivation)

PIC outputs contain signals at very different characteristic timescales:

- **electron-scale fluctuations** (fast waves, ωpe-scale)
- **ion/proton-scale dynamics** (slow envelopes, ωpi-scale)

Instead of relying on Smilei’s internal averaging, this pipeline performs **independent offline averaging** so you can:
- preserve fast dynamics with small windows,
- suppress electron-scale noise to reveal slow ion-scale trends with large windows,
- compare results *across windows on the exact same raw data*.

---

## Requirements

- Python 3.10+ recommended (works with newer versions too)
- `numpy`
- `matplotlib`
- `scipy`
- Smilei `happi` (available with your Smilei installation)
- Optional: `pint` (only for unit convenience; pipeline works without it)

## Repository layout

Typical layout (example):

```
Research/
  full_sim_output/                # Smilei results (DO NOT COMMIT)
  Smilei/                         # Smilei source + happi
  smilei_offline_pipeline/        # this repo
    main_driver.py
    pipeline_config.py
    extractors/
    analysis/
    utils/
    validate_step4.py
    README.md
```

---

## Installation

### 1) Place the folder
Put `smilei_offline_pipeline/` inside your main project directory (e.g. `Research/`).

Example:
```
Research/
  full_sim_output/                # Smilei results directory
  Smilei/                         # Smilei source + happi
  smilei_offline_pipeline/        # this project
```

### 2) Create/activate a venv and install deps
From the `Research/` directory:
```bash
python -m venv .venv
source .venv/bin/activate

pip install numpy matplotlib scipy
```

`happi` is imported from your Smilei installation; ensure Smilei’s Python path is available (as in your existing analysis scripts).

---

## Quick start

### Run everything (extract if needed, then plot)
```bash
python -u smilei_offline_pipeline/main_driver.py --run-dir full_sim_output
```

### Analysis only (never touches heavy Smilei files)
```bash
python -u smilei_offline_pipeline/main_driver.py --run-dir full_sim_output --analyze-only
```

### ParticleBinning only
```bash
python -u smilei_offline_pipeline/main_driver.py --run-dir full_sim_output --only-pb
```

---

## Output locations

Inside your run directory (`--run-dir full_sim_output`):

- **Cache**
  - `full_sim_output/reduced_offline/reduced_data.npz`
  - `full_sim_output/reduced_offline/metadata.json`

- **Figures**
  - `full_sim_output/diagnostics_output_offline/`

The cache is meant to be small compared to your original output (GBs → MBs).

---

## Data safety (what not to commit)

This pipeline is designed for **very large** Smilei outputs. Keep your repo lightweight.

Do **NOT** commit:
- `full_sim_output/` (or any Smilei run directory)
- `reduced_offline/` (cache)
- `diagnostics_output_offline/` (figures)
- `*.h5`, `*.hdf5`, `*.npz`

Recommended `.gitignore` entries:

```gitignore
# Python
__pycache__/
*.pyc
.venv/

# Smilei heavy outputs / caches
full_sim_output/
reduced_offline/
diagnostics_output_offline/
*.h5
*.hdf5
*.npz

# macOS
.DS_Store
```

---

## Configuration (`pipeline_config.py`)

All analysis parameters live in:
```
smilei_offline_pipeline/pipeline_config.py
```

Key sections:

### PhysicsConfig
- `n_ref_m3` — reference density, used to compute ωpe for normalized frequency axes and for time conversion if needed; also used for SI scaling factors when `units_mode="si"`
- `dt_code` — Smilei timestep in code units
- `T_ref_s` — optional override; if `None`, computed from ωpe(n_ref_m3)

### UnitsConfig (plot-time SI conversion)
- `mode` — `"si"` or `"code"` (conversion applied in Stage 2 only; cache remains in code units)
- `convert_fields` — apply SI scaling for E/B/J/Rho
- `convert_particle_binning` — apply SI scaling for PB metrics
- `pb_axis_mode` — `"px"` or `"vx"` (do not rely on heuristics; set explicitly based on your PB diagnostic)

### FieldsConfig
- `field_names` — fields to reduce (Ex/Ey/Ez, Bx/By/Bz, Jx/Jy/Jz, Rho)
- `metrics` — per-dump reductions: `rms`, `maxabs`, `mean`, `std`
- `virtual_probe_fracs` — probe points as fractions of domain interior
- `exclude_frac` — boundary exclusion to avoid edge artifacts
- `progress_every` — progress printing for long runs

### ScalarsConfig
- `preferred_columns` — scalar names to extract (skips missing columns automatically)

### ParticleBinningConfig
- `diag_numbers` — which PB diagnostics to reduce
- `tail_multiple` — how tail threshold is defined (may be adjusted depending on run)
- `progress_every`, `max_dumps`

### WindowingConfig (multi-scale smoothing)
- `windows_seconds` — moving-average windows in **SI seconds**; example 6-window ladder tied to electron/proton plasma periods (Te and Ti):
  - 4.58e-10, 1.76e-9, 7.04e-9, 3.52e-8, 1.51e-7, 7.55e-7
- `mode` — `"trailing"` or `"centered"` (trailing is causal; centered is phase-preserving)

### SpectrogramConfig (STFT)
- `window_seconds` — STFT segment length in **SI seconds**, independent from smoothing windows; example: 7.04e-8 seconds
- `overlap_frac` — typical 0.5 is a good default
- `plot_omega_over_omegape` — plot ω/ωpe if physics metadata is available

### PlotConfig (layout + readability)
- `figsize_time_domain` — controls only time-domain plots
- `max_points_time_domain` — caps points only for time-domain plots (does not affect STFT/spectrogram)
- `figsize_spectrogram` — separate size for spectrogram figures (not affected by time-domain width)
- `dpi`

---

## Time axis and units (important detail)

Smilei/happi can differ in how it reports time for diagnostics:
- sometimes `getTimes()` returns **code time**
- sometimes it returns **iteration index**
- sometimes it returns a mixed representation depending on happi version

This pipeline uses robust inference and sanity checks:
- uses the diagnostic’s `getTimes()` and measured median Δt
- converts to SI seconds using `dt_code` and `T_ref_s` (computed if not provided)
- warns if time series appear inconsistent

All plots are labeled in **SI seconds** when possible.

By default, cached signals are stored in Smilei/code units. If `units_mode="si"` is enabled, Stage 2 rescales Fields (E/B/J/Rho) and PB metrics to SI **for plotting**, and labels axes accordingly. This avoids re-extraction and preserves the ability to plot in code units.

---

## What signals are produced

### Scalars
Example output keys:
- `Scalars/Utot`
- `Scalars/Ukin`
- `Scalars/Uelm`
- species-specific terms if present (e.g. `Ukin_ion1`, `Ukin_eon1`)

### Fields (reduced from spatial snapshots)
Example output keys:
- `Fields/Ex/rms`, `Fields/Ex/maxabs`, ...
- `Fields/Jx/rms`, ...
- `Fields/Rho/std`, ...
- `Fields/Ex/probe@0.500` (virtual probe)
- analogous for other fields if enabled

### ParticleBinning (reduced metrics)
Depends on diag type (histogram vs x–p phase-space), but common metrics include:
- `PB{diag}/p_mean`
- `PB{diag}/p_var` (broadening/heating)
- `PB{diag}/asym` (drift/asymmetry)
- `PB{diag}/tail_frac` (caution: may saturate if threshold too low)

---

## Caching behavior (and why it exists)

To avoid rereading ~10–70GB output:
- Stage 1 writes `reduced_data.npz`
- Stage 2 reads the `.npz` and generates plots fast

**Key design rule:**
- Changing **smoothing windows** or **spectrogram settings** should not trigger extraction.
- Only changes that affect extraction (chosen signals, PB diag list, etc.) should.
- Changing `units_mode`, plot sizes, or `max_points_time_domain` should not trigger extraction; use `--analyze-only` for plotting-only runs.

For CV/public repos, the recommended workflow is: keep code in git, keep run outputs/caches locally (ignored).

If you want to force re-extraction, delete:
```
full_sim_output/reduced_offline/
```

---

## Recommended “representative” plots for reporting

1) **Fields/Jx/rms** or **Fields/Ex/rms** vs time (raw + smoothing ladder)
   - global instability activity measure  
2) Validation-style plot:  
   - `Fields/Ex/probe@0.5: raw vs selected smooth windows`  
   OR  
   - `Fields/Ex/probe@0.5: PSD check (raw/smooth/residual)`  
3) Spectrogram (optional)  
   - best used when you expect coherent bands; otherwise PSD/residual plots can be more interpretable

---

## Troubleshooting

### Pint is optional and not required
Harmless. SI plotting uses internal scaling from `n_ref_m3` and does not require Pint.

### PB units look wrong (e.g., v² vs p²)
Set `pb_axis_mode` explicitly (`px` vs `vx`).  
If PB diagnostics are x–px, you should use `px` and expect p-mean/p-var in momentum units; if x–vx, use `vx`.

### Pipeline is slow / “hangs” at start
Opening happi can take time for large HDF5 outputs (metadata scan). That’s normal.

### Cache re-extracts when I only changed windows
Use:
```bash
--analyze-only
```
If it still re-extracts, verify your pipeline version includes the cache patch.

### ParticleBinning tail metric looks weird (flat near 1)
That usually means the tail threshold is too low for the distribution. Solutions:
- increase `tail_multiple`
- switch to quantile-based metrics (p95/p99) or kurtosis-like measures

### Plots show unreadable y-axis offsets (e.g. `1e-6 + 0.9999`)
This is Matplotlib’s offset formatting. For presentation figures, disable offset or plot deltas
relative to the initial value.

---

## Extending the pipeline

### Adding Probes later
Probes are not implemented yet by design, but the architecture supports adding a new extractor:
- `extractors/extract_probes.py`
- follow the same “reduce to time series → save to npz” pattern

### Adding a new metric
- Add it during Stage 1 extraction (recommended)
- Store as a new `t__...` / `y__...` pair in `.npz`
- The analysis layer automatically treats it like any other signal

---

## Acknowledgements / credit

Built for post-processing **Smilei** PIC simulations using the official `happi` analysis interface.
If you use this pipeline in academic work, consider citing Smilei and/or your lab’s internal methodology as appropriate.

---

## Reproducibility

Every run writes:
- the reduced cache (`.npz`)
- metadata/config snapshot (`metadata.json`)
- plots in a deterministic naming scheme

This makes it safe to:
- compare across runs
- regenerate plots after changing visualization settings
- share the pipeline with new students
