from __future__ import annotations
import math
from dataclasses import dataclass

# SI constants
EPS0 = 8.8541878128e-12
QE   = 1.602176634e-19
ME   = 9.1093837015e-31
MP   = 1.67262192369e-27
C    = 299792458.0

# Useful derived constant
MU0  = 1.0 / (EPS0 * C * C)

def omega_pe(n_m3: float) -> float:
    """Electron plasma frequency [rad/s] for density n [m^-3]."""
    return math.sqrt(n_m3 * QE*QE / (EPS0 * ME))

def T_ref_from_n(n_m3: float) -> float:
    """Default T_ref [s] = 1/omega_pe(n_ref)."""
    return 1.0 / omega_pe(n_m3)


def normalization_scales(n_ref_m3: float, T_ref_s: float) -> dict:
    """Return common Smilei-style normalization scales.

    Assumes:
      - omega_ref = 1/T_ref_s
      - E_ref = (m_e * c * omega_ref) / e
      - B_ref = E_ref / c
      - J_ref = e * n_ref * c
      - rho_ref = e * n_ref

    These match the conventions used in your original analysis script for
    converting code fields into SI (V/m, T, A/m^2, C/m^3).
    """
    omega_ref = 1.0 / float(T_ref_s)
    E_ref = ME * C * omega_ref / QE
    B_ref = E_ref / C
    J_ref = QE * float(n_ref_m3) * C
    rho_ref = QE * float(n_ref_m3)
    return {
        "omega_ref": omega_ref,
        "E_ref": E_ref,
        "B_ref": B_ref,
        "J_ref": J_ref,
        "rho_ref": rho_ref,
        "me_c": ME * C,
        "c": C,
    }
