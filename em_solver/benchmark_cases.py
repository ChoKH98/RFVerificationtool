"""Benchmark case catalog and RF reference formulas."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


C0 = 3.0e8


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    structure_type: str
    description: str
    frequency_hz: tuple[float, ...]
    required_artifacts: tuple[str, ...]


def waveguide_cutoff_hz(a_m, mode_m=1, mode_n=0, b_m=None):
    """Rectangular waveguide cutoff frequency."""
    if b_m is None:
        b_m = a_m / 2.0
    return 0.5 * C0 * math.sqrt((mode_m / a_m) ** 2 + (mode_n / b_m) ** 2)


def waveguide_te10_beta(frequency_hz, a_m):
    """TE10 phase constant in a rectangular waveguide."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    k0 = 2.0 * np.pi * frequency_hz / C0
    kc = np.pi / a_m
    return np.sqrt(np.maximum(k0**2 - kc**2, 0.0))


def horn_gain_dbi(a_aperture_m, b_aperture_m, frequency_hz, efficiency=0.55):
    """Pyramidal horn aperture gain estimate."""
    wavelength = C0 / frequency_hz
    gain_linear = efficiency * 4.0 * np.pi * a_aperture_m * b_aperture_m / wavelength**2
    return 10.0 * math.log10(gain_linear)


def aperture_hpbw_deg(aperture_m, frequency_hz):
    """Large-aperture HPBW estimate from Kai Chang-style antenna rules."""
    return 70.0 * (C0 / frequency_hz) / aperture_m


def phased_array_beam_angle_deg(progressive_phase_deg, spacing_m, frequency_hz):
    """Broadside linear-array scan angle from progressive phase."""
    wavelength = C0 / frequency_hz
    k0d = 2.0 * np.pi * spacing_m / wavelength
    phase_rad = math.radians(progressive_phase_deg)
    arg = -phase_rad / k0d
    return math.degrees(math.asin(max(-1.0, min(1.0, arg))))


def grating_lobe_max_spacing_m(frequency_hz, max_scan_deg):
    """Maximum element spacing to avoid grating lobes up to max_scan_deg."""
    wavelength = C0 / frequency_hz
    return wavelength / (1.0 + math.sin(math.radians(max_scan_deg)))


BENCHMARK_CASES = {
    "array_l_s_c_x": BenchmarkCase(
        case_id="array_l_s_c_x",
        structure_type="layered_phased_array",
        description="Layered microstrip phased-array sweep across L/S/C/X representative bands.",
        frequency_hz=(1.5e9, 2.4e9, 5.8e9, 10.0e9),
        required_artifacts=("sparams.csv", "field_near.npz", "field_medium.npz", "field_far.npz"),
    ),
    "horn_xband": BenchmarkCase(
        case_id="horn_xband",
        structure_type="pyramidal_horn",
        description="X-band pyramidal horn radiation and S-parameter correlation.",
        frequency_hz=(8.2e9, 10.0e9, 12.4e9),
        required_artifacts=("sparams.csv", "field_near.npz", "field_far.npz"),
    ),
    "waveguide_family": BenchmarkCase(
        case_id="waveguide_family",
        structure_type="rectangular_waveguide",
        description="Rectangular waveguide cutoff, TE10 propagation, and S11/S21 correlation.",
        frequency_hz=(0.9e9, 1.7e9, 3.95e9, 10.0e9),
        required_artifacts=("sparams.csv", "field_near.npz"),
    ),
    "pec_corner_90": BenchmarkCase(
        case_id="pec_corner_90",
        structure_type="pec_corner_scattering",
        description="Free-space PEC 90-degree corner near-field hotspot and RCS scattering.",
        frequency_hz=(2.4e9, 5.8e9, 10.0e9),
        required_artifacts=("field_near.npz", "rcs.csv"),
    ),
    "wr90_bend_90": BenchmarkCase(
        case_id="wr90_bend_90",
        structure_type="waveguide_bend_scattering",
        description="WR-90 90-degree bend S-parameter and TE10 mode scattering.",
        frequency_hz=(8.2e9, 10.0e9, 12.4e9),
        required_artifacts=("sparams.csv", "field_near.npz"),
    ),
}


def get_case(case_id):
    try:
        return BENCHMARK_CASES[case_id]
    except KeyError as exc:
        known = ", ".join(sorted(BENCHMARK_CASES))
        raise ValueError(f"Unknown benchmark case {case_id!r}; expected one of: {known}") from exc
