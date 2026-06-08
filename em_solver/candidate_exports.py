"""Generate RFVerificationtool candidate artifacts for benchmark cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .benchmark_cases import C0, waveguide_te10_beta
except ImportError:  # Supports: python em_solver/candidate_exports.py ...
    from benchmark_cases import C0, waveguide_te10_beta


WR90_A_M = 0.9 * 0.0254
WR90_B_M = 0.4 * 0.0254
WR90_LENGTH_M = 0.08


def _write_sparams(path, frequency_hz, s11, s21):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "port_i", "port_j", "real", "imag", "db", "phase_deg"])
        for freq, value in zip(frequency_hz, s11):
            writer.writerow([freq, 1, 1, value.real, value.imag, 20.0 * np.log10(max(abs(value), 1e-30)), np.rad2deg(np.angle(value))])
        for freq, value in zip(frequency_hz, s21):
            writer.writerow([freq, 2, 1, value.real, value.imag, 20.0 * np.log10(max(abs(value), 1e-30)), np.rad2deg(np.angle(value))])


def _write_waveguide_field(path, frequency_hz, a_m, b_m, length_m):
    x = np.linspace(0.0, a_m, 9)
    y = np.linspace(0.0, b_m, 5)
    z = np.linspace(0.0, length_m, 7)
    points = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x], dtype=float)

    beta = waveguide_te10_beta(frequency_hz, a_m)
    ex = np.zeros((frequency_hz.size, points.shape[0]), dtype=complex)
    ey = np.zeros_like(ex)
    ez = np.zeros_like(ex)
    hx = np.zeros_like(ex)
    hy = np.zeros_like(ex)
    hz = np.zeros_like(ex)

    for fi, (freq, beta_i) in enumerate(zip(frequency_hz, beta)):
        k0 = 2.0 * np.pi * freq / C0
        fc = C0 / (2.0 * a_m)
        active = 1.0 if freq > fc else 0.0
        phase = np.exp(-1j * beta_i * points[:, 2])
        ey[fi] = active * np.sin(np.pi * points[:, 0] / a_m) * phase
        hx[fi] = -active * (beta_i / max(k0, 1e-30)) * ey[fi] / 377.0
        hz[fi] = active * (np.pi / a_m / max(k0, 1e-30)) * np.cos(np.pi * points[:, 0] / a_m) * phase / 377.0

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frequency_hz=frequency_hz,
        points_m=points,
        Ex=ex,
        Ey=ey,
        Ez=ez,
        Hx=hx,
        Hy=hy,
        Hz=hz,
    )


def write_waveguide_family_candidate(output_dir, f_start=8.2e9, f_stop=12.4e9, points=201, a_m=WR90_A_M, b_m=WR90_B_M, length_m=WR90_LENGTH_M):
    """Write an analytical TE10 WR-90 baseline candidate for the waveguide benchmark."""
    output_dir = Path(output_dir)
    frequency_hz = np.linspace(f_start, f_stop, points)
    beta = waveguide_te10_beta(frequency_hz, a_m)
    fc = C0 / (2.0 * a_m)
    propagating = frequency_hz > fc

    s11 = np.where(propagating, 10.0 ** (-35.0 / 20.0), 0.98) + 0j
    s21 = np.zeros_like(s11, dtype=complex)
    s21[propagating] = 10.0 ** (-0.05 / 20.0) * np.exp(-1j * beta[propagating] * length_m)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_sparams(output_dir / "sparams.csv", frequency_hz, s11, s21)
    _write_waveguide_field(output_dir / "field_near.npz", frequency_hz, a_m, b_m, length_m)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "case_id": "waveguide_family",
                "solver": "RFVerificationtool analytical TE10 baseline",
                "waveguide": "WR-90",
                "a_m": a_m,
                "b_m": b_m,
                "length_m": length_m,
                "cutoff_hz": fc,
                "note": "Initial candidate artifact; replace with full FDTD export as solver matures.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    wg_parser = subparsers.add_parser("waveguide-family", help="Write waveguide_family candidate artifacts")
    wg_parser.add_argument("--output-dir", required=True, type=Path)
    wg_parser.add_argument("--f-start", type=float, default=8.2e9)
    wg_parser.add_argument("--f-stop", type=float, default=12.4e9)
    wg_parser.add_argument("--points", type=int, default=201)
    wg_parser.set_defaults(func=lambda args: write_waveguide_family_candidate(args.output_dir, args.f_start, args.f_stop, args.points))
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.func(args)
    print(f"Wrote candidate artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
