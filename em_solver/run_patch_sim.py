"""Run a 2.4 GHz microstrip patch FDTD verification simulation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from .cpml import CPML
    from .excitation import GaussianPort
    from .fdtd_core import FDTD
    from .geometry import SUBSTRATE_THICKNESS, create_patch_geometry, patch_dimensions
except ImportError:  # Supports: python em_solver/run_patch_sim.py
    from cpml import CPML
    from excitation import GaussianPort
    from fdtd_core import FDTD
    from geometry import SUBSTRATE_THICKNESS, create_patch_geometry, patch_dimensions


def _odd_cells(length, delta, minimum):
    cells = int(np.ceil(length / delta))
    cells = max(cells, minimum)
    return cells if cells % 2 == 1 else cells + 1


def build_patch_sim(f0, dx, dy, dz, pml_cells, air_margin):
    """Build a grid large enough for the analytical patch dimensions."""
    patch_w, patch_l, _, _ = patch_dimensions(f0)

    nx = _odd_cells(patch_w + 2 * air_margin, dx, 2 * pml_cells + 21)
    ny = _odd_cells(patch_l + 2 * air_margin, dy, 2 * pml_cells + 21)

    substrate_cells = max(1, int(round(SUBSTRATE_THICKNESS / dz)))
    nz = pml_cells + 2 + substrate_cells + pml_cells + 12

    sim = FDTD(dx=dx, dy=dy, dz=dz, Nx=nx, Ny=ny, Nz=nz)
    geom = create_patch_geometry(sim, f0=f0)
    CPML(sim, thickness=pml_cells)
    port = GaussianPort(sim, geom, f0=f0)
    sim.sources.append(port)
    return sim, geom, port


def write_s11_csv(path, freqs, s11):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "s11_real", "s11_imag", "s11_db"])
        for freq, value in zip(freqs, s11):
            mag = max(abs(value), 1e-30)
            writer.writerow([freq, value.real, value.imag, 20 * np.log10(mag)])


def write_s11_plot(path, freqs, s11):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-30))
    plt.figure(figsize=(7, 4))
    plt.plot(freqs * 1e-9, s11_db)
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return True


def run(args):
    sim, geom, port = build_patch_sim(
        f0=args.f0,
        dx=args.dx,
        dy=args.dy,
        dz=args.dz,
        pml_cells=args.pml_cells,
        air_margin=args.air_margin,
    )

    print(
        f"Grid: {sim.Nx} x {sim.Ny} x {sim.Nz}, "
        f"dt={sim.dt * 1e12:.3f} ps, steps={args.steps}"
    )
    sim.run(args.steps, verbose=not args.quiet)

    freqs, s11 = port.get_s11(args.f_min, args.f_max, args.points)
    s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-30))
    min_idx = int(np.argmin(s11_db))

    csv_path = args.output_dir / "patch_s11.csv"
    png_path = args.output_dir / "patch_s11.png"
    write_s11_csv(csv_path, freqs, s11)
    plotted = write_s11_plot(png_path, freqs, s11)

    print(
        f"S11 minimum: {s11_db[min_idx]:.2f} dB at "
        f"{freqs[min_idx] * 1e-9:.4f} GHz"
    )
    print(f"Wrote {csv_path}")
    if plotted:
        print(f"Wrote {png_path}")
    else:
        print("Skipped plot because matplotlib is not installed")

    return {
        "geometry": geom,
        "freqs": freqs,
        "s11": s11,
        "csv_path": csv_path,
        "png_path": png_path if plotted else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0", type=float, default=2.4e9, help="Center frequency in Hz")
    parser.add_argument("--steps", type=int, default=3000, help="FDTD time steps")
    parser.add_argument("--dx", type=float, default=1.5e-3, help="Grid spacing in x")
    parser.add_argument("--dy", type=float, default=1.5e-3, help="Grid spacing in y")
    parser.add_argument("--dz", type=float, default=0.2e-3, help="Grid spacing in z")
    parser.add_argument("--pml-cells", type=int, default=10, help="CPML thickness")
    parser.add_argument("--air-margin", type=float, default=25e-3, help="XY air margin")
    parser.add_argument("--f-min", type=float, default=1.5e9, help="S11 sweep start")
    parser.add_argument("--f-max", type=float, default=3.5e9, help="S11 sweep stop")
    parser.add_argument("--points", type=int, default=401, help="S11 frequency points")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--quiet", action="store_true", help="Suppress step progress")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
