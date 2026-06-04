"""Run a short patch antenna simulation and write smoke-test artifacts."""

from argparse import Namespace
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from em_solver.run_patch_sim import run


def main():
    args = Namespace(
        f0=2.4e9,
        steps=20,
        dx=1.5e-3,
        dy=1.5e-3,
        dz=0.2e-3,
        pml_cells=10,
        air_margin=25e-3,
        f_min=1.5e9,
        f_max=3.5e9,
        points=21,
        output_dir=Path("results/_tmp_example"),
        quiet=True,
    )
    run(args)


if __name__ == "__main__":
    main()
