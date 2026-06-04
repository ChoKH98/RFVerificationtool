import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from em_solver.fdtd_core import FDTD
from em_solver.geometry import EPS_R_ROGERS, patch_dimensions
from em_solver.run_patch_sim import _odd_cells, build_patch_sim, run, write_s11_csv


class GeometryTests(unittest.TestCase):
    def test_patch_dimensions_are_in_expected_24ghz_range(self):
        width, length, eps_eff, delta_l = patch_dimensions()

        self.assertGreater(width, 0.040)
        self.assertLess(width, 0.043)
        self.assertGreater(length, 0.032)
        self.assertLess(length, 0.034)
        self.assertGreater(eps_eff, 3.3)
        self.assertLess(eps_eff, EPS_R_ROGERS)
        self.assertGreater(delta_l, 0.0)

    def test_odd_cells_enforces_minimum_and_odd_count(self):
        self.assertEqual(_odd_cells(0.0, 1.0, 20), 21)
        self.assertEqual(_odd_cells(10.0, 1.0, 3), 11)


class FDTDTests(unittest.TestCase):
    def test_material_update_changes_coefficients(self):
        sim = FDTD(dx=1e-3, dy=1e-3, dz=1e-3, Nx=6, Ny=6, Nz=6)
        old_ca = sim.Ca[2, 2, 2]

        sim.set_material(2, 3, 2, 3, 2, 3, eps_r=4.0, sigma=0.01)

        self.assertNotEqual(old_ca, sim.Ca[2, 2, 2])
        self.assertAlmostEqual(sim.eps[2, 2, 2], 4.0 * 8.854e-12)
        self.assertAlmostEqual(sim.sigma[2, 2, 2], 0.01)


class OutputTests(unittest.TestCase):
    def test_write_s11_csv_outputs_expected_header_and_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "s11.csv"
            freqs = np.array([1.0, 2.0])
            s11 = np.array([1.0 + 0.0j, 0.5 + 0.0j])

            write_s11_csv(path, freqs, s11)

            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows[0], ["frequency_hz", "s11_real", "s11_imag", "s11_db"])
        self.assertAlmostEqual(float(rows[1][3]), 0.0)
        self.assertAlmostEqual(float(rows[2][3]), -6.020599913279624)


class SimulationSmokeTests(unittest.TestCase):
    def test_build_patch_sim_has_valid_feed_and_cpml(self):
        with contextlib.redirect_stdout(io.StringIO()):
            sim, geom, port = build_patch_sim(
                f0=2.4e9,
                dx=1.5e-3,
                dy=1.5e-3,
                dz=0.2e-3,
                pml_cells=10,
                air_margin=25e-3,
            )

        self.assertIs(sim._cpml is not None, True)
        self.assertGreater(geom["x1"], geom["x0"])
        self.assertGreater(geom["y1"], geom["y0"])
        self.assertGreater(port.z1_idx, port.z0_idx)

    def test_short_patch_run_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                f0=2.4e9,
                steps=2,
                dx=1.5e-3,
                dy=1.5e-3,
                dz=0.2e-3,
                pml_cells=10,
                air_margin=25e-3,
                f_min=1.5e9,
                f_max=3.5e9,
                points=5,
                output_dir=Path(tmpdir),
                quiet=True,
            )

            with contextlib.redirect_stdout(io.StringIO()):
                result = run(args)

            self.assertTrue((Path(tmpdir) / "patch_s11.csv").exists())
            self.assertEqual(len(result["freqs"]), 5)
            self.assertEqual(len(result["s11"]), 5)


if __name__ == "__main__":
    unittest.main()
