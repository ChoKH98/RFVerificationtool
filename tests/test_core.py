import csv
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from em_solver.compare_s11 import compare_s11, read_s11_csv
from em_solver.benchmark import compare_case
from em_solver.benchmark_cases import (
    BENCHMARK_CASES,
    aperture_hpbw_deg,
    grating_lobe_max_spacing_m,
    horn_gain_dbi,
    phased_array_beam_angle_deg,
    waveguide_cutoff_hz,
)
from em_solver.fdtd_core import FDTD
from em_solver.geometry import EPS_R_ROGERS, patch_dimensions
from em_solver.run_patch_sim import _odd_cells, build_patch_sim, run, write_s11_csv
from em_solver.validation import (
    compare_far_fields,
    compare_fields,
    compare_rcs,
    compare_sparameters,
    read_far_field_npz,
    read_field_npz,
    read_rcs_csv,
    read_sparameters_csv,
    write_metrics_json,
    write_summary_csv,
)
from em_solver.external_solvers import detect_solvers, write_openems_runner
from em_solver.optimization import recommendations_from_metrics
from em_solver.bridge import create_bridge
from em_solver.workflow import inspect_case, run_case
from em_solver.candidate_exports import write_waveguide_family_candidate


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

    def test_compare_s11_reports_resonance_and_db_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate_path = Path(tmpdir) / "candidate.csv"
            reference_path = Path(tmpdir) / "reference.csv"
            freqs = np.array([2.3e9, 2.4e9, 2.5e9])

            write_s11_csv(candidate_path, freqs, np.array([0.8, 0.2, 0.7], dtype=complex))
            write_s11_csv(reference_path, freqs, np.array([0.9, 0.25, 0.8], dtype=complex))

            comparison = compare_s11(read_s11_csv(candidate_path), read_s11_csv(reference_path))

        self.assertAlmostEqual(comparison.candidate_resonance_hz, 2.4e9)
        self.assertAlmostEqual(comparison.reference_resonance_hz, 2.4e9)
        self.assertGreater(comparison.mean_abs_db_error, 0.0)
        self.assertTrue(comparison.passes(mean_db_tolerance=3.0))

    def test_standard_sparameter_compare_passes_for_small_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = Path(tmpdir) / "candidate.csv"
            reference = Path(tmpdir) / "reference.csv"
            header = "frequency_hz,port_i,port_j,real,imag,db,phase_deg\n"
            rows_ref = "1000000000,1,1,0.5,0,0,0\n2000000000,1,1,0.25,0,0,0\n"
            rows_cand = "1000000000,1,1,0.49,0,0,0\n2000000000,1,1,0.26,0,0,0\n"
            reference.write_text(header + rows_ref, encoding="utf-8")
            candidate.write_text(header + rows_cand, encoding="utf-8")

            metrics = compare_sparameters(read_sparameters_csv(candidate), read_sparameters_csv(reference))

        self.assertTrue(all(metric.passed for metric in metrics))

    def test_benchmark_compare_flags_missing_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            candidate = tmp / "candidate"
            hfss = tmp / "hfss"
            candidate.mkdir()
            hfss.mkdir()

            comparisons = compare_case("horn_xband", candidate, hfss)

        self.assertGreater(len(comparisons), 0)
        self.assertFalse(all(comparison.passed for comparison in comparisons))
        self.assertEqual(comparisons[0].metrics[0].name, "candidate_artifact_missing")

    def test_field_far_and_rcs_comparisons_use_standard_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
            field_shape = (1, 2)
            np.savez(
                tmp / "field_ref.npz",
                frequency_hz=np.array([1.0e9]),
                points_m=points,
                Ex=np.ones(field_shape, dtype=complex),
            )
            np.savez(
                tmp / "field_cand.npz",
                frequency_hz=np.array([1.0e9]),
                points_m=points,
                Ex=np.ones(field_shape, dtype=complex) * 1.02,
            )
            field_metrics = compare_fields(
                read_field_npz(tmp / "field_cand.npz"),
                read_field_npz(tmp / "field_ref.npz"),
                "field_near.npz",
            )

            theta = np.array([0.0, 10.0])
            phi = np.array([0.0, 0.0])
            np.savez(
                tmp / "far_ref.npz",
                frequency_hz=np.array([1.0e9]),
                theta_deg=theta,
                phi_deg=phi,
                Etheta=np.array([1.0 + 0j, 0.5 + 0j]),
                Ephi=np.array([0.0 + 0j, 0.0 + 0j]),
                gain_dbi=np.array([10.0, 8.0]),
            )
            np.savez(
                tmp / "far_cand.npz",
                frequency_hz=np.array([1.0e9]),
                theta_deg=theta,
                phi_deg=phi,
                Etheta=np.array([1.0 + 0j, 0.5 + 0j]),
                Ephi=np.array([0.0 + 0j, 0.0 + 0j]),
                gain_dbi=np.array([9.5, 8.0]),
            )
            far_metrics = compare_far_fields(read_far_field_npz(tmp / "far_cand.npz"), read_far_field_npz(tmp / "far_ref.npz"))

            rcs_header = "frequency_hz,theta_deg,phi_deg,rcs_dbsm\n"
            (tmp / "rcs_ref.csv").write_text(rcs_header + "1000000000,0,0,3\n1000000000,10,0,0\n", encoding="utf-8")
            (tmp / "rcs_cand.csv").write_text(rcs_header + "1000000000,0,0,2.5\n1000000000,10,0,-0.2\n", encoding="utf-8")
            rcs_metrics = compare_rcs(read_rcs_csv(tmp / "rcs_cand.csv"), read_rcs_csv(tmp / "rcs_ref.csv"))

            write_metrics_json(tmp / "metrics.json", "synthetic", [])
            write_summary_csv(tmp / "summary.csv", [])

        self.assertTrue(all(metric.passed for metric in field_metrics))
        self.assertTrue(all(metric.passed for metric in far_metrics))
        self.assertTrue(all(metric.passed for metric in rcs_metrics))


class BenchmarkFormulaTests(unittest.TestCase):
    def test_benchmark_catalog_contains_required_cases(self):
        self.assertIn("array_l_s_c_x", BENCHMARK_CASES)
        self.assertIn("horn_xband", BENCHMARK_CASES)
        self.assertIn("waveguide_family", BENCHMARK_CASES)
        self.assertIn("pec_corner_90", BENCHMARK_CASES)
        self.assertIn("wr90_bend_90", BENCHMARK_CASES)

    def test_rf_reference_formulas_are_in_expected_ranges(self):
        wr90_a = 0.9 * 0.0254
        self.assertAlmostEqual(waveguide_cutoff_hz(wr90_a) / 1e9, 6.56, places=1)
        self.assertGreater(horn_gain_dbi(0.22, 0.10, 10.0e9), 20.0)
        self.assertGreater(aperture_hpbw_deg(0.22, 10.0e9), 5.0)
        self.assertLess(abs(phased_array_beam_angle_deg(0.0, 0.5 * 3e8 / 2.4e9, 2.4e9)), 1e-12)
        self.assertLess(grating_lobe_max_spacing_m(2.4e9, 60.0), 3e8 / 2.4e9)


class ExternalSolverWorkflowTests(unittest.TestCase):
    def test_solver_detection_returns_status_object(self):
        status = detect_solvers()
        self.assertTrue(hasattr(status, "can_run_openems_scripts"))
        self.assertTrue(hasattr(status, "can_run_hfss"))

    def test_openems_runner_generation_writes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner, metadata = write_openems_runner("horn_xband", tmpdir)

            runner_text = runner.read_text(encoding="utf-8")
            metadata_text = metadata.read_text(encoding="utf-8")

        self.assertIn("Horn_Antenna.m", runner_text)
        self.assertIn("horn_xband", metadata_text)

    def test_optimization_recommendations_map_failed_metrics(self):
        payload = {
            "comparisons": [
                {
                    "artifact": "field_far.npz",
                    "reference": "HFSS",
                    "metrics": [
                        {
                            "name": "far_beam_pointing_error",
                            "value": 3.0,
                            "tolerance": 2.0,
                            "units": "deg",
                            "passed": False,
                        }
                    ],
                }
            ]
        }

        recommendations = recommendations_from_metrics(payload)

        self.assertEqual(len(recommendations), 1)
        self.assertIn("phased-array", recommendations[0]["recommendation"])

    def test_bridge_creation_writes_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = create_bridge(Path(tmpdir) / "RFVerificationBridge")

            self.assertTrue((root / "README.md").exists())
            self.assertTrue((root / "runners" / "hfss" / "hfss_status.py").exists())
            self.assertTrue((root / "runners" / "hfss" / "run_hfss_waveguide_family.py").exists())
            self.assertTrue((root / "references" / "horn_xband" / "hfss").is_dir())
            self.assertTrue((root / "results" / "wr90_bend_90" / "benchmark").is_dir())

    def test_workflow_inspect_and_run_emit_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = create_bridge(Path(tmpdir) / "RFVerificationBridge")
            status = inspect_case(root, "pec_corner_90")
            comparisons, recommendations = run_case(root, "pec_corner_90")

            benchmark_dir = root / "results" / "pec_corner_90" / "benchmark"
            metrics_exists = (benchmark_dir / "metrics.json").exists()

        self.assertEqual(status["case_id"], "pec_corner_90")
        self.assertGreater(len(comparisons), 0)
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(metrics_exists)

    def test_waveguide_candidate_export_writes_standard_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_waveguide_family_candidate(Path(tmpdir) / "fdtd", points=5)
            sparams = read_sparameters_csv(output / "sparams.csv")
            field = read_field_npz(output / "field_near.npz")

        self.assertEqual(len(set(zip(sparams.port_i, sparams.port_j))), 2)
        self.assertEqual(field.components["Ey"].shape[0], 5)


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
