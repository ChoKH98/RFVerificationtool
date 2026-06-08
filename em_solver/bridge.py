"""Create a Windows/WSL bridge folder for HFSS and RFVerificationtool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_WINDOWS_DESKTOP = Path("/mnt/c/Users/whqkr/Desktop")
DEFAULT_BRIDGE_NAME = "RFVerificationBridge"


HFSS_STATUS_SCRIPT = r'''"""Check Windows HFSS/PyAEDT availability from Windows Python."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


AEDT_CANDIDATES = [
    r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\AnsysEM\ansysedtng.exe",
    r"C:\Program Files\ANSYS Inc\ANSYS Student\v252\AnsysEM\ansysedtsv.exe",
    r"C:\Program Files\ANSYS Inc\v252\AnsysEM\ansysedt.exe",
    r"C:\Program Files\ANSYS Inc\v252\AnsysEM\ansysedtng.exe",
]


def main():
    pyaedt_ok = False
    pyaedt_error = None
    try:
        import ansys.aedt.core  # noqa: F401
        pyaedt_ok = True
    except Exception as exc:
        pyaedt_error = str(exc)

    found = [path for path in AEDT_CANDIDATES if Path(path).exists()]
    payload = {
        "cwd": os.getcwd(),
        "python": shutil.which("python") or shutil.which("py"),
        "pyaedt_available": pyaedt_ok,
        "pyaedt_error": pyaedt_error,
        "aedt_candidates_found": found,
    }
    print(json.dumps(payload, indent=2))
    return 0 if pyaedt_ok and found else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


HFSS_RUNNER_TEMPLATE = r'''"""HFSS/PyAEDT export runner template.

Run this from Windows Python, not WSL Python:

    py -m pip install ansys-aedt-core
    py run_hfss_export_template.py --case horn_xband

This template verifies PyAEDT can launch AEDT. Geometry creation/export should
be filled per case, then exported to the standard RFVerificationtool artifacts:

    sparams.csv
    field_near.npz
    field_medium.npz
    field_far.npz
    rcs.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--bridge-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--aedt-version", default="2025.2")
    parser.add_argument("--non-graphical", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from ansys.aedt.core import Hfss

    output_dir = args.bridge_root / "references" / args.case / "hfss"
    output_dir.mkdir(parents=True, exist_ok=True)

    hfss = Hfss(
        version=args.aedt_version,
        non_graphical=args.non_graphical,
        student_version=True,
        new_desktop=True,
        close_on_exit=True,
    )
    try:
        metadata = {
            "case_id": args.case,
            "solver": "HFSS",
            "aedt_version": args.aedt_version,
            "standard_artifacts_dir": str(output_dir),
            "status": "AEDT launched; case geometry/export implementation pending",
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
    finally:
        hfss.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    raise SystemExit(main())
'''


HFSS_WAVEGUIDE_FAMILY_RUNNER = r'''"""Run the HFSS WR-90 waveguide benchmark and export RFVerificationtool artifacts.

Run from Windows PowerShell:

    cd "$env:USERPROFILE\Desktop\RFVerificationBridge\runners\hfss"
    py -m pip install ansys-aedt-core numpy
    py .\run_hfss_waveguide_family.py --non-graphical

The script creates a simple PEC rectangular waveguide with two TE10 wave ports,
solves a driven modal sweep, exports Touchstone, and converts it to:

    references\waveguide_family\hfss\sparams.csv

Near-field export is intentionally left for the second iteration because AEDT
field export setup is more installation/version dependent than Touchstone.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


WR90_A_M = 0.9 * 0.0254
WR90_B_M = 0.4 * 0.0254
WR90_LENGTH_M = 0.08


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--aedt-version", default="2025.2")
    parser.add_argument("--non-graphical", action="store_true")
    parser.add_argument("--f-start-ghz", type=float, default=8.2)
    parser.add_argument("--f-stop-ghz", type=float, default=12.4)
    parser.add_argument("--f-solve-ghz", type=float, default=10.0)
    return parser.parse_args()


def _touchstone_values(path):
    """Yield (freq_hz, s11, s21) from a 2-port Touchstone file."""
    unit_scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    data_format = "ma"
    scale = 1.0
    rows = []
    for raw_line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            parts = line[1:].lower().split()
            if parts:
                scale = unit_scale.get(parts[0], 1.0)
            if "ri" in parts:
                data_format = "ri"
            elif "db" in parts:
                data_format = "db"
            else:
                data_format = "ma"
            continue
        values = [float(item) for item in line.split()]
        if len(values) >= 9:
            rows.append(values[:9])

    def pair_to_complex(a, b):
        if data_format == "ri":
            return complex(a, b)
        if data_format == "db":
            return 10.0 ** (a / 20.0) * complex(math.cos(math.radians(b)), math.sin(math.radians(b)))
        return a * complex(math.cos(math.radians(b)), math.sin(math.radians(b)))

    for row in rows:
        freq = row[0] * scale
        s11 = pair_to_complex(row[1], row[2])
        s21 = pair_to_complex(row[3], row[4])
        yield freq, s11, s21


def _write_sparams_csv(path, touchstone_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "port_i", "port_j", "real", "imag", "db", "phase_deg"])
        for freq, s11, s21 in _touchstone_values(touchstone_path):
            for port_i, port_j, value in ((1, 1, s11), (2, 1, s21)):
                writer.writerow([
                    freq,
                    port_i,
                    port_j,
                    value.real,
                    value.imag,
                    20.0 * math.log10(max(abs(value), 1e-30)),
                    math.degrees(math.atan2(value.imag, value.real)),
                ])


def _face_by_axis_center(obj, axis, target):
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    faces = list(obj.faces)
    return min(faces, key=lambda face: abs(float(face.center[idx]) - target))


def main():
    args = parse_args()
    from ansys.aedt.core import Hfss

    output_dir = args.bridge_root / "references" / "waveguide_family" / "hfss"
    output_dir.mkdir(parents=True, exist_ok=True)

    hfss = Hfss(
        project=str(output_dir / "waveguide_family.aedt"),
        design="WR90_TE10",
        solution_type="Modal",
        version=args.aedt_version,
        non_graphical=args.non_graphical,
        student_version=True,
        new_desktop=True,
        close_on_exit=True,
        remove_lock=True,
    )
    try:
        hfss.modeler.model_units = "mm"
        a = WR90_A_M * 1000.0
        b = WR90_B_M * 1000.0
        length = WR90_LENGTH_M * 1000.0

        air = hfss.modeler.create_box([0, 0, 0], [a, b, length], name="WR90_Air", material="vacuum")
        side_faces = [
            _face_by_axis_center(air, "x", 0.0),
            _face_by_axis_center(air, "x", a),
            _face_by_axis_center(air, "y", 0.0),
            _face_by_axis_center(air, "y", b),
        ]
        hfss.assign_perfect_e(side_faces, name="PEC_Walls")

        port1 = _face_by_axis_center(air, "z", 0.0)
        port2 = _face_by_axis_center(air, "z", length)
        hfss.wave_port(port1, integration_line=hfss.AxisDir.XPos, modes=1, impedance=50, name="P1", characteristic_impedance="Zwave")
        hfss.wave_port(port2, integration_line=hfss.AxisDir.XPos, modes=1, impedance=50, name="P2", characteristic_impedance="Zwave")

        setup = hfss.create_setup(name="Setup1", setup_type="HFSSDriven", Frequency=f"{args.f_solve_ghz}GHz", MaximumPasses=8, MaxDeltaS=0.02)
        sweep = setup.add_sweep(name="Sweep1", sweep_type="Interpolating")
        sweep.props["RangeType"] = "LinearStep"
        sweep.props["RangeStart"] = f"{args.f_start_ghz}GHz"
        sweep.props["RangeEnd"] = f"{args.f_stop_ghz}GHz"
        sweep.props["RangeStep"] = "0.05GHz"
        sweep.update()

        if not hfss.analyze_setup("Setup1", blocking=True):
            raise RuntimeError("HFSS analyze_setup failed")

        touchstone = output_dir / "waveguide_family.s2p"
        exported = hfss.export_touchstone(setup="Setup1", sweep="Sweep1", output_file=str(touchstone), renormalization=False)
        if not exported:
            raise RuntimeError("HFSS export_touchstone failed")
        _write_sparams_csv(output_dir / "sparams.csv", touchstone)
        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "case_id": "waveguide_family",
                    "solver": "HFSS",
                    "aedt_version": args.aedt_version,
                    "waveguide": "WR-90",
                    "a_m": WR90_A_M,
                    "b_m": WR90_B_M,
                    "length_m": WR90_LENGTH_M,
                    "touchstone": str(touchstone),
                    "artifacts": ["sparams.csv"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {output_dir / 'sparams.csv'}")
    finally:
        hfss.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    raise SystemExit(main())
'''


BRIDGE_README = """# RFVerificationBridge

This folder connects Windows HFSS/PyAEDT with the WSL RFVerificationtool.

## Layout

```text
references/<case>/hfss/      HFSS golden artifacts
references/<case>/openems/   openEMS cross-check artifacts
results/<case>/fdtd/         RFVerificationtool candidate artifacts
results/<case>/benchmark/    comparison metrics and optimization recommendations
runners/hfss/                Windows-side PyAEDT scripts
logs/                        run logs
```

## Windows HFSS Check

Open PowerShell in `runners\\hfss`:

```powershell
py -m pip install ansys-aedt-core
py .\\hfss_status.py
py .\\run_hfss_waveguide_family.py --non-graphical
```

## WSL Benchmark Example

```bash
cd /home/whqkrel/RFVerificationtool
python -m em_solver.benchmark compare \\
  --case waveguide_family \\
  --candidate-dir /mnt/c/Users/whqkr/Desktop/RFVerificationBridge/results/waveguide_family/fdtd \\
  --hfss-dir /mnt/c/Users/whqkr/Desktop/RFVerificationBridge/references/waveguide_family/hfss \\
  --openems-dir /mnt/c/Users/whqkr/Desktop/RFVerificationBridge/references/waveguide_family/openems \\
  --output-dir /mnt/c/Users/whqkr/Desktop/RFVerificationBridge/results/waveguide_family/benchmark
```
"""


CASES = (
    "array_l_s_c_x",
    "horn_xband",
    "waveguide_family",
    "pec_corner_90",
    "wr90_bend_90",
)


def create_bridge(root):
    root = Path(root)
    for case in CASES:
        for path in (
            root / "references" / case / "hfss",
            root / "references" / case / "openems",
            root / "results" / case / "fdtd",
            root / "results" / case / "benchmark",
        ):
            path.mkdir(parents=True, exist_ok=True)
    (root / "runners" / "hfss").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    (root / "README.md").write_text(BRIDGE_README, encoding="utf-8")
    (root / "runners" / "hfss" / "hfss_status.py").write_text(HFSS_STATUS_SCRIPT, encoding="utf-8")
    (root / "runners" / "hfss" / "run_hfss_export_template.py").write_text(HFSS_RUNNER_TEMPLATE, encoding="utf-8")
    (root / "runners" / "hfss" / "run_hfss_waveguide_family.py").write_text(HFSS_WAVEGUIDE_FAMILY_RUNNER, encoding="utf-8")
    (root / "bridge_manifest.json").write_text(
        json.dumps(
            {
                "bridge_root": str(root),
                "cases": list(CASES),
                "hfss_runner_dir": str(root / "runners" / "hfss"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_WINDOWS_DESKTOP / DEFAULT_BRIDGE_NAME,
        help="Bridge root as seen from WSL.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = create_bridge(args.root)
    print(f"Wrote bridge folder: {root}")
    print(f"Windows path: C:\\Users\\whqkr\\Desktop\\{root.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
