"""Run the HFSS/openEMS/FDTD comparison workflow across the Windows bridge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .benchmark import compare_case
    from .benchmark_cases import BENCHMARK_CASES, get_case
    from .bridge import DEFAULT_WINDOWS_DESKTOP, DEFAULT_BRIDGE_NAME, create_bridge
    from .candidate_exports import write_waveguide_family_candidate
    from .external_solvers import detect_solvers, write_openems_runner
    from .optimization import recommendations_from_metrics, write_recommendations_csv, write_recommendations_json
    from .validation import write_metrics_json, write_summary_csv
except ImportError:  # Supports: python em_solver/workflow.py ...
    from benchmark import compare_case
    from benchmark_cases import BENCHMARK_CASES, get_case
    from bridge import DEFAULT_WINDOWS_DESKTOP, DEFAULT_BRIDGE_NAME, create_bridge
    from candidate_exports import write_waveguide_family_candidate
    from external_solvers import detect_solvers, write_openems_runner
    from optimization import recommendations_from_metrics, write_recommendations_csv, write_recommendations_json
    from validation import write_metrics_json, write_summary_csv


DEFAULT_BRIDGE_ROOT = DEFAULT_WINDOWS_DESKTOP / DEFAULT_BRIDGE_NAME


def case_paths(bridge_root, case_id):
    bridge_root = Path(bridge_root)
    return {
        "candidate_dir": bridge_root / "results" / case_id / "fdtd",
        "benchmark_dir": bridge_root / "results" / case_id / "benchmark",
        "hfss_dir": bridge_root / "references" / case_id / "hfss",
        "openems_dir": bridge_root / "references" / case_id / "openems",
        "openems_runner_dir": bridge_root / "runners" / "openems" / case_id,
    }


def inspect_case(bridge_root, case_id):
    case = get_case(case_id)
    paths = case_paths(bridge_root, case_id)
    rows = []
    for artifact in case.required_artifacts:
        rows.append(
            {
                "artifact": artifact,
                "candidate_exists": (paths["candidate_dir"] / artifact).exists(),
                "hfss_exists": (paths["hfss_dir"] / artifact).exists(),
                "openems_exists": (paths["openems_dir"] / artifact).exists(),
            }
        )
    return {"case_id": case_id, "paths": {key: str(value) for key, value in paths.items()}, "artifacts": rows}


def _comparisons_payload(case_id, comparisons):
    return {
        "case_id": case_id,
        "comparisons": [
            {
                "artifact": comparison.artifact,
                "reference": comparison.reference,
                "metrics": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "tolerance": metric.tolerance,
                        "units": metric.units,
                        "passed": metric.passed,
                    }
                    for metric in comparison.metrics
                ],
            }
            for comparison in comparisons
        ],
    }


def run_case(bridge_root, case_id):
    paths = case_paths(bridge_root, case_id)
    comparisons = compare_case(case_id, paths["candidate_dir"], paths["hfss_dir"], paths["openems_dir"])
    paths["benchmark_dir"].mkdir(parents=True, exist_ok=True)
    write_metrics_json(paths["benchmark_dir"] / "metrics.json", case_id, comparisons)
    write_summary_csv(paths["benchmark_dir"] / "summary.csv", comparisons)
    payload = _comparisons_payload(case_id, comparisons)
    recommendations = recommendations_from_metrics(payload)
    write_recommendations_json(paths["benchmark_dir"] / "optimization_recommendations.json", recommendations)
    write_recommendations_csv(paths["benchmark_dir"] / "optimization_recommendations.csv", recommendations)
    return comparisons, recommendations


def _cmd_setup(args):
    root = create_bridge(args.bridge_root)
    for case_id in BENCHMARK_CASES:
        runner_dir = root / "runners" / "openems" / case_id
        try:
            write_openems_runner(case_id, runner_dir)
        except ValueError:
            pass
    print(f"Bridge ready: {root}")
    return 0


def _cmd_status(args):
    status = detect_solvers()
    payload = {
        "bridge_root": str(args.bridge_root),
        "solver_status": {
            "openems": status.openems,
            "appcsxcad": status.appcsxcad,
            "octave": status.octave,
            "matlab": status.matlab,
            "hfss_ansysedt": status.hfss_ansysedt,
            "can_run_openems_scripts": status.can_run_openems_scripts,
            "can_run_hfss": status.can_run_hfss,
        },
        "cases": [inspect_case(args.bridge_root, case_id) for case_id in sorted(BENCHMARK_CASES)],
    }
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_run(args):
    case_ids = sorted(BENCHMARK_CASES) if args.case == "all" else [args.case]
    all_passed = True
    for case_id in case_ids:
        comparisons, recommendations = run_case(args.bridge_root, case_id)
        passed = all(comparison.passed for comparison in comparisons)
        all_passed = all_passed and passed
        output_dir = case_paths(args.bridge_root, case_id)["benchmark_dir"]
        print(f"{case_id}: {'PASS' if passed else 'FAIL'}; recommendations={len(recommendations)}; output={output_dir}")
    return 0 if all_passed else 1


def _cmd_generate_candidate(args):
    if args.case != "waveguide_family":
        raise SystemExit(f"Candidate generation is currently implemented for waveguide_family, got {args.case}")
    output_dir = case_paths(args.bridge_root, args.case)["candidate_dir"]
    write_waveguide_family_candidate(output_dir)
    print(f"Wrote {args.case} candidate artifacts: {output_dir}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-root", type=Path, default=DEFAULT_BRIDGE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="Create bridge folders and solver runner templates")
    setup_parser.set_defaults(func=_cmd_setup)

    status_parser = subparsers.add_parser("status", help="Inspect solver availability and artifact coverage")
    status_parser.set_defaults(func=_cmd_status)

    run_parser = subparsers.add_parser("run", help="Run benchmark comparison for one case or all cases")
    run_parser.add_argument("--case", required=True, choices=["all", *sorted(BENCHMARK_CASES)])
    run_parser.set_defaults(func=_cmd_run)

    cand_parser = subparsers.add_parser("generate-candidate", help="Generate RFVerificationtool candidate artifacts")
    cand_parser.add_argument("--case", required=True, choices=sorted(BENCHMARK_CASES))
    cand_parser.set_defaults(func=_cmd_generate_candidate)
    return parser.parse_args()


def main():
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
