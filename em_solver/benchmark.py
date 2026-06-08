"""Benchmark comparison CLI for HFSS/openEMS correlation artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .benchmark_cases import BENCHMARK_CASES, get_case
    from .validation import (
        ArtifactComparison,
        MetricResult,
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
    from .optimization import write_recommendations_csv, write_recommendations_json, recommendations_from_metrics
except ImportError:  # Supports: python em_solver/benchmark.py ...
    from benchmark_cases import BENCHMARK_CASES, get_case
    from validation import (
        ArtifactComparison,
        MetricResult,
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
    from optimization import write_recommendations_csv, write_recommendations_json, recommendations_from_metrics


def _compare_artifact(artifact, candidate_path, reference_path, reference_name):
    if artifact == "sparams.csv":
        metrics = compare_sparameters(read_sparameters_csv(candidate_path), read_sparameters_csv(reference_path))
    elif artifact in ("field_near.npz", "field_medium.npz"):
        metrics = compare_fields(read_field_npz(candidate_path), read_field_npz(reference_path), artifact)
    elif artifact == "field_far.npz":
        metrics = compare_far_fields(read_far_field_npz(candidate_path), read_far_field_npz(reference_path))
    elif artifact == "rcs.csv":
        metrics = compare_rcs(read_rcs_csv(candidate_path), read_rcs_csv(reference_path))
    else:
        raise ValueError(f"Unsupported artifact {artifact}")
    return ArtifactComparison(artifact, str(candidate_path), reference_name, metrics)


def compare_case(case_id, candidate_dir, hfss_dir, openems_dir=None):
    case = get_case(case_id)
    comparisons = []
    for artifact in case.required_artifacts:
        candidate_path = Path(candidate_dir) / artifact
        hfss_path = Path(hfss_dir) / artifact
        if not candidate_path.exists():
            comparisons.append(ArtifactComparison(
                artifact,
                str(candidate_path),
                "HFSS",
                (MetricResult("candidate_artifact_missing", float("inf"), 0.0, "bool"),),
            ))
            continue
        if not hfss_path.exists():
            comparisons.append(ArtifactComparison(
                artifact,
                str(candidate_path),
                "HFSS",
                (MetricResult("hfss_reference_missing", float("inf"), 0.0, "bool"),),
            ))
            continue
        if candidate_path.exists() and hfss_path.exists():
            comparisons.append(_compare_artifact(artifact, candidate_path, hfss_path, "HFSS"))
        if openems_dir is not None:
            openems_path = Path(openems_dir) / artifact
            if openems_path.exists() and hfss_path.exists():
                comparisons.append(_compare_artifact(artifact, openems_path, hfss_path, "HFSS(openEMS check)"))
    return comparisons


def _cmd_list_cases(_args):
    for case_id, case in sorted(BENCHMARK_CASES.items()):
        freqs = ", ".join(f"{freq * 1e-9:g}GHz" for freq in case.frequency_hz)
        artifacts = ", ".join(case.required_artifacts)
        print(f"{case_id}: {case.structure_type}; freqs=[{freqs}]; artifacts=[{artifacts}]")
    return 0


def _cmd_compare(args):
    comparisons = compare_case(args.case, args.candidate_dir, args.hfss_dir, args.openems_dir)
    if not comparisons:
        raise SystemExit("No comparable artifacts found. Check candidate/HFSS/openEMS directories.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(args.output_dir / "metrics.json", args.case, comparisons)
    write_summary_csv(args.output_dir / "summary.csv", comparisons)
    metrics_payload = {
        "case_id": args.case,
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
    recommendations = recommendations_from_metrics(metrics_payload)
    write_recommendations_json(args.output_dir / "optimization_recommendations.json", recommendations)
    write_recommendations_csv(args.output_dir / "optimization_recommendations.csv", recommendations)
    passed = all(comparison.passed for comparison in comparisons)
    print(f"{args.case}: {'PASS' if passed else 'FAIL'}")
    print(f"Wrote {args.output_dir / 'metrics.json'}")
    print(f"Wrote {args.output_dir / 'summary.csv'}")
    print(f"Wrote {args.output_dir / 'optimization_recommendations.json'}")
    print(f"Wrote {args.output_dir / 'optimization_recommendations.csv'}")
    return 0 if passed else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-cases", help="List benchmark cases")
    list_parser.set_defaults(func=_cmd_list_cases)

    compare_parser = subparsers.add_parser("compare", help="Compare case artifacts")
    compare_parser.add_argument("--case", required=True, choices=sorted(BENCHMARK_CASES))
    compare_parser.add_argument("--candidate-dir", required=True, type=Path)
    compare_parser.add_argument("--hfss-dir", required=True, type=Path)
    compare_parser.add_argument("--openems-dir", type=Path)
    compare_parser.add_argument("--output-dir", required=True, type=Path)
    compare_parser.set_defaults(func=_cmd_compare)
    return parser.parse_args()


def main():
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
