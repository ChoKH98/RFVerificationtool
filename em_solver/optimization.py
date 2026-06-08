"""Generate optimization recommendations from benchmark metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RECOMMENDATION_RULES = (
    ("mean_db_error", "Refine local mesh around ports/discontinuities and re-check port normalization."),
    ("mean_phase_error", "Reduce time step margin or extend run time until port transients decay."),
    ("feature_frequency_error", "Calibrate material epsilon/tan-delta and verify geometry dimensions against HFSS."),
    ("high_energy_db_error", "Increase near-field sampling density and refine mesh at dielectric/PEC edges."),
    ("phase_error", "Use consistent field phase reference and verify frequency-domain DFT windowing."),
    ("far_peak_gain_error", "Validate NF2FF integration surface, aperture extent, and accepted input power normalization."),
    ("far_beam_pointing_error", "Check phased-array feed phase convention, element spacing, and theta/phi coordinate system."),
    ("rcs_high_energy_db_error", "Refine corner edge mesh and verify incident/scattered-field separation."),
    ("hfss_reference_missing", "Run the Windows PyAEDT HFSS runner or convert HFSS exports into the case reference directory."),
    ("artifact_missing", "Generate or convert the missing HFSS/openEMS/candidate artifact before optimizing solver settings."),
)


def _recommend_for_metric(metric_name):
    for token, recommendation in RECOMMENDATION_RULES:
        if token in metric_name:
            return recommendation
    return "Inspect this metric manually; no specific optimization rule is registered."


def recommendations_from_metrics(metrics_payload):
    """Return failing metrics with deterministic optimization recommendations."""
    rows = []
    for comparison in metrics_payload.get("comparisons", []):
        for metric in comparison.get("metrics", []):
            if metric.get("passed"):
                continue
            rows.append(
                {
                    "artifact": comparison.get("artifact"),
                    "reference": comparison.get("reference"),
                    "metric": metric.get("name"),
                    "value": metric.get("value"),
                    "tolerance": metric.get("tolerance"),
                    "units": metric.get("units"),
                    "recommendation": _recommend_for_metric(metric.get("name", "")),
                }
            )
    return rows


def write_recommendations_csv(path, recommendations):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact", "reference", "metric", "value", "tolerance", "units", "recommendation"],
        )
        writer.writeheader()
        writer.writerows(recommendations)


def write_recommendations_json(path, recommendations):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"recommendations": recommendations}, indent=2), encoding="utf-8")


def _cmd_recommend(args):
    payload = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    recommendations = recommendations_from_metrics(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_recommendations_json(args.output_dir / "optimization_recommendations.json", recommendations)
    write_recommendations_csv(args.output_dir / "optimization_recommendations.csv", recommendations)
    print(f"Wrote {args.output_dir / 'optimization_recommendations.json'}")
    print(f"Wrote {args.output_dir / 'optimization_recommendations.csv'}")
    return 0 if not recommendations else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    rec_parser = subparsers.add_parser("recommend", help="Recommend optimizations from metrics.json")
    rec_parser.add_argument("--metrics-json", required=True, type=Path)
    rec_parser.add_argument("--output-dir", required=True, type=Path)
    rec_parser.set_defaults(func=_cmd_recommend)
    return parser.parse_args()


def main():
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
