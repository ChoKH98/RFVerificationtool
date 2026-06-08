"""Compare S11 CSV results from this FDTD prototype and reference solvers."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class S11Trace:
    """Frequency-domain S11 trace."""

    frequency_hz: np.ndarray
    s11: np.ndarray

    @property
    def s11_db(self):
        return 20.0 * np.log10(np.maximum(np.abs(self.s11), 1e-30))

    @property
    def resonance_hz(self):
        return float(self.frequency_hz[int(np.argmin(self.s11_db))])

    @property
    def min_s11_db(self):
        return float(np.min(self.s11_db))


def read_s11_csv(path):
    """Read frequency_hz plus either s11_real/s11_imag or s11_db columns."""
    freqs = []
    values = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "frequency_hz" not in fieldnames:
            raise ValueError(f"{path} is missing frequency_hz")

        has_complex = {"s11_real", "s11_imag"}.issubset(fieldnames)
        has_db = "s11_db" in fieldnames
        if not has_complex and not has_db:
            raise ValueError(f"{path} must contain s11_real/s11_imag or s11_db")

        for row in reader:
            freqs.append(float(row["frequency_hz"]))
            if has_complex:
                values.append(complex(float(row["s11_real"]), float(row["s11_imag"])))
            else:
                values.append(10.0 ** (float(row["s11_db"]) / 20.0) + 0j)

    if not freqs:
        raise ValueError(f"{path} contains no S11 rows")
    return S11Trace(np.array(freqs, dtype=float), np.array(values, dtype=complex))


@dataclass(frozen=True)
class S11Comparison:
    candidate_resonance_hz: float
    reference_resonance_hz: float
    resonance_error_hz: float
    candidate_min_s11_db: float
    reference_min_s11_db: float
    mean_abs_db_error: float
    max_abs_db_error: float

    def passes(self, resonance_tolerance_hz=50e6, mean_db_tolerance=1.0):
        return (
            abs(self.resonance_error_hz) <= resonance_tolerance_hz
            and self.mean_abs_db_error <= mean_db_tolerance
        )


def compare_s11(candidate, reference):
    """Compare a candidate trace against a reference trace on reference frequencies."""
    cand_db = np.interp(reference.frequency_hz, candidate.frequency_hz, candidate.s11_db)
    diff_db = cand_db - reference.s11_db
    return S11Comparison(
        candidate_resonance_hz=candidate.resonance_hz,
        reference_resonance_hz=reference.resonance_hz,
        resonance_error_hz=candidate.resonance_hz - reference.resonance_hz,
        candidate_min_s11_db=candidate.min_s11_db,
        reference_min_s11_db=reference.min_s11_db,
        mean_abs_db_error=float(np.mean(np.abs(diff_db))),
        max_abs_db_error=float(np.max(np.abs(diff_db))),
    )


def format_comparison(label, comparison):
    status = "PASS" if comparison.passes() else "FAIL"
    return "\n".join(
        [
            f"{label}: {status}",
            f"  resonance: candidate={comparison.candidate_resonance_hz * 1e-9:.6f} GHz, "
            f"reference={comparison.reference_resonance_hz * 1e-9:.6f} GHz, "
            f"error={comparison.resonance_error_hz * 1e-6:.2f} MHz",
            f"  min S11: candidate={comparison.candidate_min_s11_db:.2f} dB, "
            f"reference={comparison.reference_min_s11_db:.2f} dB",
            f"  S11 dB error: mean={comparison.mean_abs_db_error:.2f} dB, "
            f"max={comparison.max_abs_db_error:.2f} dB",
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="Candidate S11 CSV")
    parser.add_argument("reference", type=Path, help="HFSS/openEMS reference S11 CSV")
    parser.add_argument("--label", default="S11 comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    candidate = read_s11_csv(args.candidate)
    reference = read_s11_csv(args.reference)
    comparison = compare_s11(candidate, reference)
    print(format_comparison(args.label, comparison))
    return 0 if comparison.passes() else 1


if __name__ == "__main__":
    raise SystemExit(main())
