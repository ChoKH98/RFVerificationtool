"""Convert HFSS Touchstone .s2p to sparams.csv (bridge format).

Reads !Data-is-not-renormalized Touchstone (MA format) and writes
references/waveguide_family/hfss/sparams.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

_BRIDGE = Path("/mnt/c/Users/whqkr/Desktop/RFVerificationBridge")
_DEFAULT_S2P = (_BRIDGE / "runners" / "hfss"
                / "waveguide_family_modified2.s2p")
_DEFAULT_OUT = (_BRIDGE / "references" / "waveguide_family" / "hfss"
                / "sparams.csv")


def _to_complex(fmt: str, a: float, b: float) -> complex:
    if fmt == "ri":
        return complex(a, b)
    if fmt == "db":
        mag = 10 ** (a / 20)
        return complex(mag * math.cos(math.radians(b)),
                       mag * math.sin(math.radians(b)))
    # "ma" (magnitude-angle)
    return complex(a * math.cos(math.radians(b)),
                   a * math.sin(math.radians(b)))


def convert(s2p_path: Path, out_path: Path) -> int:
    s2p_path = Path(s2p_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = "ma"
    freq_scale = 1e9
    rows: list[tuple[float, complex, complex]] = []

    for raw in s2p_path.open(encoding="ascii", errors="ignore"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            low = line.lower()
            if "ghz" in low:
                freq_scale = 1e9
            elif "mhz" in low:
                freq_scale = 1e6
            elif "khz" in low:
                freq_scale = 1e3
            if " ri" in low:
                fmt = "ri"
            elif " db" in low:
                fmt = "db"
            continue
        if line.startswith("!"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        freq_hz = float(parts[0]) * freq_scale
        s11 = _to_complex(fmt, float(parts[1]), float(parts[2]))
        s21 = _to_complex(fmt, float(parts[3]), float(parts[4]))
        rows.append((freq_hz, s11, s21))

    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "port_i", "port_j",
                    "real", "imag", "db", "phase_deg"])
        for freq_hz, s11, s21 in rows:
            for pi, pj, val in [(1, 1, s11), (2, 1, s21)]:
                db = 20 * math.log10(max(abs(val), 1e-30))
                ph = math.degrees(math.atan2(val.imag, val.real))
                w.writerow([freq_hz, pi, pj,
                            val.real, val.imag, db, ph])

    print(f"Converted {len(rows)} freq points  →  {out_path}")
    return len(rows)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--s2p", type=Path, default=_DEFAULT_S2P)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUT)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert(args.s2p, args.output)
