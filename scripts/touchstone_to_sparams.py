"""Convert HFSS Touchstone .s2p to RFVerificationtool sparams.csv format.
Usage: python touchstone_to_sparams.py input.s2p output.csv
"""
import csv
import math
import sys
from pathlib import Path


def convert(ts_path, csv_path):
    unit_scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
    data_format = "ma"
    scale = 1.0
    rows = []
    for line in Path(ts_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            parts = line[1:].lower().split()
            scale = unit_scale.get(parts[0] if parts else "ghz", 1e9)
            if "ri" in parts:
                data_format = "ri"
            elif "db" in parts:
                data_format = "db"
            else:
                data_format = "ma"
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) >= 9:
            rows.append(vals)

    def to_complex(a, b):
        if data_format == "ri":
            return complex(a, b)
        if data_format == "db":
            mag = 10 ** (a / 20)
            return mag * complex(math.cos(math.radians(b)), math.sin(math.radians(b)))
        return a * complex(math.cos(math.radians(b)), math.sin(math.radians(b)))

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frequency_hz", "port_i", "port_j", "real", "imag", "db", "phase_deg"])
        for row in rows:
            freq = row[0] * scale
            port_pairs = [(1, 1, row[1], row[2]), (2, 1, row[3], row[4]),
                          (1, 2, row[5], row[6]), (2, 2, row[7], row[8])]
            for pi, pj, a, b in port_pairs:
                s = to_complex(a, b)
                db = 20 * math.log10(max(abs(s), 1e-30))
                ph = math.degrees(math.atan2(s.imag, s.real))
                writer.writerow([freq, pi, pj, s.real, s.imag, db, ph])
    print(f"Converted {ts_path} -> {csv_path}: {len(rows)} frequency points")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python touchstone_to_sparams.py input.s2p output.csv")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
