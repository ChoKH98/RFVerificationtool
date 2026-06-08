"""Validation data loaders and comparison metrics for RF solver correlation."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


STRICT_SPARAM_DB_TOL = 0.5
STRICT_PHASE_DEG_TOL = 5.0
STRICT_FREQ_REL_TOL = 0.005
STRICT_FIELD_DB_TOL = 0.5
STRICT_FIELD_PHASE_DEG_TOL = 5.0
STRICT_FAR_GAIN_DB_TOL = 1.0
STRICT_FAR_POINTING_DEG_TOL = 2.0
STRICT_RCS_DB_TOL = 1.0


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    tolerance: float
    units: str

    @property
    def passed(self):
        return self.value <= self.tolerance


@dataclass(frozen=True)
class ArtifactComparison:
    artifact: str
    candidate: str
    reference: str
    metrics: tuple[MetricResult, ...]

    @property
    def passed(self):
        return all(metric.passed for metric in self.metrics)


def _complex_from_real_imag(real, imag):
    return np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)


def _db(values):
    return 20.0 * np.log10(np.maximum(np.abs(values), 1e-30))


def _phase_deg(values):
    return np.rad2deg(np.unwrap(np.angle(values)))


def _mean_abs_phase_error_deg(candidate, reference):
    diff = _phase_deg(candidate) - _phase_deg(reference)
    return float(np.mean(np.abs(diff)))


@dataclass(frozen=True)
class SParameterData:
    frequency_hz: np.ndarray
    port_i: np.ndarray
    port_j: np.ndarray
    values: np.ndarray


def read_sparameters_csv(path):
    """Read standard S-parameter CSV rows."""
    frequency_hz = []
    port_i = []
    port_j = []
    values = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"frequency_hz", "port_i", "port_j"}
        if not required.issubset(fieldnames):
            raise ValueError(f"{path} must include {sorted(required)}")
        has_complex = {"real", "imag"}.issubset(fieldnames)
        has_db_phase = {"db", "phase_deg"}.issubset(fieldnames)
        if not has_complex and not has_db_phase:
            raise ValueError(f"{path} must include real/imag or db/phase_deg")

        for row in reader:
            frequency_hz.append(float(row["frequency_hz"]))
            port_i.append(int(row["port_i"]))
            port_j.append(int(row["port_j"]))
            if has_complex:
                values.append(complex(float(row["real"]), float(row["imag"])))
            else:
                mag = 10.0 ** (float(row["db"]) / 20.0)
                phase = np.deg2rad(float(row["phase_deg"]))
                values.append(mag * np.exp(1j * phase))

    if not frequency_hz:
        raise ValueError(f"{path} contains no S-parameter rows")
    return SParameterData(
        np.array(frequency_hz, dtype=float),
        np.array(port_i, dtype=int),
        np.array(port_j, dtype=int),
        np.array(values, dtype=complex),
    )


def compare_sparameters(candidate, reference):
    """Compare S-parameters by port pair on reference frequencies."""
    metrics = []
    for port_i, port_j in sorted(set(zip(reference.port_i, reference.port_j))):
        ref_mask = (reference.port_i == port_i) & (reference.port_j == port_j)
        cand_mask = (candidate.port_i == port_i) & (candidate.port_j == port_j)
        if not np.any(cand_mask):
            metrics.append(MetricResult(f"S{port_i}{port_j}_missing", float("inf"), 0.0, "bool"))
            continue

        ref_freq = reference.frequency_hz[ref_mask]
        ref_values = reference.values[ref_mask]
        cand_freq = candidate.frequency_hz[cand_mask]
        cand_values = candidate.values[cand_mask]
        cand_real = np.interp(ref_freq, cand_freq, cand_values.real)
        cand_imag = np.interp(ref_freq, cand_freq, cand_values.imag)
        cand_interp = cand_real + 1j * cand_imag

        db_error = float(np.mean(np.abs(_db(cand_interp) - _db(ref_values))))
        phase_error = _mean_abs_phase_error_deg(cand_interp, ref_values)
        metrics.append(MetricResult(f"S{port_i}{port_j}_mean_db_error", db_error, STRICT_SPARAM_DB_TOL, "dB"))
        metrics.append(MetricResult(f"S{port_i}{port_j}_mean_phase_error", phase_error, STRICT_PHASE_DEG_TOL, "deg"))

        cand_min_freq = float(cand_freq[int(np.argmin(_db(cand_values)))])
        ref_min_freq = float(ref_freq[int(np.argmin(_db(ref_values)))])
        rel_error = abs(cand_min_freq - ref_min_freq) / max(ref_min_freq, 1.0)
        metrics.append(MetricResult(f"S{port_i}{port_j}_feature_frequency_error", rel_error, STRICT_FREQ_REL_TOL, "relative"))
    return tuple(metrics)


@dataclass(frozen=True)
class FieldData:
    frequency_hz: np.ndarray
    points_m: np.ndarray
    components: dict[str, np.ndarray]


def read_field_npz(path):
    """Read near/medium field NPZ with points_m and complex E/H components."""
    with np.load(path) as data:
        required = {"frequency_hz", "points_m"}
        if not required.issubset(data.files):
            raise ValueError(f"{path} must include frequency_hz and points_m")
        frequency_hz = np.asarray(data["frequency_hz"], dtype=float)
        points_m = np.asarray(data["points_m"], dtype=float)
        components = {}
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            if name in data.files:
                components[name] = np.asarray(data[name])
    if not components:
        raise ValueError(f"{path} contains no field components")
    return FieldData(frequency_hz, points_m, components)


def compare_fields(candidate, reference, artifact_name):
    """Compare complex near/medium fields on identical point samples."""
    if candidate.points_m.shape != reference.points_m.shape or not np.allclose(candidate.points_m, reference.points_m):
        raise ValueError(f"{artifact_name}: candidate/reference points_m grids differ")
    metrics = []
    for name, ref_values in reference.components.items():
        if name not in candidate.components:
            metrics.append(MetricResult(f"{name}_missing", float("inf"), 0.0, "bool"))
            continue
        cand_values = candidate.components[name]
        ref_mag = np.abs(ref_values)
        threshold = 0.1 * np.max(ref_mag)
        mask = ref_mag >= threshold
        if not np.any(mask):
            mask = np.ones(ref_mag.shape, dtype=bool)
        db_error = float(np.mean(np.abs(_db(cand_values[mask]) - _db(ref_values[mask]))))
        phase_error = _mean_abs_phase_error_deg(cand_values[mask], ref_values[mask])
        metrics.append(MetricResult(f"{name}_high_energy_db_error", db_error, STRICT_FIELD_DB_TOL, "dB"))
        metrics.append(MetricResult(f"{name}_phase_error", phase_error, STRICT_FIELD_PHASE_DEG_TOL, "deg"))
    return tuple(metrics)


@dataclass(frozen=True)
class FarFieldData:
    frequency_hz: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    etheta: np.ndarray
    ephi: np.ndarray
    gain_dbi: np.ndarray | None


def read_far_field_npz(path):
    """Read far-field NPZ with angular grid and complex Etheta/Ephi."""
    with np.load(path) as data:
        required = {"frequency_hz", "theta_deg", "phi_deg", "Etheta", "Ephi"}
        if not required.issubset(data.files):
            raise ValueError(f"{path} must include {sorted(required)}")
        gain = np.asarray(data["gain_dbi"], dtype=float) if "gain_dbi" in data.files else None
        return FarFieldData(
            np.asarray(data["frequency_hz"], dtype=float),
            np.asarray(data["theta_deg"], dtype=float),
            np.asarray(data["phi_deg"], dtype=float),
            np.asarray(data["Etheta"]),
            np.asarray(data["Ephi"]),
            gain,
        )


def _radiation_power_db(data):
    if data.gain_dbi is not None:
        return np.asarray(data.gain_dbi, dtype=float)
    return _db(np.sqrt(np.abs(data.etheta) ** 2 + np.abs(data.ephi) ** 2))


def _far_angle_grids(data):
    power = _radiation_power_db(data)
    theta = np.asarray(data.theta_deg, dtype=float)
    phi = np.asarray(data.phi_deg, dtype=float)
    if theta.shape == power.shape and phi.shape == power.shape:
        return theta, phi
    if theta.ndim == 1 and phi.ndim == 1 and power.shape[-2:] == (theta.size, phi.size):
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
        if power.ndim > 2:
            theta_grid = np.broadcast_to(theta_grid, power.shape)
            phi_grid = np.broadcast_to(phi_grid, power.shape)
        return theta_grid, phi_grid
    raise ValueError("far-field theta/phi grid is incompatible with field shape")


def _peak_direction(data):
    power = _radiation_power_db(data)
    theta, phi = _far_angle_grids(data)
    idx = int(np.argmax(power))
    return float(np.ravel(theta)[idx]), float(np.ravel(phi)[idx]), float(np.ravel(power)[idx])


def compare_far_fields(candidate, reference):
    """Compare far-field peak gain and beam pointing on matching angular grids."""
    if candidate.theta_deg.shape != reference.theta_deg.shape or not np.allclose(candidate.theta_deg, reference.theta_deg):
        raise ValueError("candidate/reference theta grids differ")
    if candidate.phi_deg.shape != reference.phi_deg.shape or not np.allclose(candidate.phi_deg, reference.phi_deg):
        raise ValueError("candidate/reference phi grids differ")
    cand_theta, cand_phi, cand_peak = _peak_direction(candidate)
    ref_theta, ref_phi, ref_peak = _peak_direction(reference)
    pointing = float(np.hypot(cand_theta - ref_theta, cand_phi - ref_phi))
    gain_error = abs(cand_peak - ref_peak)
    return (
        MetricResult("far_peak_gain_error", gain_error, STRICT_FAR_GAIN_DB_TOL, "dB"),
        MetricResult("far_beam_pointing_error", pointing, STRICT_FAR_POINTING_DEG_TOL, "deg"),
    )


@dataclass(frozen=True)
class RcsData:
    frequency_hz: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    rcs_dbsm: np.ndarray


def read_rcs_csv(path):
    rows = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frequency_hz", "theta_deg", "phi_deg", "rcs_dbsm"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must include {sorted(required)}")
        for row in reader:
            rows.append((float(row["frequency_hz"]), float(row["theta_deg"]), float(row["phi_deg"]), float(row["rcs_dbsm"])))
    if not rows:
        raise ValueError(f"{path} contains no RCS rows")
    arr = np.array(rows, dtype=float)
    return RcsData(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])


def compare_rcs(candidate, reference):
    ref_key = np.column_stack((reference.frequency_hz, reference.theta_deg, reference.phi_deg))
    cand_key = np.column_stack((candidate.frequency_hz, candidate.theta_deg, candidate.phi_deg))
    if ref_key.shape != cand_key.shape or not np.allclose(ref_key, cand_key):
        raise ValueError("candidate/reference RCS sample grids differ")
    threshold = np.max(reference.rcs_dbsm) - 10.0
    mask = reference.rcs_dbsm >= threshold
    if not np.any(mask):
        mask = np.ones(reference.rcs_dbsm.shape, dtype=bool)
    mean_error = float(np.mean(np.abs(candidate.rcs_dbsm[mask] - reference.rcs_dbsm[mask])))
    return (MetricResult("rcs_high_energy_db_error", mean_error, STRICT_RCS_DB_TOL, "dB"),)


def comparison_to_dict(comparison):
    return {
        "artifact": comparison.artifact,
        "candidate": comparison.candidate,
        "reference": comparison.reference,
        "passed": comparison.passed,
        "metrics": [{**asdict(metric), "passed": metric.passed} for metric in comparison.metrics],
    }


def write_metrics_json(path, case_id, comparisons):
    payload = {
        "case_id": case_id,
        "passed": all(comparison.passed for comparison in comparisons),
        "comparisons": [comparison_to_dict(comparison) for comparison in comparisons],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_csv(path, comparisons):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "candidate", "reference", "metric", "value", "tolerance", "units", "passed"])
        for comparison in comparisons:
            for metric in comparison.metrics:
                writer.writerow([
                    comparison.artifact,
                    comparison.candidate,
                    comparison.reference,
                    metric.name,
                    metric.value,
                    metric.tolerance,
                    metric.units,
                    metric.passed,
                ])
