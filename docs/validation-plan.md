# Validation Plan

## Fast Regression

Run:

```bash
python -m unittest discover -s tests
```

This checks deterministic geometry formulas, FDTD coefficient updates, S11 CSV
serialization, and a small two-step patch simulation.

## Numerical Smoke

Run:

```bash
python em_solver/run_patch_sim.py --steps 20 --quiet --points 21 --output-dir results/_tmp_smoke
```

Expected result:

- the command exits successfully
- `patch_s11.csv` is written
- `patch_s11.png` is written when matplotlib is installed

## Accuracy Validation

The target acceptance criteria are:

- single-layer patch S11 resonance within +/- 50 MHz of a trusted reference
- multilayer PCB coupling within +/- 1 dB
- via impedance within +/- 5%

## Multi-Structure HFSS/openEMS Correlation

List supported benchmark cases:

```bash
python -m em_solver.external_solvers status
python -m em_solver.benchmark list-cases
python -m em_solver.bridge --root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge
```

The Windows bridge root is:

```text
C:\Users\whqkr\Desktop\RFVerificationBridge
```

Use Windows PowerShell for PyAEDT/HFSS checks:

```powershell
cd "$env:USERPROFILE\Desktop\RFVerificationBridge\runners\hfss"
py -m pip install ansys-aedt-core
py .\hfss_status.py
py .\run_hfss_waveguide_family.py --non-graphical
```

Use the WSL workflow command to inspect and run the full comparison loop:

```bash
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge setup
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge status
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge generate-candidate --case waveguide_family
python -m em_solver.workflow --bridge-root /mnt/c/Users/whqkr/Desktop/RFVerificationBridge run --case waveguide_family
```

Generate an openEMS MATLAB/Octave runner for a case:

```bash
python -m em_solver.external_solvers generate-openems \
  --case horn_xband \
  --output-dir references/horn_xband/openems_runner
```

Run a benchmark:

```bash
python -m em_solver.benchmark compare \
  --case horn_xband \
  --candidate-dir results/horn_xband/fdtd \
  --hfss-dir references/horn_xband/hfss \
  --openems-dir references/horn_xband/openems \
  --output-dir results/horn_xband/benchmark
```

The CLI writes:

- `metrics.json`
- `summary.csv`
- `optimization_recommendations.json`
- `optimization_recommendations.csv`

HFSS is treated as the golden reference. openEMS is compared against HFSS as an
independent cross-check when `--openems-dir` is supplied.

## Standard Artifact Formats

Reference traces from HFSS or openEMS should be normalized into this S-parameter
CSV format:

```text
frequency_hz,port_i,port_j,real,imag,db,phase_deg
```

Near/medium fields use NPZ:

```text
frequency_hz
points_m
Ex,Ey,Ez,Hx,Hy,Hz
```

Far fields use NPZ:

```text
frequency_hz
theta_deg,phi_deg
Etheta,Ephi
gain_dbi optional
```

Corner scattering RCS uses CSV:

```text
frequency_hz,theta_deg,phi_deg,rcs_dbsm
```

Legacy single-port S11 comparison is still available:

```bash
python -m em_solver.compare_s11 results/patch_s11.csv references/openems_patch_s11.csv --label openEMS
python -m em_solver.compare_s11 results/patch_s11.csv references/hfss_patch_s11.csv --label HFSS
```

The current repository does not include the HFSS/openEMS reference projects or
exports needed to prove those criteria. Add those fixtures before treating the
solver as a calibrated verification tool.
