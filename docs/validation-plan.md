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

The current repository does not include the OpenEMS reference projects needed
to prove those criteria. Add those fixtures before treating the solver as a
calibrated verification tool.
