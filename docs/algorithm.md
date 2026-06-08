# Algorithm Notes

## FDTD Core

The solver uses a 3D Yee-style time-domain update with collocated NumPy arrays
for the field components. The time step is derived from the Courant limit:

```text
dt <= 1 / (c0 * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
```

Material regions are represented by per-cell `eps`, `mu`, and `sigma` arrays.
Calling `set_material()` updates the region values and recomputes the electric
and magnetic update coefficients.

## Patch Geometry

`patch_dimensions()` uses the standard transmission-line approximation for a
rectangular microstrip patch. `create_patch_geometry()` places:

- copper-like ground plane
- Rogers 4003C substrate
- copper-like rectangular patch

The grid is centered in x/y and reserves cells near the z-boundary for the PML
and air region.

## Port And S11

`GaussianPort` injects a soft Ez gap-voltage source along a feed column between
the ground and patch metallization.
It records an analytic incident voltage and the total sampled voltage, then
computes S11 with a direct DFT:

```text
S11 = (V_total - V_incident) / V_incident
```

This is useful for fast design iteration, but final RF sign-off still needs
OpenEMS or measurement correlation.

## CPML Status

The CPML class applies correction terms on x-, y-, and z-normal faces. Treat it
as an absorber prototype until it is validated against a plane-wave reflection
test and correlated against openEMS/HFSS patch results.

## Benchmark Correlation

`em_solver.benchmark` compares normalized candidate solver outputs against HFSS
and optionally openEMS. It supports:

- layered phased-array antenna cases
- X-band horn antenna cases
- rectangular waveguide cutoff/propagation cases
- PEC corner RCS scattering cases
- WR-90 bend scattering cases

`validation.py` contains the artifact loaders and strict metrics for
S-parameters, near/medium complex fields, far-field beam properties, and RCS.

`external_solvers.py` detects openEMS/AppCSXCAD/Octave/MATLAB/HFSS availability
and can generate openEMS MATLAB/Octave runner scripts from installed tutorial
templates. `optimization.py` converts failing benchmark metrics into concrete
solver tuning recommendations such as mesh refinement, port normalization,
NF2FF surface checks, and corner-edge refinement.

`workflow.py` ties the bridge together: it creates/updates the Windows bridge,
inspects artifact coverage per benchmark case, runs comparisons, and emits
optimization recommendation files into each case's benchmark output directory.
