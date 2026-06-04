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

`GaussianPort` injects a soft Ey source along a feed column at the patch edge.
It records an analytic incident voltage and the total sampled voltage, then
computes S11 with a direct DFT:

```text
S11 = (V_total - V_incident) / V_incident
```

This is useful for fast design iteration, but final RF sign-off still needs
OpenEMS or measurement correlation.

## CPML Status

The CPML class currently applies correction terms on y-normal faces and prepares
the profile and memory arrays for x/z face extensions. Treat it as an absorber
prototype until all six face corrections are implemented and validated.
