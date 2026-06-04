"""PCB geometry builder — Rogers 4003C patch antenna."""
import numpy as np


# Rogers 4003C constants
EPS_R_ROGERS = 3.55
TAN_DELTA = 0.0027
SUBSTRATE_THICKNESS = 0.8e-3   # 0.8 mm
SIGMA_COPPER = 5.8e7           # S/m

EPS0 = 8.854e-12
C0   = 3e8


def patch_dimensions(f0=2.4e9, eps_r=EPS_R_ROGERS, h=SUBSTRATE_THICKNESS):
    """Calculate patch W, L using transmission-line model (Kai Chang Ch.3)."""
    # Width W
    W = C0 / (2 * f0) * np.sqrt(2 / (eps_r + 1))

    # Effective dielectric constant
    eps_eff = (eps_r + 1) / 2 + (eps_r - 1) / 2 * (1 + 12 * h / W) ** (-0.5)

    # Fringing extension ΔL
    dL = 0.412 * h * (eps_eff + 0.3) * (W/h + 0.264) / ((eps_eff - 0.258) * (W/h + 0.8))

    # Physical length L
    L = C0 / (2 * f0 * np.sqrt(eps_eff)) - 2 * dL

    return W, L, eps_eff, dL


def create_patch_geometry(sim, f0=2.4e9):
    """Place substrate + patch + ground plane into FDTD grid."""
    dx, dy, dz = sim.dx, sim.dy, sim.dz
    Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz

    W, L, eps_eff, dL = patch_dimensions(f0)

    # Determine layer positions in z-cells
    pml_d = 10
    z_gnd   = pml_d + 2                       # ground plane z-index
    z_sub_top = z_gnd + int(round(SUBSTRATE_THICKNESS / dz))
    z_patch = z_sub_top                        # patch on top of substrate

    # Center patch in xy
    cx = Nx // 2
    cy = Ny // 2

    W_cells = max(1, int(round(W / dx)))
    L_cells = max(1, int(round(L / dy)))

    x0 = cx - W_cells // 2;  x1 = x0 + W_cells
    y0 = cy - L_cells // 2;  y1 = y0 + L_cells

    # Ground plane (PEC)
    sim.set_material(0, Nx, 0, Ny, z_gnd, z_gnd+1, eps_r=1.0, sigma=SIGMA_COPPER)

    # Dielectric substrate
    sigma_diel = 2 * np.pi * f0 * EPS_R_ROGERS * EPS0 * TAN_DELTA
    sim.set_material(0, Nx, 0, Ny, z_gnd+1, z_sub_top+1,
                     eps_r=EPS_R_ROGERS, sigma=sigma_diel)

    # Patch (PEC)
    sim.set_material(x0, x1, y0, y1, z_patch, z_patch+1,
                     eps_r=1.0, sigma=SIGMA_COPPER)

    info = dict(W=W, L=L, eps_eff=eps_eff,
                z_feed=z_gnd+1, x0=x0, x1=x1, y0=y0, y1=y1,
                z_gnd=z_gnd, z_sub_top=z_sub_top, z_patch=z_patch,
                cx=cx, cy=cy)
    print(f"Patch: W={W*1e3:.2f}mm  L={L*1e3:.2f}mm  εeff={eps_eff:.3f}")
    print(f"  grid: x[{x0}:{x1}]  y[{y0}:{y1}]  z_patch={z_patch}")
    return info
