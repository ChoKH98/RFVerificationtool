"""CPML (Convolutional PML) absorbing boundary.

The current correction terms are implemented for y-normal faces. The profile
and memory arrays for x/z faces are prepared so the remaining corrections can
be added without changing the public constructor.
"""
import numpy as np


class CPML:
    """CPML profile and y-face correction terms for the attached FDTD simulation."""

    def __init__(self, sim, thickness=10, m=3, sigma_factor=0.8, kappa_max=1.0, alpha_max=0.05):
        self.sim = sim
        self.d = thickness
        sim._cpml = self

        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dx, dy, dz = sim.dx, sim.dy, sim.dz
        dt = sim.dt
        eta0 = 377.0

        def _sigma_max(delta):
            return sigma_factor * (m + 1) / (2 * eta0 * delta)

        def _profile(n_cells, delta):
            """Return κ, σ, α arrays for one PML face (length = n_cells)."""
            i = np.arange(n_cells, dtype=float)
            rho = (i / (n_cells - 1)) ** m  # 0..1 polynomial grading
            sigma = _sigma_max(delta) * rho
            kappa = 1.0 + (kappa_max - 1.0) * rho
            alpha = alpha_max * (1.0 - rho)
            return kappa, sigma, alpha

        def _b_c(kappa, sigma, alpha):
            b = np.exp(-(sigma / kappa + alpha) * dt / 8.854e-12)
            denom = sigma + kappa * alpha
            c = np.where(np.abs(denom) < 1e-30, 0.0,
                         sigma * (b - 1.0) / (kappa * denom))
            return b, c

        d = thickness

        # Precompute b, c for x, y, z faces
        kx, sx, ax = _profile(d, dx)
        ky, sy, ay = _profile(d, dy)
        kz, sz, az = _profile(d, dz)

        self.bx, self.cx = _b_c(kx, sx, ax)
        self.by, self.cy = _b_c(ky, sy, ay)
        self.bz, self.cz = _b_c(kz, sz, az)

        # κ stretch factors for curl updates (applied to 1/Δ)
        self.kappa_x = np.ones(Nx)
        self.kappa_y = np.ones(Ny)
        self.kappa_z = np.ones(Nz)
        self.kappa_x[:d]  = kx[::-1]
        self.kappa_x[-d:] = kx
        self.kappa_y[:d]  = ky[::-1]
        self.kappa_y[-d:] = ky
        self.kappa_z[:d]  = kz[::-1]
        self.kappa_z[-d:] = kz

        # Convolution memory variables ψ  (6 × E,H × x,y,z faces)
        shape = (Nx, Ny, Nz)
        self.psi_Ex_y = np.zeros(shape); self.psi_Ex_z = np.zeros(shape)
        self.psi_Ey_x = np.zeros(shape); self.psi_Ey_z = np.zeros(shape)
        self.psi_Ez_x = np.zeros(shape); self.psi_Ez_y = np.zeros(shape)
        self.psi_Hx_y = np.zeros(shape); self.psi_Hx_z = np.zeros(shape)
        self.psi_Hy_x = np.zeros(shape); self.psi_Hy_z = np.zeros(shape)
        self.psi_Hz_x = np.zeros(shape); self.psi_Hz_y = np.zeros(shape)

    def _face_mask_y(self):
        d = self.d
        Ny = self.sim.Ny
        mask = np.zeros(Ny, dtype=bool)
        mask[:d] = True; mask[-d:] = True
        return mask

    def _face_mask_x(self):
        d = self.d
        Nx = self.sim.Nx
        mask = np.zeros(Nx, dtype=bool)
        mask[:d] = True; mask[-d:] = True
        return mask

    def _face_mask_z(self):
        d = self.d
        Nz = self.sim.Nz
        mask = np.zeros(Nz, dtype=bool)
        mask[:d] = True; mask[-d:] = True
        return mask

    def update_H_psi(self):
        sim = self.sim
        d = self.d
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dt = sim.dt

        # --- ψ_Hx_y (y-face PML, Hx update correction) ---
        for j in list(range(d)) + list(range(Ny-d, Ny)):
            pj = j if j < d else Ny - 1 - j
            b, c = self.by[pj], self.cy[pj]
            dEz = (sim.Ez[:Nx-1, j+1 if j+1 < Ny else j, :Nz-1] - sim.Ez[:Nx-1, j, :Nz-1]) / sim.dy
            self.psi_Hx_y[:Nx-1, j, :Nz-1] = b * self.psi_Hx_y[:Nx-1, j, :Nz-1] + c * dEz
            sim.Hx[:Nx-1, j, :Nz-1] -= dt / (sim.mu[:Nx-1, j, :Nz-1]) * self.psi_Hx_y[:Nx-1, j, :Nz-1]

    def update_E_psi(self):
        sim = self.sim
        d = self.d
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dt = sim.dt

        # --- ψ_Ex_y (y-face PML, Ex update correction) ---
        for j in list(range(d)) + list(range(Ny-d, Ny)):
            pj = j if j < d else Ny - 1 - j
            b, c = self.by[pj], self.cy[pj]
            dHz = (sim.Hz[1:Nx, j, 1:Nz-1] - sim.Hz[1:Nx, j-1 if j > 0 else 0, 1:Nz-1]) / sim.dy
            self.psi_Ex_y[1:Nx, j, 1:Nz-1] = b * self.psi_Ex_y[1:Nx, j, 1:Nz-1] + c * dHz
            sim.Ex[1:Nx, j, 1:Nz-1] += dt / (sim.eps[1:Nx, j, 1:Nz-1]) * self.psi_Ex_y[1:Nx, j, 1:Nz-1]
