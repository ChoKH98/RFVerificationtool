"""CPML (Convolutional PML) absorbing boundary - vectorized NumPy implementation."""
import numpy as np


class CPML:
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
            i = np.arange(n_cells, dtype=float)
            rho = (i / (n_cells - 1)) ** m
            sigma = _sigma_max(delta) * rho
            kappa = 1.0 + (kappa_max - 1.0) * rho
            alpha = alpha_max * (1.0 - rho)
            return kappa, sigma, alpha

        def _b_c(kappa, sigma, alpha):
            b = np.exp(-(sigma / kappa + alpha) * dt / 8.854e-12)
            denom = sigma + kappa * alpha
            c = np.where(np.abs(denom) < 1e-30, 0.0, sigma * (b - 1.0) / (kappa * denom))
            return b, c

        d = thickness
        kx, sx, ax = _profile(d, dx)
        ky, sy, ay = _profile(d, dy)
        kz, sz, az = _profile(d, dz)

        self.bx, self.cx = _b_c(kx, sx, ax)
        self.by, self.cy = _b_c(ky, sy, ay)
        self.bz, self.cz = _b_c(kz, sz, az)

        self.kappa_x = np.ones(Nx); self.kappa_y = np.ones(Ny); self.kappa_z = np.ones(Nz)
        self.kappa_x[:d] = kx[::-1]; self.kappa_x[-d:] = kx
        self.kappa_y[:d] = ky[::-1]; self.kappa_y[-d:] = ky
        self.kappa_z[:d] = kz[::-1]; self.kappa_z[-d:] = kz

        # Precomputed 1D b/c for vectorized face updates (shape: d-1 each)
        # Low face j=1..d-1: pj=d-1-j -> by[d-2..0] (reversed)
        # High face j=N-d..N-2: pj=j-(N-d) -> by[0..d-2]
        self.b_y_lo = self.by[d-2::-1]; self.c_y_lo = self.cy[d-2::-1]
        self.b_y_hi = self.by[:d-1];    self.c_y_hi = self.cy[:d-1]
        self.b_z_lo = self.bz[d-2::-1]; self.c_z_lo = self.cz[d-2::-1]
        self.b_z_hi = self.bz[:d-1];    self.c_z_hi = self.cz[:d-1]
        self.b_x_lo = self.bx[d-2::-1]; self.c_x_lo = self.cx[d-2::-1]
        self.b_x_hi = self.bx[:d-1];    self.c_x_hi = self.cx[:d-1]

        shape = (Nx, Ny, Nz)
        self.psi_Ex_y = np.zeros(shape); self.psi_Ex_z = np.zeros(shape)
        self.psi_Ey_x = np.zeros(shape); self.psi_Ey_z = np.zeros(shape)
        self.psi_Ez_x = np.zeros(shape); self.psi_Ez_y = np.zeros(shape)
        self.psi_Hx_y = np.zeros(shape); self.psi_Hx_z = np.zeros(shape)
        self.psi_Hy_x = np.zeros(shape); self.psi_Hy_z = np.zeros(shape)
        self.psi_Hz_x = np.zeros(shape); self.psi_Hz_y = np.zeros(shape)

    def _face_mask_y(self):
        d = self.d; Ny = self.sim.Ny
        mask = np.zeros(Ny, dtype=bool); mask[:d] = True; mask[-d:] = True
        return mask

    def _face_mask_x(self):
        d = self.d; Nx = self.sim.Nx
        mask = np.zeros(Nx, dtype=bool); mask[:d] = True; mask[-d:] = True
        return mask

    def _face_mask_z(self):
        d = self.d; Nz = self.sim.Nz
        mask = np.zeros(Nz, dtype=bool); mask[:d] = True; mask[-d:] = True
        return mask

    def update_H_psi(self):
        sim = self.sim
        d = self.d
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dt = sim.dt

        # y-face low (j=1..d-1)
        dEz = (sim.Ez[:Nx-1, 2:d+1, :Nz-1] - sim.Ez[:Nx-1, 1:d, :Nz-1]) / sim.dy
        self.psi_Hx_y[:Nx-1, 1:d, :Nz-1] = self.b_y_lo[None,:,None] * self.psi_Hx_y[:Nx-1, 1:d, :Nz-1] + self.c_y_lo[None,:,None] * dEz
        sim.Hx[:Nx-1, 1:d, :Nz-1] -= dt / sim.mu[:Nx-1, 1:d, :Nz-1] * self.psi_Hx_y[:Nx-1, 1:d, :Nz-1]
        dEx = (sim.Ex[:Nx-1, 2:d+1, :Nz-1] - sim.Ex[:Nx-1, 1:d, :Nz-1]) / sim.dy
        self.psi_Hz_y[:Nx-1, 1:d, :Nz-1] = self.b_y_lo[None,:,None] * self.psi_Hz_y[:Nx-1, 1:d, :Nz-1] + self.c_y_lo[None,:,None] * dEx
        sim.Hz[:Nx-1, 1:d, :Nz-1] += dt / sim.mu[:Nx-1, 1:d, :Nz-1] * self.psi_Hz_y[:Nx-1, 1:d, :Nz-1]

        # y-face high (j=Ny-d..Ny-2)
        dEz = (sim.Ez[:Nx-1, Ny-d+1:Ny, :Nz-1] - sim.Ez[:Nx-1, Ny-d:Ny-1, :Nz-1]) / sim.dy
        self.psi_Hx_y[:Nx-1, Ny-d:Ny-1, :Nz-1] = self.b_y_hi[None,:,None] * self.psi_Hx_y[:Nx-1, Ny-d:Ny-1, :Nz-1] + self.c_y_hi[None,:,None] * dEz
        sim.Hx[:Nx-1, Ny-d:Ny-1, :Nz-1] -= dt / sim.mu[:Nx-1, Ny-d:Ny-1, :Nz-1] * self.psi_Hx_y[:Nx-1, Ny-d:Ny-1, :Nz-1]
        dEx = (sim.Ex[:Nx-1, Ny-d+1:Ny, :Nz-1] - sim.Ex[:Nx-1, Ny-d:Ny-1, :Nz-1]) / sim.dy
        self.psi_Hz_y[:Nx-1, Ny-d:Ny-1, :Nz-1] = self.b_y_hi[None,:,None] * self.psi_Hz_y[:Nx-1, Ny-d:Ny-1, :Nz-1] + self.c_y_hi[None,:,None] * dEx
        sim.Hz[:Nx-1, Ny-d:Ny-1, :Nz-1] += dt / sim.mu[:Nx-1, Ny-d:Ny-1, :Nz-1] * self.psi_Hz_y[:Nx-1, Ny-d:Ny-1, :Nz-1]

        # z-face low (k=1..d-1)
        dEy = (sim.Ey[:Nx-1, :Ny-1, 2:d+1] - sim.Ey[:Nx-1, :Ny-1, 1:d]) / sim.dz
        self.psi_Hx_z[:Nx-1, :Ny-1, 1:d] = self.b_z_lo[None,None,:] * self.psi_Hx_z[:Nx-1, :Ny-1, 1:d] + self.c_z_lo[None,None,:] * dEy
        sim.Hx[:Nx-1, :Ny-1, 1:d] += dt / sim.mu[:Nx-1, :Ny-1, 1:d] * self.psi_Hx_z[:Nx-1, :Ny-1, 1:d]
        dEx = (sim.Ex[:Nx-1, :Ny-1, 2:d+1] - sim.Ex[:Nx-1, :Ny-1, 1:d]) / sim.dz
        self.psi_Hy_z[:Nx-1, :Ny-1, 1:d] = self.b_z_lo[None,None,:] * self.psi_Hy_z[:Nx-1, :Ny-1, 1:d] + self.c_z_lo[None,None,:] * dEx
        sim.Hy[:Nx-1, :Ny-1, 1:d] -= dt / sim.mu[:Nx-1, :Ny-1, 1:d] * self.psi_Hy_z[:Nx-1, :Ny-1, 1:d]

        # z-face high (k=Nz-d..Nz-2)
        dEy = (sim.Ey[:Nx-1, :Ny-1, Nz-d+1:Nz] - sim.Ey[:Nx-1, :Ny-1, Nz-d:Nz-1]) / sim.dz
        self.psi_Hx_z[:Nx-1, :Ny-1, Nz-d:Nz-1] = self.b_z_hi[None,None,:] * self.psi_Hx_z[:Nx-1, :Ny-1, Nz-d:Nz-1] + self.c_z_hi[None,None,:] * dEy
        sim.Hx[:Nx-1, :Ny-1, Nz-d:Nz-1] += dt / sim.mu[:Nx-1, :Ny-1, Nz-d:Nz-1] * self.psi_Hx_z[:Nx-1, :Ny-1, Nz-d:Nz-1]
        dEx = (sim.Ex[:Nx-1, :Ny-1, Nz-d+1:Nz] - sim.Ex[:Nx-1, :Ny-1, Nz-d:Nz-1]) / sim.dz
        self.psi_Hy_z[:Nx-1, :Ny-1, Nz-d:Nz-1] = self.b_z_hi[None,None,:] * self.psi_Hy_z[:Nx-1, :Ny-1, Nz-d:Nz-1] + self.c_z_hi[None,None,:] * dEx
        sim.Hy[:Nx-1, :Ny-1, Nz-d:Nz-1] -= dt / sim.mu[:Nx-1, :Ny-1, Nz-d:Nz-1] * self.psi_Hy_z[:Nx-1, :Ny-1, Nz-d:Nz-1]

        # x-face low (i=1..d-1)
        dEz = (sim.Ez[2:d+1, :Ny-1, :Nz-1] - sim.Ez[1:d, :Ny-1, :Nz-1]) / sim.dx
        self.psi_Hy_x[1:d, :Ny-1, :Nz-1] = self.b_x_lo[:,None,None] * self.psi_Hy_x[1:d, :Ny-1, :Nz-1] + self.c_x_lo[:,None,None] * dEz
        sim.Hy[1:d, :Ny-1, :Nz-1] += dt / sim.mu[1:d, :Ny-1, :Nz-1] * self.psi_Hy_x[1:d, :Ny-1, :Nz-1]
        dEy = (sim.Ey[2:d+1, :Ny-1, :Nz-1] - sim.Ey[1:d, :Ny-1, :Nz-1]) / sim.dx
        self.psi_Hz_x[1:d, :Ny-1, :Nz-1] = self.b_x_lo[:,None,None] * self.psi_Hz_x[1:d, :Ny-1, :Nz-1] + self.c_x_lo[:,None,None] * dEy
        sim.Hz[1:d, :Ny-1, :Nz-1] -= dt / sim.mu[1:d, :Ny-1, :Nz-1] * self.psi_Hz_x[1:d, :Ny-1, :Nz-1]

        # x-face high (i=Nx-d..Nx-2)
        dEz = (sim.Ez[Nx-d+1:Nx, :Ny-1, :Nz-1] - sim.Ez[Nx-d:Nx-1, :Ny-1, :Nz-1]) / sim.dx
        self.psi_Hy_x[Nx-d:Nx-1, :Ny-1, :Nz-1] = self.b_x_hi[:,None,None] * self.psi_Hy_x[Nx-d:Nx-1, :Ny-1, :Nz-1] + self.c_x_hi[:,None,None] * dEz
        sim.Hy[Nx-d:Nx-1, :Ny-1, :Nz-1] += dt / sim.mu[Nx-d:Nx-1, :Ny-1, :Nz-1] * self.psi_Hy_x[Nx-d:Nx-1, :Ny-1, :Nz-1]
        dEy = (sim.Ey[Nx-d+1:Nx, :Ny-1, :Nz-1] - sim.Ey[Nx-d:Nx-1, :Ny-1, :Nz-1]) / sim.dx
        self.psi_Hz_x[Nx-d:Nx-1, :Ny-1, :Nz-1] = self.b_x_hi[:,None,None] * self.psi_Hz_x[Nx-d:Nx-1, :Ny-1, :Nz-1] + self.c_x_hi[:,None,None] * dEy
        sim.Hz[Nx-d:Nx-1, :Ny-1, :Nz-1] -= dt / sim.mu[Nx-d:Nx-1, :Ny-1, :Nz-1] * self.psi_Hz_x[Nx-d:Nx-1, :Ny-1, :Nz-1]

    def update_E_psi(self):
        sim = self.sim
        d = self.d
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dt = sim.dt

        # y-face low (j=1..d-1, j-1=0..d-2)
        dHz = (sim.Hz[1:Nx, 1:d, 1:Nz-1] - sim.Hz[1:Nx, 0:d-1, 1:Nz-1]) / sim.dy
        self.psi_Ex_y[1:Nx, 1:d, 1:Nz-1] = self.b_y_lo[None,:,None] * self.psi_Ex_y[1:Nx, 1:d, 1:Nz-1] + self.c_y_lo[None,:,None] * dHz
        sim.Ex[1:Nx, 1:d, 1:Nz-1] += dt / sim.eps[1:Nx, 1:d, 1:Nz-1] * self.psi_Ex_y[1:Nx, 1:d, 1:Nz-1]
        dHx = (sim.Hx[1:Nx-1, 1:d, 1:Nz] - sim.Hx[1:Nx-1, 0:d-1, 1:Nz]) / sim.dy
        self.psi_Ez_y[1:Nx-1, 1:d, 1:Nz] = self.b_y_lo[None,:,None] * self.psi_Ez_y[1:Nx-1, 1:d, 1:Nz] + self.c_y_lo[None,:,None] * dHx
        sim.Ez[1:Nx-1, 1:d, 1:Nz] -= dt / sim.eps[1:Nx-1, 1:d, 1:Nz] * self.psi_Ez_y[1:Nx-1, 1:d, 1:Nz]

        # y-face high (j=Ny-d..Ny-2, j-1=Ny-d-1..Ny-3)
        dHz = (sim.Hz[1:Nx, Ny-d:Ny-1, 1:Nz-1] - sim.Hz[1:Nx, Ny-d-1:Ny-2, 1:Nz-1]) / sim.dy
        self.psi_Ex_y[1:Nx, Ny-d:Ny-1, 1:Nz-1] = self.b_y_hi[None,:,None] * self.psi_Ex_y[1:Nx, Ny-d:Ny-1, 1:Nz-1] + self.c_y_hi[None,:,None] * dHz
        sim.Ex[1:Nx, Ny-d:Ny-1, 1:Nz-1] += dt / sim.eps[1:Nx, Ny-d:Ny-1, 1:Nz-1] * self.psi_Ex_y[1:Nx, Ny-d:Ny-1, 1:Nz-1]
        dHx = (sim.Hx[1:Nx-1, Ny-d:Ny-1, 1:Nz] - sim.Hx[1:Nx-1, Ny-d-1:Ny-2, 1:Nz]) / sim.dy
        self.psi_Ez_y[1:Nx-1, Ny-d:Ny-1, 1:Nz] = self.b_y_hi[None,:,None] * self.psi_Ez_y[1:Nx-1, Ny-d:Ny-1, 1:Nz] + self.c_y_hi[None,:,None] * dHx
        sim.Ez[1:Nx-1, Ny-d:Ny-1, 1:Nz] -= dt / sim.eps[1:Nx-1, Ny-d:Ny-1, 1:Nz] * self.psi_Ez_y[1:Nx-1, Ny-d:Ny-1, 1:Nz]

        # z-face low (k=1..d-1, k-1=0..d-2)
        dHy = (sim.Hy[1:Nx, 1:Ny-1, 1:d] - sim.Hy[1:Nx, 1:Ny-1, 0:d-1]) / sim.dz
        self.psi_Ex_z[1:Nx, 1:Ny-1, 1:d] = self.b_z_lo[None,None,:] * self.psi_Ex_z[1:Nx, 1:Ny-1, 1:d] + self.c_z_lo[None,None,:] * dHy
        sim.Ex[1:Nx, 1:Ny-1, 1:d] -= dt / sim.eps[1:Nx, 1:Ny-1, 1:d] * self.psi_Ex_z[1:Nx, 1:Ny-1, 1:d]
        dHx = (sim.Hx[:Nx-1, 1:Ny, 1:d] - sim.Hx[:Nx-1, 1:Ny, 0:d-1]) / sim.dz
        self.psi_Ey_z[:Nx-1, 1:Ny, 1:d] = self.b_z_lo[None,None,:] * self.psi_Ey_z[:Nx-1, 1:Ny, 1:d] + self.c_z_lo[None,None,:] * dHx
        sim.Ey[:Nx-1, 1:Ny, 1:d] += dt / sim.eps[:Nx-1, 1:Ny, 1:d] * self.psi_Ey_z[:Nx-1, 1:Ny, 1:d]

        # z-face high (k=Nz-d..Nz-2, k-1=Nz-d-1..Nz-3)
        dHy = (sim.Hy[1:Nx, 1:Ny-1, Nz-d:Nz-1] - sim.Hy[1:Nx, 1:Ny-1, Nz-d-1:Nz-2]) / sim.dz
        self.psi_Ex_z[1:Nx, 1:Ny-1, Nz-d:Nz-1] = self.b_z_hi[None,None,:] * self.psi_Ex_z[1:Nx, 1:Ny-1, Nz-d:Nz-1] + self.c_z_hi[None,None,:] * dHy
        sim.Ex[1:Nx, 1:Ny-1, Nz-d:Nz-1] -= dt / sim.eps[1:Nx, 1:Ny-1, Nz-d:Nz-1] * self.psi_Ex_z[1:Nx, 1:Ny-1, Nz-d:Nz-1]
        dHx = (sim.Hx[:Nx-1, 1:Ny, Nz-d:Nz-1] - sim.Hx[:Nx-1, 1:Ny, Nz-d-1:Nz-2]) / sim.dz
        self.psi_Ey_z[:Nx-1, 1:Ny, Nz-d:Nz-1] = self.b_z_hi[None,None,:] * self.psi_Ey_z[:Nx-1, 1:Ny, Nz-d:Nz-1] + self.c_z_hi[None,None,:] * dHx
        sim.Ey[:Nx-1, 1:Ny, Nz-d:Nz-1] += dt / sim.eps[:Nx-1, 1:Ny, Nz-d:Nz-1] * self.psi_Ey_z[:Nx-1, 1:Ny, Nz-d:Nz-1]

        # x-face low (i=1..d-1, i-1=0..d-2)
        dHz = (sim.Hz[1:d, 1:Ny, 1:Nz-1] - sim.Hz[0:d-1, 1:Ny, 1:Nz-1]) / sim.dx
        self.psi_Ey_x[1:d, 1:Ny, 1:Nz-1] = self.b_x_lo[:,None,None] * self.psi_Ey_x[1:d, 1:Ny, 1:Nz-1] + self.c_x_lo[:,None,None] * dHz
        sim.Ey[1:d, 1:Ny, 1:Nz-1] -= dt / sim.eps[1:d, 1:Ny, 1:Nz-1] * self.psi_Ey_x[1:d, 1:Ny, 1:Nz-1]
        dHy = (sim.Hy[1:d, 1:Ny-1, 1:Nz] - sim.Hy[0:d-1, 1:Ny-1, 1:Nz]) / sim.dx
        self.psi_Ez_x[1:d, 1:Ny-1, 1:Nz] = self.b_x_lo[:,None,None] * self.psi_Ez_x[1:d, 1:Ny-1, 1:Nz] + self.c_x_lo[:,None,None] * dHy
        sim.Ez[1:d, 1:Ny-1, 1:Nz] += dt / sim.eps[1:d, 1:Ny-1, 1:Nz] * self.psi_Ez_x[1:d, 1:Ny-1, 1:Nz]

        # x-face high (i=Nx-d..Nx-2, i-1=Nx-d-1..Nx-3)
        dHz = (sim.Hz[Nx-d:Nx-1, 1:Ny, 1:Nz-1] - sim.Hz[Nx-d-1:Nx-2, 1:Ny, 1:Nz-1]) / sim.dx
        self.psi_Ey_x[Nx-d:Nx-1, 1:Ny, 1:Nz-1] = self.b_x_hi[:,None,None] * self.psi_Ey_x[Nx-d:Nx-1, 1:Ny, 1:Nz-1] + self.c_x_hi[:,None,None] * dHz
        sim.Ey[Nx-d:Nx-1, 1:Ny, 1:Nz-1] -= dt / sim.eps[Nx-d:Nx-1, 1:Ny, 1:Nz-1] * self.psi_Ey_x[Nx-d:Nx-1, 1:Ny, 1:Nz-1]
        dHy = (sim.Hy[Nx-d:Nx-1, 1:Ny-1, 1:Nz] - sim.Hy[Nx-d-1:Nx-2, 1:Ny-1, 1:Nz]) / sim.dx
        self.psi_Ez_x[Nx-d:Nx-1, 1:Ny-1, 1:Nz] = self.b_x_hi[:,None,None] * self.psi_Ez_x[Nx-d:Nx-1, 1:Ny-1, 1:Nz] + self.c_x_hi[:,None,None] * dHy
        sim.Ez[Nx-d:Nx-1, 1:Ny-1, 1:Nz] += dt / sim.eps[Nx-d:Nx-1, 1:Ny-1, 1:Nz] * self.psi_Ez_x[Nx-d:Nx-1, 1:Ny-1, 1:Nz]
