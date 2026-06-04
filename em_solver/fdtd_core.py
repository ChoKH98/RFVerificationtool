"""3D FDTD Core — Yee grid time-domain EM solver."""
import numpy as np


class FDTD:
    def __init__(self, dx, dy, dz, Nx, Ny, Nz, dt=None):
        self.dx, self.dy, self.dz = dx, dy, dz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz

        # Courant stability: dt <= 1/(c*sqrt(1/dx²+1/dy²+1/dz²))
        c0 = 3e8
        dt_max = 1.0 / (c0 * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
        self.dt = dt if dt else 0.99 * dt_max

        # Material arrays (default: free space)
        self.eps = np.ones((Nx, Ny, Nz)) * 8.854e-12
        self.mu  = np.ones((Nx, Ny, Nz)) * 4*np.pi*1e-7
        self.sigma = np.zeros((Nx, Ny, Nz))

        # Field arrays
        shape = (Nx, Ny, Nz)
        self.Ex = np.zeros(shape); self.Ey = np.zeros(shape); self.Ez = np.zeros(shape)
        self.Hx = np.zeros(shape); self.Hy = np.zeros(shape); self.Hz = np.zeros(shape)

        # Update coefficients
        self._compute_coeffs()

        self.time_step = 0
        self.sources = []
        self.probes = []
        self._cpml = None

    def _compute_coeffs(self):
        dt = self.dt
        eps, sigma = self.eps, self.sigma

        # E update: Ca = (1 - σΔt/2ε)/(1 + σΔt/2ε), Cb = Δt/ε/(1 + σΔt/2ε)
        denom = 1.0 + sigma * dt / (2.0 * eps)
        self.Ca = (1.0 - sigma * dt / (2.0 * eps)) / denom
        self.CbX = (dt / (self.eps * self.dx)) / denom
        self.CbY = (dt / (self.eps * self.dy)) / denom
        self.CbZ = (dt / (self.eps * self.dz)) / denom

        # H update coefficients (lossless magnetic)
        mu = self.mu
        self.Da = np.ones_like(mu)
        self.DbX = dt / (mu * self.dx)
        self.DbY = dt / (mu * self.dy)
        self.DbZ = dt / (mu * self.dz)

    def set_material(self, x0, x1, y0, y1, z0, z1, eps_r=1.0, sigma=0.0, mu_r=1.0):
        self.eps[x0:x1, y0:y1, z0:z1] = eps_r * 8.854e-12
        self.sigma[x0:x1, y0:y1, z0:z1] = sigma
        self.mu[x0:x1, y0:y1, z0:z1] = mu_r * 4*np.pi*1e-7
        self._compute_coeffs()

    def _update_E(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        # Ex: dHz/dy - dHy/dz
        self.Ex[1:Nx, 1:Ny-1, 1:Nz-1] = (
            self.Ca[1:Nx, 1:Ny-1, 1:Nz-1] * self.Ex[1:Nx, 1:Ny-1, 1:Nz-1]
            + self.CbY[1:Nx, 1:Ny-1, 1:Nz-1] * (self.Hz[1:Nx, 1:Ny-1, 1:Nz-1] - self.Hz[1:Nx, 0:Ny-2, 1:Nz-1])
            - self.CbZ[1:Nx, 1:Ny-1, 1:Nz-1] * (self.Hy[1:Nx, 1:Ny-1, 1:Nz-1] - self.Hy[1:Nx, 1:Ny-1, 0:Nz-2])
        )
        # Ey: dHx/dz - dHz/dx
        self.Ey[1:Nx-1, 1:Ny, 1:Nz-1] = (
            self.Ca[1:Nx-1, 1:Ny, 1:Nz-1] * self.Ey[1:Nx-1, 1:Ny, 1:Nz-1]
            + self.CbZ[1:Nx-1, 1:Ny, 1:Nz-1] * (self.Hx[1:Nx-1, 1:Ny, 1:Nz-1] - self.Hx[1:Nx-1, 1:Ny, 0:Nz-2])
            - self.CbX[1:Nx-1, 1:Ny, 1:Nz-1] * (self.Hz[1:Nx-1, 1:Ny, 1:Nz-1] - self.Hz[0:Nx-2, 1:Ny, 1:Nz-1])
        )
        # Ez: dHy/dx - dHx/dy
        self.Ez[1:Nx-1, 1:Ny-1, 1:Nz] = (
            self.Ca[1:Nx-1, 1:Ny-1, 1:Nz] * self.Ez[1:Nx-1, 1:Ny-1, 1:Nz]
            + self.CbX[1:Nx-1, 1:Ny-1, 1:Nz] * (self.Hy[1:Nx-1, 1:Ny-1, 1:Nz] - self.Hy[0:Nx-2, 1:Ny-1, 1:Nz])
            - self.CbY[1:Nx-1, 1:Ny-1, 1:Nz] * (self.Hx[1:Nx-1, 1:Ny-1, 1:Nz] - self.Hx[1:Nx-1, 0:Ny-2, 1:Nz])
        )

    def _update_H(self):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        # Hx: dEy/dz - dEz/dy
        self.Hx[0:Nx-1, 0:Ny-1, 0:Nz-1] -= (
            self.DbY[0:Nx-1, 0:Ny-1, 0:Nz-1] * (self.Ez[0:Nx-1, 1:Ny, 0:Nz-1] - self.Ez[0:Nx-1, 0:Ny-1, 0:Nz-1])
            - self.DbZ[0:Nx-1, 0:Ny-1, 0:Nz-1] * (self.Ey[0:Nx-1, 0:Ny-1, 1:Nz] - self.Ey[0:Nx-1, 0:Ny-1, 0:Nz-1])
        )
        # Hy: dEz/dx - dEx/dz
        self.Hy[0:Nx-1, 0:Ny-1, 0:Nz-1] -= (
            self.DbZ[0:Nx-1, 0:Ny-1, 0:Nz-1] * (self.Ex[0:Nx-1, 0:Ny-1, 1:Nz] - self.Ex[0:Nx-1, 0:Ny-1, 0:Nz-1])
            - self.DbX[0:Nx-1, 0:Ny-1, 0:Nz-1] * (self.Ez[1:Nx, 0:Ny-1, 0:Nz-1] - self.Ez[0:Nx-1, 0:Ny-1, 0:Nz-1])
        )
        # Hz: dEx/dy - dEy/dx
        self.Hz[0:Nx-1, 0:Ny-1, 0:Nz-1] -= (
            self.DbX[0:Nx-1, 0:Ny-1, 0:Nz-1] * (self.Ey[1:Nx, 0:Ny-1, 0:Nz-1] - self.Ey[0:Nx-1, 0:Ny-1, 0:Nz-1])
            - self.DbY[0:Nx-1, 0:Ny-1, 0:Nz-1] * (self.Ex[0:Nx-1, 1:Ny, 0:Nz-1] - self.Ex[0:Nx-1, 0:Ny-1, 0:Nz-1])
        )

    def step(self):
        self._update_H()
        if self._cpml:
            self._cpml.update_H_psi()
        self._update_E()
        if self._cpml:
            self._cpml.update_E_psi()
        for src in self.sources:
            src.apply(self)
        for prb in self.probes:
            prb.record(self)
        self.time_step += 1

    def run(self, n_steps, verbose=True):
        for n in range(n_steps):
            self.step()
            if verbose and n % 500 == 0:
                print(f"  Step {n}/{n_steps}  t={n*self.dt*1e9:.2f} ns")
