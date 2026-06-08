"""WR-90 waveguide S-parameter simulation using the FDTD/CPML solver.

TE10 mode injection at source plane; DFT extraction for S21.
Writes results/waveguide_family/fdtd/sparams.csv to the bridge folder.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

try:
    from .fdtd_core import FDTD
except ImportError:
    from fdtd_core import FDTD

C0 = 3e8
_EPS0 = 8.854e-12
_BRIDGE = Path("/mnt/c/Users/whqkr/Desktop/RFVerificationBridge")


class _ZOnlyCPML:
    """Z-face-only CPML for waveguide simulations.

    Applies absorbing corrections only on the ±z faces so that the natural
    PEC boundary conditions on the x/y waveguide walls are left intact.
    The standard CPML class corrupts TE10 by absorbing the guide interior
    (no air buffer exists between the guide wall and the grid boundary).

    Also fixes the z-hi derivative: the Ey psi must use backward dHx
    (consistent with the main FDTD Ey update), not forward dHx.
    """

    def __init__(self, sim, thickness: int = 10, m: int = 3,
                 sigma_factor: float = 0.8, alpha_max: float = 0.05):
        self.sim = sim
        self.d = thickness
        sim._cpml = self

        d = thickness
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dz = sim.dz
        dt = sim.dt

        sigma_max = sigma_factor * (m + 1) / (2 * 377.0 * dz)
        i_arr = np.arange(d, dtype=float)
        rho   = (i_arr / (d - 1)) ** m
        sigma = sigma_max * rho
        alpha = alpha_max * (1.0 - rho)

        b = np.exp(-(sigma + alpha) * dt / _EPS0)
        denom = sigma + alpha
        c = np.where(np.abs(denom) < 1e-30, 0.0, sigma * (b - 1.0) / denom)

        # b_lo[0] → outermost z-lo cell (k=1), b_lo[d-2] → innermost (k=d-1)
        self.b_lo = b[d-2::-1]
        self.c_lo = c[d-2::-1]
        # b_hi[0] → innermost z-hi cell (k=Nz-d), b_hi[d-2] → outermost (k=Nz-2)
        self.b_hi = b[:d-1]
        self.c_hi = c[:d-1]

        shape = (Nx, Ny, Nz)
        self.psi_Hx_z = np.zeros(shape)
        self.psi_Hy_z = np.zeros(shape)
        self.psi_Ex_z = np.zeros(shape)
        self.psi_Ey_z = np.zeros(shape)

    def update_H_psi(self):
        """H psi step: forward dE/dz (matches main FDTD H update)."""
        sim = self.sim
        d   = self.d
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dt  = sim.dt

        for slc, bv, cv in (
            (slice(1, d),       self.b_lo, self.c_lo),
            (slice(Nz-d, Nz-1), self.b_hi, self.c_hi),
        ):
            dEy = (sim.Ey[:Nx-1, :Ny-1, slc.start+1:slc.stop+1]
                   - sim.Ey[:Nx-1, :Ny-1, slc]) / sim.dz
            self.psi_Hx_z[:Nx-1, :Ny-1, slc] = (
                bv[None, None, :] * self.psi_Hx_z[:Nx-1, :Ny-1, slc]
                + cv[None, None, :] * dEy)
            sim.Hx[:Nx-1, :Ny-1, slc] += (
                dt / sim.mu[:Nx-1, :Ny-1, slc] * self.psi_Hx_z[:Nx-1, :Ny-1, slc])

            dEx = (sim.Ex[:Nx-1, :Ny-1, slc.start+1:slc.stop+1]
                   - sim.Ex[:Nx-1, :Ny-1, slc]) / sim.dz
            self.psi_Hy_z[:Nx-1, :Ny-1, slc] = (
                bv[None, None, :] * self.psi_Hy_z[:Nx-1, :Ny-1, slc]
                + cv[None, None, :] * dEx)
            sim.Hy[:Nx-1, :Ny-1, slc] -= (
                dt / sim.mu[:Nx-1, :Ny-1, slc] * self.psi_Hy_z[:Nx-1, :Ny-1, slc])

    def update_E_psi(self):
        """E psi step: backward dH/dz (matches main FDTD E update)."""
        sim = self.sim
        d   = self.d
        Nx, Ny, Nz = sim.Nx, sim.Ny, sim.Nz
        dt  = sim.dt

        for slc, slc_m1, bv, cv in (
            (slice(1, d),       slice(0, d-1),     self.b_lo, self.c_lo),
            (slice(Nz-d, Nz-1), slice(Nz-d-1, Nz-2), self.b_hi, self.c_hi),
        ):
            dHy = (sim.Hy[1:Nx, 1:Ny-1, slc] - sim.Hy[1:Nx, 1:Ny-1, slc_m1]) / sim.dz
            self.psi_Ex_z[1:Nx, 1:Ny-1, slc] = (
                bv[None, None, :] * self.psi_Ex_z[1:Nx, 1:Ny-1, slc]
                + cv[None, None, :] * dHy)
            sim.Ex[1:Nx, 1:Ny-1, slc] -= (
                dt / sim.eps[1:Nx, 1:Ny-1, slc] * self.psi_Ex_z[1:Nx, 1:Ny-1, slc])

            # Backward dHx — consistent with main FDTD Ey update
            dHx = (sim.Hx[:Nx-1, 1:Ny, slc] - sim.Hx[:Nx-1, 1:Ny, slc_m1]) / sim.dz
            self.psi_Ey_z[:Nx-1, 1:Ny, slc] = (
                bv[None, None, :] * self.psi_Ey_z[:Nx-1, 1:Ny, slc]
                + cv[None, None, :] * dHx)
            sim.Ey[:Nx-1, 1:Ny, slc] += (
                dt / sim.eps[:Nx-1, 1:Ny, slc] * self.psi_Ey_z[:Nx-1, 1:Ny, slc])
_DEFAULT_OUT = _BRIDGE / "results" / "waveguide_family" / "fdtd"


def _write_sparams(path: Path, freqs, S11, S21):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "port_i", "port_j", "real", "imag", "db", "phase_deg"])
        for freq, s11v, s21v in zip(freqs, S11, S21):
            for pi, pj, val in [(1, 1, s11v), (2, 1, s21v)]:
                db = 20 * np.log10(max(abs(val), 1e-30))
                ph = float(np.degrees(np.angle(val)))
                w.writerow([float(freq), pi, pj, float(val.real), float(val.imag), db, ph])


def run(steps: int = 1200, output_dir: Path = _DEFAULT_OUT):
    output_dir = Path(output_dir)

    # ── Grid ──────────────────────────────────────────────────────────────────
    dx = dy = dz = 1e-3          # 1 mm cells
    pml  = 10                    # CPML thickness
    guard = 4                    # guard cells at each z-port
    GL   = 80                    # guide length cells (= 80 mm)
    Nx   = 25                    # broad wall: a_eff = 24 mm (walls at ix=0 and ix=24)
    Ny   = 12                    # narrow wall: b_eff = 11 mm (walls at iy=0 and iy=11)
    Nz   = pml + guard + GL + guard + pml   # = 108

    a_eff = (Nx - 1) * dx        # 24 mm effective width
    fc    = C0 / (2 * a_eff)     # cutoff freq

    z_src = pml + guard          # = 14  (source injection plane)
    z_trn = pml + guard + GL - 2 # = 92  (transmission probe)

    print(f"WR-90 FDTD: {Nx}x{Ny}x{Nz},  z_src={z_src}, z_trn={z_trn}")
    print(f"  a_eff={a_eff*1e3:.1f} mm,  fc={fc/1e9:.4f} GHz")

    # ── TE10 mode profile: sin(pi*x/a) ───────────────────────────────────────
    # Ey is updated at ix=1..Nx-2, iy=1..Ny-1 (natural PEC at boundaries)
    ix = np.arange(Nx)
    mode_x = np.sin(np.pi * ix / (Nx - 1))   # shape (Nx,)
    mode_inj = mode_x[1:Nx-1]                  # shape (Nx-2,) — active cells
    # 2-D mode for broadcast: (Nx-2, Ny-1)
    mode_2d  = mode_inj[:, np.newaxis]          # broadcasts over y dimension
    mode_norm = float(np.sum(mode_inj**2) * (Ny - 1))

    # ── Gaussian pulse ────────────────────────────────────────────────────────
    f0  = 10e9
    bw  = 4e9
    tau = 0.5 / bw
    t0  = 5 * tau

    # ── Build sim ─────────────────────────────────────────────────────────────
    sim = FDTD(dx=dx, dy=dy, dz=dz, Nx=Nx, Ny=Ny, Nz=Nz)
    _ZOnlyCPML(sim, thickness=pml)

    dt = sim.dt
    print(f"  dt={dt*1e12:.3f} ps,  steps={steps},  "
          f"T_sim={steps*dt*1e9:.2f} ns")

    # ── Sources and probes ────────────────────────────────────────────────────
    # Reference plane: 5 cells forward from source. Only the forward-going
    # wave exists here (backward wave travels to -z and is absorbed by CPML).
    # Using modal projection for both ref and trn ensures the same
    # normalization, so |S21| = |V_trn / V_ref| = 1 for a lossless guide.
    z_ref = z_src + 5

    v_ref = np.zeros(steps)
    v_trn = np.zeros(steps)

    class _TE10Source:
        def apply(self, s):
            n = s.time_step
            if n >= steps:
                return
            t = n * dt
            pulse = (np.exp(-((t - t0) / tau) ** 2)
                     * np.cos(2 * np.pi * f0 * (t - t0)))
            s.Ey[1:Nx-1, 1:Ny, z_src] += pulse * mode_2d

    class _TE10Probe:
        def record(self, s):
            n = s.time_step
            if n >= steps:
                return
            v_ref[n] = (np.sum(s.Ey[1:Nx-1, 1:Ny, z_ref] * mode_2d)
                        / mode_norm)
            v_trn[n] = (np.sum(s.Ey[1:Nx-1, 1:Ny, z_trn] * mode_2d)
                        / mode_norm)

    sim.sources.append(_TE10Source())
    sim.probes.append(_TE10Probe())

    # ── Time loop ─────────────────────────────────────────────────────────────
    t_start = time.time()
    sim.run(steps, verbose=True)
    elapsed = time.time() - t_start
    print(f"  Completed {steps} steps in {elapsed:.1f} s "
          f"({elapsed / steps * 1000:.2f} ms/step)")

    # ── DFT → S-parameters ───────────────────────────────────────────────────
    f_min, f_max, n_pts = 8.2e9, 12.4e9, 43
    freqs = np.linspace(f_min, f_max, n_pts)
    t_arr = np.arange(steps) * dt
    window = np.ones(steps)

    exp_mat = np.exp(-2j * np.pi * np.outer(freqs, t_arr))
    V_ref = exp_mat @ (v_ref * window)
    V_trn = exp_mat @ (v_trn * window)

    safe = np.where(np.abs(V_ref) > 1e-30, V_ref, 1.0 + 0j)
    S21  = np.where(np.abs(V_ref) > 1e-30, V_trn / safe, 0j)
    # S11: energy conservation estimate
    S11  = np.sqrt(np.maximum(0.0, 1.0 - np.abs(S21) ** 2)) + 0j

    # Summary
    mask = freqs > fc
    S21_db = 20 * np.log10(np.maximum(np.abs(S21), 1e-30))
    print(f"  S21 (f>fc): mean={np.mean(S21_db[mask]):.3f} dB  "
          f"(ideal: 0 dB for lossless guide)")

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = output_dir / "sparams.csv"
    _write_sparams(out_path, freqs, S11, S21)
    print(f"  Wrote {out_path}")
    return freqs, S11, S21


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=1200,
                   help="FDTD time steps (default: 1200)")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT,
                   help="Directory for sparams.csv output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.steps, args.output_dir)
