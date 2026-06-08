"""Gaussian pulse port excitation + S11 extraction via DFT."""
import numpy as np


C0 = 3e8


class GaussianPort:
    """Microstrip edge gap-voltage feed: excite Ez (vertical) between ground and patch bottom edge."""

    def __init__(self, sim, geom, f0=2.4e9, bw=2e9, z0=50.0):
        self.sim = sim
        self.f0, self.bw, self.z0 = f0, bw, z0

        # Feed position: one cell inside patch bottom edge (avoids aperture boundary
        # coupling to TM10 mode; TM01 coupling remains ~cos(?/22)??.99 of maximum)
        self.xf = geom['cx']
        self.yf = geom['y0'] + 1      # one cell inside patch, away from aperture
        self.z0_idx = geom.get('z_feed', geom['z_gnd'] + 1)
        self.z1_idx = geom['z_sub_top']
        if self.z1_idx <= self.z0_idx:
            raise ValueError("Feed path must span at least one dielectric cell")

        # Pulse timing
        tau = 0.5 / bw
        self.tau = tau
        self.t0  = 5 * tau            # delay so pulse starts at 0

        # Time recording
        dt = sim.dt
        self.dt = dt
        self.v_inc = []   # incident voltage (analytic)
        self.v_tot = []   # total voltage at port

    def apply(self, sim):
        t = sim.time_step * self.dt
        tau, t0, f0 = self.tau, self.t0, self.f0
        pulse = np.exp(-((t - t0) / tau) ** 2) * np.cos(2 * np.pi * f0 * (t - t0))

        # Inject Ez at feed column (soft gap-voltage source)
        for zi in range(self.z0_idx, self.z1_idx):
            sim.Ez[self.xf, self.yf, zi] += pulse

        # Record incident (analytic) and total voltage
        v_inc = pulse * (self.z1_idx - self.z0_idx) * sim.dz
        v_tot = np.sum(sim.Ez[self.xf, self.yf, self.z0_idx:self.z1_idx]) * sim.dz
        self.v_inc.append(v_inc)
        self.v_tot.append(v_tot)

    def get_s11(self, f_min=1.5e9, f_max=3.5e9, n_pts=401):
        """Compute S11 via DFT of time-domain port signals."""
        dt = self.dt
        v_inc = np.array(self.v_inc)
        v_tot = np.array(self.v_tot)
        n = len(v_inc)
        t = np.arange(n) * dt

        freqs = np.linspace(f_min, f_max, n_pts)
        # Vectorized DFT with Hann window to reduce spectral leakage
        window = np.hanning(n)
        exp_factor = np.exp(-2j * np.pi * np.outer(freqs, t))
        V_inc = exp_factor @ (v_inc * window)
        V_tot = exp_factor @ (v_tot * window)

        # S11 = (V_tot - V_inc) / V_inc  (reflected / incident)
        safe_inc = np.where(np.abs(V_inc) > 1e-30, V_inc, 1.0 + 0j)
        S11 = np.where(np.abs(V_inc) > 1e-30, (V_tot - V_inc) / safe_inc, np.zeros_like(V_inc))
        return freqs, S11


