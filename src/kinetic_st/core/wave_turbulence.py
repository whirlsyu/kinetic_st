"""Wave-kinetic analysis of tissue-level gene expression waves.

Computes the wave action spectrum n(k, t) = |u_hat(k, t)|^2 from tissue
fields, estimates cascade direction and steady-state spectral slope, and
provides a simplified four-wave collision propagator for illustration.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass, field

from kinetic_st.core.fluid_limit import TissueField


@dataclass
class WaveSpectrum:
    """Wave action spectrum at one time."""
    k_magnitudes: np.ndarray
    spectrum: np.ndarray
    time: float


@dataclass
class WaveKineticResult:
    """Wave-kinetic analysis output."""
    spectra: list
    spectral_flux: np.ndarray
    spectral_slope: float
    cascade_direction: str
    k_range: np.ndarray
    info: Dict = field(default_factory=dict)


class WaveKineticAnalyzer:
    """Analyze tissue expression waves via wave-turbulence diagnostics.

    Parameters
    ----------
    n_k_bins : wave-number magnitude bins
    grid_shape : (H, W) for regular grid; None = infer from coords
    k_max : maximum wave number; None = Nyquist
    four_wave_coupling : coupling strength for propagator
    """

    def __init__(self, n_k_bins: int = 20, grid_shape: Optional[tuple] = None,
                 k_max: Optional[float] = None, four_wave_coupling: float = 0.1):
        self.n_k_bins = n_k_bins
        self.grid_shape = grid_shape
        self.k_max = k_max
        self.four_wave_coupling = four_wave_coupling

    def analyze(self, fields, coords, component=0, verbose=False) -> WaveKineticResult:
        """Analyze a time series of tissue fields."""
        arr = (np.stack([f.mean_state for f in fields], axis=0)
               if isinstance(fields, list) and fields and isinstance(fields[0], TissueField)
               else np.asarray(fields))
        T, n, d = arr.shape
        comp = arr[:, :, component]

        grid_shape = self.grid_shape or _infer_grid(coords)
        H, W = grid_shape

        ky = np.fft.fftfreq(H)[:, None]
        kx = np.fft.fftfreq(W)[None, :]
        k_mag = np.sqrt(kx ** 2 + ky ** 2)
        k_max_eff = self.k_max or k_mag.max()
        k_bins = np.linspace(0, k_max_eff, self.n_k_bins + 1)
        k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

        spectra = []
        nkt = np.zeros((T, self.n_k_bins))
        for t in range(T):
            field_grid = _scatter_to_grid(comp[t], coords, grid_shape)
            n2 = np.abs(np.fft.fft2(field_grid)) ** 2
            flat_k, flat_n = k_mag.ravel(), n2.ravel()
            for bi in range(self.n_k_bins):
                mask = (flat_k >= k_bins[bi]) & (flat_k < k_bins[bi + 1])
                nkt[t, bi] = flat_n[mask].mean() if mask.any() else 0.0
            spectra.append(WaveSpectrum(k_centers.copy(), nkt[t].copy(), float(t)))

        flux = -np.diff(np.cumsum(nkt, axis=1), axis=0)
        mean_flux = flux.mean(axis=0)

        n_mean = nkt.mean(axis=0)
        valid = (k_centers > 1e-6) & (n_mean > 1e-12) & (k_centers < 0.9 * k_max_eff)
        slope = float(np.polyfit(np.log(k_centers[valid]), np.log(n_mean[valid]), 1)[0]) \
            if valid.sum() >= 3 else 0.0

        half = len(mean_flux) // 2
        cascade = ("forward" if mean_flux[:half].mean() > 0
                   else "inverse" if mean_flux[:half].mean() < 0 else "mixed")

        if verbose:
            print(f"  [WaveKinetic] slope={slope:.3f} cascade={cascade}")

        return WaveKineticResult(
            spectra=spectra, spectral_flux=flux, spectral_slope=slope,
            cascade_direction=cascade, k_range=k_centers,
            info={"n_k_bins": self.n_k_bins, "grid_shape": grid_shape,
                  "T": T, "component": component})

    def propagate_spectrum(self, n_init, n_steps=100, dt=0.1) -> np.ndarray:
        """Simplified four-wave collision propagator (illustrative).

        Local-in-k neighbor exchange preserving total wave action.
        """
        n = n_init.copy().astype(np.float64)
        for _ in range(n_steps):
            diff = np.zeros_like(n)
            diff[1:-1] = n[2:] + n[:-2] - 2 * n[1:-1]
            diff[0], diff[-1] = n[1] - n[0], n[-2] - n[-1]
            n = np.maximum(n + self.four_wave_coupling * dt * diff, 1e-12)
            n *= n_init.sum() / n.sum()
        return n


def _infer_grid(coords):
    xs = np.unique(np.round(coords[:, 0], 4))
    ys = np.unique(np.round(coords[:, 1], 4))
    return (max(len(ys), 2), max(len(xs), 2))


def _scatter_to_grid(values, coords, grid_shape):
    H, W = grid_shape
    xs, ys = coords[:, 0], coords[:, 1]
    ix = np.clip(((xs - xs.min()) / (xs.max() - xs.min() + 1e-9) * (W - 1)).round().astype(int), 0, W - 1)
    iy = np.clip(((ys - ys.min()) / (ys.max() - ys.min() + 1e-9) * (H - 1)).round().astype(int), 0, H - 1)
    grid = np.zeros((H, W))
    counts = np.zeros((H, W))
    for v, j, i in zip(values, ix, iy):
        grid[i, j] += v
        counts[i, j] += 1
    return grid / np.maximum(counts, 1)
