"""Stochastic kinetic solver and invariant measure estimation.

Adds Brownian state noise to the deterministic kinetic equation, modeling
transcriptional bursting. The invariant measure mu(v) defines spatial domains.
"""

import numpy as np
from scipy import sparse
from typing import Optional
from dataclasses import dataclass

from kinetic_st.core.collision import CollisionKernel
from kinetic_st.core.boltzmann import KineticConfig, KineticTrajectory


@dataclass
class StochasticConfig(KineticConfig):
    """Stochastic solver configuration."""
    diffusion: float = 0.05
    noise_type: str = "additive"  # or "multiplicative"


class StochasticKineticSolver:
    """Integrate the stochastic kinetic equation (Euler-Maruyama)."""

    def __init__(self, kernel: CollisionKernel, config: Optional[StochasticConfig] = None):
        self.kernel = kernel
        self.config = config or StochasticConfig()
        self.trajectory_ = None

    def solve(self, states0, adjacency, Y_lr=None, verbose=False) -> KineticTrajectory:
        cfg = self.config
        rng = np.random.RandomState(cfg.seed)
        states = np.array(states0, dtype=np.float64, copy=True)
        n, d = states.shape
        sqrt2D = np.sqrt(2.0 * cfg.diffusion)

        A = adjacency.tocsr().astype(np.float64)
        deg = np.maximum(np.asarray(A.sum(axis=1)).flatten(), 1.0)
        P = (sparse.diags(1.0 / deg) @ A).tocsr()

        history = [states.copy()]
        times = [0.0]
        for step in range(cfg.n_steps):
            t = (step + 1) * cfg.dt
            states += cfg.dt * cfg.jump_rate * (P @ states - states)
            collided = self.kernel.apply_collisions(
                states, adjacency, Y_lr=Y_lr,
                n_pairs_per_spot=cfg.n_pairs_per_spot, rng=rng)
            states += (cfg.dt / cfg.tau) * (collided - states)
            if cfg.drift_field is not None:
                states += cfg.dt * cfg.drift_field(states, t)
            if cfg.diffusion > 0:
                sigma = sqrt2D * (np.abs(states) + 1e-3) if cfg.noise_type == "multiplicative" else sqrt2D
                states += np.sqrt(cfg.dt) * sigma * rng.standard_normal(states.shape)
            if (step + 1) % cfg.record_every == 0:
                history.append(states.copy())
                times.append(t)
                if verbose and (step + 1) % (cfg.record_every * 20) == 0:
                    print(f"  [StochasticKinetic] t={t:.3f} step={step+1}/{cfg.n_steps}")

        self.trajectory_ = KineticTrajectory(
            states=states, history=np.array(history), times=np.array(times),
            info={"solver": "stochastic", "diffusion": cfg.diffusion,
                  "n_steps": cfg.n_steps, "dt": cfg.dt, "n_spots": n, "dim": d})
        return self.trajectory_


class InvariantMeasureEstimator:
    """Estimate the non-equilibrium invariant measure from a trajectory.

    Parameters
    ----------
    representation : "kde" or "gmm"
    bandwidth : KDE bandwidth; None = Silverman's rule
    n_components : GMM components
    burnin : fraction of trajectory to discard as transient
    max_samples : cap on samples for estimation speed
    seed : random seed
    """

    def __init__(self, representation: str = "kde", bandwidth: Optional[float] = None,
                 n_components: int = 8, burnin: float = 0.2,
                 max_samples: int = 20000, seed: int = 0):
        self.representation = representation
        self.bandwidth = bandwidth
        self.n_components = n_components
        self.burnin = burnin
        self.max_samples = max_samples
        self.seed = seed
        self.samples_ = None
        self.gmm_ = None
        self._fitted = False

    def fit(self, trajectory: KineticTrajectory) -> "InvariantMeasureEstimator":
        history = trajectory.history
        T, n, d = history.shape
        t0 = max(1, int(T * self.burnin))
        samples = history[t0:].reshape(-1, d)
        if len(samples) > self.max_samples:
            rng = np.random.RandomState(self.seed)
            samples = samples[rng.choice(len(samples), self.max_samples, replace=False)]
        self.samples_ = samples
        self._dim = d

        if self.representation == "kde":
            std = samples.std(axis=0) + 1e-8
            self.bandwidth_ = self.bandwidth or np.median(std) * (len(samples) ** (-1.0 / (d + 4)))
        elif self.representation == "gmm":
            from sklearn.mixture import GaussianMixture
            k = max(min(self.n_components, len(samples) // 20), 1)
            self.gmm_ = GaussianMixture(
                n_components=k, covariance_type="diag",
                random_state=self.seed, max_iter=100, n_init=2).fit(samples)
        else:
            raise ValueError(f"Unknown representation: {self.representation}")
        self._fitted = True
        return self

    def log_density(self, states: np.ndarray) -> np.ndarray:
        """log mu(v) at each row."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        if self.representation == "kde":
            return _kde_logpdf(states, self.samples_, self.bandwidth_)
        return self.gmm_.score_samples(states)

    def density(self, states: np.ndarray) -> np.ndarray:
        """mu(v) at each row."""
        return np.exp(self.log_density(states))

    def sample(self, n_samples: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Draw samples from mu."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        rng = rng or np.random.RandomState(self.seed)
        if self.representation == "kde":
            idx = rng.choice(len(self.samples_), n_samples, replace=True)
            return self.samples_[idx] + rng.standard_normal((n_samples, self._dim)) * self.bandwidth_
        return self.gmm_.sample(n_samples)[0]


def _kde_logpdf(X, samples, bw):
    """Gaussian KDE log-density."""
    n, d = X.shape
    m = samples.shape[0]
    log_norm = -d * np.log(bw) - 0.5 * d * np.log(2 * np.pi) - np.log(m)
    out = np.empty(n)
    chunk = max(1, 2_000_000 // max(m, 1))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d2 = np.sum((X[s:e, None, :] - samples[None, :, :]) ** 2, axis=2)
        logk = -0.5 * d2 / (bw ** 2)
        mx = logk.max(axis=1)
        out[s:e] = log_norm + mx + np.log(np.exp(logk - mx[:, None]).mean(axis=1))
    return out
