"""Macroscopic tissue fields via moment closure of the cell-state density.

Moments: density rho, mean state u (expression field), covariance P
(heterogeneity tensor). A relaxation_indicator reports how close each spot
is to local quasi-equilibrium (fluid-limit regime).
"""

import numpy as np
from scipy import sparse
from typing import Optional
from dataclasses import dataclass

from kinetic_st.core.boltzmann import KineticTrajectory


@dataclass
class TissueField:
    """Macroscopic fields at one time point."""
    density: np.ndarray
    mean_state: np.ndarray
    covariance: np.ndarray
    heterogeneity: np.ndarray
    time: float


class FluidLimiter:
    """Compute tissue fields by taking moments of the cell-state density.

    Parameters
    ----------
    bandwidth : KDE bandwidth; None = Silverman's rule
    n_neighbors : spatial neighborhood size for moment support
    use_local_kde : estimate f_i as Gaussian KDE over neighborhood
    """

    def __init__(self, bandwidth: Optional[float] = None,
                 n_neighbors: int = 12, use_local_kde: bool = True):
        self.bandwidth = bandwidth
        self.n_neighbors = n_neighbors
        self.use_local_kde = use_local_kde
        self.fields_ = None

    def compute(self, trajectory: KineticTrajectory, adjacency: sparse.spmatrix) -> list:
        """Compute fields for every recorded time."""
        A = adjacency.tocsr()
        n = A.shape[0]
        nbr_lists = _neighbor_lists(A, self.n_neighbors)
        history, times = trajectory.history, trajectory.times
        T, _, d = history.shape

        if self.bandwidth is None:
            all_states = history.reshape(-1, d)
            self.bandwidth_ = np.median(all_states.std(axis=0) + 1e-8) * (n ** (-1.0 / (d + 4)))
        else:
            self.bandwidth_ = self.bandwidth

        fields = []
        for ti in range(T):
            states = history[ti]
            if self.use_local_kde:
                mean_state, cov_diag = _local_moments(states, nbr_lists, self.bandwidth_)
            else:
                mean_state, cov_diag = states.copy(), np.zeros((n, d))
            covariance = np.array([np.diag(c) for c in cov_diag])
            fields.append(TissueField(
                density=np.ones(n), mean_state=mean_state,
                covariance=covariance, heterogeneity=cov_diag.sum(axis=1),
                time=float(times[ti])))
        self.fields_ = fields
        return fields

    def relaxation_indicator(self, trajectory: KineticTrajectory) -> np.ndarray:
        """Local-equilibrium indicator per spot (0 = quasi-equilibrium)."""
        if self.fields_ is None:
            raise RuntimeError("Call compute() first.")
        all_h = np.array([f.heterogeneity for f in self.fields_])
        return (all_h / (all_h.mean() + 1e-8)).mean(axis=0)


def compute_moments(states, adjacency, n_neighbors=12) -> TissueField:
    """One-shot moment computation for a single snapshot."""
    limiter = FluidLimiter(n_neighbors=n_neighbors, use_local_kde=True)
    A = adjacency.tocsr()
    nbr_lists = _neighbor_lists(A, n_neighbors)
    n, d = states.shape
    bw = np.median(states.std(axis=0) + 1e-8) * (n ** (-1.0 / (d + 4)))
    mean_state, cov_diag = _local_moments(states, nbr_lists, bw)
    return TissueField(
        density=np.ones(n), mean_state=mean_state,
        covariance=np.array([np.diag(c) for c in cov_diag]),
        heterogeneity=cov_diag.sum(axis=1), time=0.0)


def _neighbor_lists(A, k):
    """Self + up to k neighbors per node."""
    A = A.tocsr()
    out = []
    for i in range(A.shape[0]):
        nbrs = A.indices[A.indptr[i]:A.indptr[i + 1]][:k]
        out.append(np.concatenate([[i], nbrs]))
    return out


def _local_moments(states, nbr_lists, bandwidth):
    """Weighted local mean and variance (Gaussian KDE)."""
    n, d = states.shape
    mean_state = np.zeros((n, d))
    cov_diag = np.zeros((n, d))
    bw2 = 2.0 * bandwidth ** 2 + 1e-12
    for i in range(n):
        local = states[nbr_lists[i]]
        w = np.exp(-np.sum((local - states[i]) ** 2, axis=1) / bw2)
        w /= w.sum() + 1e-12
        mean_state[i] = (w[:, None] * local).sum(axis=0)
        cov_diag[i] = (w[:, None] * (local - mean_state[i]) ** 2).sum(axis=0)
    return mean_state, cov_diag
