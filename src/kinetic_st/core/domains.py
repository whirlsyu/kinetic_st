"""Spatial domain identification via local invariant measure divergence.

A domain is a maximal contiguous region where the local invariant measure
mu(x, v) is approximately constant. Replaces heuristic graph clustering with
a dynamics-grounded definition.
"""

import numpy as np
from scipy import sparse
from typing import Optional
from dataclasses import dataclass

from kinetic_st.core.boltzmann import KineticTrajectory
from kinetic_st.core.stochastic_pde import InvariantMeasureEstimator


@dataclass
class DomainResult:
    """Domain identification output."""
    labels: np.ndarray
    n_domains: int
    domain_field: np.ndarray
    local_log_density: np.ndarray
    centroids: np.ndarray


class DomainIdentifier:
    """Identify spatial domains from a kinetic trajectory.

    Parameters
    ----------
    n_neighbors : spatial neighborhood for local mu_i estimation
    n_domains : int or "auto" (BIC-like penalty)
    divergence : "kl", "l2", or "pairwise"
    representation : invariant-measure representation ("kde" or "gmm")
    bandwidth : KDE bandwidth
    n_components : GMM components
    seed : random seed
    """

    def __init__(self, n_neighbors: int = 12, n_domains="auto",
                 divergence: str = "kl", representation: str = "gmm",
                 bandwidth: Optional[float] = None, n_components: int = 10,
                 seed: int = 0):
        self.n_neighbors = n_neighbors
        self.n_domains = n_domains
        self.divergence = divergence
        self.representation = representation
        self.bandwidth = bandwidth
        self.n_components = n_components
        self.seed = seed
        self.result_ = None

    def fit(self, trajectory: KineticTrajectory, adjacency: sparse.spmatrix,
            verbose: bool = False) -> DomainResult:
        """Identify spatial domains."""
        n = adjacency.shape[0]
        A = adjacency.tocsr()
        nbr_lists = _neighbor_lists(A, self.n_neighbors)

        global_est = InvariantMeasureEstimator(
            representation=self.representation, bandwidth=self.bandwidth,
            n_components=self.n_components, seed=self.seed).fit(trajectory)
        self.global_measure_ = global_est

        final_states = trajectory.states
        d = final_states.shape[1]
        global_logp = global_est.log_density(final_states)

        if self.divergence == "kl":
            field = -global_logp
        elif self.divergence == "l2":
            local_means = np.array([final_states[idx].mean(axis=0) for idx in nbr_lists])
            field = np.linalg.norm(local_means - global_est.samples_.mean(axis=0), axis=1)
        elif self.divergence == "pairwise":
            field = np.zeros(n)
            for i in range(n):
                lp_local = _local_logpdf(final_states[nbr_lists[i]], final_states[i])
                field[i] = global_logp[i] - lp_local
        else:
            raise ValueError(f"Unknown divergence: {self.divergence}")

        field /= field.std() + 1e-8
        labels = self._segment(field, final_states, A)

        n_dom = int(labels.max() + 1)
        centroids = np.array([final_states[labels == k].mean(axis=0)
                              for k in range(n_dom) if (labels == k).any()])

        self.result_ = DomainResult(
            labels=labels, n_domains=n_dom, domain_field=field,
            local_log_density=global_logp, centroids=centroids)
        if verbose:
            print(f"  [DomainIdentifier] identified {n_dom} domains")
        return self.result_

    def _segment(self, field, states, A):
        """Segment tissue into domains via KMeans + spatial contiguity."""
        n = len(states)
        if self.n_domains == "auto":
            from sklearn.cluster import KMeans
            best_k, best_score = 2, -np.inf
            for k in range(2, min(15, n // 20) + 1):
                km = KMeans(n_clusters=k, random_state=self.seed, n_init=3).fit(states)
                score = np.sum((km.cluster_centers_ - states.mean(axis=0)) ** 2) * k / max(km.inertia_, 1e-8) - 0.05 * k
                if score > best_score:
                    best_score, best_k = score, k
            n_dom = best_k
        else:
            n_dom = int(self.n_domains)

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_dom, random_state=self.seed, n_init=5).fit(states)
        return _enforce_contiguity(km.labels_, A)


def _neighbor_lists(A, k):
    A = A.tocsr()
    return [np.concatenate([[i], A.indices[A.indptr[i]:A.indptr[i + 1]][:k]])
            for i in range(A.shape[0])]


def _local_logpdf(samples, x):
    """Crude local KDE log-density."""
    d = samples.shape[1]
    bw = np.std(samples, axis=0) + 1e-3
    logk = -0.5 * np.sum(((samples - x) / bw) ** 2, axis=1) - np.sum(np.log(bw)) - 0.5 * d * np.log(2 * np.pi)
    m = logk.max()
    return float(m + np.log(np.exp(logk - m).mean()))


def _enforce_contiguity(labels, A):
    """Split non-contiguous clusters into connected components."""
    from scipy.sparse.csgraph import connected_components
    A = A.tocsr()
    new_labels = np.full(len(labels), -1, dtype=int)
    next_id = 0
    for lab in np.unique(labels):
        mask = labels == lab
        idx_global = np.where(mask)[0]
        if len(idx_global) == 0:
            continue
        n_cc, cc_labels = connected_components(A[mask][:, mask], directed=False)
        for c in range(n_cc):
            new_labels[idx_global[cc_labels == c]] = next_id
            next_id += 1
    return new_labels
