"""Ligand-receptor collision operator on latent cell states.

A "collision" is a ligand-receptor-mediated interaction updating two
neighboring cell states. Irreversibility is baked in via attractor bias
(gamma), since biological interactions are intrinsically irreversible.
"""

import numpy as np
from typing import Optional, Tuple
from scipy import sparse


class LigandReceptorRule:
    """Collision rule: v' = (1-a)*v + b*v* + g*a(v), and vice versa.

    Parameters
    ----------
    alpha : self-relaxation rate (default 0.1)
    beta : cross-influence rate (default 0.2)
    gamma : attractor bias, irreversible differentiation (default 0.05)
    attractors : (n_types, d) cell-type centroids; None = irreversibility off
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.2,
                 gamma: float = 0.05, attractors: Optional[np.ndarray] = None):
        for v in (alpha, beta, gamma):
            if not 0.0 <= v <= 1.0:
                raise ValueError("alpha, beta, gamma must be in [0, 1].")
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.attractors = attractors

    def apply(self, v: np.ndarray, vs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply collision to cell pairs. v, vs: (m, d) -> vp, vsp: (m, d)."""
        vp = (1.0 - self.alpha) * v + self.beta * vs
        vsp = (1.0 - self.alpha) * vs + self.beta * v
        if self.gamma > 0 and self.attractors is not None:
            vp += self.gamma * (_nearest_attractor(vp, self.attractors) - vp)
            vsp += self.gamma * (_nearest_attractor(vsp, self.attractors) - vsp)
        return vp, vsp


class CollisionKernel:
    """Collision operator combining a rule with a cross-section sigma(v, v*).

    sigma = sigma_LR * sigma_space, where sigma_LR depends on ligand-receptor
    expression and sigma_space on spatial adjacency.

    Parameters
    ----------
    rule : LigandReceptorRule
    lr_matrix : (n_lr,) per-pair strengths; None = all 1
    lr_index : (ligand_idx, receptor_idx) for expression-based sigma
    sigma_scale : overall collision rate scale
    """

    def __init__(self, rule: LigandReceptorRule,
                 lr_matrix: Optional[np.ndarray] = None,
                 lr_index: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 sigma_scale: float = 1.0):
        self.rule = rule
        self.lr_matrix = lr_matrix
        self.lr_index = lr_index
        self.sigma_scale = sigma_scale

    def cross_section(self, states, adjacency, Y_lr=None) -> sparse.spmatrix:
        """Sparse cross-section matrix over spatial edges."""
        A = adjacency.tocoo()
        n = adjacency.shape[0]
        if self.lr_index is not None and Y_lr is not None:
            lig_idx, rec_idx = self.lr_index
            w = np.asarray((Y_lr[A.row][:, lig_idx] * Y_lr[A.col][:, rec_idx]).sum(axis=1)).flatten()
        else:
            d2 = np.sum((states[A.row] - states[A.col]) ** 2, axis=1)
            w = np.exp(-0.5 * d2 / (np.median(d2) + 1e-8))
        if self.lr_matrix is not None:
            w *= float(np.mean(self.lr_matrix))
        w *= self.sigma_scale
        return sparse.coo_matrix((w, (A.row, A.col)), shape=(n, n)).tocsr()

    def apply_collisions(self, states, adjacency, Y_lr=None,
                         n_pairs_per_spot=4, rng=None) -> np.ndarray:
        """One Monte-Carlo collision step on all cells."""
        rng = rng or np.random.RandomState(0)
        n, d = states.shape
        Sigma = self.cross_section(states, adjacency, Y_lr).tocsr()
        new_states = states.copy()
        for i in range(n):
            s, e = Sigma.indptr[i], Sigma.indptr[i + 1]
            if e == s:
                continue
            nbrs, w = Sigma.indices[s:e], Sigma.data[s:e]
            wsum = w.sum()
            if wsum <= 0:
                continue
            m = min(n_pairs_per_spot, len(nbrs))
            chosen = rng.choice(nbrs, size=m, replace=True, p=w / wsum)
            vp, _ = self.rule.apply(
                np.broadcast_to(states[i], (m, d)), states[chosen])
            new_states[i] = vp.mean(axis=0)
        return new_states


def _nearest_attractor(states, attractors):
    """Nearest attractor centroid for each state."""
    d2 = ((states[:, None, :] - attractors[None, :, :]) ** 2).sum(axis=2)
    return attractors[np.argmin(d2, axis=1)]
