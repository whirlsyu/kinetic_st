"""Supervised marker discovery on the kinetic-smoothed expression field.

Scores markers on the dynamically denoised macroscopic field recovered by
moment closure, rather than the raw noisy log-CPM matrix. The one-vs-rest
z-score is identical to classical methods; only the expression source differs.
"""

from typing import Dict, List, Optional
import numpy as np


def kinetic_marker_scores(smooth_expr: np.ndarray, labels: np.ndarray):
    """One-vs-rest z-score of every gene for every label.

    Parameters
    ----------
    smooth_expr : (n_spots, n_genes) smoothed expression
    labels : (n_spots,) group labels

    Returns
    -------
    scores : (n_types, n_genes) z-scores
    unique_labels : (n_types,)
    """
    smooth_expr = np.asarray(smooth_expr, dtype=np.float64)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    H = np.array([smooth_expr[labels == l].mean(axis=0) for l in unique_labels])
    gene_std = smooth_expr.std(axis=0) + 1e-10
    scores = np.array([(H[k] - (H.sum(axis=0) - H[k]) / max(K - 1, 1)) / gene_std
                        for k in range(K)])
    return scores, unique_labels


class KineticMarkerFinder:
    """Rank marker genes per group on the kinetic-smoothed expression field.

    Parameters
    ----------
    n_top : top markers per group (default 20)
    """

    def __init__(self, n_top: int = 20):
        self.n_top = n_top
        self.markers_: Dict[str, List[str]] = {}
        self.scores_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None

    def fit(self, smooth_expr: np.ndarray, labels: np.ndarray,
            gene_names: Optional[np.ndarray] = None) -> "KineticMarkerFinder":
        smooth_expr = np.asarray(smooth_expr, dtype=np.float64)
        n_genes = smooth_expr.shape[1]
        gene_arr = (np.asarray([str(g) for g in gene_names])
                    if gene_names is not None
                    else np.array([f"G_{i}" for i in range(n_genes)]))
        scores, unique_labels = kinetic_marker_scores(smooth_expr, labels)
        self.scores_ = scores
        self.labels_ = unique_labels
        self.markers_ = {str(l): [str(gene_arr[i]) for i in np.argsort(scores[k])[::-1][:self.n_top]]
                          for k, l in enumerate(unique_labels)}
        return self

    def score_for(self, label, gene_index: int) -> float:
        """z-score of a gene for a given label."""
        if self.scores_ is None:
            raise RuntimeError("Not fitted.")
        return float(self.scores_[list(self.labels_).index(label), gene_index])


def find_kinetic_markers(smooth_expr, labels, gene_names=None, n_top=20):
    """Return {label: [top genes]}."""
    return KineticMarkerFinder(n_top=n_top).fit(smooth_expr, labels, gene_names).markers_


def markers_from_model(model, labels, gene_names=None, n_top=20, time_idx=-1):
    """Compute markers directly from a fitted KineticST model.

    Reads the macroscopic field at time_idx, decodes to gene space, and ranks.
    """
    if not getattr(model, "_fitted", False):
        raise RuntimeError("Model is not fitted.")
    smooth = model.decode_to_genes(model.fields_[time_idx].mean_state)
    return find_kinetic_markers(smooth, labels, gene_names, n_top=n_top)
