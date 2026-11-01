"""Latent embedding of cell states via PCA on log-CPM expression."""

import numpy as np
from typing import Optional, Tuple
from scipy import sparse


class LatentEmbedding:
    """Embed gene expression into a low-dimensional cell-state space.

    Serves as the "velocity variable" of the kinetic equation.

    Parameters
    ----------
    n_components : latent dimension (default 16)
    method : "pca" (only option)
    normalize : standardize latent dims to unit variance (default True)
    """

    def __init__(self, n_components: int = 16, method: str = "pca",
                 normalize: bool = True):
        self.n_components = n_components
        self.method = method
        self.normalize = normalize
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.scale_ = None
        self._fitted = False

    def fit(self, Y, gene_names: Optional[np.ndarray] = None) -> "LatentEmbedding":
        """Fit on expression matrix Y (n_spots, n_genes)."""
        Yn = _logcpm(Y)
        Yd = np.asarray(Yn.toarray() if sparse.issparse(Yn) else Yn, dtype=np.float64)
        self._fit_pca(Yd)
        self.gene_names_ = gene_names
        self._fitted = True
        return self

    def transform(self, Y) -> np.ndarray:
        """Project expression matrix into latent space."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        Yn = _logcpm(Y)
        Yd = np.asarray(Yn.toarray() if sparse.issparse(Yn) else Yn, dtype=np.float64)
        Z = (Yd - self.mean_) @ self.components_.T
        if self.normalize:
            Z = Z / self.scale_[np.newaxis, :]
        return Z

    def fit_transform(self, Y, gene_names: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(Y, gene_names)
        return self.transform(Y)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map latent states back to gene-expression space."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        if self.normalize:
            Z = Z * self.scale_[np.newaxis, :]
        return Z @ self.components_ + self.mean_

    def _fit_pca(self, Yd):
        k = min(self.n_components, *Yd.shape)
        self.mean_ = Yd.mean(axis=0)
        U, s, Vt = np.linalg.svd(Yd - self.mean_, full_matrices=False)
        k = min(k, len(s))
        self.components_ = Vt[:k]
        self.explained_variance_ = (s[:k] ** 2) / max(Yd.shape[0] - 1, 1)
        self.scale_ = np.sqrt(self.explained_variance_) + 1e-8 if self.normalize \
            else np.ones(k)


def fit_latent(Y, n_components: int = 16, method: str = "pca"):
    """Fit and return (embedder, latent codes)."""
    emb = LatentEmbedding(n_components=n_components, method=method)
    Z = emb.fit_transform(Y)
    return emb, Z


def _logcpm(Y):
    """Log-CPM normalization."""
    if sparse.issparse(Y):
        lib = np.maximum(np.asarray(Y.sum(axis=1)).flatten(), 1.0)
        Yn = sparse.diags(1e4 / lib) @ Y
        Yn.data = np.log1p(Yn.data)
        return Yn
    Yd = np.asarray(Y, dtype=np.float64)
    lib = np.maximum(Yd.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(Yd / lib * 1e4)
