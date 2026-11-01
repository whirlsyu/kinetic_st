"""Spatial graph construction from spot coordinates."""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from typing import Optional


def coords_to_adjacency(
    coords: np.ndarray,
    method: str = "knn",
    k: int = 6,
    radius: Optional[float] = None,
) -> sparse.spmatrix:
    """Build a sparse spatial adjacency matrix from coordinates.

    Parameters
    ----------
    coords : (n_spots, 2) or (n_spots, 3)
    method : "knn", "radius", or "grid"
    k : neighbors for KNN
    radius : radius for radius-based graph

    Returns
    -------
    A : sparse.csr_matrix (n_spots, n_spots), binary
    """
    coords = np.asarray(coords, dtype=np.float64)
    if method == "grid":
        return _grid_adjacency(coords)
    elif method == "radius":
        if radius is None:
            raise ValueError("radius required for method='radius'.")
        return _radius_adjacency(coords, radius)
    elif method == "knn":
        return _knn_adjacency(coords, k)
    else:
        raise ValueError(f"Unknown method: {method}")


def _knn_adjacency(coords, k):
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    rows, cols = [], []
    for i in range(len(coords)):
        for j in indices[i, 1:]:
            rows.append(i)
            cols.append(int(j))
    A = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(len(coords), len(coords)),
    )
    return A.maximum(A.T).tocsr()


def _radius_adjacency(coords, radius):
    tree = cKDTree(coords)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return sparse.csr_matrix((len(coords), len(coords)))
    A = sparse.coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
        shape=(len(coords), len(coords)),
    )
    return A.tocsr()


def _grid_adjacency(coords):
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    delta = np.median(dists[:, 1])
    return _radius_adjacency(coords, delta * 1.5)
