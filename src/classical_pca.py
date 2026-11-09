"""Uncentered truncated SVD for the known-rank matrix recovery baseline."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import svd


def truncated_svd(matrix: ArrayLike, rank: int) -> NDArray[np.float64]:
    """Return a best rank-at-most-rank approximation in Frobenius norm."""
    observed = np.asarray(matrix, dtype=np.float64)
    if observed.ndim != 2 or not 0 <= rank <= min(observed.shape):
        raise ValueError("expected a matrix and a rank within its dimensions")
    u, values, vt = svd(observed, full_matrices=False)
    return (u[:, :rank] * values[:rank]) @ vt[:rank]
