"""Solve equality-constrained principal component pursuit with scaled ADMM."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import svd

Matrix = NDArray[np.float64]
DEFAULT_TOLERANCE = 1e-7
DEFAULT_MAX_ITERATIONS = 5000
PENALTY_SCALE = 0.3


@dataclass(frozen=True)
class Iteration:
    """Relative primal and dual stationarity residuals after one update."""

    iteration: int
    relative_primal: float
    relative_dual: float


@dataclass(frozen=True)
class PCPResult:
    """Recovered components and solver diagnostics."""

    low_rank: Matrix
    sparse: Matrix
    rank: int
    penalty: float
    regularization: float
    converged: bool
    history: tuple[Iteration, ...]


def soft_threshold(matrix: ArrayLike, threshold: float) -> Matrix:
    """Apply the proximal map of the scaled elementwise L1 norm."""
    values = np.asarray(matrix, dtype=np.float64)
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def singular_value_threshold(matrix: Matrix, threshold: float) -> tuple[Matrix, int]:
    """Shrink the singular values and return the matrix and retained rank."""
    u, values, vt = svd(matrix, full_matrices=False, check_finite=False)
    kept = int(np.count_nonzero(values > threshold))
    return (u[:, :kept] * (values[:kept] - threshold)) @ vt[:kept], kept


def principal_component_pursuit(
    matrix: ArrayLike,
    *,
    regularization: float | None = None,
    penalty: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> PCPResult:
    """Minimize ||L||_* + lambda ||S||_1 subject to M = L + S.

    The scaled dual ascends on M - L - S. Stationarity in the L block
    is monitored by the norm of mu times the change in S. Both normalized
    residuals must pass the tolerance. The zero input has an exact solution.
    """
    observed = np.asarray(matrix, dtype=np.float64)
    if observed.ndim != 2 or not observed.size or not np.isfinite(observed).all():
        raise ValueError("expected a nonempty finite matrix")
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("expected a positive tolerance and iteration limit")

    norm_observed = float(np.linalg.norm(observed, "fro"))
    rms = norm_observed / np.sqrt(observed.size)
    lam = float(1.0 / np.sqrt(max(observed.shape)) if regularization is None else regularization)
    mu = float(PENALTY_SCALE / (rms or 1.0) if penalty is None else penalty)
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(mu) or mu <= 0:
        raise ValueError("regularization and penalty must be finite and positive")

    sparse = np.zeros_like(observed)
    dual = np.zeros_like(observed)
    if norm_observed == 0:
        return PCPResult(dual, sparse, 0, mu, lam, True, (Iteration(0, 0.0, 0.0),))

    history = []
    converged = False
    for iteration in range(1, max_iterations + 1):
        low_rank, rank = singular_value_threshold(observed - sparse + dual, 1.0 / mu)
        next_sparse = soft_threshold(observed - low_rank + dual, lam / mu)
        residual = observed - low_rank - next_sparse
        dual += residual

        relative_primal = float(np.linalg.norm(residual, "fro") / norm_observed)
        dual_scale = max(1.0, mu * float(np.linalg.norm(dual, "fro")))
        relative_dual = float(mu * np.linalg.norm(next_sparse - sparse, "fro") / dual_scale)
        history.append(Iteration(iteration, relative_primal, relative_dual))
        sparse = next_sparse
        if max(relative_primal, relative_dual) <= tolerance:
            converged = True
            break

    return PCPResult(low_rank, sparse, rank, mu, lam, converged, tuple(history))
