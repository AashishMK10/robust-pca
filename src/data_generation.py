"""Generate Gaussian low-rank matrices and uniformly located gross errors."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticTrial:
    """Ground truth, sparse errors, and the observed matrix for a trial."""

    low_rank: Matrix
    sparse: Matrix
    observed: Matrix


def generate_trial(
    rng: np.random.Generator,
    m: int,
    n: int,
    rank: int,
    density: float,
    magnitude: float = 10.0,
) -> SyntheticTrial:
    """Draw independent Gaussian factors and a fixed-size signed support."""
    if not 1 <= rank <= min(m, n):
        raise ValueError("rank must lie between 1 and the smaller dimension")
    if not 0 <= density <= 1:
        raise ValueError("density must lie in [0, 1]")
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("magnitude must be finite and positive")

    left = rng.standard_normal((m, rank))
    right = rng.standard_normal((n, rank))
    low_rank = (left @ right.T) / np.sqrt(rank)

    count = round(density * m * n)
    support = rng.choice(m * n, size=count, replace=False)
    sparse = np.zeros((m, n), dtype=np.float64)
    sparse.flat[support] = magnitude * rng.choice((-1.0, 1.0), size=count)
    return SyntheticTrial(low_rank, sparse, low_rank + sparse)
