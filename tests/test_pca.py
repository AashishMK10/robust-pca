import numpy as np
import pytest

from src.classical_pca import truncated_svd
from src.data_generation import generate_trial


def test_uncentered_rank_recovery_and_shape():
    sample = generate_trial(np.random.default_rng(4), 20, 15, 3, 0)
    observed = sample.observed + 2.0
    estimate = truncated_svd(observed, 4)
    assert estimate.shape == (20, 15)
    np.testing.assert_allclose(estimate, observed, atol=1e-12, rtol=0)


def test_residual_is_monotone_and_equals_spectral_tail():
    matrix = np.random.default_rng(3).normal(size=(12, 8))
    errors = [np.linalg.norm(truncated_svd(matrix, r) - matrix) for r in range(9)]
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    assert np.all(np.diff(errors) <= 1e-12)
    np.testing.assert_allclose(
        errors, [np.linalg.norm(singular_values[r:]) for r in range(9)], atol=1e-12
    )


def test_rank_above_smaller_dimension_is_rejected():
    with pytest.raises(ValueError, match="rank"):
        truncated_svd(np.zeros((4, 3)), 4)
