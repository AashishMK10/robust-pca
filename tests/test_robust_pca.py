import numpy as np

from src.robust_pca import (
    principal_component_pursuit,
    singular_value_threshold,
    soft_threshold,
)


def test_soft_threshold():
    np.testing.assert_array_equal(
        soft_threshold([[-3, -1, 0, 1, 4]], 1), [[-2, 0, 0, 0, 3]]
    )


def test_singular_value_threshold():
    low_rank, rank = singular_value_threshold(np.diag([5.0, -3.0, 1.0]), 2.0)
    np.testing.assert_allclose(low_rank, np.diag([3, -1, 0]), atol=1e-12)
    assert rank == 2


def test_zero_matrix():
    zero = np.zeros((5, 7))
    result = principal_component_pursuit(zero)
    assert result.converged and result.rank == 0 and result.history[-1].iteration == 0
    np.testing.assert_array_equal(result.low_rank, zero)
    np.testing.assert_array_equal(result.sparse, zero)


def test_convergence_and_recovery_for_incoherent_rank_one():
    low_rank = np.ones((24, 24))
    sparse = np.zeros_like(low_rank)
    sparse[[0, 5, 13, 21], [18, 11, 2, 7]] = [10, -10, 10, -10]
    observed = low_rank + sparse
    tolerance = 1e-8
    result = principal_component_pursuit(observed, tolerance=tolerance)
    final = result.history[-1]
    assert result.converged and result.rank == 1
    assert final.relative_primal <= tolerance and final.relative_dual <= tolerance
    np.testing.assert_allclose(
        final.relative_primal,
        np.linalg.norm(observed - result.low_rank - result.sparse) / np.linalg.norm(observed),
    )
    np.testing.assert_allclose(result.low_rank, low_rank, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result.sparse, sparse, atol=2e-6, rtol=0)


def test_primal_pass_without_dual_pass_is_not_convergence():
    # With a very large penalty, the first update fits M almost exactly,
    # but the first change in S still gives a large dual residual.
    matrix = np.ones((12, 12))
    matrix[0, 0] = 21
    result = principal_component_pursuit(matrix, penalty=1e5, tolerance=1e-6, max_iterations=1)
    final = result.history[-1]
    assert final.relative_primal < 1e-6 < final.relative_dual
    assert not result.converged and final.iteration == 1
