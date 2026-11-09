import numpy as np
import pytest

from src.data_generation import generate_trial


def test_dimensions_rank_and_observation():
    sample = generate_trial(np.random.default_rng(42), 30, 40, 4, 0.1)
    assert sample.low_rank.shape == sample.sparse.shape == sample.observed.shape == (30, 40)
    assert np.linalg.matrix_rank(sample.low_rank) == 4
    assert np.count_nonzero(sample.sparse) == 120
    assert set(np.unique(sample.sparse)) == {-10.0, 0.0, 10.0}
    np.testing.assert_array_equal(sample.observed, sample.low_rank + sample.sparse)


def test_fixed_seed_repeats_stream_without_resetting_between_trials():
    first = np.random.default_rng(23)
    second = np.random.default_rng(23)
    trial_a = generate_trial(first, 20, 20, 3, 0.2)
    trial_b = generate_trial(first, 20, 20, 3, 0.2)
    repeat_a = generate_trial(second, 20, 20, 3, 0.2)
    repeat_b = generate_trial(second, 20, 20, 3, 0.2)
    np.testing.assert_array_equal(trial_a.observed, repeat_a.observed)
    np.testing.assert_array_equal(trial_b.observed, repeat_b.observed)
    assert not np.array_equal(trial_a.sparse, trial_b.sparse)
    assert not np.array_equal(trial_a.low_rank, trial_b.low_rank)


@pytest.mark.parametrize(("density", "count"), [(0, 0), (0.09, 9), (1, 104)])
def test_support_size_rounding_and_boundaries(density, count):
    sample = generate_trial(np.random.default_rng(3), 8, 13, 2, density)
    assert np.count_nonzero(sample.sparse) == count
