import numpy as np
import pytest

from src.metrics import relative_frobenius, support_metrics


def test_relative_frobenius():
    assert relative_frobenius([[3.0, 3.0]], [[3.0, 4.0]]) == pytest.approx(0.2)


def test_relative_frobenius_rejects_zero_reference():
    with pytest.raises(ValueError, match="nonzero"):
        relative_frobenius(np.zeros((3, 3)), np.zeros((3, 3)))


def test_support_detection_rates():
    scores = support_metrics(
        [True, True, True, False, False, False], [True, True, False, True, True, False]
    )
    assert (scores.tp, scores.fp, scores.fn, scores.tn) == (2, 2, 1, 1)
    assert (scores.precision, scores.recall, scores.f1, scores.fpr) == pytest.approx(
        (1 / 2, 2 / 3, 4 / 7, 2 / 3)
    )


def test_empty_truth_and_predictions():
    scores = support_metrics([False, False], [False, False])
    assert np.isnan(scores.precision) and np.isnan(scores.recall) and np.isnan(scores.f1)
    assert scores.fpr == 0


def test_empty_truth_with_false_alarm():
    scores = support_metrics([False, False], [True, False])
    assert scores.precision == 0 and scores.fpr == 0.5
    assert np.isnan(scores.recall) and np.isnan(scores.f1)
