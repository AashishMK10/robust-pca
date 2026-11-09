"""Relative recovery errors and support detection metrics."""

from dataclasses import dataclass
from math import nan

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Detection:
    """Confusion counts and derived metrics at a fixed detection threshold."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    fpr: float


def relative_frobenius(estimate: ArrayLike, truth: ArrayLike) -> float:
    """Return relative Frobenius error for a nonzero reference matrix."""
    reference = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if estimate.shape != reference.shape:
        raise ValueError("estimate and truth must have the same shape")
    scale = np.linalg.norm(reference, "fro")
    if scale == 0:
        raise ValueError("relative error requires nonzero ground truth")
    return float(np.linalg.norm(estimate - reference, "fro") / scale)


def support_metrics(truth: ArrayLike, predicted: ArrayLike) -> Detection:
    """Score two masks. Recall and F1 are undefined when truth has no positives."""
    actual = np.asarray(truth, dtype=bool)
    estimated = np.asarray(predicted, dtype=bool)
    if actual.shape != estimated.shape:
        raise ValueError("truth and predicted masks must have the same shape")
    tp = int(np.count_nonzero(actual & estimated))
    fp = int(np.count_nonzero(~actual & estimated))
    fn = int(np.count_nonzero(actual & ~estimated))
    tn = int(np.count_nonzero(~actual & ~estimated))
    return Detection(
        tp, fp, fn, tn,
        precision=tp / (tp + fp) if tp + fp else nan,
        recall=tp / (tp + fn) if tp + fn else nan,
        f1=2 * tp / (2 * tp + fp + fn) if tp + fn else nan,
        fpr=fp / (fp + tn) if fp + tn else nan,
    )
