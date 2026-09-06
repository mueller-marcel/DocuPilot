"""The measuring instrument: one-to-one matching, and the chance floor."""

import numpy as np
import pytest

from docupilot.evaluation import metrics


def test_perfect_and_empty_predictions():
    gt = [1.0, 5.0, 9.0]
    assert metrics.match(gt, gt, 0.5).f1 == 1.0
    empty = metrics.match(gt, [], 1.0)
    assert (empty.tp, empty.fp, empty.fn) == (0, 0, 3) and empty.f1 == 0.0
    spurious = metrics.match([], [1.0, 2.0], 1.0)
    assert (spurious.tp, spurious.fp, spurious.fn) == (0, 2, 0)


def test_a_prediction_beyond_the_tolerance_is_a_miss():
    assert metrics.match([10.0], [10.9], 1.0).tp == 1
    assert metrics.match([10.0], [11.1], 1.0).tp == 0


def test_one_prediction_cannot_satisfy_two_boundaries():
    # Without the one-to-one rule this would score recall 1.0 on a single guess.
    m = metrics.match([10.0, 10.4], [10.2], 1.0)
    assert (m.tp, m.fp, m.fn) == (1, 0, 1)


def test_matching_maximises_hits_rather_than_taking_the_nearest_first():
    # Greedy nearest-first pairs 10.0 with 10.1 and then loses 10.6, scoring one
    # hit. The optimal assignment scores two.
    m = metrics.match([10.0, 10.6], [10.1, 11.2], 1.0)
    assert m.tp == 2
    assert sorted(round(p, 1) for _, p in m.pairs) == [10.1, 11.2]


def test_pairs_report_which_boundary_was_found():
    m = metrics.match([2.0, 8.0], [8.2], 0.5)
    assert m.pairs == [(8.0, 8.2)]


def test_precision_recall_f1_follow_from_the_counts():
    m = metrics.match([1.0, 5.0, 9.0], [1.1, 5.1, 20.0], 0.5)
    assert (m.tp, m.fp, m.fn) == (2, 1, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)


def test_tolerance_must_be_positive():
    with pytest.raises(ValueError):
        metrics.match([1.0], [1.0], 0.0)


def test_chance_level_grows_with_tolerance_and_is_reproducible():
    gt, duration = [10.0, 30.0, 60.0, 90.0], 120.0
    levels = [metrics.chance_level(gt, duration, tau, n_draws=200) for tau in (0.5, 2.0, 5.0)]
    assert levels == sorted(levels)
    assert all(0.0 <= v <= 1.0 for v in levels)
    assert metrics.chance_level(gt, duration, 1.0, n_draws=200) == \
        metrics.chance_level(gt, duration, 1.0, n_draws=200)
    assert metrics.chance_level([], 10.0, 1.0) == 0.0
    assert metrics.chance_level(gt, 0.0, 1.0) == 0.0


def test_chance_level_is_far_below_a_real_detector():
    # The floor exists so an F1 can be read at all; on a sparse corpus it must
    # stay small, otherwise the reported F1 says nothing.
    gt = np.linspace(5, 115, 8).tolist()
    assert metrics.chance_level(gt, 120.0, 1.0, n_draws=300) < 0.25
