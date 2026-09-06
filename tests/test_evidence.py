"""The two drawing primitives and the sampling grid — pure array maths."""

import math

import numpy as np
import pytest

from docupilot.segmentation import evidence as ev


def test_grid_spans_the_duration_at_the_declared_rate():
    g = ev.grid(2.0, hz=50.0)
    assert len(g) == 100
    assert g[0] == 0.0
    assert g[1] - g[0] == pytest.approx(1 / 50.0)
    assert ev.grid(0.0).size == 0
    assert ev.grid(-1.0).size == 0


def test_gaussian_is_centred_and_its_width_is_spread_over_two():
    score = np.zeros(101, dtype=np.float32)
    ev.apply_gaussian(score, center=50, value=1.0, spread=20)
    assert score[50] == pytest.approx(1.0)
    # sigma = spread/2, so one sigma out the value has fallen to exp(-0.5).
    assert score[60] == pytest.approx(math.exp(-0.5), abs=1e-6)
    assert score[40] == pytest.approx(score[60], abs=1e-7)     # symmetric
    assert score[29] == 0.0 and score[71] == 0.0               # cut at +/- spread


def test_overlapping_peaks_take_the_maximum_not_the_sum():
    score = np.zeros(50, dtype=np.float32)
    ev.apply_gaussian(score, 25, 0.6, 10)
    ev.apply_gaussian(score, 25, 0.3, 10)
    # Evidence must not accumulate: two weak judgements next to each other would
    # otherwise claim a boundary neither of them asserts.
    assert score[25] == pytest.approx(0.6)


def test_gaussian_outside_the_curve_is_ignored():
    score = np.zeros(10, dtype=np.float32)
    ev.apply_gaussian(score, -1, 1.0, 5)
    ev.apply_gaussian(score, 10, 1.0, 5)
    assert not score.any()


def test_window_is_zero_at_both_edges_and_peaks_where_asked():
    score = np.zeros(200, dtype=np.float32)
    peak = ev.apply_window(score, lo=20, hi=120, peak=95, value=0.8)
    assert peak == 95
    assert score[95] == pytest.approx(0.8)
    assert score[20] == 0.0 and score[120] == 0.0
    # Zero at the edges is what keeps adjacent windows from fusing into one
    # plateau, which would yield one candidate per RUN of steps instead of per step.
    assert np.all(np.diff(score[20:96]) >= -1e-7)      # rises to the peak
    assert np.all(np.diff(score[95:121]) <= 1e-7)      # falls after it


def test_window_peak_is_clamped_into_the_interval():
    score = np.zeros(100, dtype=np.float32)
    assert ev.apply_window(score, lo=10, hi=50, peak=90, value=0.5) == 50
    assert ev.apply_window(score, lo=10, hi=50, peak=0, value=0.5) == 10


def test_window_ignores_empty_intervals_and_zero_evidence():
    score = np.zeros(100, dtype=np.float32)
    ev.apply_window(score, lo=50, hi=40, peak=45, value=0.9)
    ev.apply_window(score, lo=10, hi=20, peak=15, value=0.0)
    assert not score.any()


def test_empty_evidence_is_well_formed():
    e = ev.BoundaryEvidence.empty()
    assert e.times_s.shape == (0,) and e.score.shape == (0,) and e.boundaries_s == []
