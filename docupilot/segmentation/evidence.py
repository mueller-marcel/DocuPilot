"""
What every modality produces, and the two ways of drawing evidence onto a curve.

This module is the segmentation package's shared vocabulary. It knows nothing
about screens, microphones or keyboards — which is the point: the three modality
modules must not learn about each other, and a shared type is the only thing they
are allowed to have in common.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sampling grid for modalities that have no frames of their own (audio, events).
# Video uses its own frame rate. The value is arbitrary and only sets the
# quantisation of a reported boundary (±10 ms here) — orders of magnitude below
# what any modality can actually resolve.
GRID_HZ = 50.0

# At or above this probability a modality commits to a boundary. One threshold,
# applied once per modality, inside the extractor that owns the evidence.
BOUNDARY_THRESHOLD = 0.5


@dataclass(frozen=True)
class BoundaryEvidence:
    """
    One modality's answer to "where are the action boundaries?".

    `score` over `times_s` is the graded evidence curve — for display, and the
    raw material any later analysis aggregates. `boundaries_s` is what the
    modality actually commits to.

    Each extractor derives its own boundaries, because only it knows what its
    curve MEANS. A video verdict is a spike at the frame the screen settled on;
    an audio verdict is a wide bump over an announced step, whose peak is the
    estimate; an events verdict sits on the last input of a burst. Reading
    boundaries off all three with one rule needs a rule that fits none of them.
    """

    times_s: np.ndarray          # (T,) float64 — timestamp per sample
    score: np.ndarray            # (T,) float32 — graded evidence in [0, 1]
    boundaries_s: list[float]

    @classmethod
    def empty(cls) -> BoundaryEvidence:
        return cls(
            times_s=np.zeros(0, dtype=np.float64),
            score=np.zeros(0, dtype=np.float32),
            boundaries_s=[],
        )


def grid(duration_s: float, hz: float = GRID_HZ) -> np.ndarray:
    """Timestamps of a `hz`-spaced grid spanning [0, duration_s]."""
    return np.arange(max(0, int(round(duration_s * hz))), dtype=np.float64) / hz


def apply_gaussian(score: np.ndarray, center: int, value: float, spread: int) -> None:
    """
    Write a symmetric Gaussian peak into `score`, centred on `center`.

    np.maximum so overlapping peaks don't cancel — only the stronger wins.
    """
    n = len(score)
    if not 0 <= center < n:
        return
    lo, hi = max(0, center - spread), min(n, center + spread + 1)
    offsets = np.arange(lo, hi) - center
    g = np.exp(-0.5 * (offsets / max(spread / 2.0, 1.0)) ** 2)
    score[lo:hi] = np.maximum(score[lo:hi], (value * g).astype(np.float32))


def apply_window(score: np.ndarray, lo: int, hi: int, peak: int, value: float) -> int:
    """
    Write a raised-cosine bump over [lo, hi] peaking at `peak`, scaled by `value`.

    Zero at both edges, `value` at the peak, asymmetric when the peak is off
    centre. np.maximum, same rule as apply_gaussian.

    :return: the clamped peak index — the window can be cut short by the end of
        the recording, and the caller needs the index that was actually written,
        not the one it asked for.
    """
    n = len(score)
    lo, hi = max(0, lo), min(n - 1, hi)
    if hi <= lo or value <= 0.0:
        return min(max(peak, 0), max(n - 1, 0))
    peak = min(max(peak, lo), hi)

    x = np.arange(lo, hi + 1, dtype=np.float64)
    shape = np.zeros_like(x)

    rise = x <= peak
    if peak > lo:
        shape[rise] = 0.5 * (1.0 - np.cos(np.pi * (x[rise] - lo) / (peak - lo)))
    else:
        shape[rise] = 1.0                      # peak sits on the left edge

    fall = ~rise
    if hi > peak:
        shape[fall] = 0.5 * (1.0 + np.cos(np.pi * (x[fall] - peak) / (hi - peak)))

    score[lo:hi + 1] = np.maximum(score[lo:hi + 1], (value * shape).astype(np.float32))
    return peak
