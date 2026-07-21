"""
How a set of modalities becomes one prediction: collect the moments they propose,
describe each with what those modalities say about it, then decide.

Everything here reads only the modalities of the subset it was given. A single
look at an excluded modality would make the Shapley values measure the leak.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks

from docupilot.segmentation.evidence import BoundaryEvidence

# How far around a candidate a modality's score is read. Set to the primary
# tolerance: the events arm anchors a candidate at the end of an input burst
# while the annotation sits at the visual settling that follows, so a point
# sample would miss a score that is plainly there. Absorbs that offset without
# fitting it to the data.
FEATURE_WINDOW_S = 1.0

# Two predictions closer than this cannot both be hits anyway — matching is
# one-to-one — so the weaker one is suppressed.
SUPPRESS_RADIUS_S = 1.0

Evidence = Mapping[str, BoundaryEvidence]


def candidate_times(evidence: Evidence, subset: Sequence[str]) -> np.ndarray:
    """
    The moments the given modalities propose: every local maximum of their score
    curves, pooled and sorted.

    Peaks rather than the modalities' own `boundaries_s`, which are already
    thresholded: taking those would let the classifier remove boundaries but
    never add one, capping recall at each modality's internal threshold.

    :param evidence: one BoundaryEvidence per modality.
    :param subset: which modalities may contribute — no others are read.
    :return: candidate timestamps in seconds, ascending. Empty for an empty subset.
    """
    times: list[float] = []
    for modality in subset:
        ev = evidence[modality]
        if len(ev.score) < 3:
            continue
        peaks, _ = find_peaks(ev.score)
        times.extend(float(ev.times_s[i]) for i in peaks)
    return np.sort(np.asarray(times, dtype=np.float64))


def _window_max(times_s: np.ndarray, score: np.ndarray, t: float, window_s: float) -> float:
    """Highest score within +/- window_s of t; 0.0 when the window is empty."""
    lo = int(np.searchsorted(times_s, t - window_s, side="left"))
    hi = int(np.searchsorted(times_s, t + window_s, side="right"))
    return float(score[lo:hi].max()) if hi > lo else 0.0


def feature_matrix(
    times: np.ndarray,
    evidence: Evidence,
    subset: Sequence[str],
    window_s: float = FEATURE_WINDOW_S,
) -> np.ndarray:
    """
    One row per candidate, one column per modality in the subset.

    :param times: candidate timestamps in seconds.
    :param evidence: one BoundaryEvidence per modality.
    :param subset: which modalities become columns, in the given order.
    :param window_s: half-width of the window a score is read over.
    :return: array of shape (len(times), len(subset)).
    """
    if len(times) == 0 or not subset:
        return np.zeros((len(times), len(subset)), dtype=np.float64)

    return np.column_stack([
        [
            _window_max(evidence[m].times_s, evidence[m].score, float(t), window_s)
            for t in times
        ]
        for m in subset
    ])


def label_candidates(times: np.ndarray, gt_s: Sequence[float], tau_s: float) -> np.ndarray:
    """
    True for every candidate that sits within `tau_s` of an annotated boundary.

    Deliberately not one-to-one: this is the training signal, not the score. Two
    candidates near the same boundary are both legitimately positive; the
    one-to-one rule belongs to the evaluation, where it is enforced by matching.

    :return: boolean array, one entry per candidate.
    """
    if len(times) == 0:
        return np.zeros(0, dtype=bool)
    gt = np.sort(np.asarray(gt_s, dtype=np.float64))
    if len(gt) == 0:
        return np.zeros(len(times), dtype=bool)
    nearest = np.abs(gt[np.clip(np.searchsorted(gt, times), 0, len(gt) - 1)] - times)
    left = np.abs(gt[np.clip(np.searchsorted(gt, times) - 1, 0, len(gt) - 1)] - times)
    return np.minimum(nearest, left) <= tau_s


def suppress(
    times: np.ndarray, scores: np.ndarray, radius_s: float = SUPPRESS_RADIUS_S
) -> list[float]:
    """
    Keep the strongest candidate in each neighbourhood, drop the rest.

    Without this a single boundary proposed by several modalities would produce a
    cluster of predictions, of which matching accepts one and counts the others
    as false positives.

    :param times: candidate timestamps in seconds.
    :param scores: one probability per candidate.
    :param radius_s: candidates closer than this compete.
    :return: the surviving timestamps, ascending.
    """
    kept: list[float] = []
    for i in np.argsort(scores)[::-1]:          # strongest first
        t = float(times[i])
        if all(abs(t - k) > radius_s for k in kept):
            kept.append(t)
    return sorted(kept)


# ── Deciders ──────────────────────────────────────────────────────────────────

class RuleFuser:
    """
    Untrained baseline: a candidate is a boundary when any of the subset's
    modalities is confident enough about it.

    Needs no training data and therefore no cross-validation, so it carries no
    model variance. Its purpose is the cross-check — if it puts the modalities in
    the same Shapley order as the forest, that ordering does not depend on the
    choice of classifier.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RuleFuser":
        return self                                    # nothing to learn

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros(len(features), dtype=np.float64)
        return features.max(axis=1)


class ForestFuser:
    """
    Random forest over the subset's scores.

    Learns how the modalities combine instead of assuming it, which is what the
    question about *information* content asks for. `class_weight="balanced"`
    handles the imbalance so no decision threshold has to be tuned — a threshold
    fitted per subset would be one more place for the test session to leak in.
    """

    def __init__(self, n_estimators: int = 300, seed: int = 0) -> None:
        self._n_estimators = n_estimators
        self._seed = seed
        self._model = None
        self._constant = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ForestFuser":
        from sklearn.ensemble import RandomForestClassifier

        # A fold whose candidates carry only one label cannot be learned from —
        # but the answer is that label, not zero. A modality whose proposals are
        # all correct would otherwise be scored as predicting nothing.
        classes = np.unique(labels)
        if len(classes) < 2:
            self._model = None
            self._constant = float(classes[0]) if len(classes) else 0.0
            return self

        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            class_weight="balanced",
            random_state=self._seed,
            n_jobs=-1,
        )
        self._model.fit(features, labels)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros(len(features), dtype=np.float64)
        if self._model is None:
            return np.full(len(features), self._constant, dtype=np.float64)
        return self._model.predict_proba(features)[:, 1]
