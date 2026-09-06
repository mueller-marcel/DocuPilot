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
from scipy.stats import rankdata

from docupilot.evaluation import metrics
from docupilot.segmentation.evidence import BoundaryEvidence

# Half-widths the score is read over around a candidate. Several widths, so the
# decider can tell a sharp peak (video: the settling instant) from a broad
# plateau (audio: an execution window) — the temporal precision of a modality
# becomes a feature instead of a handicap. The set is the SAME for every
# modality: an asymmetric recipe would move the Shapley values by design
# effort, not by information.
FEATURE_WINDOWS_S: tuple[float, ...] = (0.5, 1.0, 2.0)

# Names of the feature columns one modality contributes, in order.
FEATURE_NAMES: tuple[str, ...] = (
    "point",                                   # score at the candidate instant
    *(f"max_{w:g}s" for w in FEATURE_WINDOWS_S),
    "rank",                                    # session-relative rank of max_1s
)
FEATURES_PER_MODALITY = len(FEATURE_NAMES)

# Window whose maximum the rank feature is taken over. The primary tolerance,
# so the rank is "how strong is this modality here, on the scale it is scored
# at" — a per-session normalisation that costs no label.
_RANK_WINDOW_S = 1.0

# Two predictions closer than this cannot both be hits anyway — matching is
# one-to-one — so the weaker one is suppressed.
SUPPRESS_RADIUS_S = 1.0

# Decision thresholds the calibration may pick from. Coarse on purpose: a finer
# grid would fit the training folds' noise, and 0.05 steps already move F1 by
# less than the corpus can resolve.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.05 * k, 2) for k in range(1, 20))

Evidence = Mapping[str, BoundaryEvidence]


# ── Candidates ────────────────────────────────────────────────────────────────

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


# ── Features ──────────────────────────────────────────────────────────────────

def _window_max(times_s: np.ndarray, score: np.ndarray, t: float, window_s: float) -> float:
    """Highest score within +/- window_s of t; 0.0 when the window is empty."""
    lo = int(np.searchsorted(times_s, t - window_s, side="left"))
    hi = int(np.searchsorted(times_s, t + window_s, side="right"))
    return float(score[lo:hi].max()) if hi > lo else 0.0


def _value_at(times_s: np.ndarray, score: np.ndarray, t: float) -> float:
    """Score of the sample nearest to t; 0.0 for an empty lane."""
    if len(times_s) == 0:
        return 0.0
    i = int(np.searchsorted(times_s, t))
    if i > 0 and (i == len(times_s) or abs(times_s[i - 1] - t) <= abs(times_s[i] - t)):
        i -= 1
    return float(score[i])


def _modality_features(times: np.ndarray, ev: BoundaryEvidence) -> np.ndarray:
    """One modality's block: (n_candidates, FEATURES_PER_MODALITY)."""
    n = len(times)
    block = np.zeros((n, FEATURES_PER_MODALITY), dtype=np.float64)
    if n == 0:
        return block
    block[:, 0] = [_value_at(ev.times_s, ev.score, float(t)) for t in times]
    for column, window in enumerate(FEATURE_WINDOWS_S, start=1):
        block[:, column] = [_window_max(ev.times_s, ev.score, float(t), window) for t in times]
    rank_source = [_window_max(ev.times_s, ev.score, float(t), _RANK_WINDOW_S) for t in times]
    # Average ranks scaled to (0, 1]: ties (many zeros) share a rank instead of
    # being ordered by position, which would leak the candidate index.
    block[:, -1] = rankdata(rank_source, method="average") / n
    return block


def feature_matrix(times: np.ndarray, evidence: Evidence, subset: Sequence[str]) -> np.ndarray:
    """
    One row per candidate, FEATURES_PER_MODALITY columns per modality in the
    subset, blocks in the given order.

    :param times: candidate timestamps in seconds.
    :param evidence: one BoundaryEvidence per modality.
    :param subset: which modalities become column blocks, in the given order.
    :return: array of shape (len(times), FEATURES_PER_MODALITY * len(subset)).
    """
    if not subset:
        return np.zeros((len(times), 0), dtype=np.float64)
    return np.hstack([_modality_features(times, evidence[m]) for m in subset])


def block_columns(position: int) -> list[int]:
    """The column indices of the modality at `position` in a feature matrix."""
    start = position * FEATURES_PER_MODALITY
    return list(range(start, start + FEATURES_PER_MODALITY))


# ── Labels and decisions ──────────────────────────────────────────────────────

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


def decide(times: np.ndarray, proba: np.ndarray, threshold: float) -> list[float]:
    """Candidates at or above the threshold, one per neighbourhood."""
    keep = proba >= threshold
    return suppress(times[keep], proba[keep])


def choose_threshold(
    folds: Sequence[tuple[np.ndarray, np.ndarray, Sequence[float]]],
    tau_s: float,
    grid: Sequence[float] = THRESHOLD_GRID,
) -> float:
    """
    The decision threshold that maximises macro F1 over the given sessions.

    Meant to be fed with TRAINING sessions and out-of-bag probabilities, so the
    held-out session never touches the operating point. Macro F1 because the
    characteristic function of the experiment is a macro average — the
    threshold optimises the quantity that is later attributed.

    Ties go to the value nearest 0.5: a flat optimum should not pull the
    operating point to an extreme by accident of grid order.

    :param folds: (candidate times, probabilities, ground truth) per session.
    :param tau_s: tolerance the F1 is scored at.
    :return: a value from `grid`; 0.5 when there is nothing to score.
    """
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1s = [
            metrics.match(gt, decide(times, proba, t), tau_s).f1
            for times, proba, gt in folds
        ]
        f1 = float(np.mean(f1s)) if f1s else 0.0
        better = f1 > best_f1 + 1e-12
        tie = abs(f1 - best_f1) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)
        if better or tie:
            best_t, best_f1 = float(t), f1
    return best_t


# ── Decider ───────────────────────────────────────────────────────────────────

class ForestFuser:
    """
    Random forest over the subset's features.

    Learns how the modalities combine instead of assuming it, which is what the
    question about *information* content asks for. `class_weight="balanced"`
    handles the imbalance; the operating point is still calibrated afterwards,
    on out-of-bag predictions, because a balanced weighting does not make 0.5
    the F1-optimal cut.
    """

    def __init__(self, n_estimators: int = 300, seed: int = 0) -> None:
        self._n_estimators = n_estimators
        self._seed = seed
        self._model = None
        self._constant = 0.0
        self._oob: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "ForestFuser":
        from sklearn.ensemble import RandomForestClassifier

        # A fold whose candidates carry only one label cannot be learned from —
        # but the answer is that label, not zero. A modality whose proposals are
        # all correct would otherwise be scored as predicting nothing.
        classes = np.unique(labels)
        if len(classes) < 2:
            self._model = None
            self._constant = float(classes[0]) if len(classes) else 0.0
            self._oob = np.full(len(labels), self._constant, dtype=np.float64)
            return self

        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            class_weight="balanced",
            random_state=self._seed,
            oob_score=True,
            n_jobs=-1,
        )
        self._model.fit(features, labels)
        oob = self._model.oob_decision_function_[:, 1]
        # A row no tree left out has no out-of-bag vote (NaN); with hundreds of
        # trees that is practically never, and "undecided" is the honest fill.
        self._oob = np.where(np.isnan(oob), 0.5, oob)
        return self

    def oob_proba(self) -> np.ndarray:
        """
        P(boundary) for every training row, from trees that did not see it —
        a leakage-free estimate of the model's behaviour on the training
        sessions, available without a single extra fit.

        :raises RuntimeError: before fit().
        """
        if self._oob is None:
            raise RuntimeError("oob_proba() vor fit() aufgerufen")
        return self._oob

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros(len(features), dtype=np.float64)
        if self._model is None:
            return np.full(len(features), self._constant, dtype=np.float64)
        return self._model.predict_proba(features)[:, 1]
