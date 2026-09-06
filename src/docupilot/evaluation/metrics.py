"""
How a predicted set of boundaries is scored against the annotated one: a
prediction counts as a hit when it falls within a tolerance of a ground-truth
boundary, matched one-to-one.

The measuring instrument of the whole experiment — it decides nothing about
modalities, it only compares two lists of timestamps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Match:
    """One session's hit/miss balance at one tolerance."""

    tp: int
    fp: int
    fn: int
    pairs: list[tuple[float, float]]   # (t_gt, t_pred) of every hit, for inspection

    @property
    def precision(self) -> float:
        """Share of predictions that were real boundaries; 0.0 when none were made."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        """Share of real boundaries that were found; 0.0 when there were none."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def match(gt_s: Sequence[float], pred_s: Sequence[float], tau_s: float) -> Match:
    """
    Pair predicted boundaries with annotated ones, at most one each way.

    One-to-one is not cosmetic: without it a single prediction could satisfy
    several ground-truth boundaries and inflate recall. The optimal assignment
    (Kuhn-Munkres) is used rather than nearest-neighbour greedy, which can lock
    in a close pair and thereby lose a larger set of valid ones.

    Pairs further apart than `tau_s` are priced above any achievable total of
    valid pairs, so the solver only falls back on them when nothing valid is
    left; they are dropped afterwards. The number of hits is therefore
    maximised first, their total distance minimised second.

    :param gt_s: annotated boundaries in seconds.
    :param pred_s: predicted boundaries in seconds.
    :param tau_s: tolerance in seconds; a hit lies within it.
    :return: the hit/miss balance.
    :raises ValueError: when the tolerance is not positive.
    """
    if tau_s <= 0:
        raise ValueError(f"Toleranz muss positiv sein, war {tau_s}")

    gt = np.sort(np.asarray(gt_s, dtype=np.float64))
    pred = np.sort(np.asarray(pred_s, dtype=np.float64))
    n, m = len(gt), len(pred)
    if n == 0 or m == 0:
        return Match(tp=0, fp=m, fn=n, pairs=[])

    distance = np.abs(gt[:, None] - pred[None, :])
    # Above every total a fully valid assignment could reach, so one forbidden
    # pair always costs more than any distance it could save.
    penalty = min(n, m) * tau_s + 1.0
    rows, cols = linear_sum_assignment(np.where(distance <= tau_s, distance, penalty))

    pairs = [
        (float(gt[i]), float(pred[j]))
        for i, j in zip(rows, cols)
        if distance[i, j] <= tau_s
    ]
    return Match(tp=len(pairs), fp=m - len(pairs), fn=n - len(pairs), pairs=pairs)


def chance_level(
    gt_s: Sequence[float],
    duration_s: float,
    tau_s: float,
    n_draws: int = 1000,
    seed: int = 0,
) -> float:
    """
    F1 reached by guessing: boundaries drawn uniformly, as many as were annotated.

    Without this floor an F1 is uninterpretable — with a generous tolerance and
    many boundaries, chance alone already scores. Seeded, so the number is
    reproducible.

    :param gt_s: annotated boundaries in seconds.
    :param duration_s: length of the recording the guesses are spread over.
    :param tau_s: tolerance in seconds.
    :param n_draws: how many random sets to average over.
    :param seed: fixes the draw.
    :return: mean F1 over the draws.
    """
    gt = np.asarray(gt_s, dtype=np.float64)
    if len(gt) == 0 or duration_s <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(np.mean([
        match(gt, rng.uniform(0.0, duration_s, len(gt)), tau_s).f1
        for _ in range(n_draws)
    ]))
