"""
How much a modality's raw candidates already coincide with the ground truth,
before any classifier — the confound behind a modality that shares its timing
with the annotation (criterion contamination; Kaufman et al. 2012, Kriegeskorte
et al. 2009).

The annotation marks the instant an action's result has settled on screen; the
video modality proposes exactly such instants. Coverage alone cannot separate
"informative" from "definitionally tied", so three things are measured that
can: how precisely the nearest candidate aligns (a shared signal aligns to a
fraction of the tolerance, an independent judgement does not), how much of the
coverage would be reached by placing the same number of candidates at random,
and — via the candidate-pool ablation in `experiment` — how much a modality
gains from the others' timing alone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from docupilot.evaluation import fusion

if TYPE_CHECKING:
    from docupilot.evaluation.experiment import SessionData

# Share of the tolerance below which an alignment counts as "fine": a candidate
# this close to the annotation on most boundaries means the annotator and the
# modality reacted to the same signal.
FINE_FRACTION = 0.25

_CHANCE_DRAWS = 200


@dataclass(frozen=True)
class CouplingStats:
    """One modality's structural closeness to the annotation, averaged over sessions."""

    coverage: float
    """Share of annotated boundaries with a candidate within ±τ."""
    chance_coverage: float
    """Coverage the same number of candidates would reach placed at random."""
    rate: float
    """Candidates per annotated boundary."""
    fine_share: float
    """Share of boundaries with a candidate within ±τ·FINE_FRACTION."""
    offset_median_s: float
    """Median of (nearest candidate − boundary) over covered boundaries; negative
    = the modality marks an earlier stage of the action. NaN when nothing is
    covered."""
    offset_p25_s: float
    offset_p75_s: float

    @property
    def lift(self) -> float:
        """Coverage above chance — the part that is not explained by density."""
        return self.coverage - self.chance_coverage

    @property
    def offset_iqr_s(self) -> float:
        """Spread of the alignment: the signature of a shared signal is a narrow one."""
        return self.offset_p75_s - self.offset_p25_s


def _nearest_deltas(candidates: np.ndarray, gt: Sequence[float]) -> list[float]:
    if not candidates.size:
        return []
    return [float(candidates[np.abs(candidates - g).argmin()] - g) for g in gt]


def _coverage(candidates: np.ndarray, gt: Sequence[float], tau_s: float) -> float:
    deltas = _nearest_deltas(candidates, gt)
    return sum(1 for d in deltas if abs(d) <= tau_s) / len(gt)


def _chance_coverage(
    n_candidates: int, gt: Sequence[float], duration_s: float, tau_s: float,
    n_draws: int = _CHANCE_DRAWS, seed: int = 0,
) -> float:
    """Expected coverage of `n_candidates` uniform random moments, by simulation."""
    if n_candidates == 0 or duration_s <= 0:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(np.mean([
        _coverage(np.sort(rng.uniform(0.0, duration_s, n_candidates)), gt, tau_s)
        for _ in range(n_draws)
    ]))


def modality_coupling(
    data: Sequence[SessionData], modality: str, tau_s: float
) -> CouplingStats:
    """
    Coupling of one modality's RAW candidates with the annotated boundaries.

    Computed before the classifier — it measures the modality's structural
    closeness to the label, not what a model can learn from it. Every statistic
    is taken per session and then averaged, so a long session weighs no more
    than a short one — the same unit the experiment uses.

    :param data: the corpus.
    :param modality: the modality to characterise.
    :param tau_s: the tolerance.
    """
    return _coupling(data, [modality], tau_s)


def union_coupling(
    data: Sequence[SessionData], players: Sequence[str], tau_s: float
) -> CouplingStats:
    """Coupling of ALL players' candidates pooled — its coverage is the recall
    ceiling of the whole experiment: a boundary no modality proposes cannot be
    found by any decider."""
    return _coupling(data, list(players), tau_s)


def _coupling(data: Sequence[SessionData], pool: Sequence[str], tau_s: float) -> CouplingStats:
    coverages, chances, rates, fines = [], [], [], []
    medians, p25s, p75s = [], [], []
    for session in data:
        gt = session.gt_s
        if not gt:
            continue
        candidates = fusion.candidate_times(session.evidence, pool)
        deltas = _nearest_deltas(candidates, gt)
        covered = [d for d in deltas if abs(d) <= tau_s]
        coverages.append(len(covered) / len(gt))
        chances.append(_chance_coverage(candidates.size, gt, session.duration_s, tau_s))
        rates.append(candidates.size / len(gt))
        fines.append(sum(1 for d in deltas if abs(d) <= tau_s * FINE_FRACTION) / len(gt))
        if covered:
            medians.append(float(np.median(covered)))
            p25s.append(float(np.percentile(covered, 25)))
            p75s.append(float(np.percentile(covered, 75)))

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return CouplingStats(
        coverage=mean(coverages) if coverages else 0.0,
        chance_coverage=mean(chances) if chances else 0.0,
        rate=mean(rates) if rates else 0.0,
        fine_share=mean(fines) if fines else 0.0,
        offset_median_s=mean(medians),
        offset_p25_s=mean(p25s),
        offset_p75_s=mean(p75s),
    )


def coupling_table(
    data: Sequence[SessionData], players: Sequence[str], tau_s: float
) -> dict[str, CouplingStats]:
    """One CouplingStats per modality — the confound summary for one corpus."""
    return {m: modality_coupling(data, m, tau_s) for m in players}
