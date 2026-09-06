"""
Whether a measured difference is real or an accident of the 25 sessions that
happened to be recorded.

Everything here resamples SESSIONS, because the session is the unit of the
experiment: a different corpus would have produced different numbers, and the
question is how much different.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import norm

from docupilot.evaluation.analysis import interaction, shapley

# Session name -> that session's score for one subset.
PerSession = Mapping[str, float]
PerSessionBySubset = Mapping[frozenset[str], PerSession]


@dataclass(frozen=True)
class Interval:
    """A point estimate with the range the corpus can actually support."""

    point: float
    lo: float
    hi: float

    @property
    def excludes_zero(self) -> bool:
        """True when the effect has a direction the data can commit to."""
        return self.lo > 0.0 or self.hi < 0.0

    def below(self, threshold: float) -> bool:
        """The WHOLE interval is smaller than the threshold — the equivalence
        claim (|effect| < SESOI proven), as opposed to mere non-significance."""
        return max(abs(self.lo), abs(self.hi)) < threshold

    def __str__(self) -> str:
        return f"{self.point:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}]"


def bootstrap_intervals(
    units: Sequence[str],
    statistic: Callable[[Sequence[str]], Sequence[float]],
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[Interval]:
    """
    BCa confidence intervals for SEVERAL statistics from ONE resampling.

    The resample is decided by the seed, not by the statistic, so quantities
    computed from the same sessions — the three Shapley values, the three
    interaction indices, all eight subset means — share every replicate. Drawing
    them together costs one bootstrap instead of one per quantity and yields
    exactly the intervals separate runs would.

    BCa rather than plain percentiles: F1 is bounded at 1 and its sampling
    distribution is skewed near the top, which shifts a percentile interval.
    BCa corrects for that skew and for the bias of the estimator (Efron &
    Tibshirani 1993).

    :param units: the session names the corpus consists of.
    :param statistic: maps a (re)sample of session names — possibly with
        repeats — to one number per quantity, always in the same order.
    :param n_draws: bootstrap replicates.
    :param alpha: 0.05 gives a 95 % interval.
    :param seed: fixes the resampling.
    :return: one interval per quantity, in the statistic's order.
    """
    names = list(units)
    n = len(names)
    observed = np.asarray(statistic(names), dtype=np.float64)
    if n < 2:
        return [Interval(float(v), float(v), float(v)) for v in observed]

    rng = np.random.default_rng(seed)
    replicates = np.array([
        statistic([names[i] for i in rng.integers(0, n, n)]) for _ in range(n_draws)
    ], dtype=np.float64)
    jack = np.array(
        [statistic(names[:i] + names[i + 1:]) for i in range(n)], dtype=np.float64
    )

    intervals: list[Interval] = []
    for column in range(len(observed)):
        intervals.append(_bca(
            observed[column], replicates[:, column], jack[:, column], n_draws, alpha
        ))
    return intervals


def _bca(
    observed: float, replicates: np.ndarray, jack: np.ndarray, n_draws: int, alpha: float
) -> Interval:
    """Bias-corrected and accelerated interval for one quantity."""
    # Bias correction: where the observed value sits among the replicates.
    below = float((replicates < observed).mean())
    z0 = norm.ppf(min(max(below, 1.0 / n_draws), 1.0 - 1.0 / n_draws))

    # Acceleration from the jackknife: how skewed the statistic is.
    centred = jack.mean() - jack
    denominator = 6.0 * (float((centred ** 2).sum()) ** 1.5)
    acceleration = float((centred ** 3).sum()) / denominator if denominator > 0 else 0.0

    def adjusted(z_alpha: float) -> float:
        z = z0 + z_alpha
        return float(norm.cdf(z0 + z / (1.0 - acceleration * z)))

    lo_q = adjusted(norm.ppf(alpha / 2.0))
    hi_q = adjusted(norm.ppf(1.0 - alpha / 2.0))
    return Interval(
        point=float(observed),
        lo=float(np.quantile(replicates, min(lo_q, hi_q))),
        hi=float(np.quantile(replicates, max(lo_q, hi_q))),
    )


def bootstrap_ci(
    units: Sequence[str],
    statistic: Callable[[Sequence[str]], float],
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """
    Confidence interval for ONE statistic by resampling sessions.

    `statistic` receives a list of session names — possibly with repeats — and
    returns one number. Anything derived from the corpus can be passed: a
    subset's mean F1, a paired difference, or a Shapley value.

    :return: point estimate with lower and upper bound.
    """
    return bootstrap_intervals(
        units, lambda names: [statistic(names)], n_draws, alpha, seed
    )[0]


def _mean_over(scores: PerSession, picked: Sequence[str]) -> float:
    return float(np.mean([scores[s] for s in picked]))


def _paired_names(before: PerSession, after: PerSession) -> list[str]:
    names = sorted(set(before) & set(after))
    if len(names) != len(before) or len(names) != len(after):
        raise KeyError("Beide Teilmengen müssen auf denselben Sessions bewertet sein")
    return names


def subset_ci(
    per_session: PerSession, n_draws: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """How precisely one subset's mean F1 is pinned down by the corpus."""
    return bootstrap_ci(
        list(per_session), lambda names: _mean_over(per_session, names),
        n_draws, alpha, seed,
    )


def subset_cis(
    per_session_by_subset: PerSessionBySubset,
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[frozenset[str], Interval]:
    """
    `subset_ci` for every subset at once, from one shared resampling.

    :raises KeyError: when the subsets were not all scored on the same sessions.
    """
    keys = list(per_session_by_subset)
    if not keys:
        return {}
    names = list(per_session_by_subset[keys[0]])
    # The resample indexes into the unit list, so sharing one draw is only the
    # same as separate draws when every subset lists its sessions in the same
    # order. Otherwise fall back to one bootstrap per subset — slower, identical.
    if any(list(per_session_by_subset[k]) != names for k in keys[1:]):
        return {k: subset_ci(per_session_by_subset[k], n_draws, alpha, seed) for k in keys}
    intervals = bootstrap_intervals(
        names,
        lambda picked: [_mean_over(per_session_by_subset[k], picked) for k in keys],
        n_draws, alpha, seed,
    )
    return dict(zip(keys, intervals))


def delta_ci(
    before: PerSession,
    after: PerSession,
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """
    The gain from `before` to `after`, per session and then resampled.

    Paired on purpose: session difficulty cancels out, which is what makes an
    effect visible at all with a corpus this size.

    :raises KeyError: when the two subsets were not scored on the same sessions.
    """
    names = _paired_names(before, after)
    return bootstrap_ci(
        names,
        lambda picked: float(np.mean([after[s] - before[s] for s in picked])),
        n_draws, alpha, seed,
    )


def _resampled_values(
    per_session_by_subset: PerSessionBySubset, picked: Sequence[str]
) -> dict[frozenset[str], float]:
    """The characteristic function v(S) as the resampled corpus would report it."""
    return {
        subset: _mean_over(scores, picked)
        for subset, scores in per_session_by_subset.items()
    }


def shapley_ci(
    per_session_by_subset: PerSessionBySubset,
    players: Sequence[str],
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Interval]:
    """
    Confidence intervals for the Shapley values themselves.

    The whole attribution is recomputed inside every resample, so the interval
    carries the uncertainty of all 8 subsets at once rather than treating the
    Shapley value as if it were a directly measured quantity.

    :param per_session_by_subset: per-session F1 for every subset.
    :param players: the modalities.
    :return: one interval per modality.
    """
    names = list(next(iter(per_session_by_subset.values())))

    def statistic(picked: Sequence[str]) -> list[float]:
        phi = shapley(_resampled_values(per_session_by_subset, picked), players)
        return [phi[p] for p in players]

    return dict(zip(players, bootstrap_intervals(names, statistic, n_draws, alpha, seed)))


def interaction_ci(
    per_session_by_subset: PerSessionBySubset,
    players: Sequence[str],
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[tuple[str, str], Interval]:
    """
    Confidence intervals for the pairwise interaction indices.

    Same construction as shapley_ci: the whole index is recomputed inside every
    session resample, so the interval carries the uncertainty of all 8 subsets.
    Without it a value a hair above zero reads as "synergy" when it is
    indistinguishable from noise — the same overreach the Shapley CIs prevent.

    :param per_session_by_subset: per-session F1 for every subset.
    :param players: the modalities.
    :return: one interval per unordered pair, keyed like analysis.interaction.
    """
    names = list(next(iter(per_session_by_subset.values())))
    pairs = list(combinations(players, 2))

    def statistic(picked: Sequence[str]) -> list[float]:
        index = interaction(_resampled_values(per_session_by_subset, picked), players)
        return [index[pair] for pair in pairs]

    return dict(zip(pairs, bootstrap_intervals(names, statistic, n_draws, alpha, seed)))


def _per_session_size_means(
    per_session_by_subset: PerSessionBySubset,
    players: Sequence[str],
    size: int,
    names: Sequence[str],
) -> dict[str, float]:
    """Each session's mean F1 over all subsets of one size — one saturation point."""
    subs = [frozenset(combo) for combo in combinations(players, size)]
    return {
        s: float(np.mean([per_session_by_subset[sub][s] for sub in subs]))
        for s in names
    }


def _size_steps(
    per_session_by_subset: PerSessionBySubset, players: Sequence[str]
) -> tuple[list[str], dict[int, tuple[dict[str, float], dict[str, float]]]]:
    """Per step k the (before, after) per-session means the step is the gain of."""
    names = list(next(iter(per_session_by_subset.values())))
    means = {
        k: _per_session_size_means(per_session_by_subset, players, k, names)
        for k in range(len(players) + 1)
    }
    return names, {k: (means[k - 1], means[k]) for k in range(1, len(players) + 1)}


def saturation_step_ci(
    per_session_by_subset: PerSessionBySubset,
    players: Sequence[str],
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[int, Interval]:
    """
    Paired confidence interval for each step of the saturation curve.

    The curve's value at size k is the mean F1 over all subsets of that size; the
    step k is that mean minus the one at k-1. Averaging the subsets *per session*
    first turns the step into a paired difference, so the interval can say
    whether adding the k-th modality helps by more than the corpus noise — the
    significance half of the saturation criterion, next to the effect size.

    :param per_session_by_subset: per-session F1 for every subset.
    :param players: the modalities.
    :return: one interval per step k = 1..n.
    """
    names, steps = _size_steps(per_session_by_subset, players)
    for before, after in steps.values():
        _paired_names(before, after)

    def statistic(picked: Sequence[str]) -> list[float]:
        return [
            float(np.mean([after[s] - before[s] for s in picked]))
            for before, after in steps.values()
        ]

    # Sorted, like every paired difference (`delta_ci`): the unit order decides
    # which sessions a resample draws, so it is part of the result.
    return dict(zip(
        steps, bootstrap_intervals(sorted(names), statistic, n_draws, alpha, seed)
    ))


def saturation_step_required_n(
    per_session_by_subset: PerSessionBySubset,
    players: Sequence[str],
    sesoi: float,
) -> dict[int, int]:
    """
    Sessions needed to resolve each saturation step at the given SESOI.

    The quantified companion to an inconclusive step: a step whose CI covers
    both 0 and the relevance threshold supports neither "helps" nor "does not
    help" — this states what corpus size the equivalence claim would take
    (Lakens 2022), instead of leaving "n.s." to be misread as absence.

    :param per_session_by_subset: per-session F1 for every subset.
    :param players: the modalities.
    :param sesoi: smallest effect size of interest.
    :return: required session count per step k = 1..n.
    """
    _, steps = _size_steps(per_session_by_subset, players)
    return {
        k: required_sessions(paired_sd(before, after), sesoi)
        for k, (before, after) in steps.items()
    }


# ── Justifying the corpus size ────────────────────────────────────────────────

def minimum_detectable_effect(
    sd: float, n: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    """
    Smallest paired difference a corpus of `n` sessions could show.

    The sample-size justification for data already collected: instead of
    assuming an effect size and asking how many sessions are needed, it states
    what the sessions at hand can resolve (Lakens 2022). `sd` is measured from
    the per-session differences, not assumed.

    :param sd: standard deviation of the paired differences.
    :param n: number of sessions.
    :return: the smallest difference detectable at the given power.
    """
    if n < 2:
        return float("inf")
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return float(z * sd / math.sqrt(n))


def required_sessions(
    sd: float, sesoi: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """
    Sessions needed to resolve an effect of size `sesoi`.

    Reported for every comparison the corpus cannot settle: naming the number
    turns a limitation into a quantified one.

    :param sd: standard deviation of the paired differences.
    :param sesoi: smallest effect worth detecting.
    :return: required number of sessions.
    """
    if sesoi <= 0:
        return 0
    z = norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power)
    return int(math.ceil((z * sd / sesoi) ** 2))


def paired_sd(before: PerSession, after: PerSession) -> float:
    """Standard deviation of the per-session differences — the input to both above."""
    names = sorted(set(before) & set(after))
    if len(names) < 2:
        return 0.0
    return float(np.std([after[s] - before[s] for s in names], ddof=1))
