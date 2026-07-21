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

import numpy as np
from scipy.stats import norm

from docupilot.evaluation.analysis import shapley

# Session name -> that session's score for one subset.
PerSession = Mapping[str, float]


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

    @property
    def below(self) -> Callable[[float], bool]:
        """`interval.below(sesoi)` — the whole effect is smaller than a threshold."""
        return lambda threshold: max(abs(self.lo), abs(self.hi)) < threshold

    def __str__(self) -> str:
        return f"{self.point:+.3f} [{self.lo:+.3f}, {self.hi:+.3f}]"


def bootstrap_ci(
    units: Sequence[str],
    statistic: Callable[[Sequence[str]], float],
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """
    Confidence interval by resampling sessions, bias-corrected and accelerated.

    BCa rather than plain percentiles: F1 is bounded at 1 and its sampling
    distribution is skewed near the top, which shifts a percentile interval.
    BCa corrects for that skew and for the bias of the estimator (Efron &
    Tibshirani 1993).

    `statistic` receives a list of session names — possibly with repeats — and
    returns one number. Anything derived from the corpus can be passed: a
    subset's mean F1, a paired difference, or a Shapley value.

    :param units: the session names the corpus consists of.
    :param statistic: computes the quantity from a (re)sample of sessions.
    :param n_draws: bootstrap replicates.
    :param alpha: 0.05 gives a 95 % interval.
    :param seed: fixes the resampling.
    :return: point estimate with lower and upper bound.
    """
    names = list(units)
    n = len(names)
    observed = statistic(names)
    if n < 2:
        return Interval(observed, observed, observed)

    rng = np.random.default_rng(seed)
    replicates = np.array([
        statistic([names[i] for i in rng.integers(0, n, n)]) for _ in range(n_draws)
    ])

    # Bias correction: where the observed value sits among the replicates.
    below = float((replicates < observed).mean())
    z0 = norm.ppf(min(max(below, 1.0 / n_draws), 1.0 - 1.0 / n_draws))

    # Acceleration from the jackknife: how skewed the statistic is.
    jack = np.array([statistic(names[:i] + names[i + 1:]) for i in range(n)])
    centred = jack.mean() - jack
    denominator = 6.0 * (float((centred ** 2).sum()) ** 1.5)
    acceleration = float((centred ** 3).sum()) / denominator if denominator > 0 else 0.0

    def adjusted(z_alpha: float) -> float:
        z = z0 + z_alpha
        return float(norm.cdf(z0 + z / (1.0 - acceleration * z)))

    lo_q = adjusted(norm.ppf(alpha / 2.0))
    hi_q = adjusted(norm.ppf(1.0 - alpha / 2.0))
    return Interval(
        point=observed,
        lo=float(np.quantile(replicates, min(lo_q, hi_q))),
        hi=float(np.quantile(replicates, max(lo_q, hi_q))),
    )


def subset_ci(
    per_session: PerSession, n_draws: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """How precisely one subset's mean F1 is pinned down by the corpus."""
    return bootstrap_ci(
        list(per_session),
        lambda names: float(np.mean([per_session[s] for s in names])),
        n_draws, alpha, seed,
    )


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
    names = sorted(set(before) & set(after))
    if len(names) != len(before) or len(names) != len(after):
        raise KeyError("Beide Teilmengen müssen auf denselben Sessions bewertet sein")
    return bootstrap_ci(
        names,
        lambda picked: float(np.mean([after[s] - before[s] for s in picked])),
        n_draws, alpha, seed,
    )


def shapley_ci(
    per_session_by_subset: Mapping[frozenset[str], PerSession],
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
    any_subset = next(iter(per_session_by_subset.values()))
    names = list(any_subset)

    def value_for(player: str) -> Callable[[Sequence[str]], float]:
        def statistic(picked: Sequence[str]) -> float:
            values = {
                subset: float(np.mean([scores[s] for s in picked]))
                for subset, scores in per_session_by_subset.items()
            }
            return shapley(values, players)[player]

        return statistic

    return {
        player: bootstrap_ci(names, value_for(player), n_draws, alpha, seed)
        for player in players
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
