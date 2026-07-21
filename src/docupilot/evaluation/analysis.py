"""
Turn the performance of every modality subset into an attribution: how much each
modality contributed, where two of them are redundant, and where adding another
stops paying off.

With three modalities all 8 coalitions are evaluated in full, so the Shapley
values are exact — there is no sampling error to justify.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from math import factorial

import numpy as np

# One subset's score. frozenset keys because a coalition has no order.
Values = Mapping[frozenset[str], float]


def subsets(players: Sequence[str]) -> list[frozenset[str]]:
    """Every coalition including the empty one — the full factorial design."""
    return [
        frozenset(combo)
        for size in range(len(players) + 1)
        for combo in combinations(players, size)
    ]


def shapley(values: Values, players: Sequence[str]) -> dict[str, float]:
    """
    Each modality's average marginal contribution over all orders of joining.

    The Shapley value is the unique attribution satisfying efficiency, symmetry,
    dummy and additivity (Shapley 1953) — the choice is forced by those
    requirements, not a matter of taste.

    :param values: score of every subset; must contain all 2^n of them.
    :param players: the modalities.
    :return: one value per modality; they sum to v(all) - v(empty).
    :raises KeyError: when a subset is missing from `values`.
    """
    n = len(players)
    result: dict[str, float] = {}

    for player in players:
        others = [p for p in players if p != player]
        total = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for combo in combinations(others, size):
                coalition = frozenset(combo)
                total += weight * (values[coalition | {player}] - values[coalition])
        result[player] = total

    return result


def efficiency_error(
    values: Values, players: Sequence[str], phi: Mapping[str, float]
) -> float:
    """
    How far the Shapley values miss summing to v(all) - v(empty).

    A mathematical identity, so anything above rounding noise means the
    implementation is wrong — the result checks itself.
    """
    expected = values[frozenset(players)] - values[frozenset()]
    return abs(sum(phi.values()) - expected)


def interaction(values: Values, players: Sequence[str]) -> dict[tuple[str, str], float]:
    """
    Pairwise synergy (positive) or redundancy (negative) between modalities.

    The Shapley interaction index (Grabisch & Roubens 1999). Shapley values alone
    hide this: they say how much each modality contributes, not whether two of
    them are saying the same thing — which is exactly the redundancy question.

    :param values: score of every subset.
    :param players: the modalities.
    :return: one value per unordered pair.
    """
    n = len(players)
    result: dict[tuple[str, str], float] = {}

    for first, second in combinations(players, 2):
        rest = [p for p in players if p not in (first, second)]
        total = 0.0
        for size in range(len(rest) + 1):
            weight = (
                factorial(size) * factorial(n - size - 2) / factorial(n - 1)
            )
            for combo in combinations(rest, size):
                s = frozenset(combo)
                total += weight * (
                    values[s | {first, second}]
                    - values[s | {first}]
                    - values[s | {second}]
                    + values[s]
                )
        result[(first, second)] = total

    return result


def saturation(values: Values, players: Sequence[str]) -> dict[int, float]:
    """
    Mean score per number of modalities involved — the saturation curve.

    Averaging over all subsets of a size removes which modalities were picked,
    leaving only how many. The step from one size to the next is the marginal
    gain whose shrinking is what saturation means.

    :return: mean score keyed by subset size 0..n.
    """
    return {
        size: float(np.mean([
            values[frozenset(combo)] for combo in combinations(players, size)
        ]))
        for size in range(len(players) + 1)
    }


def marginal_gain(curve: Mapping[int, float]) -> dict[int, float]:
    """Gain of each additional modality: curve[k] - curve[k-1]."""
    return {k: curve[k] - curve[k - 1] for k in sorted(curve) if k - 1 in curve}
