"""
Shapley attribution, interaction index and saturation curve.

Checked against the AXIOMS that make the Shapley value the right choice — a
frozen number would only say "it changed", the axioms say what must hold.
"""

import pytest

from docupilot.evaluation import analysis

PLAYERS = ("a", "b", "c")


def values(**scores: float) -> dict[frozenset[str], float]:
    """Characteristic function from labels like ab=0.7; '' is the empty coalition."""
    return {frozenset(label): score for label, score in scores.items()}


ADDITIVE = values(**{"": 0.0, "a": 0.3, "b": 0.2, "c": 0.1,
                     "ab": 0.5, "ac": 0.4, "bc": 0.3, "abc": 0.6})


def test_subsets_are_the_full_factorial_design():
    subs = analysis.subsets(PLAYERS)
    assert len(subs) == 8
    assert frozenset() in subs and frozenset(PLAYERS) in subs
    assert len(set(subs)) == 8


def test_efficiency_the_values_sum_to_the_grand_coalition():
    phi = analysis.shapley(ADDITIVE, PLAYERS)
    assert sum(phi.values()) == pytest.approx(ADDITIVE[frozenset(PLAYERS)])
    assert analysis.efficiency_error(ADDITIVE, PLAYERS, phi) < 1e-12


def test_a_purely_additive_game_pays_each_player_its_own_worth():
    phi = analysis.shapley(ADDITIVE, PLAYERS)
    assert phi["a"] == pytest.approx(0.3)
    assert phi["b"] == pytest.approx(0.2)
    assert phi["c"] == pytest.approx(0.1)


def test_symmetry_interchangeable_players_are_paid_equally():
    symmetric = values(**{"": 0.0, "a": 0.4, "b": 0.4, "c": 0.1,
                          "ab": 0.6, "ac": 0.5, "bc": 0.5, "abc": 0.8})
    phi = analysis.shapley(symmetric, PLAYERS)
    assert phi["a"] == pytest.approx(phi["b"])


def test_dummy_a_player_that_adds_nothing_is_paid_nothing():
    dummy = values(**{"": 0.0, "a": 0.5, "b": 0.3, "c": 0.0,
                      "ab": 0.7, "ac": 0.5, "bc": 0.3, "abc": 0.7})
    assert analysis.shapley(dummy, PLAYERS)["c"] == pytest.approx(0.0)


def test_interaction_sign_separates_redundancy_from_synergy():
    # Two players that say the same thing: together no better than the best alone.
    redundant = values(**{"": 0.0, "a": 0.5, "b": 0.5, "c": 0.1,
                          "ab": 0.5, "ac": 0.6, "bc": 0.6, "abc": 0.6})
    assert analysis.interaction(redundant, PLAYERS)[("a", "b")] < 0

    # Two players that only pay off jointly.
    synergistic = values(**{"": 0.0, "a": 0.1, "b": 0.1, "c": 0.1,
                            "ab": 0.8, "ac": 0.2, "bc": 0.2, "abc": 0.9})
    assert analysis.interaction(synergistic, PLAYERS)[("a", "b")] > 0

    assert analysis.interaction(ADDITIVE, PLAYERS)[("a", "b")] == pytest.approx(0.0)


def test_interaction_covers_every_unordered_pair_once():
    index = analysis.interaction(ADDITIVE, PLAYERS)
    assert set(index) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_saturation_averages_over_subsets_of_each_size():
    curve = analysis.saturation(ADDITIVE, PLAYERS)
    assert curve[0] == 0.0
    assert curve[1] == pytest.approx((0.3 + 0.2 + 0.1) / 3)
    assert curve[2] == pytest.approx((0.5 + 0.4 + 0.3) / 3)
    assert curve[3] == pytest.approx(0.6)


def test_marginal_gain_is_the_step_between_neighbouring_sizes():
    curve = analysis.saturation(ADDITIVE, PLAYERS)
    gains = analysis.marginal_gain(curve)
    assert set(gains) == {1, 2, 3}
    assert gains[2] == pytest.approx(curve[2] - curve[1])
    assert sum(gains.values()) == pytest.approx(curve[3] - curve[0])


def test_shapley_works_for_a_two_player_game():
    two = values(**{"": 0.0, "a": 0.4, "b": 0.2, "ab": 0.8})
    phi = analysis.shapley(two, ("a", "b"))
    # Each gets its solo worth plus half of what the pair adds beyond both.
    assert phi["a"] == pytest.approx(0.4 + 0.1)
    assert phi["b"] == pytest.approx(0.2 + 0.1)


def test_a_missing_coalition_is_an_error_not_a_silent_zero():
    with pytest.raises(KeyError):
        analysis.shapley(values(**{"": 0.0, "a": 0.1}), PLAYERS)
