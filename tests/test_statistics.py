"""
Bootstrap intervals and the sample-size arithmetic.

The bootstrap is seeded, so the properties that matter are reproducibility, the
relation between an interval and its point estimate, and that a shared
resampling gives what separate ones would.
"""

import numpy as np
import pytest

from docupilot.evaluation import statistics

PLAYERS = ("a", "b", "c")


def per_session(*scores: float) -> dict[str, float]:
    return {f"s{i}": v for i, v in enumerate(scores)}


@pytest.fixture
def by_subset():
    """A per-session table for all 8 coalitions of three players."""
    # Every added modality is worth exactly 0.1 on every session, so the
    # saturation steps have a known answer while the sessions still differ.
    rng = np.random.default_rng(7)
    names = [f"s{i}" for i in range(12)]
    base = {n: rng.uniform(0.2, 0.6) for n in names}
    out = {}
    for size, subsets in enumerate([
        [frozenset()],
        [frozenset({p}) for p in PLAYERS],
        [frozenset(pair) for pair in (("a", "b"), ("a", "c"), ("b", "c"))],
        [frozenset(PLAYERS)],
    ]):
        for subset in subsets:
            out[subset] = {n: base[n] + 0.1 * size for n in names}
    return out


class TestInterval:
    def test_excludes_zero_only_when_the_whole_range_is_one_sided(self):
        assert statistics.Interval(0.1, 0.02, 0.2).excludes_zero
        assert statistics.Interval(-0.1, -0.2, -0.02).excludes_zero
        assert not statistics.Interval(0.1, -0.01, 0.2).excludes_zero

    def test_below_is_an_equivalence_claim_not_mere_non_significance(self):
        # Wholly inside +/- 0.05: the effect is provably small.
        assert statistics.Interval(0.01, -0.02, 0.03).below(0.05)
        # Covers zero but also reaches 0.4: undecided, not negligible.
        assert not statistics.Interval(0.01, -0.02, 0.4).below(0.05)

    def test_str_shows_the_point_and_the_range(self):
        assert str(statistics.Interval(0.1234, -0.01, 0.2)) == "+0.123 [-0.010, +0.200]"


class TestBootstrap:
    def test_point_estimate_is_the_statistic_on_the_real_sample(self):
        scores = per_session(0.1, 0.4, 0.5, 0.6, 0.9)
        ci = statistics.subset_ci(scores)
        assert ci.point == pytest.approx(float(np.mean(list(scores.values()))))
        assert ci.lo <= ci.point <= ci.hi

    def test_same_seed_same_interval(self):
        scores = per_session(0.1, 0.4, 0.5, 0.6, 0.9)
        assert statistics.subset_ci(scores) == statistics.subset_ci(scores)

    def test_a_wider_confidence_level_gives_a_wider_interval(self):
        scores = per_session(0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4)
        narrow = statistics.subset_ci(scores, alpha=0.20)
        wide = statistics.subset_ci(scores, alpha=0.01)
        assert wide.lo <= narrow.lo and wide.hi >= narrow.hi

    def test_a_constant_sample_has_no_uncertainty(self):
        ci = statistics.subset_ci(per_session(0.5, 0.5, 0.5, 0.5))
        assert (ci.point, ci.lo, ci.hi) == (0.5, 0.5, 0.5)

    def test_a_single_unit_degenerates_instead_of_failing(self):
        ci = statistics.bootstrap_ci(["only"], lambda names: 1.0)
        assert (ci.point, ci.lo, ci.hi) == (1.0, 1.0, 1.0)

    def test_shared_resampling_equals_separate_runs(self, by_subset):
        # subset_cis draws once for all coalitions; that must be identical to
        # bootstrapping each on its own, or the saving would change the result.
        shared = statistics.subset_cis(by_subset)
        for subset, scores in by_subset.items():
            assert shared[subset] == statistics.subset_ci(scores)

    def test_mismatched_session_order_falls_back_instead_of_mixing_units(self):
        a = {"s1": 0.1, "s2": 0.5, "s3": 0.9}
        b = {"s3": 0.2, "s1": 0.4, "s2": 0.6}
        shared = statistics.subset_cis({frozenset({"x"}): a, frozenset({"y"}): b})
        assert shared[frozenset({"x"})] == statistics.subset_ci(a)
        assert shared[frozenset({"y"})] == statistics.subset_ci(b)


class TestPairedDifferences:
    def test_delta_is_the_mean_per_session_difference(self):
        before = per_session(0.1, 0.2, 0.3)
        after = per_session(0.2, 0.4, 0.6)
        assert statistics.delta_ci(before, after).point == pytest.approx(0.2)

    def test_a_consistent_gain_yields_an_interval_above_zero(self):
        before = per_session(*[0.30, 0.32, 0.28, 0.31, 0.29, 0.30, 0.33, 0.27])
        after = per_session(*[0.50, 0.53, 0.48, 0.52, 0.49, 0.51, 0.54, 0.47])
        assert statistics.delta_ci(before, after).excludes_zero

    def test_noise_around_zero_does_not_claim_a_direction(self):
        before = per_session(0.4, 0.5, 0.6, 0.3, 0.55, 0.45, 0.5, 0.52)
        after = per_session(0.5, 0.4, 0.55, 0.4, 0.5, 0.5, 0.45, 0.55)
        assert not statistics.delta_ci(before, after).excludes_zero

    def test_both_subsets_must_be_scored_on_the_same_sessions(self):
        with pytest.raises(KeyError):
            statistics.delta_ci({"a": 1.0, "b": 2.0}, {"a": 1.0})

    def test_paired_sd_needs_at_least_two_sessions(self):
        assert statistics.paired_sd({"a": 1.0}, {"a": 2.0}) == 0.0
        assert statistics.paired_sd(per_session(0.1, 0.3), per_session(0.2, 0.8)) > 0


class TestAttributionIntervals:
    def test_shapley_intervals_bracket_the_exact_values(self, by_subset):
        from docupilot.evaluation import analysis

        cis = statistics.shapley_ci(by_subset, PLAYERS)
        exact = analysis.shapley(
            {s: float(np.mean(list(v.values()))) for s, v in by_subset.items()}, PLAYERS
        )
        assert set(cis) == set(PLAYERS)
        for player, ci in cis.items():
            assert ci.point == pytest.approx(exact[player])
            assert ci.lo <= ci.point <= ci.hi

    def test_interaction_intervals_cover_every_pair(self, by_subset):
        cis = statistics.interaction_ci(by_subset, PLAYERS)
        assert set(cis) == {("a", "b"), ("a", "c"), ("b", "c")}

    def test_saturation_steps_are_paired_differences_of_the_size_means(self, by_subset):
        steps = statistics.saturation_step_ci(by_subset, PLAYERS)
        assert set(steps) == {1, 2, 3}
        # Every added modality is worth 0.1 by construction of the fixture.
        for k in steps:
            assert steps[k].point == pytest.approx(0.1, abs=1e-9)

    def test_required_sessions_per_step_is_reported_for_every_step(self, by_subset):
        needed = statistics.saturation_step_required_n(by_subset, PLAYERS, 0.05)
        assert set(needed) == {1, 2, 3}
        assert all(n >= 0 for n in needed.values())


class TestSampleSize:
    def test_more_sessions_resolve_smaller_effects(self):
        assert statistics.minimum_detectable_effect(0.2, 100) < \
            statistics.minimum_detectable_effect(0.2, 25)

    def test_noisier_differences_resolve_worse(self):
        assert statistics.minimum_detectable_effect(0.4, 25) > \
            statistics.minimum_detectable_effect(0.1, 25)

    def test_a_single_session_can_resolve_nothing(self):
        assert statistics.minimum_detectable_effect(0.2, 1) == float("inf")

    def test_required_sessions_grows_as_the_effect_of_interest_shrinks(self):
        assert statistics.required_sessions(0.2, 0.05) > statistics.required_sessions(0.2, 0.2)
        assert statistics.required_sessions(0.2, 0.0) == 0

    def test_the_two_are_consistent_with_each_other(self):
        # Feeding the MDE back in must ask for about the corpus one already has.
        sd, n = 0.25, 30
        mde = statistics.minimum_detectable_effect(sd, n)
        assert statistics.required_sessions(sd, mde) == pytest.approx(n, abs=1)
