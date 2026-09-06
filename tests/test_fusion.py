"""
How a set of modalities becomes one prediction: candidates, the features that
describe them, the calibrated operating point, and the forest.
"""

import numpy as np
import pytest

from docupilot.evaluation import fusion
from docupilot.segmentation.evidence import BoundaryEvidence
from synthetic import corpus


@pytest.fixture(scope="module")
def session():
    return corpus()[0]


def lane(times, scores) -> BoundaryEvidence:
    return BoundaryEvidence(
        np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float32), []
    )


class TestCandidates:
    def test_candidates_are_the_peaks_of_the_given_modalities_only(self, session):
        video_only = fusion.candidate_times(session.evidence, ["video"])
        both = fusion.candidate_times(session.evidence, ["video", "audio"])
        assert set(video_only) <= set(both)
        assert len(both) > len(video_only)

    def test_candidates_come_back_sorted(self, session):
        times = fusion.candidate_times(session.evidence, ["video", "audio", "events"])
        assert np.all(np.diff(times) >= 0)

    def test_an_empty_subset_proposes_nothing(self, session):
        assert fusion.candidate_times(session.evidence, []).size == 0

    def test_peaks_are_taken_rather_than_the_modality_s_own_boundaries(self):
        # A modality commits at 0.5; taking those would cap recall at each
        # modality's threshold and let the classifier only ever remove.
        evidence = {"m": lane([0.0, 1.0, 2.0], [0.0, 0.3, 0.0])}
        assert fusion.candidate_times(evidence, ["m"]).tolist() == [1.0]

    def test_a_lane_too_short_to_have_a_peak_is_skipped(self):
        assert fusion.candidate_times({"m": lane([0.0, 1.0], [0.1, 0.9])}, ["m"]).size == 0


class TestFeatures:
    def test_one_block_of_columns_per_modality_in_the_given_order(self, session):
        times = fusion.candidate_times(session.evidence, ["video"])
        both = fusion.feature_matrix(times, session.evidence, ["audio", "video"])
        assert both.shape == (len(times), 2 * fusion.FEATURES_PER_MODALITY)
        alone = fusion.feature_matrix(times, session.evidence, ["audio"])
        # A modality's columns depend on its own lane alone — that independence
        # is what makes the ablation mean anything.
        np.testing.assert_array_equal(both[:, fusion.block_columns(0)], alone)

    def test_block_columns_addresses_the_right_slice(self):
        assert fusion.block_columns(0)[0] == 0
        assert fusion.block_columns(1)[0] == fusion.FEATURES_PER_MODALITY
        assert len(fusion.block_columns(2)) == fusion.FEATURES_PER_MODALITY

    def test_the_point_feature_is_the_score_at_the_candidate(self):
        evidence = {"m": lane([0.0, 1.0, 2.0], [0.1, 0.8, 0.2])}
        row = fusion.feature_matrix(np.array([1.0]), evidence, ["m"])[0]
        assert row[list(fusion.FEATURE_NAMES).index("point")] == pytest.approx(0.8)

    def test_wider_windows_can_only_report_more(self):
        evidence = {"m": lane(np.arange(0, 5, 0.1), np.zeros(50))}
        evidence["m"].score[45] = 0.9              # a peak 2.5 s away from t=2.0
        row = fusion.feature_matrix(np.array([2.0]), evidence, ["m"])[0]
        names = list(fusion.FEATURE_NAMES)
        widths = [row[names.index(f"max_{w:g}s")] for w in fusion.FEATURE_WINDOWS_S]
        assert widths == sorted(widths)

    def test_the_window_set_lets_a_sharp_peak_be_told_from_a_plateau(self):
        grid = np.arange(0, 6, 0.1)
        sharp = np.zeros(60)
        sharp[30] = 1.0
        broad = np.zeros(60)
        broad[10:50] = 1.0
        names = list(fusion.FEATURE_NAMES)
        rows = {
            name: fusion.feature_matrix(np.array([3.0]), {name: lane(grid, curve)}, [name])[0]
            for name, curve in (("sharp", sharp), ("broad", broad))
        }
        # Same peak height, different width — video's precision and audio's
        # interval must be distinguishable, or the forest cannot use either.
        assert rows["sharp"][names.index("max_0.5s")] == rows["broad"][names.index("max_0.5s")]
        narrow_sharp = rows["sharp"][names.index("point")]
        assert narrow_sharp == pytest.approx(rows["broad"][names.index("point")])
        assert fusion.FEATURE_WINDOWS_S == (0.5, 1.0, 2.0)

    def test_the_rank_feature_is_relative_within_the_session(self):
        grid = np.arange(0, 10, 0.1)
        curve = np.zeros(100)
        for i, v in zip((10, 30, 50), (0.2, 0.5, 0.9)):
            curve[i] = v
        row = fusion.feature_matrix(
            np.array([1.0, 3.0, 5.0]), {"m": lane(grid, curve)}, ["m"]
        )
        rank = row[:, list(fusion.FEATURE_NAMES).index("rank")]
        assert list(rank) == sorted(rank)
        assert rank.max() == pytest.approx(1.0) and rank.min() > 0

    def test_no_candidates_or_no_modalities_give_an_empty_matrix(self, session):
        times = fusion.candidate_times(session.evidence, ["video"])
        assert fusion.feature_matrix(times, session.evidence, []).shape == (len(times), 0)
        assert fusion.feature_matrix(
            np.zeros(0), session.evidence, ["video"]
        ).shape == (0, fusion.FEATURES_PER_MODALITY)

    def test_reading_off_the_end_of_a_lane_is_zero_not_an_error(self):
        evidence = {"m": lane([0.0, 1.0], [0.5, 0.5])}
        row = fusion.feature_matrix(np.array([99.0]), evidence, ["m"])[0]
        assert row[list(fusion.FEATURE_NAMES).index("max_0.5s")] == 0.0


class TestLabelsAndDecision:
    def test_a_candidate_within_the_tolerance_is_positive(self):
        labels = fusion.label_candidates(np.array([1.0, 1.4, 5.0, 9.0]), [1.2, 8.5], 0.5)
        assert labels.tolist() == [True, True, False, True]

    def test_labelling_is_deliberately_not_one_to_one(self):
        # Two candidates near one boundary are both legitimately positive; the
        # one-to-one rule belongs to the evaluation, not the training signal.
        assert fusion.label_candidates(np.array([1.0, 1.1]), [1.05], 0.5).all()

    def test_nothing_to_label_yields_nothing(self):
        assert fusion.label_candidates(np.zeros(0), [1.0], 1.0).size == 0
        assert not fusion.label_candidates(np.array([1.0]), [], 1.0).any()

    def test_suppression_keeps_the_strongest_of_a_cluster(self):
        times = np.array([1.0, 1.4, 5.0, 9.0])
        assert fusion.suppress(times, np.array([0.5, 0.9, 0.1, 0.7])) == [1.4, 5.0, 9.0]

    def test_suppression_returns_ascending_times(self):
        times = np.array([9.0, 1.0, 5.0])
        assert fusion.suppress(times, np.array([0.9, 0.8, 0.7])) == [1.0, 5.0, 9.0]

    def test_decide_thresholds_then_suppresses(self):
        times = np.array([1.0, 1.4, 5.0, 9.0])
        proba = np.array([0.5, 0.9, 0.1, 0.7])
        assert fusion.decide(times, proba, 0.6) == [1.4, 9.0]
        assert fusion.decide(times, proba, 0.99) == []


class TestThresholdCalibration:
    def test_the_threshold_that_maximises_f1_is_chosen(self):
        times = np.array([1.0, 5.0, 9.0])
        # Only the middle candidate is a boundary; the cut has to fall between
        # 0.4 and 0.8 to score a perfect F1.
        folds = [(times, np.array([0.4, 0.8, 0.4]), [5.0])]
        chosen = fusion.choose_threshold(folds, 1.0)
        assert 0.4 < chosen <= 0.8
        assert fusion.decide(times, np.array([0.4, 0.8, 0.4]), chosen) == [5.0]

    def test_a_flat_optimum_falls_back_to_the_centre(self):
        times = np.array([1.0, 5.0, 9.0])
        folds = [(times, np.array([0.7, 0.7, 0.7]), [1.0, 5.0, 9.0])]
        # Every threshold up to 0.7 scores the same; an arbitrary grid order
        # must not push the operating point to an extreme.
        assert fusion.choose_threshold(folds, 1.0) == pytest.approx(0.5)

    def test_the_threshold_is_averaged_over_the_folds_not_fitted_per_session(self):
        times = np.array([1.0, 5.0])
        folds = [
            (times, np.array([0.9, 0.1]), [1.0]),
            (times, np.array([0.3, 0.9]), [5.0]),
        ]
        chosen = fusion.choose_threshold(folds, 1.0)
        assert 0.3 < chosen <= 0.9

    def test_nothing_to_calibrate_on_leaves_the_neutral_point(self):
        assert fusion.choose_threshold([], 1.0) == 0.5

    def test_the_grid_is_coarse_and_excludes_the_degenerate_ends(self):
        # A finer grid would fit the training folds' noise; 0.05 steps move F1
        # by less than the corpus can resolve anyway.
        assert min(fusion.THRESHOLD_GRID) == 0.05
        assert max(fusion.THRESHOLD_GRID) == 0.95
        assert 0.5 in fusion.THRESHOLD_GRID


class TestForestFuser:
    def test_it_learns_a_separable_signal(self):
        rng = np.random.default_rng(0)
        features = rng.uniform(size=(120, 3))
        labels = features[:, 0] > 0.5
        fuser = fusion.ForestFuser(n_estimators=60).fit(features, labels)
        proba = fuser.predict_proba(features)
        assert proba[labels].mean() > proba[~labels].mean() + 0.3

    def test_out_of_bag_probabilities_cover_every_training_row(self):
        rng = np.random.default_rng(1)
        features = rng.uniform(size=(80, 3))
        fuser = fusion.ForestFuser(n_estimators=60).fit(features, features[:, 0] > 0.5)
        oob = fuser.oob_proba()
        # Leakage-free predictions for the training sessions, without a second fit.
        assert oob.shape == (80,) and np.all((oob >= 0.0) & (oob <= 1.0))

    def test_asking_for_out_of_bag_before_fitting_is_an_error(self):
        with pytest.raises(RuntimeError):
            fusion.ForestFuser().oob_proba()

    def test_a_fold_with_one_label_answers_with_that_label(self):
        # A modality whose proposals are all correct must not be scored as
        # predicting nothing.
        fuser = fusion.ForestFuser().fit(np.zeros((3, 2)), np.ones(3, dtype=bool))
        np.testing.assert_array_equal(fuser.predict_proba(np.zeros((2, 2))), [1.0, 1.0])
        np.testing.assert_array_equal(fuser.oob_proba(), [1.0, 1.0, 1.0])

        never = fusion.ForestFuser().fit(np.zeros((3, 2)), np.zeros(3, dtype=bool))
        np.testing.assert_array_equal(never.predict_proba(np.zeros((1, 2))), [0.0])

    def test_predicting_on_nothing_returns_nothing(self):
        assert fusion.ForestFuser().predict_proba(np.zeros((0, 3))).shape == (0,)

    def test_the_same_seed_gives_the_same_model(self):
        rng = np.random.default_rng(2)
        features = rng.uniform(size=(60, 3))
        labels = features[:, 1] > 0.5
        a = fusion.ForestFuser(n_estimators=40, seed=7).fit(features, labels)
        b = fusion.ForestFuser(n_estimators=40, seed=7).fit(features, labels)
        np.testing.assert_allclose(a.predict_proba(features), b.predict_proba(features))
