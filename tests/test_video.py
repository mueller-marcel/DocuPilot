"""
The video modality without a video: the activity signal is a function of a frame
sequence, the dwells of that signal, and the anchor walk of the dwells plus two
callables. Decoding is covered separately in test_media_adapters.py.
"""

import numpy as np
import pytest

from docupilot.segmentation import video
from docupilot.segmentation.video_scoring import Judgement


def grey(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size), value, dtype=np.uint8)


def textured(seed: int, size: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(size, size), dtype=np.uint8)


class TestActivitySignal:
    def test_a_still_sequence_has_no_activity(self):
        frames = [grey(120)] * 5
        assert video.frame_activity(frames).tolist() == [0.0] * 5

    def test_the_first_frame_has_no_predecessor_and_scores_zero(self):
        assert video.frame_activity([textured(1), textured(2)])[0] == 0.0

    def test_a_changed_frame_registers_activity(self):
        activity = video.frame_activity([textured(1), textured(1), textured(2)])
        assert activity[1] == 0.0                    # identical to its predecessor
        assert activity[2] > video.ACTIVITY_QUIET

    def test_a_change_in_one_tile_is_not_averaged_away(self):
        # The whole point of tiles plus a maximum: a small region changing must
        # mark the frame active, even though 63 of 64 tiles are untouched.
        before = grey(128)
        after = before.copy()
        after[:8, :8] = textured(3, 8)
        activity = video.frame_activity([before, after])
        assert activity[1] > video.ACTIVITY_QUIET

    def test_activity_stays_in_the_unit_interval(self):
        activity = video.frame_activity([textured(i) for i in range(6)])
        assert np.all((activity >= 0.0) & (activity <= 1.0))

    def test_an_empty_sequence_yields_an_empty_signal(self):
        assert video.frame_activity([]).shape == (0,)


class TestDwells:
    def test_runs_below_the_quiet_threshold_become_dwells(self):
        activity = np.array([0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0])
        assert video.dwells(activity, min_frames=2) == [(0, 2), (5, 7)]

    def test_runs_shorter_than_the_minimum_are_discarded(self):
        # The first guard against over-segmentation: a rest too short to be a
        # settled state is not one.
        activity = np.array([0.0, 0.9, 0.0, 0.9, 0.0, 0.0, 0.0])
        assert video.dwells(activity, min_frames=3) == [(4, 6)]

    def test_bounds_are_inclusive_and_runs_do_not_overlap(self):
        activity = np.array([0.0] * 4 + [0.5] + [0.0] * 4)
        found = video.dwells(activity, min_frames=1, quiet=0.1)
        assert found == [(0, 3), (5, 8)]

    def test_a_fully_active_recording_has_no_dwells(self):
        assert video.dwells(np.full(10, 0.9), min_frames=1) == []

    def test_settled_frame_is_sampled_into_the_dwell_but_never_past_it(self):
        # A long dwell is sampled 0.2 s in; a one-frame dwell stays where it is.
        assert video.settled_frames([(0, 100)], fps=10.0) == [2]
        assert video.settled_frames([(7, 7)], fps=10.0) == [7]


class TestChangedRegion:
    def test_identical_frames_have_no_region(self):
        assert video.changed_region(grey(100).astype(np.int16),
                                    grey(100).astype(np.int16)) is None

    def test_a_corner_change_is_boxed_in_that_corner(self):
        before = grey(100).astype(np.int16)
        after = before.copy()
        after[:8, :8] = 200
        box = video.changed_region(before, after)
        assert box is not None
        x0, y0, x1, y1 = box
        assert (x0, y0) == (0.0, 0.0)
        assert x1 <= 0.2 and y1 <= 0.2           # one tile of an 8x8 grid

    def test_the_box_spans_every_changed_tile(self):
        before = grey(100).astype(np.int16)
        after = before.copy()
        after[:8, :8] = 200
        after[56:, 56:] = 0
        x0, y0, x1, y1 = video.changed_region(before, after)
        assert (x0, y0) == (0.0, 0.0) and (x1, y1) == (1.0, 1.0)

    def test_noise_below_the_epsilon_does_not_open_a_region(self):
        before = grey(100).astype(np.int16)
        assert video.changed_region(before, before + 1) is None


class TestAnchorWalk:
    """The core decision: the anchor advances only on an accepted boundary."""

    @staticmethod
    def steps(n: int) -> list[tuple[int, int]]:
        return [(i * 10, i) for i in range(n)]

    def test_the_first_dwell_becomes_the_anchor_and_is_never_judged(self):
        asked = []
        video.walk_dwells(self.steps(3), lambda a, c: True,
                          lambda a, c: (asked.append((a, c)), Judgement("X", 0.1))[1])
        assert [a for a, _ in asked] == [0, 0]

    def test_a_rejected_state_leaves_the_anchor_where_it_was(self):
        # The menu case: opening a menu and stepping into a submenu are both
        # rejected, so the deciding click is judged against the state the user
        # started from, not against the submenu.
        asked = []

        def judge(anchor, current):
            asked.append((anchor, current))
            return Judgement("TRANSIENT_UI", 0.2)

        video.walk_dwells(self.steps(4), lambda a, c: True, judge)
        assert asked == [(0, 1), (0, 2), (0, 3)]

    def test_an_accepted_state_becomes_the_new_anchor(self):
        asked = []

        def judge(anchor, current):
            asked.append((anchor, current))
            return Judgement("ACTION_COMPLETED", 0.9)

        video.walk_dwells(self.steps(4), lambda a, c: True, judge)
        assert asked == [(0, 1), (1, 2), (2, 3)]

    def test_verdicts_carry_the_dwell_start_not_the_judged_frame(self):
        verdicts = video.walk_dwells(
            self.steps(3), lambda a, c: True, lambda a, c: Judgement("A", 0.9)
        )
        assert [v.dwell_start for v in verdicts] == [10, 20]

    def test_an_unchanged_pair_costs_no_model_call(self):
        calls = []
        verdicts = video.walk_dwells(
            self.steps(4), lambda a, c: False,
            lambda a, c: calls.append(1) or Judgement("A", 0.9),
        )
        assert calls == [] and verdicts == []

    def test_an_unusable_answer_leaves_no_evidence_but_still_costs_a_call(self):
        seen = []
        verdicts = video.walk_dwells(
            self.steps(3), lambda a, c: True,
            lambda a, c: None, on_call=seen.append,
        )
        assert verdicts == [] and seen == [1, 2]

    def test_the_call_budget_stops_the_walk_and_says_so(self):
        limits = []
        verdicts = video.walk_dwells(
            self.steps(10), lambda a, c: True, lambda a, c: Judgement("X", 0.1),
            max_calls=3, on_limit=lambda: limits.append(True),
        )
        # A silent cap would read as "everything checked".
        assert len(verdicts) == 3 and limits == [True]

    def test_cancelling_keeps_what_was_already_judged(self):
        judged = []

        def judge(anchor, current):
            judged.append(current)
            return Judgement("X", 0.1)

        verdicts = video.walk_dwells(
            self.steps(10), lambda a, c: True, judge,
            is_cancelled=lambda: len(judged) >= 2,
        )
        assert len(verdicts) == 2

    def test_the_threshold_decides_acceptance(self):
        exactly_at = video.walk_dwells(
            self.steps(3), lambda a, c: True,
            lambda a, c: Judgement("A", 0.5), threshold=0.5,
        )
        assert all(v.judgement.p_boundary >= 0.5 for v in exactly_at)


class TestEvidenceFromVerdicts:
    def test_accepted_verdicts_become_boundaries_at_their_dwell_start(self):
        times = np.arange(0, 10, 0.1)
        verdicts = [
            video.DwellVerdict(10, Judgement("A", 0.9)),
            video.DwellVerdict(50, Judgement("T", 0.2)),
        ]
        ev = video.evidence_from_verdicts(verdicts, times, fps=10.0)
        assert ev.boundaries_s == [pytest.approx(1.0)]
        assert ev.score[10] == pytest.approx(0.9)
        assert ev.score[50] == pytest.approx(0.2)

    def test_the_curve_is_graded_never_a_decision(self):
        times = np.arange(0, 5, 0.1)
        ev = video.evidence_from_verdicts(
            [video.DwellVerdict(20, Judgement("T", 0.35))], times, fps=10.0
        )
        # A rejected state still leaves its evidence: the fusion downstream must
        # be free to accept what this modality alone would not.
        assert ev.boundaries_s == [] and ev.score.max() == pytest.approx(0.35)


class TestFrameTimes:
    def test_times_are_zeroed_on_the_first_frame(self):
        times = video.parse_frame_times("10.0\n10.1\n10.2\n", 3)
        assert times.tolist() == pytest.approx([0.0, 0.1, 0.2])

    def test_blank_lines_are_ignored(self):
        assert len(video.parse_frame_times("1.0\n\n2.0\n\n", 2)) == 2

    def test_a_count_mismatch_is_loud(self):
        # Container and decoder disagreeing would put the modality on a wrong
        # time axis; guessing which one is right is not an option.
        with pytest.raises(RuntimeError, match="inkonsistente Aufnahme"):
            video.parse_frame_times("1.0\n2.0\n", 5)


class TestActivityCache:
    """
    The stored scan, with the decoder stubbed out.

    What is worth testing here is when a stored scan may be reused, not whether
    OpenCV can read an MP4 — that is the library's own business and is verified
    by running the application.
    """

    @staticmethod
    def stub(monkeypatch, calls: list, n_frames: int = 5):
        monkeypatch.setattr(video, "_scan", lambda path: (
            calls.append(path) or (n_frames, np.linspace(0, 1, n_frames, dtype=np.float32), 10.0)
        ))
        monkeypatch.setattr(
            video, "_frame_times_s", lambda path, n: np.arange(n, dtype=np.float64) / 10.0
        )

    def test_a_scan_is_stored_and_reused(self, session, monkeypatch):
        calls: list = []
        self.stub(monkeypatch, calls)

        first = video.scan_activity(session)
        assert (session.session_dir / "video_activity.npz").exists()
        second = video.scan_activity(session)

        assert len(calls) == 1                       # decoded once, not twice
        assert second.n_frames == first.n_frames and second.fps == first.fps
        np.testing.assert_array_equal(second.activity, first.activity)
        np.testing.assert_array_equal(second.times_s, first.times_s)

    def test_a_changed_recording_is_scanned_again(self, session, monkeypatch):
        calls: list = []
        self.stub(monkeypatch, calls)
        video.scan_activity(session)
        before = video._activity_key(session)

        session.recording_path.write_bytes(b"different bytes")
        # The key is the recording's CONTENT, so a changed file can never be
        # served the previous scan.
        assert video._activity_key(session) != before
        video.scan_activity(session)
        assert len(calls) == 2

    def test_a_damaged_store_is_a_miss_never_a_crash(self, session, monkeypatch):
        calls: list = []
        self.stub(monkeypatch, calls)
        video.scan_activity(session)
        (session.session_dir / "video_activity.npz").write_bytes(b"not an npz")
        video.scan_activity(session)
        assert len(calls) == 2

    def test_use_cache_false_neither_reads_nor_writes(self, session, monkeypatch):
        calls: list = []
        self.stub(monkeypatch, calls)
        video.scan_activity(session, use_cache=False)
        assert not (session.session_dir / "video_activity.npz").exists()
        video.scan_activity(session, use_cache=False)
        assert len(calls) == 2


class TestExtractGuards:
    def test_extract_refuses_to_run_without_a_backend(self, session, monkeypatch):
        from docupilot.segmentation import video_scoring

        monkeypatch.setattr(video_scoring, "is_available", lambda: False)
        # Failing loudly beats emitting a structural-only lane, which would
        # measure how many pixels moved rather than what it means.
        with pytest.raises(RuntimeError, match="nicht nutzbar"):
            video.extract(session, use_cache=False)

    def test_a_recording_without_dwells_yields_an_empty_lane(self, session, monkeypatch):
        from docupilot.segmentation import video_scoring

        monkeypatch.setattr(video_scoring, "is_available", lambda: True)
        monkeypatch.setattr(video, "scan_activity", lambda s, use_cache=True: video.ActivityScan(
            n_frames=5, activity=np.full(5, 0.9, dtype=np.float32), fps=10.0,
            times_s=np.arange(5) / 10.0,
        ))
        ev = video.extract(session, use_cache=False)
        assert ev.boundaries_s == [] and not ev.score.any()

    def test_asking_for_no_frames_decodes_nothing(self):
        # Returns before OpenCV is even imported, so an empty dwell list costs
        # nothing on a machine without the decoder.
        assert video._read_frames("does-not-exist.mp4", set()) == {}
