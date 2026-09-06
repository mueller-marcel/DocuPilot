"""
The events modality: input bursts split at a pause, each scored by the rest
after it. Reads one JSON file, so every case is written out in full.
"""

import numpy as np
import pytest

from conftest import write_session
from docupilot.segmentation import events


def clicks_at(*seconds: float, ends_at: float | None = None) -> list[dict]:
    """
    An event log with a click at each given second.

    The stop marker is placed explicitly, because the LAST burst is scored
    against the end of the log — a generous tail would silently give it a full
    rest and mask what the test is about.
    """
    log = [{"type": "recording_started", "t_ms": 0.0}]
    log += [{"type": "mouse_click", "t_ms": s * 1000.0, "x": 1, "y": 2} for s in seconds]
    end = ends_at if ends_at is not None else max(seconds, default=0.0) + 0.5
    log.append({"type": "recording_stopped", "t_ms": end * 1000.0})
    return log


def extract_from(tmp_path, *seconds: float, ends_at: float | None = None):
    log = clicks_at(*seconds, ends_at=ends_at)
    return events.extract(write_session(tmp_path / "s", events=log))


class TestBursts:
    def test_inputs_closer_than_the_pause_form_one_burst(self):
        assert events._bursts([0.0, 0.5, 1.0]) == [(0.0, 1.0)]

    def test_a_pause_beyond_the_threshold_splits_the_burst(self):
        # 2 s is imported from keystroke-logging research, not fitted here.
        assert events._bursts([0.0, 0.5, 5.0, 5.2]) == [(0.0, 0.5), (5.0, 5.2)]

    def test_no_input_no_bursts(self):
        assert events._bursts([]) == []

    def test_a_single_input_is_its_own_burst(self):
        assert events._bursts([3.0]) == [(3.0, 3.0)]


class TestScoring:
    def test_a_candidate_sits_at_the_end_of_a_burst(self, tmp_path):
        ev = extract_from(tmp_path, 1.0, 1.2, 1.4)
        # The action completes after the last input of the burst, not the first.
        assert ev.score.argmax() / 50.0 == pytest.approx(1.4, abs=0.05)

    def test_a_longer_rest_scores_higher(self, tmp_path):
        short = extract_from(tmp_path / "a", 1.0, 4.0, ends_at=4.5)
        long = extract_from(tmp_path / "b", 1.0, 9.0, ends_at=9.5)
        assert long.score.max() > short.score.max()

    def test_the_score_saturates_at_the_full_rest(self, tmp_path):
        end = 1.0 + events._REST_FULL_S + 5.0
        ev = extract_from(tmp_path, 1.0, end, ends_at=end + 0.5)
        assert ev.score.max() == pytest.approx(1.0)

    def test_a_burst_followed_by_a_short_rest_is_no_boundary(self, tmp_path):
        ev = extract_from(tmp_path, 1.0, 4.0, ends_at=4.5)
        # rest = 3 s of 8 s -> 0.375, below the modality's own threshold; the
        # last burst rests only 0.5 s, so nothing here is committed to.
        assert ev.boundaries_s == []
        assert 0.0 < ev.score.max() < 0.5

    def test_a_long_rest_makes_the_modality_commit(self, tmp_path):
        ev = extract_from(tmp_path, 1.0, 9.0, ends_at=9.5)
        assert len(ev.boundaries_s) == 1
        assert ev.boundaries_s[0] == pytest.approx(1.0, abs=0.05)

    def test_the_last_burst_is_scored_against_the_end_of_the_log(self, tmp_path):
        # There is no following burst to measure the rest against; the log's own
        # end closes it, so the modality never reads another modality's length.
        brief = extract_from(tmp_path / "a", 1.0, ends_at=2.0)
        ample = extract_from(tmp_path / "b", 1.0, ends_at=20.0)
        assert brief.score.max() < ample.score.max() == pytest.approx(1.0)


class TestExtract:
    def test_an_empty_log_yields_empty_evidence(self, tmp_path):
        ev = events.extract(write_session(tmp_path / "s", events=[]))
        assert ev.times_s.size == 0 and ev.boundaries_s == []

    def test_the_log_states_its_own_duration(self, tmp_path):
        # Taking the length from the video would make this modality depend on
        # another one, which the ablation forbids.
        session = write_session(tmp_path / "s", events=[
            {"type": "mouse_click", "t_ms": 1000.0},
            {"type": "recording_stopped", "t_ms": 30000.0},
        ])
        assert events._log_duration_s(session) == pytest.approx(30.0)
        assert events.extract(session).times_s[-1] == pytest.approx(30.0, abs=0.05)

    def test_without_a_lifecycle_marker_the_last_event_ends_the_log(self, tmp_path):
        session = write_session(tmp_path / "s", events=[
            {"type": "mouse_click", "t_ms": 1000.0},
            {"type": "key_press", "t_ms": 7000.0},
        ])
        assert events._log_duration_s(session) == pytest.approx(7.0)

    def test_only_user_input_counts_as_a_candidate(self, tmp_path):
        session = write_session(tmp_path / "s", events=[
            {"type": "recording_started", "t_ms": 0.0},
            {"type": "av_started", "t_ms": 100.0},
            {"type": "mouse_click", "t_ms": 5000.0},
            {"type": "recording_stopped", "t_ms": 20000.0},
        ])
        assert [t for t, _ in events.input_markers(session)] == [5000.0]

    def test_the_curve_is_graded_and_bounded(self, tmp_path):
        ev = extract_from(tmp_path, 1.0, 5.0, 12.0, 30.0, ends_at=40.0)
        assert np.all((ev.score >= 0.0) & (ev.score <= 1.0))
        assert ev.score.dtype == np.float32
