"""
The ffprobe adapter, the corpus scan, and the synchronisation arithmetic.

ffprobe itself is replaced by its recorded output; what is tested is the parsing
and the decisions built on it.
"""

import json

import numpy as np
import pytest

from conftest import write_session
from docupilot.evaluation import corpus, dataset, media, synchronization

FFPROBE_JSON = json.dumps({
    "streams": [
        {"codec_type": "video", "start_time": "0.000000"},
        {"codec_type": "audio", "start_time": "0.021333"},
    ],
    "format": {"duration": "134.017000"},
})


class TestMediaInfo:
    def test_duration_and_stream_starts_are_read(self):
        info = media.MediaInfo.parse(FFPROBE_JSON)
        assert info.duration_s == pytest.approx(134.017)
        assert info.stream_start_s["video"] == pytest.approx(0.0)
        assert info.stream_start_s["audio"] == pytest.approx(0.021333)

    def test_a_missing_stream_is_absent_rather_than_an_error(self):
        info = media.MediaInfo.parse(json.dumps({"streams": [{"codec_type": "video"}]}))
        assert "audio" not in info.stream_start_s
        assert np.isnan(info.stream_start_s["video"])

    def test_an_unreadable_duration_is_raised_not_guessed(self):
        with pytest.raises(RuntimeError, match="ffprobe lieferte"):
            _ = media.MediaInfo.parse("{}").duration_s
        with pytest.raises(RuntimeError):
            _ = media.MediaInfo.parse("").duration_s

    def test_probe_is_memoised_per_file(self, tmp_path, monkeypatch):
        path = tmp_path / "recording.mp4"
        path.write_bytes(b"some bytes")
        calls = []

        class Result:
            stdout = FFPROBE_JSON

        def fake_run(*args, **kwargs):
            calls.append(args)
            return Result()

        monkeypatch.setattr(media.subprocess, "run", fake_run)
        first = media.probe(path)
        second = media.probe(path)
        # Duration and stream offset are asked for separately per session; one
        # process start is enough for both.
        assert first is second and len(calls) == 1

    def test_a_changed_file_is_probed_again(self, tmp_path, monkeypatch):
        path = tmp_path / "recording.mp4"
        path.write_bytes(b"one")
        monkeypatch.setattr(
            media.subprocess, "run",
            lambda *a, **k: type("R", (), {"stdout": FFPROBE_JSON})(),
        )
        first = media.probe(path)
        path.write_bytes(b"different length")
        assert media.probe(path) is not first


class TestDataset:
    def test_duration_comes_from_the_container(self, tmp_path, monkeypatch):
        session = write_session(tmp_path / "s", recording=b"x")
        monkeypatch.setattr(media, "probe", lambda p: media.MediaInfo.parse(FFPROBE_JSON))
        assert dataset.duration_s(session) == pytest.approx(134.017)

    def test_an_unreadable_duration_names_the_file(self, tmp_path, monkeypatch):
        session = write_session(tmp_path / "s")
        monkeypatch.setattr(media, "probe", lambda p: media.MediaInfo.parse("{}"))
        with pytest.raises(RuntimeError, match="recording.mp4"):
            dataset.duration_s(session)

    def test_ground_truth_is_converted_to_seconds_and_sorted(self, tmp_path):
        session = write_session(tmp_path / "s", ground_truth=[
            {"t_ms": 5000.0, "kind": "end"},
            {"t_ms": 1500.0, "kind": "end"},
            {"t_ms": 9000.0, "kind": "start"},
        ])
        assert dataset.ground_truth_s(session) == [1.5, 5.0]
        assert dataset.ground_truth_s(session, "start") == [9.0]


class TestSynchronisation:
    def test_the_stream_offset_is_the_gap_between_the_two_starts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(media, "probe", lambda p: media.MediaInfo.parse(FFPROBE_JSON))
        assert synchronization.stream_offset_ms(tmp_path / "r.mp4") == pytest.approx(21.333)

    def test_a_missing_audio_stream_leaves_the_offset_unknown(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            media, "probe",
            lambda p: media.MediaInfo.parse(json.dumps({"streams": [{"codec_type": "video"}]})),
        )
        assert np.isnan(synchronization.stream_offset_ms(tmp_path / "r.mp4"))

    def test_rising_edges_are_transitions_not_active_frames(self):
        activity = np.array([0.0, 0.0, 0.9, 0.9, 0.0, 0.9])
        times = np.arange(6, dtype=np.float64)
        # Frames 2, 3 and 5 are active; only 2 and 5 START a burst of activity.
        assert synchronization.rising_edges_s(activity, times, quiet=0.5).tolist() == [2.0, 5.0]

    def test_too_short_a_signal_has_no_edges(self):
        assert synchronization.rising_edges_s(np.array([0.9]), np.array([0.0]), 0.5).size == 0

    def test_a_click_is_matched_to_the_first_reaction_after_it(self):
        offsets = synchronization.reaction_offsets_s(np.array([1.2, 1.8]), [1.0])
        assert offsets.tolist() == [pytest.approx(0.2)]

    def test_motion_already_under_way_is_not_credited_to_the_click(self):
        # An edge well before the click belongs to something else; only the
        # small backward window for measurement noise is allowed.
        assert synchronization.reaction_offsets_s(np.array([0.1]), [1.0]).size == 0
        assert synchronization.reaction_offsets_s(np.array([0.9]), [1.0]).size == 1

    def test_a_click_without_any_reaction_is_dropped(self):
        assert synchronization.reaction_offsets_s(np.array([9.0]), [1.0]).size == 0

    def test_a_session_without_clicks_needs_no_scan(self, tmp_path):
        session = write_session(tmp_path / "s", events=[{"type": "key_press", "t_ms": 1.0}])
        assert synchronization.click_offsets_s(session).size == 0

    def test_the_report_summarises_the_offsets(self, tmp_path, monkeypatch):
        session = write_session(tmp_path / "s")
        monkeypatch.setattr(
            synchronization, "click_offsets_s",
            lambda s, **kw: np.array([0.10, 0.14, 0.20, 0.50]),
        )
        monkeypatch.setattr(synchronization, "stream_offset_ms", lambda p: 0.0)
        report = synchronization.report(session)
        assert report["n"] == 4
        assert report["click_median_ms"] == pytest.approx(170.0)
        assert report["click_absmax_ms"] == pytest.approx(500.0)
        assert report["click_iqr_lo_ms"] <= report["click_median_ms"] <= report["click_iqr_hi_ms"]

    def test_a_session_without_reactions_reports_only_the_stream_offset(
        self, tmp_path, monkeypatch
    ):
        session = write_session(tmp_path / "s")
        monkeypatch.setattr(synchronization, "click_offsets_s", lambda s, **kw: np.zeros(0))
        monkeypatch.setattr(synchronization, "stream_offset_ms", lambda p: 3.0)
        assert synchronization.report(session) == {"stream_offset_ms": 3.0, "n": 0}


class TestCorpusScan:
    def test_a_single_session_directory_is_scanned_as_itself(self, tmp_path):
        directory = tmp_path / "session_one"
        write_session(directory, ground_truth=[{"t_ms": 1.0}, {"t_ms": 2.0}])
        scan = corpus.scan(directory)
        assert [s.name for s in scan.sessions] == ["session_one"]
        assert scan.usable == [directory]
        assert scan.sessions[0].n_boundaries == 2

    def test_a_folder_of_sessions_is_scanned_in_name_order(self, tmp_path):
        for name in ("session_b", "session_a"):
            write_session(tmp_path / name, ground_truth=[{"t_ms": 1.0}])
        (tmp_path / "not_a_session").mkdir()
        scan = corpus.scan(tmp_path)
        assert [s.name for s in scan.sessions] == ["session_a", "session_b"]

    def test_a_session_without_an_annotation_cannot_be_scored(self, tmp_path):
        write_session(tmp_path / "session_a")
        write_session(tmp_path / "session_b", ground_truth=[{"t_ms": 1.0}])
        scan = corpus.scan(tmp_path)
        assert [s.annotated for s in scan.sessions] == [False, True]
        assert scan.usable == [tmp_path / "session_b"]

    def test_the_two_definitions_are_counted_separately(self, tmp_path):
        write_session(tmp_path / "session_a", ground_truth=[
            {"t_ms": 1.0}, {"t_ms": 2.0, "kind": "end"}, {"t_ms": 3.0, "kind": "start"},
        ])
        info, = corpus.scan(tmp_path).sessions
        assert info.n_boundaries == 2 and info.n_start_boundaries == 1

    def test_the_sensitivity_run_needs_every_session_annotated_both_ways(self, tmp_path):
        write_session(tmp_path / "session_a", ground_truth=[
            {"t_ms": 1.0}, {"t_ms": 3.0, "kind": "start"},
        ])
        assert corpus.scan(tmp_path).start_definition_available
        write_session(tmp_path / "session_b", ground_truth=[{"t_ms": 1.0}])
        assert not corpus.scan(tmp_path).start_definition_available

    def test_sessions_without_a_video_cache_are_counted(self, tmp_path):
        write_session(tmp_path / "session_a", ground_truth=[{"t_ms": 1.0}])
        cached = tmp_path / "session_b"
        write_session(cached, ground_truth=[{"t_ms": 1.0}])
        (cached / "gui_vlm_cache.json").write_text("{}", encoding="utf-8")
        # The count is what warns about the model cost of the next run.
        assert corpus.scan(tmp_path).without_video_cache == 1

    def test_one_session_cannot_be_evaluated(self, tmp_path):
        write_session(tmp_path / "session_a", ground_truth=[{"t_ms": 1.0}])
        # Leave-one-session-out has no fold that excludes the scored session.
        assert not corpus.scan(tmp_path).can_evaluate
        write_session(tmp_path / "session_b", ground_truth=[{"t_ms": 1.0}])
        assert corpus.scan(tmp_path).can_evaluate


class TestCorpusDescription:
    """The summary line the window shows; here so its wording needs no display."""

    def test_it_names_the_usable_count_and_what_was_skipped(self, tmp_path):
        write_session(tmp_path / "session_a", ground_truth=[{"t_ms": 1.0}])
        write_session(tmp_path / "session_b")
        text = corpus.describe(corpus.scan(tmp_path))
        assert "1 Sessions verwendbar" in text
        assert "1 ohne Ground Truth übersprungen" in text

    def test_it_warns_about_model_cost_only_when_a_cache_is_missing(self, tmp_path):
        directory = tmp_path / "session_a"
        write_session(directory, ground_truth=[{"t_ms": 1.0}])
        assert "ohne Video-Cache" in corpus.describe(corpus.scan(tmp_path))
        (directory / "gui_vlm_cache.json").write_text("{}", encoding="utf-8")
        assert "keine Modellkosten" in corpus.describe(corpus.scan(tmp_path))

    def test_it_says_whether_the_sensitivity_run_can_happen(self, tmp_path):
        write_session(tmp_path / "session_a", ground_truth=[{"t_ms": 1.0}])
        assert "kein Sensitivitätslauf" in corpus.describe(corpus.scan(tmp_path))
        write_session(tmp_path / "session_b", ground_truth=[
            {"t_ms": 1.0}, {"t_ms": 2.0, "kind": "start"},
        ])
        write_session(tmp_path / "session_a", ground_truth=[
            {"t_ms": 1.0}, {"t_ms": 2.0, "kind": "start"},
        ])
        assert "Sensitivitätslauf aktiv" in corpus.describe(corpus.scan(tmp_path))

    def test_a_corpus_too_small_to_score_says_so(self, tmp_path):
        write_session(tmp_path / "session_a", ground_truth=[{"t_ms": 1.0}])
        assert "mindestens zwei Sessions" in corpus.describe(corpus.scan(tmp_path))
