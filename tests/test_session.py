"""Session I/O: the event log, the two boundary definitions, the event writer."""

import json

import pytest

from conftest import write_session
from docupilot.recording.session import EventWriter, RecordingSession


class TestOpening:
    def test_a_directory_without_a_recording_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="recording.mp4"):
            RecordingSession.from_directory(tmp_path)

    def test_a_directory_without_an_event_log_is_refused(self, tmp_path):
        (tmp_path / "recording.mp4").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="events.json"):
            RecordingSession.from_directory(tmp_path)

    def test_ground_truth_is_optional(self, tmp_path):
        session = write_session(tmp_path / "s")
        assert session.ground_truth_data == []

    def test_paths_hang_off_the_session_directory(self, tmp_path):
        session = write_session(tmp_path / "s")
        assert session.recording_path.parent == session.session_dir
        assert session.events_path.name == "events.json"


class TestEventLog:
    def test_lifecycle_events_are_not_user_input(self, tmp_path):
        session = write_session(tmp_path / "s", events=[
            {"type": "recording_started", "t_ms": 0.0},
            {"type": "mouse_click", "t_ms": 10.0},
            {"type": "key_press", "t_ms": 20.0},
            {"type": "key_release", "t_ms": 25.0},
            {"type": "mouse_scroll", "t_ms": 30.0},
            {"type": "av_stopped", "t_ms": 40.0},
        ])
        assert len(session.read_events()) == 6
        assert [e["t_ms"] for e in session.input_events()] == [10.0, 20.0, 25.0, 30.0]

    def test_filtering_a_log_in_hand_matches_reading_it_again(self, tmp_path):
        session = write_session(tmp_path / "s", events=[
            {"type": "mouse_click", "t_ms": 1.0}, {"type": "av_started", "t_ms": 2.0},
        ])
        assert RecordingSession.input_events_of(session.read_events()) == session.input_events()

    def test_a_broken_log_reads_as_empty_rather_than_raising(self, tmp_path):
        directory = tmp_path / "s"
        write_session(directory)
        (directory / "events.json").write_text("{ not json", encoding="utf-8")
        assert RecordingSession.from_directory(directory).read_events() == []

    def test_the_log_is_not_cached_a_running_session_would_go_stale(self, tmp_path):
        directory = tmp_path / "s"
        session = write_session(directory, events=[])
        assert session.read_events() == []
        (directory / "events.json").write_text(
            json.dumps([{"type": "mouse_click", "t_ms": 1.0}]), encoding="utf-8"
        )
        assert len(session.read_events()) == 1


class TestGroundTruth:
    def test_a_boundary_is_stored_with_its_schema_and_saved_at_once(self, tmp_path):
        directory = tmp_path / "s"
        session = write_session(directory)
        session.add_ground_truth_boundary(1234.5, label="00:01.234")
        entry, = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))
        assert entry["t_ms"] == 1234.5
        assert entry["label"] == "00:01.234"
        assert entry["kind"] == "end"
        assert entry["created_at_utc"]

    def test_markers_come_back_in_chronological_order(self, tmp_path):
        session = write_session(tmp_path / "s")
        session.add_ground_truth_boundary(5000.0)
        session.add_ground_truth_boundary(1000.0)
        assert [t for t, _ in session.ground_truth_markers()] == [1000.0, 5000.0]

    def test_the_two_definitions_are_kept_apart(self, tmp_path):
        session = write_session(tmp_path / "s")
        session.add_ground_truth_boundary(5000.0, kind="end")
        session.add_ground_truth_boundary(6000.0, kind="start")
        assert session.count_boundaries("end") == 1
        assert session.count_boundaries("start") == 1
        assert [t for t, _ in session.ground_truth_markers("end")] == [5000.0]
        assert [t for t, _ in session.ground_truth_markers("start")] == [6000.0]

    def test_an_unknown_definition_is_rejected(self, tmp_path):
        session = write_session(tmp_path / "s")
        with pytest.raises(ValueError, match="Grenzart"):
            session.add_ground_truth_boundary(1.0, kind="middle")

    def test_entries_written_before_the_distinction_count_as_end(self, tmp_path):
        session = write_session(tmp_path / "s", ground_truth=[{"t_ms": 1000.0}])
        assert RecordingSession.boundary_kind({"t_ms": 1.0}) == "end"
        assert session.count_boundaries("end") == 1

    def test_replacing_the_list_persists(self, tmp_path):
        directory = tmp_path / "s"
        session = write_session(directory)
        session.add_ground_truth_boundary(1.0)
        session.set_ground_truth_boundaries([{"t_ms": 42.0}])
        reopened = RecordingSession.from_directory(directory)
        assert [t for t, _ in reopened.ground_truth_markers()] == [42.0]

    def test_a_broken_annotation_file_leaves_the_list_empty(self, tmp_path):
        directory = tmp_path / "s"
        write_session(directory)
        (directory / "ground_truth.json").write_text("{", encoding="utf-8")
        assert RecordingSession.from_directory(directory).ground_truth_data == []


class TestEventWriter:
    def test_the_output_is_a_valid_json_array_of_stamped_events(self, tmp_path):
        path = tmp_path / "events.json"
        writer = EventWriter(path)
        writer.open()
        writer.write({"type": "a", "v": "ä"}, 1.5)
        writer.write({"type": "b"}, 2.0)
        writer.close()
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written == [{"type": "a", "v": "ä", "t_ms": 1.5}, {"type": "b", "t_ms": 2.0}]

    def test_writing_before_open_or_after_close_is_ignored(self, tmp_path):
        path = tmp_path / "events.json"
        writer = EventWriter(path)
        writer.write({"type": "lost"}, 0.0)
        writer.open()
        writer.write({"type": "kept"}, 1.0)
        writer.close()
        writer.write({"type": "lost too"}, 2.0)
        writer.close()
        assert [e["type"] for e in json.loads(path.read_text(encoding="utf-8"))] == ["kept"]

    def test_events_are_readable_before_the_file_is_closed(self, tmp_path):
        # A crash mid-recording must not cost every event up to that point.
        path = tmp_path / "events.json"
        writer = EventWriter(path)
        writer.open()
        writer.write({"type": "a"}, 1.0)
        partial = path.read_text(encoding="utf-8")
        writer.close()
        assert '"type": "a"' in partial
