"""
The lane store and the pipeline that fills it.

A stored lane claims "this is what the extractor would produce for exactly
these inputs", so the tests are about when that claim must be withdrawn.
"""

import numpy as np
import pytest

from conftest import write_session
from docupilot.segmentation import MODALITIES, pipeline, store
from docupilot.segmentation.evidence import BoundaryEvidence


def lane(*values: float) -> BoundaryEvidence:
    return BoundaryEvidence(
        times_s=np.arange(len(values), dtype=np.float64),
        score=np.asarray(values, dtype=np.float32),
        boundaries_s=[float(i) for i, v in enumerate(values) if v >= 0.5],
    )


class TestFileDigest:
    def test_the_same_bytes_give_the_same_digest(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.write_bytes(b"identical")
        b.write_bytes(b"identical")
        assert store.file_digest(a) == store.file_digest(b)

    def test_different_bytes_give_different_digests(self, tmp_path):
        path = tmp_path / "f"
        path.write_bytes(b"one")
        first = store.file_digest(path)
        path.write_bytes(b"two")
        assert store.file_digest(path) != first

    def test_a_missing_file_is_a_stable_value_not_an_error(self, tmp_path):
        assert store.file_digest(tmp_path / "absent") == "<missing>"


class TestStore:
    def test_a_lane_survives_a_round_trip(self, tmp_path):
        session = write_session(tmp_path / "s")
        store.save(session, "events", lane(0.1, 0.9, 0.2))
        restored = store.load(session, "events")
        np.testing.assert_allclose(restored.score, [0.1, 0.9, 0.2], rtol=1e-6)
        np.testing.assert_allclose(restored.times_s, [0.0, 1.0, 2.0])
        assert restored.boundaries_s == [1.0]

    def test_nothing_stored_is_a_miss(self, tmp_path):
        assert store.load(write_session(tmp_path / "s"), "events") is None

    def test_changed_inputs_invalidate_the_lane(self, tmp_path):
        directory = tmp_path / "s"
        session = write_session(directory, events=[{"type": "mouse_click", "t_ms": 1.0}])
        store.save(session, "events", lane(0.9))
        assert store.load(session, "events") is not None
        (directory / "events.json").write_text("[]", encoding="utf-8")
        # Serving a lane for inputs it was not computed from is worse than any
        # scan it saves.
        assert store.load(session, "events") is None

    def test_the_annotation_is_deliberately_not_part_of_the_key(self, tmp_path):
        directory = tmp_path / "s"
        session = write_session(directory)
        store.save(session, "events", lane(0.9))
        session.add_ground_truth_boundary(1000.0)
        # Annotating changes what a lane is compared against, never what it
        # contains; re-annotating must not cost a re-extraction.
        assert store.load(session, "events") is not None

    def test_a_damaged_file_is_a_miss_never_a_crash(self, tmp_path):
        session = write_session(tmp_path / "s")
        store.save(session, "events", lane(0.9))
        store.path_for(session, "events").write_bytes(b"not an npz")
        assert store.load(session, "events") is None

    def test_lanes_of_different_modalities_do_not_collide(self, tmp_path):
        session = write_session(tmp_path / "s")
        store.save(session, "events", lane(0.1))
        store.save(session, "audio", lane(0.9))
        assert store.load(session, "events").score.tolist() == [pytest.approx(0.1)]
        assert store.load(session, "audio").score.tolist() == [pytest.approx(0.9)]

    def test_a_precomputed_fingerprint_is_honoured(self, tmp_path):
        session = write_session(tmp_path / "s")
        fingerprint = store.fingerprint(session, "events")
        store.save(session, "events", lane(0.9), fingerprint)
        assert store.load(session, "events", fingerprint) is not None
        assert store.load(session, "events", "some other value") is None

    def test_write_npz_is_atomic_and_leaves_no_temporary_behind(self, tmp_path):
        path = tmp_path / "out.npz"
        store.write_npz(path, a=np.arange(3))
        assert path.exists()
        assert not list(tmp_path.glob("*.tmp"))


class TestPipeline:
    def test_the_cheapest_modality_is_reported_first(self):
        # A caller rendering results as they arrive fills the screen in this order.
        assert MODALITIES == ("events", "video", "audio")

    def test_every_modality_is_reported_and_one_failure_does_not_stop_the_rest(
        self, tmp_path, monkeypatch
    ):
        from docupilot.segmentation import audio, video

        session = write_session(tmp_path / "s")
        monkeypatch.setattr(video, "extract", lambda s, **kw: lane(0.9))
        monkeypatch.setattr(
            audio, "extract",
            lambda s, **kw: (_ for _ in ()).throw(RuntimeError("kein Backend")),
        )
        results, errors = {}, {}
        pipeline.segment(session, results.__setitem__, errors.__setitem__)
        assert set(results) == {"events", "video"}
        assert "kein Backend" in errors["audio"]

    def test_a_stored_lane_is_served_instead_of_recomputed(self, tmp_path, monkeypatch):
        from docupilot.segmentation import audio, video

        session = write_session(tmp_path / "s")
        calls = []
        monkeypatch.setattr(video, "extract", lambda s, **kw: calls.append(1) or lane(0.9))
        monkeypatch.setattr(audio, "extract", lambda s, **kw: lane(0.1))

        pipeline.segment(session, lambda *a: None, lambda *a: None)
        pipeline.segment(session, lambda *a: None, lambda *a: None)
        assert calls == [1]

    def test_a_cancelled_extraction_is_never_stored(self, tmp_path, monkeypatch):
        from docupilot.segmentation import audio, video

        session = write_session(tmp_path / "s")
        monkeypatch.setattr(video, "extract", lambda s, **kw: lane(0.9))
        monkeypatch.setattr(audio, "extract", lambda s, **kw: lane(0.1))
        # A truncated lane is indistinguishable from a complete one once read back.
        pipeline.segment(session, lambda *a: None, lambda *a: None,
                         is_cancelled=lambda: True)
        assert not store.path_for(session, "video").exists()

    def test_use_cache_false_neither_reads_nor_writes(self, tmp_path, monkeypatch):
        from docupilot.segmentation import audio, video

        session = write_session(tmp_path / "s")
        monkeypatch.setattr(video, "extract", lambda s, **kw: lane(0.9))
        monkeypatch.setattr(audio, "extract", lambda s, **kw: lane(0.1))
        pipeline.segment(session, lambda *a: None, lambda *a: None, use_cache=False)
        assert not store.path_for(session, "video").exists()

    def test_a_failing_result_callback_is_not_reported_as_a_modality_error(
        self, tmp_path, monkeypatch
    ):
        from docupilot.segmentation import audio, video

        session = write_session(tmp_path / "s")
        monkeypatch.setattr(video, "extract", lambda s, **kw: lane(0.9))
        monkeypatch.setattr(audio, "extract", lambda s, **kw: lane(0.1))
        errors = {}
        with pytest.raises(ValueError):
            pipeline.segment(
                session,
                lambda m, e: (_ for _ in ()).throw(ValueError("caller broke")),
                errors.__setitem__,
            )
        assert errors == {}
