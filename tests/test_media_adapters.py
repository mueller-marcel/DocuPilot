"""
The only tests that touch OpenCV and ffprobe: decoding a real file.

The clip is generated here rather than checked in — a binary fixture in the
repository would have to be trusted, while a generated one is described by the
code that makes it. Skipped when the tools are unavailable.

The generated picture looks like a user interface (flat areas, a toolbar, a few
rules) rather than noise, because that is what the recorder captures and what a
video codec can reproduce faithfully. Per-pixel noise is the worst case for any
encoder and would test the codec, not the pipeline.
"""

import shutil

import numpy as np
import pytest

from conftest import write_session
from docupilot.evaluation import dataset, media
from docupilot.segmentation import video

cv2 = pytest.importorskip("cv2", reason="OpenCV nicht installiert")
pytestmark = pytest.mark.skipif(
    shutil.which("ffprobe") is None, reason="ffprobe nicht im PATH"
)

FPS = 10.0
WIDTH, HEIGHT = 160, 120
N_FRAMES = 40
CHANGE_AT = 20             # the one frame where something visibly happens

# Where the change is drawn, as a share of the frame — what the region box
# should point at.
CHANGE_BOX = (90 / WIDTH, 60 / HEIGHT, 140 / WIDTH, 90 / HEIGHT)


def _interface() -> np.ndarray:
    """A still frame that looks like an application window."""
    frame = np.full((HEIGHT, WIDTH, 3), 245, np.uint8)
    cv2.rectangle(frame, (0, 0), (WIDTH, 14), (200, 200, 200), -1)      # toolbar
    for y in range(24, 110, 12):                                        # table rules
        cv2.line(frame, (8, y), (WIDTH - 8, y), (180, 180, 180), 1)
    cv2.rectangle(frame, (8, 20), (70, 32), (120, 120, 120), -1)        # a filled cell
    return frame


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A short clip: a still interface, one visible change halfway through."""
    path = tmp_path_factory.mktemp("generated") / "recording.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    frame = _interface()
    for index in range(N_FRAMES):
        if index == CHANGE_AT:
            cv2.rectangle(frame, (90, 60), (140, 90), (40, 40, 220), -1)
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("Dieser OpenCV-Build kann kein MP4 schreiben")
    return path


class TestFfprobe:
    def test_the_duration_is_read_from_the_container(self, clip):
        assert media.probe(clip).duration_s == pytest.approx(N_FRAMES / FPS, abs=0.2)

    def test_the_session_reads_the_same_duration(self, clip):
        session = write_session(clip.parent / "s", recording=clip.read_bytes())
        assert dataset.duration_s(session) == pytest.approx(N_FRAMES / FPS, abs=0.2)

    def test_frame_times_match_the_decoded_frames_and_start_at_zero(self, clip):
        n_frames, _, _ = video._scan(str(clip))
        times = video._frame_times_s(str(clip), n_frames)
        assert len(times) == n_frames
        assert times[0] == 0.0
        assert np.all(np.diff(times) > 0)


class TestDecoding:
    def test_every_frame_yields_one_activity_value(self, clip):
        n_frames, activity, fps = video._scan(str(clip))
        assert n_frames == len(activity) == N_FRAMES
        assert fps == pytest.approx(FPS, abs=0.5)

    def test_the_strongest_activity_is_the_visible_change(self, clip):
        _, activity, _ = video._scan(str(clip))
        assert int(activity.argmax()) == CHANGE_AT
        assert activity[CHANGE_AT] > video.ACTIVITY_QUIET

    def test_the_change_separates_two_settled_states(self, clip):
        _, activity, fps = video._scan(str(clip))
        found = video.dwells(activity, min_frames=max(1, round(0.5 * fps)))
        # Before and after; the encoder needs a few frames to settle at the
        # start, which is why the first dwell does not begin at frame 0.
        assert len(found) == 2
        assert found[0][1] < CHANGE_AT <= found[1][0]

    def test_requested_frames_are_read_downscaled_in_one_pass(self, clip):
        frames = video._read_frames(str(clip), {0, CHANGE_AT})
        assert set(frames) == {0, CHANGE_AT}
        assert frames[0].shape[1] <= WIDTH

    def test_asking_for_nothing_decodes_nothing(self, clip):
        assert video._read_frames(str(clip), set()) == {}

    def test_a_missing_file_scans_as_empty_rather_than_raising(self):
        n_frames, activity, fps = video._scan("does-not-exist.mp4")
        assert n_frames == 0 and activity.size == 0 and fps > 0


class TestChangedRegion:
    def test_the_box_points_at_the_change_and_not_at_the_whole_frame(self, clip):
        frames = video._read_frames(str(clip), {0, CHANGE_AT})
        box = video.changed_region(
            video._gray16(frames[0]), video._gray16(frames[CHANGE_AT])
        )
        assert box is not None
        x0, y0, x1, y1 = box
        want_x0, want_y0, want_x1, want_y1 = CHANGE_BOX
        # Tile granularity makes the box coarser than the drawn rectangle, but
        # it must still contain it and cover well under half the screen.
        assert x0 <= want_x0 and y0 <= want_y0
        assert x1 >= want_x1 and y1 >= want_y1
        assert (x1 - x0) * (y1 - y0) < 0.5


class TestActivityCache:
    def test_a_stored_scan_is_reused_instead_of_decoding_again(self, clip, tmp_path):
        session = write_session(tmp_path / "s", recording=clip.read_bytes())
        first = video.scan_activity(session)
        assert (session.session_dir / "video_activity.npz").exists()

        original = video._scan
        try:
            video._scan = lambda path: pytest.fail("decoded despite a stored scan")
            second = video.scan_activity(session)
        finally:
            video._scan = original
        np.testing.assert_array_equal(first.activity, second.activity)
        np.testing.assert_array_equal(first.times_s, second.times_s)

    def test_a_different_recording_invalidates_the_stored_scan(self, clip, tmp_path):
        session = write_session(tmp_path / "s", recording=clip.read_bytes())
        video.scan_activity(session)
        before = video._activity_key(session)

        session.recording_path.write_bytes(clip.read_bytes() + b"\x00")
        # The key is the recording's CONTENT, so a changed file can never be
        # served the previous scan.
        assert video._activity_key(session) != before

    def test_use_cache_false_neither_reads_nor_writes(self, clip, tmp_path):
        session = write_session(tmp_path / "s", recording=clip.read_bytes())
        video.scan_activity(session, use_cache=False)
        assert not (session.session_dir / "video_activity.npz").exists()
