"""
Measure the modality sync instead of asserting it.

The recorder puts events on time.monotonic() (zeroed at arm(), just before the
first frame) and lets ffmpeg stamp video frames with its wall clock. Those are
different clocks that should share an origin — this module checks the residual
on a real recording, which is the number the thesis needs rather than a claim.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation import video


def stream_offset_ms(recording_path: Path | str) -> float:
    """
    Audio start minus video start in the muxed file, in milliseconds.

    Whisper zeroes on the first audio sample, the video extractor on the first
    video frame — a non-zero offset here would put the two modalities on shifted
    axes. Worth checking because `-use_wallclock_as_timestamps` stamps only the
    video input; the audio device carries its own timing.

    :return: offset in ms (0.0 = streams share the origin), NaN if a stream is
        missing.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,start_time", "-of", "json", str(recording_path)],
        capture_output=True, text=True,
    ).stdout
    starts = {
        s.get("codec_type"): float(s.get("start_time", "nan"))
        for s in json.loads(out or "{}").get("streams", [])
    }
    if "audio" not in starts or "video" not in starts:
        return float("nan")
    return (starts["audio"] - starts["video"]) * 1000.0


def click_offsets_s(
    session: RecordingSession, look_back_s: float = 0.3, look_ahead_s: float = 1.5
) -> np.ndarray:
    """
    Per mouse click, the seconds until the screen first reacts.

    A click cannot cause a change before it happens, so a clean recording shows a
    small positive median (UI latency) and no large systematic shift; a large
    negative median would mean events are logged late relative to the video — a
    clock desync. Uses the same activity signal the video extractor sees, so it
    measures sync as the pipeline actually experiences it.

    The event a click is matched to is the first quiet->active RISING EDGE in the
    window, not merely the first active frame: that isolates the click's own
    reaction from motion already under way, which would otherwise pull the
    estimate negative.

    :return: one offset per click that had a following change; empty if none.
    """
    mp4 = str(session.recording_path)
    n_frames, activity = video._scan(mp4)
    times_s = video._frame_times_s(mp4, n_frames)
    clicks = [
        e["t_ms"] / 1000.0
        for e in session.input_events()
        if e.get("type") == "mouse_click"
    ]
    if n_frames < 2 or not clicks:
        return np.zeros(0, dtype=np.float64)

    active = activity >= video._ACTIVITY_QUIET
    rise_times = times_s[1:][(~active[:-1]) & active[1:]]   # quiet -> active

    offsets: list[float] = []
    for click in clicks:
        window = rise_times[
            (rise_times >= click - look_back_s) & (rise_times <= click + look_ahead_s)
        ]
        if len(window):
            offsets.append(float(window[0] - click))
    return np.asarray(offsets, dtype=np.float64)


def report(session: RecordingSession) -> dict[str, float]:
    """
    One line of sync evidence for a session: stream offset and click->reaction.

    :return: dict with stream_offset_ms, click median/IQR/max in ms, and n.
    """
    offsets = click_offsets_s(session)
    if len(offsets) == 0:
        return {"stream_offset_ms": stream_offset_ms(session.recording_path), "n": 0}
    return {
        "stream_offset_ms": stream_offset_ms(session.recording_path),
        "click_median_ms": float(np.median(offsets) * 1000),
        "click_iqr_lo_ms": float(np.percentile(offsets, 25) * 1000),
        "click_iqr_hi_ms": float(np.percentile(offsets, 75) * 1000),
        "click_absmax_ms": float(np.abs(offsets).max() * 1000),
        "n": len(offsets),
    }
