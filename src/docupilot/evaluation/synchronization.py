"""
Measure the modality sync instead of asserting it.

The recorder puts events on time.monotonic() (zeroed at arm(), just before the
first frame) and lets ffmpeg stamp video frames with its wall clock. Those are
different clocks that should share an origin — this module checks the residual
on a real recording, which is the number the thesis needs rather than a claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from docupilot.evaluation import media
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
    starts = media.probe(recording_path).stream_start_s
    if "audio" not in starts or "video" not in starts:
        return float("nan")
    return (starts["audio"] - starts["video"]) * 1000.0


def rising_edges_s(activity: np.ndarray, times_s: np.ndarray, quiet: float) -> np.ndarray:
    """
    The moments the screen goes from still to active.

    A rising EDGE, not merely an active frame: that isolates a click's own
    reaction from motion already under way, which would otherwise let a click
    be matched to a change that started before it and pull the estimate negative.
    """
    if len(activity) < 2:
        return np.zeros(0, dtype=np.float64)
    active = activity >= quiet
    return times_s[1:][(~active[:-1]) & active[1:]]


def reaction_offsets_s(
    rise_times_s: np.ndarray,
    click_times_s: Sequence[float],
    look_back_s: float = 0.3,
    look_ahead_s: float = 1.5,
) -> np.ndarray:
    """
    Per click, the seconds until the first rising edge near it.

    A click cannot cause a change before it happens, so a clean recording shows a
    small positive median (UI latency) and no large systematic shift; a large
    negative median would mean events are logged late relative to the video — a
    clock desync. The small backward window absorbs exactly that measurement
    noise without admitting an unrelated earlier change.

    :return: one offset per click that had a following change; empty if none.
    """
    offsets: list[float] = []
    for click in click_times_s:
        window = rise_times_s[
            (rise_times_s >= click - look_back_s) & (rise_times_s <= click + look_ahead_s)
        ]
        if len(window):
            offsets.append(float(window[0] - click))
    return np.asarray(offsets, dtype=np.float64)


def click_offsets_s(
    session: RecordingSession, look_back_s: float = 0.3, look_ahead_s: float = 1.5
) -> np.ndarray:
    """
    Per mouse click, the seconds until the screen first reacts.

    Uses the same activity signal the video extractor sees, so it measures sync
    as the pipeline actually experiences it — and the same cached scan, so a
    corpus that was already segmented is not decoded a second time.

    :return: one offset per click that had a following change; empty if none.
    """
    clicks = [
        e["t_ms"] / 1000.0
        for e in session.input_events()
        if e.get("type") == "mouse_click"
    ]
    if not clicks:
        return np.zeros(0, dtype=np.float64)
    scan = video.scan_activity(session)
    rises = rising_edges_s(scan.activity, scan.times_s, video.ACTIVITY_QUIET)
    return reaction_offsets_s(rises, clicks, look_back_s, look_ahead_s)


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
        # P95 because the τ justification is a coverage claim: "τ covers the
        # typical latency" must hang on a quantile, not on a maximum that a
        # single outlier decides.
        "click_p95_ms": float(np.percentile(offsets, 95) * 1000),
        "click_absmax_ms": float(np.abs(offsets).max() * 1000),
        "n": len(offsets),
    }
