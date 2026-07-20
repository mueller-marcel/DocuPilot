"""
Action boundaries from events.json alone: input bursts, graded by the rest after
them. Opens exactly one file, so the modality stays independent for the ablation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    GRID_HZ,
    BoundaryEvidence,
    apply_gaussian,
    grid,
)

MODALITY = "events"

# Pauses >= this separate two bursts (Wengelin 2006 — imported, not fitted).
_BURST_PAUSE_S = 2.0

# Rest length at which the score saturates.
# PROVISIONAL: no literature value, the only free parameter here. Calibrate on a
# dev split, never on the evaluation set.
_REST_FULL_S = 8.0

_SPREAD_S = 1.0

# Lets the log state its own length; taking it from the video would make the last
# burst's score depend on another modality.
_END_EVENT_TYPES = frozenset({"recording_stopped", "av_stopped", "input_stopped"})


def input_markers(session: RecordingSession) -> list[tuple[float, str]]:
    """Every recorded user input as (t_ms, type), chronologically."""
    markers = [
        (float(ev.get("t_ms", 0.0)), str(ev.get("type", "")))
        for ev in session.input_events()
    ]
    markers.sort(key=lambda m: m[0])
    return markers


def _log_duration_s(session: RecordingSession) -> float:
    """
    How long the recording ran, according to the event log itself. Falls back to
    the last event of any kind when no lifecycle event is present.
    """
    events = session.read_events()
    if not events:
        return 0.0
    ends = [
        float(ev.get("t_ms", 0.0))
        for ev in events
        if ev.get("type") in _END_EVENT_TYPES
    ]
    return max(ends or [float(ev.get("t_ms", 0.0)) for ev in events]) / 1000.0


def _bursts(times_s: list[float]) -> list[tuple[float, float]]:
    """Group input times into (start, end) bursts, split at _BURST_PAUSE_S."""
    if not times_s:
        return []
    out: list[tuple[float, float]] = []
    start = prev = times_s[0]
    for t in times_s[1:]:
        if t - prev > _BURST_PAUSE_S:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def extract(
    session: RecordingSession,
    *,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> BoundaryEvidence:
    """
    Action boundaries from the event log alone: each input burst is one candidate
    on its LAST event, scored `min(pause_after / _REST_FULL_S, 1)`.

    The keyword arguments are the shared extractor contract and unused here.

    :param session: session whose .events_path points at events.json.
    """
    markers = input_markers(session)
    duration_s = _log_duration_s(session)
    times_s = grid(duration_s)
    if not markers or len(times_s) == 0:
        return BoundaryEvidence.empty()

    score = np.zeros(len(times_s), dtype=np.float32)
    spread = max(1, int(_SPREAD_S * GRID_HZ))

    bursts = _bursts([t_ms / 1000.0 for t_ms, _ in markers])
    next_starts = [start for start, _ in bursts[1:]] + [duration_s]

    boundaries_s: list[float] = []
    for (_, end_s), next_start_s in zip(bursts, next_starts):
        value = min(max(next_start_s - end_s, 0.0) / _REST_FULL_S, 1.0)
        center = min(max(int(round(end_s * GRID_HZ)), 0), len(times_s) - 1)
        apply_gaussian(score, center, value, spread)
        if value >= BOUNDARY_THRESHOLD:
            boundaries_s.append(float(times_s[center]))

    return BoundaryEvidence(times_s, score, boundaries_s)
