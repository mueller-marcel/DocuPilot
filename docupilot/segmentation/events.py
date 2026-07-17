"""
The events modality: action boundaries from events.json ALONE.

No screen, no audio — the modality stays independent for the 2^3 Shapley
ablation. This module opens exactly one file: events.json.

HOW THE THREE RULES MAP ONTO A RAW INPUT LOG (docs/annotationsleitfaden.md)
  A · user-triggered   SEEN, exactly. This is the one rule where events beat
                       video: the video extractor has to infer the trigger,
                       events record it. A boundary therefore follows an input.
  B · new persistent   BLIND. A click is a click whether it applied a filter or
      state            opened a menu. Nothing in the log distinguishes them.
  C · goal, not means  PARTLY. The definition's observable is "afterwards the
                       user pauses / turns to something else" — and a pause in
                       INPUT is exactly what this log measures.

  Being blind to rule B is the honest ceiling of this modality, and the reason
  for its false positives. It must not be papered over: the ablation is supposed
  to measure what a keystroke log knows, not what we wish it knew.

WHY BURSTS AND WHY 2 s
  Writing research has segmented raw keystroke streams into BURSTS separated by
  PAUSES for decades, with a settled convention: pauses >= 2 s are COGNITIVE
  pauses (planning, revising), while 30 ms .. 2 s reflect transcription
  (Wengelin 2006; tooling: Inputlog, Leijten & Van Waes, Written Communication
  30, 2013). The threshold is imported, not fitted — and session_30 corroborates
  it: inter-event gaps are sharply bimodal (p50 = 0.03 s, p75 = 1.74 s,
  p90 = 5.04 s), and 2 s sits in the valley.

  Grading by the pause AFTER the burst is rule C read literally. Theoretical
  backing: Event Segmentation Theory puts perceived boundaries where prediction
  breaks — at goal shifts — and finds them partly predictable from low-level
  movement cues (Zacks & Swallow 2007; Kurby & Zacks, TiCS 2008).

WHY NOT THE RPA/PROCESS-MINING SEGMENTATION
  Robotic Process Mining names exactly this problem (Leno et al., ICPM 2020;
  Agostinelli et al., RCIS 2021), but its UI logs are SEMANTIC: openWorkbook,
  copy, clickTextField, paste, each with application and target element. Their
  methods presuppose that aggregation. This log is raw input — click at (x, y),
  key, scroll — with no window title and no UI element, so those methods do not
  transfer. That is a limitation to state, not a gap to hide.

WHY THE BOUNDARY SITS ON THE BURST END, WITH NO LATENCY
  The result appears some time after the trigger, so the true boundary is a bit
  LATER than the burst end (measured on session_30: median 1.05 s, range
  0.03 .. 2.98 s). Correcting for it is deliberately NOT done. The ground truth
  is anchored on the VISUAL settling moment, so calibrating an offset against it
  would align events to video — the events arm would become a video proxy and the
  Shapley decomposition would report a correlation we built ourselves. Events
  alone cannot know the latency, so events alone do not claim it.

  (A CLOCK DRIFT between events.json and the video is a different matter: that is
  a recording bug, and fixing it is the synchronisation the study design requires
  — preprocessing, not extraction. It does not belong in here.)

DELIBERATELY ABSENT — do not reintroduce:
  - The pointer's spatial jump between bursts as a score component. It is
    theoretically attractive (EST's "location change") but measured on session_30
    it does not separate: burst #11 jumps 90 px and IS a boundary, burst #12
    jumps 1087 px and is not. A weight on it would be decoration.
  - Scoring every EVENT by the gap BEFORE it (what this extractor used to do).
    That marks where an action STARTS. Our boundary is where one COMPLETES — the
    same begin-vs-complete confusion we resolved for the ground truth.
  - A "local maximum of pause length" rule. It reaches precision ~0.87 on
    session_30 by keeping only the last burst of each action, but it is read off
    ONE session with n=7, and it is non-maximum suppression by the back door: it
    can delete a real boundary that happens to sit next to a longer pause.
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

# Pauses >= this separate two bursts. Convention from keystroke-logging research
# (Wengelin 2006), NOT fitted here.
_BURST_PAUSE_S = 2.0

# Pause at which the score saturates: how long a rest has to be before it reads
# as "the user is done and looking at the result" rather than "thinking".
#
# PROVISIONAL — there is no literature value for this, and it is the only free
# parameter here. Every inter-burst pause is >= _BURST_PAUSE_S by construction, so
# the scale has to be well above 2 s or every burst scores 1.0. At 8 s the
# decision threshold (0.5) lands on a 4 s rest = twice the cognitive-pause
# threshold. Calibrate on a dev split before freezing — never on the evaluation
# set.
_REST_FULL_S = 8.0

_SPREAD_S = 1.0

# The recorder writes lifecycle events into the same file, so the log knows its
# own length. Reading it from the media file instead would make the last burst's
# score depend on the video.
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
    How long the recording ran, according to the event log ITSELF.

    Falls back to the last event of any kind when no lifecycle event is present
    (a log from an older recorder, or a truncated one).
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
    Action boundaries for one recording, from the event log alone.

    Each burst of input is one candidate, placed on the burst's LAST event and
    graded by the rest that follows it:

        score = min(pause_after / _REST_FULL_S, 1)

    The last burst is closed by the end of the recording: the user stopped
    recording, which is the longest rest there is.

    use_cache / on_progress / is_cancelled are part of the shared extractor
    contract and unused here — this modality is a JSON parse and finishes before
    a progress bar would render.

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
