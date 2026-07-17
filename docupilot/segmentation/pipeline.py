"""
The one entry point into segmentation: run every modality over one session.

Callers say "segment this recording" and get one BoundaryEvidence per modality.
They do not learn that video needs a frame rate, that audio needs a transcript,
or that events carry their own clock — each module derives what it needs from its
own modality, which is also what keeps the 2^3 Shapley ablation honest.

Adding a fourth modality means writing a module with `MODALITY` and `extract()`
and naming it in _EXTRACTORS. No caller changes.
"""

from __future__ import annotations

from collections.abc import Callable

from docupilot.recording.session import RecordingSession
from docupilot.segmentation import audio, events, video
from docupilot.segmentation.evidence import BoundaryEvidence

# Cheapest first, slowest last: events are a JSON parse, video costs one VLM call
# per settled-state pair, audio spends minutes in Whisper before it says anything.
# A caller rendering results as they arrive fills the screen in that order.
_EXTRACTORS = (events, video, audio)

MODALITIES: tuple[str, ...] = tuple(m.MODALITY for m in _EXTRACTORS)


def segment(
    session: RecordingSession,
    on_result: Callable[[str, BoundaryEvidence], None],
    on_error: Callable[[str, str], None],
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    use_cache: bool = True,
) -> None:
    """
    Segment one recording, reporting each modality as it finishes.

    Results arrive by callback rather than as a return value because the slow
    modalities are minutes apart and a caller that waited for all three would show
    nothing for the whole run. Runs on the caller's thread and touches no UI
    toolkit — the callbacks are plain functions.

    One modality failing does not stop the others: a missing VLM key must not cost
    the events lane. Failures are reported through on_error, never swallowed.

    :param session: the recording to segment.
    :param on_result: called as (modality, evidence) when a modality finishes.
    :param on_error: called as (modality, message) when a modality raises.
    :param on_progress: called as (modality, done, total) during long modalities.
    :param is_cancelled: polled between and inside modalities; True stops early.
    :param use_cache: reuse model verdicts cached in the session directory.
    """
    for extractor in _EXTRACTORS:
        if is_cancelled is not None and is_cancelled():
            return
        modality = extractor.MODALITY
        try:
            on_result(modality, extractor.extract(
                session,
                use_cache=use_cache,
                on_progress=(
                    None if on_progress is None
                    else lambda done, total, m=modality: on_progress(m, done, total)
                ),
                is_cancelled=is_cancelled,
            ))
        except Exception as exc:                  # noqa: BLE001 — reported, not hidden
            on_error(modality, str(exc))
