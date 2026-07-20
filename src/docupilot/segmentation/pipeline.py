"""
The one entry point into segmentation: run every modality over one session.

A fourth modality means a module with `MODALITY` and `extract()`, named in
_EXTRACTORS. No caller changes.
"""

from __future__ import annotations

from collections.abc import Callable

from docupilot.recording.session import RecordingSession
from docupilot.segmentation import audio, events, video
from docupilot.segmentation.evidence import BoundaryEvidence

# Cheapest first, slowest last, so a caller rendering results as they arrive
# fills the screen in that order.
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

    Results arrive by callback because the modalities are minutes apart. Runs on
    the caller's thread; one modality failing does not stop the others.

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
