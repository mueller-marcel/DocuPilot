"""
Everything the evaluation reads off a recording, in the units it computes in.

The session stores boundaries in milliseconds while the modalities report
seconds — the conversion happens here and nowhere else, so a factor of 1000
cannot leak into a result.
"""

from __future__ import annotations

from docupilot.evaluation import media
from docupilot.recording.session import DEFAULT_BOUNDARY_KIND, RecordingSession


def ground_truth_s(session: RecordingSession, kind: str = DEFAULT_BOUNDARY_KIND) -> list[float]:
    """
    The annotated boundaries of one definition in seconds, ascending.

    :param session: the recording; its ground_truth.json holds milliseconds.
    :param kind: "end" (result settled) or "start" (next action's first input).
    :return: boundary timestamps in seconds.
    """
    return sorted(t_ms / 1000.0 for t_ms, _ in session.ground_truth_markers(kind))


def duration_s(session: RecordingSession) -> float:
    """
    Length of the recording in seconds, read from the container.

    Needed for the chance level: guesses are spread over the whole recording,
    so a wrong length would silently move that floor.

    :param session: the recording; its .recording_path points at the MP4.
    :return: duration in seconds.
    :raises RuntimeError: when the duration cannot be read.
    """
    info = media.probe(session.recording_path)
    try:
        return info.duration_s
    except RuntimeError:
        raise RuntimeError(
            f"Dauer von {session.recording_path} nicht lesbar — ffprobe lieferte "
            f"{info.duration_raw!r}."
        ) from None
