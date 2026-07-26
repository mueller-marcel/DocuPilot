"""
Everything the evaluation reads off a recording, in the units it computes in.

The session stores boundaries in milliseconds while the modalities report
seconds — the conversion happens here and nowhere else, so a factor of 1000
cannot leak into a result.
"""

from __future__ import annotations

import subprocess

from docupilot.recording.session import RecordingSession


def ground_truth_s(session: RecordingSession) -> list[float]:
    """
    The annotated boundaries in seconds, ascending.

    :param session: the recording; its ground_truth.json holds milliseconds.
    :return: boundary timestamps in seconds.
    """
    return sorted(t_ms / 1000.0 for t_ms, _ in session.ground_truth_markers())


def duration_s(session: RecordingSession) -> float:
    """
    Length of the recording in seconds, read from the container.

    Needed for the chance level: guesses are spread over the whole recording,
    so a wrong length would silently move that floor.

    :param session: the recording; its .recording_path points at the MP4.
    :return: duration in seconds.
    :raises RuntimeError: when the duration cannot be read.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(session.recording_path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        raise RuntimeError(
            f"Dauer von {session.recording_path} nicht lesbar — ffprobe lieferte "
            f"{out!r}."
        ) from None
