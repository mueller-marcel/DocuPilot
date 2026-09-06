"""
Shared test setup.

There are no checked-in fixtures and no golden files: every input a test needs
is built in the test itself, and every assertion states a property the code
must have rather than a number it once produced.

A `RecordingSession` only requires that recording.mp4 and events.json EXIST, so
the many tests that never decode media use placeholder files. The handful of
tests that really exercise OpenCV and ffprobe generate a tiny clip at runtime
(see test_media_adapters.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# Widgets are constructed in a few tests; no display is available on CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def write_session(
    directory: Path,
    events: list[dict[str, Any]] | None = None,
    ground_truth: list[dict[str, Any]] | None = None,
    recording: bytes = b"",
):
    """
    Lay out a session directory and open it.

    :param directory: where to write; created if missing.
    :param events: the event log, or an empty one.
    :param ground_truth: the annotation; omitted entirely when None.
    :param recording: bytes for recording.mp4 — empty unless a test decodes it.
    :return: the opened RecordingSession.
    """
    from docupilot.recording.session import RecordingSession

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "recording.mp4").write_bytes(recording)
    (directory / "events.json").write_text(
        json.dumps(events if events is not None else []), encoding="utf-8"
    )
    if ground_truth is not None:
        (directory / "ground_truth.json").write_text(
            json.dumps(ground_truth), encoding="utf-8"
        )
    return RecordingSession.from_directory(directory)


@pytest.fixture
def session(tmp_path):
    """An empty session on disk — enough for anything that does not decode media."""
    return write_session(tmp_path / "session_test")


@pytest.fixture(scope="session")
def app():
    """The one QApplication; Qt refuses a second one per process."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
