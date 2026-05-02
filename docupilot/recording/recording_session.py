from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docupilot.recording.protocols import Microphone, Screen


@dataclass
class RecordingSession:
    """
    Holds the state and file-path information for a single recording session.
    """

    screen: Screen
    microphone: Microphone

    session_id: uuid.UUID = field(default_factory=uuid.uuid4)
    started_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_monotonic_ns: int = field(default_factory=time.monotonic_ns)

    screen_file_name: str = "screen.mp4"
    audio_file_name: str = "voice.mp3"
    events_file_name: str = "events.json"

    base_dir: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "docupilot")

    session_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.session_dir = self.base_dir / f"session_{self.session_id}"

    @property
    def screen_path(self) -> Path:
        """
        Absolute path to the screen recording file.
        """

        return self.session_dir / self.screen_file_name

    @property
    def audio_path(self) -> Path:
        """
        Absolute path to the audio recording file.
        """

        return self.session_dir / self.audio_file_name

    @property
    def events_path(self) -> Path:
        """Absolute path to the input-events JSON file."""
        return self.session_dir / self.events_file_name

    def session_time_ms(self) -> float:
        """
        Returns the elapsed session time in milliseconds.

        Uses a monotonic clock so all modalities share a consistent timeline
        regardless of wall-clock adjustments.
        """
        return round((time.monotonic_ns() - self.start_monotonic_ns) / 1_000_000, 3)


class RecordingSessionSerializer:
    """
    Converts a RecordingSession into a metadata dictionary.
    """

    @staticmethod
    def to_metadata_dict(session: RecordingSession) -> dict[str, Any]:
        """
        Builds a JSON-serializable metadata snapshot of the given session.

        :param session: The session to serialize.
        :return: A dictionary suitable for embedding in the events file.
        """
        geo = session.screen.geometry()
        return {
            "session_id": str(session.session_id),
            "session_dir": str(session.session_dir),
            "started_at_utc": session.started_at_utc.isoformat(),
            "files": {
                "screen": session.screen_file_name,
                "audio": session.audio_file_name,
                "events": session.events_file_name,
            },
            "screen": {
                "name": session.screen.name(),
                "x": geo.x(),
                "y": geo.y(),
                "width": geo.width(),
                "height": geo.height(),
                "device_pixel_ratio": session.screen.device_pixel_ratio(),
            },
            "microphone": {
                "description": session.microphone.description(),
            },
        }
