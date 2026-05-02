from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RecordingSession:
    """
    Datenobjekt für eine konkrete Recording-Session.

    Diese Klasse enthält bewusst keine Aufnahme-Logik.
    Sie hält nur:
    - ausgewählten Bildschirm
    - ausgewähltes Mikrofon
    - Session-ID
    - Temp-Verzeichnis
    - Dateipfade
    - Startzeit
    - gemeinsame monotone Zeitbasis
    """

    screen: Any
    microphone: Any

    session_id: uuid.UUID = field(default_factory=uuid.uuid4)

    started_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_monotonic_ns: int = field(default_factory=time.monotonic_ns)

    screen_file_name: str = "screen.mp4"
    audio_file_name: str = "voice.mp3"
    events_file_name: str = "events.json"

    @property
    def session_dir(self) -> Path:
        """
        Zielverzeichnis:

        temp/docupilot/session_{GUID}
        """
        return (
            Path(tempfile.gettempdir())
            / "docupilot"
            / f"session_{self.session_id}"
        )

    @property
    def screen_path(self) -> Path:
        return self.session_dir / self.screen_file_name

    @property
    def audio_path(self) -> Path:
        return self.session_dir / self.audio_file_name

    @property
    def events_path(self) -> Path:
        return self.session_dir / self.events_file_name

    def t_ms(self) -> float:
        """
        Gemeinsame Zeitbasis für alle Modalitäten.

        t_ms = Millisekunden seit Start der RecordingSession.
        Diese Zeitbasis ist für Synchronisierung besser geeignet als datetime.now(),
        weil time.monotonic_ns() nicht von Systemzeitänderungen beeinflusst wird.
        """
        elapsed_ns = time.monotonic_ns() - self.start_monotonic_ns
        return round(elapsed_ns / 1_000_000, 3)

    def to_metadata_dict(self) -> dict[str, Any]:
        screen_geometry = self.screen.geometry()

        return {
            "session_id": str(self.session_id),
            "session_dir": str(self.session_dir),
            "started_at_utc": self.started_at_utc.isoformat(),
            "files": {
                "screen": self.screen_file_name,
                "audio": self.audio_file_name,
                "events": self.events_file_name,
            },
            "screen": {
                "name": self.screen.name(),
                "x": screen_geometry.x(),
                "y": screen_geometry.y(),
                "width": screen_geometry.width(),
                "height": screen_geometry.height(),
                "device_pixel_ratio": self.screen.devicePixelRatio(),
            },
            "microphone": {
                "description": self.microphone.description(),
            },
        }