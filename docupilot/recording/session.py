"""
docupilot.recording.session
───────────────────────────
RecordingSession  – Pfade und gemeinsame Uhr
EventWriter       – Thread-sicheres JSON-Array
Protocols         – Screen, Microphone, ModalityRecorder
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


# ── Protocols ─────────────────────────────────────────────────────────────────

class ScreenGeometry(Protocol):
    def x(self) -> int: ...
    def y(self) -> int: ...
    def width(self) -> int: ...
    def height(self) -> int: ...

class Screen(Protocol):
    def name(self) -> str: ...
    def geometry(self) -> ScreenGeometry: ...
    def devicePixelRatio(self) -> float: ...

class Microphone(Protocol):
    def description(self) -> str: ...


# ── RecordingSession ──────────────────────────────────────────────────────────

@dataclass
class RecordingSession:
    """
    Pfade und gemeinsame Uhr einer Aufnahme-Session.

    t0 wird in RecorderService.start_recording() gesetzt, unmittelbar
    bevor der ffmpeg-Prozess gestartet wird. Events werden relativ zu
    t0 gestempelt:

        mp4_position_ms ≈ event t_ms   (Toleranz < 100 ms)
    """
    screen:     Screen
    microphone: Microphone

    session_id:         uuid.UUID = field(default_factory=uuid.uuid4)
    started_at_utc:     datetime  = field(default_factory=lambda: datetime.now(timezone.utc))
    start_monotonic_ns: int       = field(default=0, init=False)
    _armed:             bool      = field(default=False, init=False, repr=False)

    recording_file_name: str  = "recording.mp4"
    events_file_name:    str  = "events.json"
    base_dir:            Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "docupilot"
    )
    session_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.session_dir = self.base_dir / f"session_{self.session_id}"

    def arm(self, t0_ns: int) -> None:
        """Setzt den gemeinsamen Uhren-Ursprung."""
        self.start_monotonic_ns = t0_ns
        self._armed = True

    @property
    def is_armed(self) -> bool:
        return self._armed

    def session_time_ms(self) -> float:
        """Millisekunden seit t0. Negativ wenn vor arm()."""
        return round((time.monotonic_ns() - self.start_monotonic_ns) / 1_000_000, 3)

    @property
    def recording_path(self) -> Path:
        return self.session_dir / self.recording_file_name

    @property
    def events_path(self) -> Path:
        return self.session_dir / self.events_file_name

    def to_metadata_dict(self) -> dict[str, Any]:
        geo = self.screen.geometry()
        return {
            "session_id":     str(self.session_id),
            "session_dir":    str(self.session_dir),
            "started_at_utc": self.started_at_utc.isoformat(),
            "files": {
                "recording": self.recording_file_name,
                "events":    self.events_file_name,
            },
            "screen": {
                "name":               self.screen.name(),
                "x":                  geo.x(),
                "y":                  geo.y(),
                "width":              geo.width(),
                "height":             geo.height(),
                "device_pixel_ratio": self.screen.devicePixelRatio(),
            },
            "microphone": {"description": self.microphone.description()},
        }


# ── EventWriter ───────────────────────────────────────────────────────────────

class EventWriter:
    """Schreibt Events thread-sicher als JSON-Array."""

    def __init__(self, path: Path) -> None:
        self._path  = path
        self._file  = None
        self._count = 0
        self._lock  = threading.Lock()

    def open(self) -> None:
        self._count = 0
        self._file  = self._path.open("w", encoding="utf-8")
        self._file.write("[\n")
        self._file.flush()

    def write(self, event: dict[str, Any], t_ms: float) -> None:
        if self._file is None:
            return
        event["t_ms"] = t_ms
        with self._lock:
            if self._count > 0:
                self._file.write(",\n")
            json.dump(event, self._file, ensure_ascii=False)
            self._file.flush()
            self._count += 1

    def close(self) -> None:
        if self._file is None:
            return
        self._file.write("\n]\n")
        self._file.flush()
        self._file.close()
        self._file = None