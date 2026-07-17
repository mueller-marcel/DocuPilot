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


# The event types InputRecorder writes for actual user input, as opposed to the
# recorder's own lifecycle events. Private: callers ask for input_events() rather
# than filtering the log themselves, so this stays one fact in one place.
_INPUT_EVENT_TYPES = frozenset(
    {"mouse_click", "key_press", "key_release", "mouse_scroll"}
)


# ── Stub-Implementierungen für das Öffnen gespeicherter Sessions ──────────────
#
# Beim Öffnen einer bereits aufgezeichneten Session (Datei > Öffnen) steht
# kein echtes Screen- und kein echtes Microphone-Objekt zur Verfügung, weil
# keine neue Aufnahme gestartet wird. Diese Stubs erfüllen die jeweiligen
# Protocols mit Dummy-Werten (Liskov-konform), damit RecordingSession ohne
# Sonderfälle konstruiert werden kann.

class _NullScreenGeometry:
    def x(self) -> int:              return 0
    def y(self) -> int:              return 0
    def width(self) -> int:          return 0
    def height(self) -> int:         return 0

class _NullScreen:
    def name(self) -> str:           return "<gespeicherte Session>"
    def geometry(self) -> _NullScreenGeometry: return _NullScreenGeometry()
    def devicePixelRatio(self) -> float: return 1.0

class _NullMicrophone:
    def description(self) -> str:    return "<gespeicherte Session>"


@dataclass
class RecordingSession:
    """
    Represents a recording session that involves screen and microphone capturing.

    This class is used to manage all aspects of a recording session, including file
    paths, metadata generation, and session timing. It handles initialization of
    session directories and maintains information relevant to both the screen and
    microphone used during the session.

    RecordingSession ist die EINZIGE Quelle der Wahrheit für Ground-Truth-
    Grenzen (ground_truth_data): egal ob eine Session gerade live aufgezeichnet
    oder über from_directory() von der Platte geöffnet wurde, jede Stelle im
    Code (AnnotationWindow, FeatureDialog, ein zukünftiger Export) liest und
    schreibt über dieselben Methoden hier — es gibt keine zweite, parallele
    Ablage mehr.
    """

    screen:     Screen
    microphone: Microphone

    session_id:         uuid.UUID = field(default_factory=uuid.uuid4)
    started_at_utc:     datetime  = field(default_factory=lambda: datetime.now(timezone.utc))
    start_monotonic_ns: int       = field(default=0, init=False)
    _armed:             bool      = field(default=False, init=False, repr=False)

    recording_file_name:    str  = "recording.mp4"
    events_file_name:       str  = "events.json"
    ground_truth_file_name: str  = "ground_truth.json"
    base_dir:               Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "docupilot"
    )
    session_dir: Path = field(init=False)

    # Geladene Ground-Truth-Grenzen. Bleibt leer, solange keine Grenze gesetzt
    # bzw. keine ground_truth.json geladen wurde — Ground Truth ist
    # grundsätzlich optional.
    ground_truth_data: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.session_dir = self.base_dir / f"session_{self.session_id}"

    @classmethod
    def from_directory(cls, directory: Path) -> "RecordingSession":
        """
        Rekonstruiert eine RecordingSession aus einem bestehenden Verzeichnis.

        Wird über "Datei > Öffnen" verwendet, um eine bereits aufgezeichnete
        Session zu laden, ohne eine neue Aufnahme zu starten. Screen und
        Microphone werden durch Stub-Objekte ersetzt, da diese Hardware beim
        Öffnen nicht verfügbar bzw. nicht relevant ist. Eine vorhandene
        ground_truth.json wird automatisch geladen; fehlt sie, bleibt
        ground_truth_data einfach leer.

        :param directory: Pfad zum Session-Verzeichnis (muss recording.mp4
            und events.json enthalten; ground_truth.json ist optional).
        :return: RecordingSession mit session_dir = directory.
        :raises FileNotFoundError: Wenn recording.mp4 oder events.json fehlen.
        """
        session = cls(screen=_NullScreen(), microphone=_NullMicrophone())
        session.session_dir = directory

        if not session.recording_path.exists():
            raise FileNotFoundError(
                f"Keine Aufnahmedatei gefunden:\n{session.recording_path}\n\n"
                "Bitte wähle ein Verzeichnis, das eine 'recording.mp4' enthält."
            )
        if not session.events_path.exists():
            raise FileNotFoundError(
                f"Keine Event-Datei gefunden:\n{session.events_path}\n\n"
                "Bitte wähle ein Verzeichnis, das eine 'events.json' enthält."
            )

        session.load_ground_truth()
        return session

    def read_events(self) -> list[dict[str, Any]]:
        """
        The raw event log, or an empty list when it is missing or unreadable.

        Not cached: the log is written during recording, so a session that is
        still running would hand out a stale snapshot.

        :return: Events as written by EventWriter, in file order.
        """
        try:
            with self.events_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def input_events(self) -> list[dict[str, Any]]:
        """
        Only the user's own input, without the recorder's lifecycle events.

        :return: Clicks, keys and scrolls, in file order.
        """
        return [e for e in self.read_events() if e.get("type") in _INPUT_EVENT_TYPES]

    def load_ground_truth(self) -> None:
        """
        (Re-)Lädt ground_truth.json aus session_dir, sofern vorhanden, und
        aktualisiert ground_truth_data entsprechend.

        Ist die Datei nicht vorhanden oder ungültig, bleibt ground_truth_data
        eine leere Liste — Ground Truth ist grundsätzlich optional und darf
        das Öffnen einer Session nicht verhindern.

        Diese Methode ist beliebig oft aufrufbar (idempotent) und wird sowohl
        beim Öffnen einer Session von der Platte als auch beim erneuten
        Anzeigen einer laufenden Session genutzt, damit beide Wege exakt das
        gleiche Verhalten zeigen.

        :return: None
        """
        if not self.ground_truth_path.exists():
            return

        try:
            with self.ground_truth_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.ground_truth_data = data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            self.ground_truth_data = []

    def save_ground_truth(self) -> None:
        """
        Speichert ground_truth_data als ground_truth.json in session_dir.

        :return: None
        """
        self.ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ground_truth_path.open("w", encoding="utf-8") as f:
            json.dump(self.ground_truth_data, f, ensure_ascii=False, indent=2)

    def add_ground_truth_boundary(self, t_ms: float, label: str | None = None) -> None:
        """
        Fügt eine neue Ground-Truth-Grenze hinzu und speichert sofort.

        Zentralisiert das Anlegen einer Grenze an einer Stelle, damit
        AnnotationWindow nicht selbst eine JSON-Datei schreiben muss und
        das Schema (t_ms, label, created_at_utc) an genau einem Ort gepflegt
        wird.

        :param t_ms: Zeitpunkt der Grenze in Millisekunden.
        :param label: Anzeigename der Grenze, z. B. ein formatierter Zeitstempel.
        :return: None
        """
        self.ground_truth_data.append({
            "t_ms": t_ms,
            "label": label or "Grenze",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.save_ground_truth()

    def set_ground_truth_boundaries(self, boundaries: list[dict[str, Any]]) -> None:
        """
        Ersetzt alle Ground-Truth-Grenzen (z. B. nach Löschen im
        Grenzen-Dialog) und speichert sofort.

        :param boundaries: Die neue, vollständige Liste der Grenzen.
        :return: None
        """
        self.ground_truth_data = boundaries
        self.save_ground_truth()

    def ground_truth_markers(self) -> list[tuple[float, str]]:
        """
        Wandelt ground_truth_data in (t_ms, label)-Tupel um, wie sie von
        FeatureTimelineWidget.set_events() erwartet werden.

        Diese Umwandlung liegt bewusst hier statt im UI-Code, damit jede
        Stelle, die Ground Truth anzeigen will (FeatureDialog, ein
        zukünftiger Export, ...), dieselbe Logik wiederverwendet, statt sie
        zu duplizieren.

        :return: Liste von (t_ms, label) Tupeln, chronologisch sortiert.
        """
        markers = [
            (float(entry.get("t_ms", 0.0)), str(entry.get("label", "Grenze")))
            for entry in self.ground_truth_data
        ]
        return sorted(markers, key=lambda m: m[0])

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

    @property
    def ground_truth_path(self) -> Path:
        return self.session_dir / self.ground_truth_file_name

    def to_metadata_dict(self) -> dict[str, Any]:
        geo = self.screen.geometry()
        return {
            "session_id":     str(self.session_id),
            "session_dir":    str(self.session_dir),
            "started_at_utc": self.started_at_utc.isoformat(),
            "files": {
                "recording":    self.recording_file_name,
                "events":       self.events_file_name,
                "ground_truth": self.ground_truth_file_name,
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