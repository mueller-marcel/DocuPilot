"""
One recording: where its files live, its shared clock, and the annotated ground
truth. The single owner of ground_truth_data — nothing else reads or writes
ground_truth.json.
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


# User input, as opposed to the recorder's own lifecycle events. Private —
# callers ask for input_events() instead of filtering the log themselves.
_INPUT_EVENT_TYPES = frozenset(
    {"mouse_click", "key_press", "key_release", "mouse_scroll"}
)

# The two boundary definitions a session can be annotated with. "end" is the
# thesis' definition (the action's result has settled on screen); "start" is
# the exposé's reading (the next action's first input). Both are kept so the
# evaluation can show how much of a modality's contribution depends on the
# choice. Entries without a kind predate the distinction and are "end".
BOUNDARY_KINDS: tuple[str, ...] = ("end", "start")
DEFAULT_BOUNDARY_KIND = "end"


# A session opened from disk has no real hardware behind it. These stubs satisfy
# the protocols so RecordingSession needs no special case for that.

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
    A recording: file paths, metadata, the shared clock and the ground truth.

    The single source of truth for ground_truth_data — recorded live or loaded
    through from_directory(), it is read and written only here.
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

    # Ground truth is optional: the list stays empty until a boundary is set or a
    # ground_truth.json is loaded.
    ground_truth_data: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.session_dir = self.base_dir / f"session_{self.session_id}"

    @classmethod
    def from_directory(cls, directory: Path) -> "RecordingSession":
        """
        Rebuild a session from a directory ("File > Open"). Screen and
        microphone are replaced by stubs.

        :param directory: the session directory (must contain recording.mp4 and
            events.json; ground_truth.json is optional).
        :return: a RecordingSession with session_dir = directory.
        :raises FileNotFoundError: when recording.mp4 or events.json is missing.
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

    # ── Event log ────────────────────────────────────────────────────────

    def read_events(self) -> list[dict[str, Any]]:
        """
        The raw event log, or an empty list when missing. Not cached — a running
        session would otherwise hand out a stale snapshot.

        :return: events as written by EventWriter, in file order.
        """
        try:
            with self.events_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    @staticmethod
    def input_events_of(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        The user's own input within an already-read log — for callers that hold
        the log and must not re-read the file to filter it.

        :return: clicks, keys and scrolls, in the given order.
        """
        return [e for e in events if e.get("type") in _INPUT_EVENT_TYPES]

    def input_events(self) -> list[dict[str, Any]]:
        """
        Only the user's own input, without the recorder's lifecycle events.

        :return: clicks, keys and scrolls, in file order.
        """
        return self.input_events_of(self.read_events())

    # ── Ground truth ─────────────────────────────────────────────────────

    def load_ground_truth(self) -> None:
        """
        (Re)load ground_truth.json when present. Idempotent; a missing or
        broken file leaves ground_truth_data empty — the ground truth is optional.
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
        """Write ground_truth_data to ground_truth.json in session_dir."""
        self.ground_truth_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ground_truth_path.open("w", encoding="utf-8") as f:
            json.dump(self.ground_truth_data, f, ensure_ascii=False, indent=2)

    def add_ground_truth_boundary(
        self, t_ms: float, label: str | None = None, kind: str = DEFAULT_BOUNDARY_KIND
    ) -> None:
        """
        Append one boundary and save at once. Keeps the entry schema
        (t_ms, label, kind, created_at_utc) in one place.

        :param t_ms: the boundary's time in milliseconds.
        :param label: display name, e.g. a formatted timestamp.
        :param kind: which definition the boundary follows, see BOUNDARY_KINDS.
        :raises ValueError: for an unknown kind.
        """
        if kind not in BOUNDARY_KINDS:
            raise ValueError(f"Unbekannte Grenzart {kind!r}; erlaubt: {BOUNDARY_KINDS}")
        self.ground_truth_data.append({
            "t_ms": t_ms,
            "label": label or "Grenze",
            "kind": kind,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.save_ground_truth()

    @staticmethod
    def boundary_kind(entry: dict[str, Any]) -> str:
        """The definition an entry follows; entries without one are "end"."""
        return str(entry.get("kind", DEFAULT_BOUNDARY_KIND))

    def count_boundaries(self, kind: str = DEFAULT_BOUNDARY_KIND) -> int:
        """How many boundaries of one kind are annotated."""
        return sum(1 for e in self.ground_truth_data if self.boundary_kind(e) == kind)

    def set_ground_truth_boundaries(self, boundaries: list[dict[str, Any]]) -> None:
        """
        Replace all boundaries and save at once.

        :param boundaries: the new, complete list.
        """
        self.ground_truth_data = boundaries
        self.save_ground_truth()

    def ground_truth_markers(self, kind: str = DEFAULT_BOUNDARY_KIND) -> list[tuple[float, str]]:
        """
        The boundaries of one definition as (t_ms, label) tuples, the shape the
        timeline draws.

        :param kind: which definition, see BOUNDARY_KINDS.
        :return: markers in chronological order.
        """
        markers = [
            (float(entry.get("t_ms", 0.0)), str(entry.get("label", "Grenze")))
            for entry in self.ground_truth_data
            if self.boundary_kind(entry) == kind
        ]
        return sorted(markers, key=lambda m: m[0])

    # ── Clock ────────────────────────────────────────────────────────────

    def arm(self, t0_ns: int) -> None:
        """Fix the shared clock origin every modality is timed against."""
        self.start_monotonic_ns = t0_ns
        self._armed = True

    @property
    def is_armed(self) -> bool:
        return self._armed

    def session_time_ms(self) -> float:
        """Milliseconds since t0. Negative before arm()."""
        return round((time.monotonic_ns() - self.start_monotonic_ns) / 1_000_000, 3)

    # ── Paths and metadata ───────────────────────────────────────────────

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
    """
    Appends events to a JSON array on disk, one line each, thread-safely.

    Written incrementally and flushed per event so a crash mid-recording leaves
    every event up to that point readable; the closing bracket is added by
    close().
    """

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
        """
        Append one event, stamped with the session time. Ignored when the writer
        is not open, so recorders may keep reporting after close() without harm.
        """
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
