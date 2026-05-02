from __future__ import annotations
from pathlib import Path
from typing import Any

import json
import threading

class EventWriter:
    """
    Writes events to a JSON array file.
    """

    def __init__(self, path: Path) -> None:
        """
        :param path: Destination file path. The parent directory must already exist.
        """

        self._path = path
        self._file = None
        self._count = 0
        self._lock = threading.Lock()

    def open(self) -> None:
        """
        Open the file and write the opening bracket of the JSON array.
        """

        self._count = 0
        self._file = self._path.open("w", encoding="utf-8")
        self._file.write("[\n")
        self._file.flush()

    def write(self, event: dict[str, Any], t_ms: float) -> None:
        """
        Append a single event object to the JSON array.

        :param event: A JSON-serializable dictionary representing the event.
        :param t_ms: Session-relative timestamp in milliseconds from RecordingSession.session_time_ms(). All modalities share the same monotonic clock, so timestamps are directly comparable.
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
        """
        Close the JSON array and flush the file to disk.
        """
        
        if self._file is None:
            return

        self._file.write("\n]\n")
        self._file.flush()
        self._file.close()
        self._file = None
