from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import (
    QAudioInput,
    QMediaCaptureSession,
    QMediaRecorder,
    QScreenCapture,
)
from pynput import keyboard, mouse

from docupilot.recording.recording_session import RecordingSession


class RecorderService(QObject):
    """
    Service für multimodale Aufzeichnung.

    Verantwortlichkeiten:
    - RecordingSession erzeugen
    - Temp-Verzeichnis erzeugen
    - Bildschirmaufnahme starten/stoppen
    - Audioaufnahme starten/stoppen
    - globale Maus-/Tastaturevents aufnehmen
    - Events synchronisiert nach events.json schreiben
    """

    recording_started = Signal(object)
    recording_stopped = Signal(object)
    recording_error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

        self.current_session: RecordingSession | None = None

        self._events_file = None
        self._event_count = 0
        self._event_lock = threading.Lock()

        self._last_mouse_move_t_ms = 0.0
        self._mouse_move_interval_ms = 50.0

        self._screen_capture_session: QMediaCaptureSession | None = None
        self._screen_capture: QScreenCapture | None = None
        self._screen_recorder: QMediaRecorder | None = None

        self._audio_capture_session: QMediaCaptureSession | None = None
        self._audio_input: QAudioInput | None = None
        self._audio_recorder: QMediaRecorder | None = None

        self._mouse_listener: mouse.Listener | None = None
        self._keyboard_listener: keyboard.Listener | None = None

    def start_recording(
        self,
        screen: Any,
        microphone: Any,
    ) -> RecordingSession:
        """
        Erstellt eine RecordingSession und startet alle Modalitäten.

        :param screen: Ausgewählter QScreen.
        :param microphone: Ausgewähltes QAudioDevice.
        :return: RecordingSession mit allen Dateipfaden.
        """

        if self.current_session is not None:
            raise RuntimeError("Es läuft bereits eine Aufnahme.")

        session = RecordingSession(
            screen=screen,
            microphone=microphone,
        )

        session.session_dir.mkdir(parents=True, exist_ok=False)

        self.current_session = session

        try:
            self._open_events_file(session)

            self._write_event(
                {
                    "type": "recording_started",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "metadata": session.to_metadata_dict(),
                }
            )

            self._start_screen_recording(session)
            self._start_audio_recording(session)
            self._start_event_recording()

            self.recording_started.emit(session)

            return session

        except Exception:
            self._cleanup_after_failed_start()
            raise

    def stop_recording(self) -> RecordingSession:
        """
        Stoppt die aktuelle Aufnahme.

        :return: Die beendete RecordingSession.
        """

        if self.current_session is None:
            raise RuntimeError("Es läuft keine Aufnahme.")

        session = self.current_session

        self._write_event(
            {
                "type": "recording_stopping",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        self._stop_event_recording()
        self._stop_audio_recording()
        self._stop_screen_recording()

        self._write_event(
            {
                "type": "recording_stopped",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        self._close_events_file()

        self.current_session = None

        self.recording_stopped.emit(session)

        return session

    def is_recording(self) -> bool:
        return self.current_session is not None

    def _cleanup_after_failed_start(self) -> None:
        try:
            self._stop_event_recording()
        except Exception:
            pass

        try:
            self._stop_audio_recording()
        except Exception:
            pass

        try:
            self._stop_screen_recording()
        except Exception:
            pass

        try:
            self._close_events_file()
        except Exception:
            pass

        self.current_session = None

    def _open_events_file(self, session: RecordingSession) -> None:
        self._event_count = 0
        self._events_file = session.events_path.open("w", encoding="utf-8")
        self._events_file.write("[\n")
        self._events_file.flush()

    def _close_events_file(self) -> None:
        if self._events_file is None:
            return

        self._events_file.write("\n]\n")
        self._events_file.flush()
        self._events_file.close()
        self._events_file = None

    def _write_event(self, event: dict[str, Any]) -> None:
        if self.current_session is None:
            return

        if self._events_file is None:
            return

        event["t_ms"] = self.current_session.t_ms()

        with self._event_lock:
            if self._event_count > 0:
                self._events_file.write(",\n")

            json.dump(event, self._events_file, ensure_ascii=False)
            self._events_file.flush()
            self._event_count += 1

    def _start_screen_recording(self, session: RecordingSession) -> None:
        self._screen_capture_session = QMediaCaptureSession(self)
        self._screen_capture = QScreenCapture(self)
        self._screen_recorder = QMediaRecorder(self)

        self._screen_capture.setScreen(session.screen)

        self._screen_recorder.setOutputLocation(
            QUrl.fromLocalFile(str(session.screen_path))
        )
        self._screen_recorder.setQuality(QMediaRecorder.Quality.HighQuality)

        # Moderate FPS für bessere Performance und kleinere Dateien.
        self._screen_recorder.setVideoFrameRate(10.0)

        self._screen_recorder.errorOccurred.connect(
            self._on_screen_recorder_error
        )

        self._screen_capture.errorOccurred.connect(
            self._on_screen_capture_error
        )

        self._screen_capture_session.setScreenCapture(self._screen_capture)
        self._screen_capture_session.setRecorder(self._screen_recorder)

        geometry = session.screen.geometry()

        self._write_event(
            {
                "type": "screen_started",
                "file": session.screen_file_name,
                "screen_name": session.screen.name(),
                "width": geometry.width(),
                "height": geometry.height(),
                "fps": 10.0,
            }
        )

        self._screen_capture.start()
        self._screen_recorder.record()

    def _stop_screen_recording(self) -> None:
        if self._screen_recorder is not None:
            self._screen_recorder.stop()

        if self._screen_capture is not None:
            self._screen_capture.stop()

        self._write_event(
            {
                "type": "screen_stopped",
            }
        )

        self._screen_recorder = None
        self._screen_capture = None
        self._screen_capture_session = None

    def _start_audio_recording(self, session: RecordingSession) -> None:
        self._audio_capture_session = QMediaCaptureSession(self)
        self._audio_input = QAudioInput(session.microphone, self)
        self._audio_recorder = QMediaRecorder(self)

        self._audio_recorder.setOutputLocation(
            QUrl.fromLocalFile(str(session.audio_path))
        )
        self._audio_recorder.setQuality(QMediaRecorder.Quality.HighQuality)

        self._audio_recorder.errorOccurred.connect(
            self._on_audio_recorder_error
        )

        self._audio_capture_session.setAudioInput(self._audio_input)
        self._audio_capture_session.setRecorder(self._audio_recorder)

        self._write_event(
            {
                "type": "audio_started",
                "file": session.audio_file_name,
                "microphone": session.microphone.description(),
            }
        )

        self._audio_recorder.record()

    def _stop_audio_recording(self) -> None:
        if self._audio_recorder is not None:
            self._audio_recorder.stop()

        self._write_event(
            {
                "type": "audio_stopped",
            }
        )

        self._audio_recorder = None
        self._audio_input = None
        self._audio_capture_session = None

    def _start_event_recording(self) -> None:
        self._write_event(
            {
                "type": "events_started",
            }
        )

        self._last_mouse_move_t_ms = 0.0

        self._mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
        )

        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )

        self._mouse_listener.start()
        self._keyboard_listener.start()

    def _stop_event_recording(self) -> None:
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None

        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

        self._write_event(
            {
                "type": "events_stopped",
            }
        )

    def _on_mouse_move(self, x: int, y: int) -> None:
        """
        Mouse-Move-Events werden gedrosselt, damit events.json nicht unnötig groß wird.
        Für die spätere Segmentierung reichen 20 Events/Sekunde meist völlig aus.
        """

        if self.current_session is None:
            return

        current_t_ms = self.current_session.t_ms()

        if current_t_ms - self._last_mouse_move_t_ms < self._mouse_move_interval_ms:
            return

        self._last_mouse_move_t_ms = current_t_ms

        self._write_event(
            {
                "type": "mouse_move",
                "x": x,
                "y": y,
            }
        )

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        self._write_event(
            {
                "type": "mouse_click",
                "x": x,
                "y": y,
                "button": str(button),
                "pressed": pressed,
            }
        )

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._write_event(
            {
                "type": "mouse_scroll",
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
            }
        )

    def _on_key_press(self, key) -> None:
        self._write_event(
            {
                "type": "key_press",
                "key": self._safe_key(key),
            }
        )

    def _on_key_release(self, key) -> None:
        self._write_event(
            {
                "type": "key_release",
                "key": self._safe_key(key),
            }
        )

    def _on_screen_recorder_error(self, error, message: str) -> None:
        self._write_event(
            {
                "type": "screen_recorder_error",
                "error": str(error),
                "message": message,
            }
        )
        self.recording_error.emit(f"Bildschirmaufnahme-Fehler: {message}")

    def _on_screen_capture_error(self, error, message: str) -> None:
        self._write_event(
            {
                "type": "screen_capture_error",
                "error": str(error),
                "message": message,
            }
        )
        self.recording_error.emit(f"Screen-Capture-Fehler: {message}")

    def _on_audio_recorder_error(self, error, message: str) -> None:
        self._write_event(
            {
                "type": "audio_recorder_error",
                "error": str(error),
                "message": message,
            }
        )
        self.recording_error.emit(f"Audioaufnahme-Fehler: {message}")

    @staticmethod
    def _safe_key(key) -> str:
        try:
            return str(key.char)
        except AttributeError:
            return str(key)