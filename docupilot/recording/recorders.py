"""
docupilot.recording.recorders
──────────────────────────────
AvRecorder    – Screen + Mikrofon → recording.mp4  (ein ffmpeg-Prozess)
InputRecorder – Maus + Tastatur   → events.json    (pynput)
RecorderService – Orchestrierung, gemeinsame Zeitbasis

Design
──────
Ein einziger ffmpeg-Prozess nimmt Bildschirm und Mikrofon gleichzeitig auf.
Kein Puffern, kein Merge, kein sounddevice.

ffmpeg liest beide Quellen nativ:
    Windows  → gdigrab (Screen) + dshow (Mikrofon)
    macOS    → avfoundation (Screen + Mikrofon)
    Linux    → x11grab (Screen) + pulse (Mikrofon)

Gemeinsame Zeitbasis
────────────────────
t0 = time.monotonic_ns(), gesetzt unmittelbar vor ffmpeg.start() und
vor InputRecorder.start(). Die ffmpeg-Startup-Latenz (~50 ms) ist die
einzige Sync-Ungenauigkeit — weit unter der 100 ms Toleranz.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone

import pynput
from PySide6.QtCore import QObject, Signal

from docupilot.recording.session import (
    EventWriter,
    Microphone,
    RecordingSession,
    Screen,
)

logger = logging.getLogger(__name__)


# ── AvRecorder ────────────────────────────────────────────────────────────────

class AvRecorder:
    """
    Nimmt Bildschirm und Mikrofon mit einem einzigen ffmpeg-Prozess auf.

    ffmpeg liest beide Quellen simultan und schreibt direkt in recording.mp4.
    Gestoppt wird sauber über 'q\\n' auf stdin.
    """

    _FPS = 10

    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        self._session = session
        self._writer  = writer
        self._proc:   subprocess.Popen | None = None

    def start(self) -> None:
        geo = self._session.screen.geometry()
        cmd = self._build_cmd(
            mic_description = self._session.microphone.description(),
            x=geo.x(), y=geo.y(),
            w=geo.width(), h=geo.height(),
            out=str(self._session.recording_path),
        )
        logger.debug("AvRecorder: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._writer.write(
            {
                "type":        "av_started",
                "file":        self._session.recording_file_name,
                "fps":         self._FPS,
                "width":       geo.width(),
                "height":      geo.height(),
                "microphone":  self._session.microphone.description(),
            },
            t_ms=self._session.session_time_ms(),
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        # ffmpeg sauber beenden: 'q' auf stdin
        try:
            self._proc.stdin.write(b"q\n")
            self._proc.stdin.flush()
            _, stderr = self._proc.communicate(timeout=15)
            if self._proc.returncode != 0:
                logger.error(
                    "AvRecorder ffmpeg exit %d:\n%s",
                    self._proc.returncode,
                    stderr.decode(errors="replace"),
                )
        except subprocess.TimeoutExpired:
            logger.warning("AvRecorder: ffmpeg did not exit cleanly, killing")
            self._proc.kill()
            self._proc.communicate()
        finally:
            self._proc = None

        self._writer.write(
            {"type": "av_stopped", "file": self._session.recording_file_name},
            t_ms=self._session.session_time_ms(),
        )
        logger.info("Recording saved → %s", self._session.recording_path)

    def _build_cmd(
        self,
        mic_description: str,
        x: int, y: int, w: int, h: int,
        out: str,
    ) -> list[str]:
        """
        Baut den plattformspezifischen ffmpeg-Befehl.

        Windows  – gdigrab für Screen, dshow für Audio
        macOS    – avfoundation für beides (Index-basiert)
        Linux    – x11grab für Screen, pulse für Audio
        """
        system = platform.system()

        base = ["ffmpeg", "-loglevel", "error", "-y"]

        if system == "Windows":
            return base + [
                # Screen: gdigrab, crop auf Monitor-Koordinaten
                "-f", "gdigrab",
                "-framerate", str(self._FPS),
                "-offset_x", str(x), "-offset_y", str(y),
                "-video_size", f"{w}x{h}",
                "-i", "desktop",
                # Mikrofon: dshow
                "-f", "dshow",
                "-i", f"audio={mic_description}",
                # Output
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                out,
            ]

        elif system == "Darwin":
            # avfoundation: "screen_index:audio_index"
            # Screen-Index 1 = primärer Monitor, Audio-Index aus description
            return base + [
                "-f", "avfoundation",
                "-framerate", str(self._FPS),
                "-capture_cursor", "1",
                "-i", f"1:{mic_description}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                out,
            ]

        else:  # Linux
            display = ":0.0"
            return base + [
                # Screen: x11grab
                "-f", "x11grab",
                "-framerate", str(self._FPS),
                "-video_size", f"{w}x{h}",
                "-i", f"{display}+{x},{y}",
                # Mikrofon: pulse
                "-f", "pulse",
                "-i", mic_description,
                # Output
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k",
                out,
            ]


# ── InputRecorder ─────────────────────────────────────────────────────────────

class InputRecorder:
    """
    Globale Maus- und Tastatur-Events via pynput.
    Mouse-Move wird auf 1 Event / 50 ms gedrosselt.
    Alle Events nutzen session_time_ms() — dieselbe Uhr wie die MP4.
    """

    _MOUSE_MOVE_INTERVAL_MS = 50.0

    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        self._session        = session
        self._writer         = writer
        self._last_move_t_ms = 0.0
        self._mouse:    pynput.mouse.Listener    | None = None
        self._keyboard: pynput.keyboard.Listener | None = None

    def start(self) -> None:
        self._last_move_t_ms = 0.0
        self._writer.write(
            {"type": "input_started"},
            t_ms=self._session.session_time_ms(),
        )
        self._mouse = pynput.mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._keyboard = pynput.keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._mouse.start()
        self._keyboard.start()

    def stop(self) -> None:
        if self._mouse:
            self._mouse.stop()
            self._mouse = None
        if self._keyboard:
            self._keyboard.stop()
            self._keyboard = None
        self._writer.write(
            {"type": "input_stopped"},
            t_ms=self._session.session_time_ms(),
        )

    def _on_move(self, x: int, y: int) -> None:
        t_ms = self._session.session_time_ms()
        if t_ms - self._last_move_t_ms < self._MOUSE_MOVE_INTERVAL_MS:
            return
        self._last_move_t_ms = t_ms
        self._writer.write({"type": "mouse_move", "x": x, "y": y}, t_ms=t_ms)

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        self._writer.write(
            {"type": "mouse_click", "x": x, "y": y,
             "button": str(button), "pressed": pressed},
            t_ms=self._session.session_time_ms(),
        )

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._writer.write(
            {"type": "mouse_scroll", "x": x, "y": y, "dx": dx, "dy": dy},
            t_ms=self._session.session_time_ms(),
        )

    def _on_press(self, key) -> None:
        self._writer.write(
            {"type": "key_press", "key": _key_str(key)},
            t_ms=self._session.session_time_ms(),
        )

    def _on_release(self, key) -> None:
        self._writer.write(
            {"type": "key_release", "key": _key_str(key)},
            t_ms=self._session.session_time_ms(),
        )


def _key_str(key) -> str:
    try:
        return str(key.char)
    except AttributeError:
        return str(key)


# ── RecorderService ───────────────────────────────────────────────────────────

class RecorderService(QObject):
    """
    Orchestriert den Lifecycle einer Aufnahme-Session.

    Gemeinsame Zeitbasis
    ────────────────────
    t0 wird mit time.monotonic_ns() gesetzt unmittelbar bevor
    AvRecorder und InputRecorder gestartet werden:

        session.arm(time.monotonic_ns())
        av_recorder.start()      # ffmpeg startet, schreibt MP4 ab t≈0
        input_recorder.start()   # Events ab t≈0

    Die ffmpeg-Startup-Latenz (~20–50 ms) ist die einzige Ungenauigkeit.

    Signals
    ───────
    recording_started(RecordingSession)
    recording_stopped(RecordingSession)
    recording_error(str)
    """

    recording_started = Signal(object)
    recording_stopped = Signal(object)
    recording_error   = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session:  RecordingSession | None = None
        self._writer:   EventWriter      | None = None
        self._av:       AvRecorder       | None = None
        self._input:    InputRecorder    | None = None

    @property
    def current_session(self) -> RecordingSession | None:
        return self._session

    def is_recording(self) -> bool:
        return self._session is not None

    def start_recording(self, screen: Screen, microphone: Microphone) -> RecordingSession:
        if self.is_recording():
            raise RuntimeError("A recording is already in progress.")

        session = RecordingSession(screen=screen, microphone=microphone)
        session.session_dir.mkdir(parents=True, exist_ok=False)
        self._session = session

        try:
            self._writer = EventWriter(session.events_path)
            self._writer.open()

            self._av    = AvRecorder(session, self._writer)
            self._input = InputRecorder(session, self._writer)

            # t0 setzen, dann sofort beide Recorder starten
            session.arm(time.monotonic_ns())
            self._av.start()
            self._input.start()

            self._writer.write(
                {
                    "type":         "recording_started",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "metadata":      session.to_metadata_dict(),
                },
                t_ms=session.session_time_ms(),
            )
            self.recording_started.emit(session)
            return session

        except Exception:
            self._cleanup()
            raise

    def stop_recording(self) -> RecordingSession:
        if self._session is None:
            raise RuntimeError("No recording in progress.")

        session = self._session
        try:
            self._writer.write(
                {"type": "recording_stopping",
                 "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                t_ms=session.session_time_ms(),
            )

            # Input zuerst stoppen (sofort), dann ffmpeg (wartet auf sauberes Ende)
            if self._input:
                self._input.stop()
            if self._av:
                self._av.stop()

            self._writer.write(
                {"type": "recording_stopped",
                 "timestamp_utc": datetime.now(timezone.utc).isoformat()},
                t_ms=session.session_time_ms(),
            )
            self.recording_stopped.emit(session)
            return session

        finally:
            self._close_writer()
            self._session = None
            self._av      = None
            self._input   = None

    def _cleanup(self) -> None:
        for r in [self._input, self._av]:
            if r:
                try:
                    r.stop()
                except Exception:
                    pass
        self._close_writer()
        self._session = None
        self._av      = None
        self._input   = None

    def _close_writer(self) -> None:
        if self._writer:
            try:
                self._writer.close()
            finally:
                self._writer = None