"""
docupilot.recording.recorders
──────────────────────────────
AvRecorder      – Screen + Mikrofon → recording.mp4  (ffmpeg, keine Pipes)
InputRecorder   – Maus + Tastatur   → events.json    (pynput)
RecorderService – Orchestrierung, gemeinsame Zeitbasis

Design
──────
ffmpeg nimmt Screen und Mikrofon vollständig selbst auf:
    Windows  → gdigrab (Screen) + dshow (Mikrofon)
    macOS    → avfoundation
    Linux    → x11grab + pulse

Kein stdin, keine Pipes, kein moov-Problem.
ffmpeg schreibt die MP4 direkt und vollständig.

Zeitbasis
─────────
t0 = time.monotonic_ns(), gesetzt unmittelbar vor ffmpeg.start().
Events werden mit session_time_ms() gestempelt (gleiche Uhr).
Offset = ffmpeg-Startup-Latenz (~20–50 ms) → innerhalb 100 ms Toleranz.
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
    """ffmpeg nimmt Screen + Mikrofon auf und schreibt direkt recording.mp4."""

    _FPS = 10

    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        self._session = session
        self._writer  = writer
        self._proc:   subprocess.Popen | None = None

    def start(self) -> None:
        cmd = self._build_cmd()
        logger.debug("AvRecorder: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._writer.write(
            {"type": "av_started", "fps": self._FPS},
            t_ms=self._session.session_time_ms(),
        )

    def stop(self) -> None:
        proc       = self._proc
        self._proc = None

        self._writer.write(
            {"type": "av_stopped"},
            t_ms=self._session.session_time_ms(),
        )

        if proc:
            # Im Hintergrund beenden — UI bleibt responsiv
            threading.Thread(
                target=self._finalize, args=(proc,),
                daemon=False, name="av-finalize",
            ).start()

    def _finalize(self, proc: subprocess.Popen) -> None:
        """
        Beendet ffmpeg sauber damit der MP4-Footer (moov-Atom) geschrieben wird.

        Auf Windows killt proc.terminate() den Prozess sofort (TerminateProcess)
        ohne den Footer zu schreiben → MP4 nicht abspielbar.
        Stattdessen senden wir 'q\n' auf stdin — ffmpeg beendet sich dann
        sauber und schreibt den Footer.
        """
        try:
            proc.stdin.write(b"q\n")
            proc.stdin.flush()
            proc.stdin.close()
            _, stderr = proc.communicate(timeout=15)
            if stderr:
                logger.warning("ffmpeg: %s", stderr.decode(errors="replace").strip())
            logger.info("Recording saved → %s", self._session.recording_path)
        except (OSError, BrokenPipeError):
            pass  # ffmpeg bereits beendet
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()

    def _build_cmd(self) -> list[str]:
        geo = self._session.screen.geometry()
        mic = self._session.microphone.description()
        out = str(self._session.recording_path)
        fps = str(self._FPS)
        sys = platform.system()

        codec = [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
        ]

        if sys == "Windows":
            return [
                "ffmpeg", "-loglevel", "warning", "-y",
                "-f", "gdigrab", "-framerate", fps,
                "-offset_x", str(geo.x()), "-offset_y", str(geo.y()),
                "-video_size", f"{geo.width()}x{geo.height()}",
                "-i", "desktop",
                "-f", "dshow", "-i", f"audio={mic}",
            ] + codec + [out]

        elif sys == "Darwin":
            return [
                "ffmpeg", "-loglevel", "warning", "-y",
                "-f", "avfoundation", "-framerate", fps,
                "-i", f"1:{mic}",
            ] + codec + [out]

        else:  # Linux
            display = ":0.0"
            return [
                "ffmpeg", "-loglevel", "warning", "-y",
                "-f", "x11grab", "-framerate", fps,
                "-video_size", f"{geo.width()}x{geo.height()}",
                "-i", f"{display}+{geo.x()},{geo.y()}",
                "-f", "pulse", "-i", mic,
            ] + codec + [out]


# ── InputRecorder ─────────────────────────────────────────────────────────────

class InputRecorder:
    """Maus + Tastatur via pynput. Mouse-Move gedrosselt auf 1/50 ms."""

    _THROTTLE_MS = 50.0

    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        self._session        = session
        self._writer         = writer
        self._last_move_t_ms = 0.0
        self._mouse:    pynput.mouse.Listener    | None = None
        self._keyboard: pynput.keyboard.Listener | None = None

    def start(self) -> None:
        self._last_move_t_ms = 0.0
        self._writer.write({"type": "input_started"},
                           t_ms=self._session.session_time_ms())
        self._mouse = pynput.mouse.Listener(
            on_move=self._on_move, on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._keyboard = pynput.keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release,
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
        self._writer.write({"type": "input_stopped"},
                           t_ms=self._session.session_time_ms())

    def _on_move(self, x: int, y: int) -> None:
        t_ms = self._session.session_time_ms()
        if t_ms - self._last_move_t_ms < self._THROTTLE_MS:
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
        self._writer.write({"type": "key_press", "key": _key_str(key)},
                           t_ms=self._session.session_time_ms())

    def _on_release(self, key) -> None:
        self._writer.write({"type": "key_release", "key": _key_str(key)},
                           t_ms=self._session.session_time_ms())


def _key_str(key) -> str:
    try:
        return str(key.char)
    except AttributeError:
        return str(key)


# ── RecorderService ───────────────────────────────────────────────────────────

class RecorderService(QObject):
    """
    Startet und stoppt eine Aufnahme-Session.

    Zeitbasis: t0 = time.monotonic_ns() unmittelbar vor av.start().
    Offset zwischen t0 und erstem ffmpeg-Frame: ~20–50 ms (< 100 ms).

    Signals: recording_started(RecordingSession)
             recording_stopped(RecordingSession)
             recording_error(str)
    """

    recording_started = Signal(object)
    recording_stopped = Signal(object)
    recording_error   = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session: RecordingSession | None = None
        self._writer:  EventWriter      | None = None
        self._av:      AvRecorder       | None = None
        self._input:   InputRecorder    | None = None

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

            # t0 setzen, dann sofort beide starten
            session.arm(time.monotonic_ns())
            self._av.start()
            self._input.start()

            self._writer.write(
                {"type": "recording_started",
                 "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                 "metadata": session.to_metadata_dict()},
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