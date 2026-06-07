"""
docupilot.recording.recorders
──────────────────────────────
AvRecorder    – Screen (mss) + Mikrofon (ffmpeg-nativ) → recording.mp4
InputRecorder – Maus + Tastatur → events.json (pynput)
RecorderService – Orchestrierung

Gemeinsame Zeitbasis (garantiert)
──────────────────────────────────
t0 wird atomisch nach dem ersten mss-Frame-Grab gesetzt:

    screenshot = sct.grab(region)    # erster Frame vollständig im Speicher
    session.arm(time.monotonic_ns()) # t0 = jetzt → MP4-PTS 0 == t_ms 0

ffmpeg empfängt Frames über stdin (rawvideo pipe) — PTS 0 ist exakt der
erste Frame, der nach arm() ankommt. Audio wird von ffmpeg nativ über die
OS-API aufgenommen und intern mit dem Video-Stream synchronisiert.

Alle Input-Events werden mit session_time_ms() gestempelt (gleiche Uhr):

    mp4_position_ms == event t_ms  (Toleranz < 5 ms)
"""
from __future__ import annotations

import logging
import platform
import subprocess
import threading
import time
from datetime import datetime, timezone

import mss
import numpy as np
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
    Nimmt Screen (mss) und Mikrofon (ffmpeg-nativ) in einer MP4 auf.

    ffmpeg läuft mit zwei Inputs:
      - pipe:0  : rohe BGR-Frames von mss (stdin)
      - OS-Audio: Mikrofon direkt über gdigrab/avfoundation/pulse

    Der erste Frame setzt t0 (session.arm()), womit MP4-PTS 0 == t_ms 0.
    """

    _FPS = 10

    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        self._session     = session
        self._writer      = writer
        self._proc:       subprocess.Popen | None = None
        self._capture_thread: threading.Thread | None = None
        self._stop_event  = threading.Event()
        geo               = session.screen.geometry()
        self._w           = geo.width()
        self._h           = geo.height()

    def start(self) -> None:
        self._stop_event.clear()
        cmd = self._build_cmd()
        logger.debug("AvRecorder ffmpeg: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Capture-Thread startet ffmpeg-Pipe und setzt t0 beim ersten Frame
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="av-capture"
        )
        self._capture_thread.start()

        # Warten bis arm() gesetzt ist, bevor start() zurückkehrt.
        # Danach haben alle Events ein gültiges (positives) t_ms.
        deadline = time.monotonic() + 2.0
        while not self._session.start_monotonic_ns:
            if time.monotonic() > deadline:
                raise RuntimeError("AvRecorder: arm() timed out")
            time.sleep(0.001)

        self._writer.write(
            {"type": "av_started", "fps": self._FPS,
             "width": self._w, "height": self._h},
            t_ms=self._session.session_time_ms(),
        )

    def stop(self) -> None:
        # Capture-Loop stoppen
        self._stop_event.set()
        if self._capture_thread:
            self._capture_thread.join(timeout=5)
            self._capture_thread = None

        # ffmpeg sauber beenden
        if self._proc:
            try:
                self._proc.stdin.close()
                _, stderr = self._proc.communicate(timeout=15)
                if self._proc.returncode not in (0, 255):
                    logger.error("AvRecorder ffmpeg exit %d:\n%s",
                                 self._proc.returncode,
                                 stderr.decode(errors="replace"))
            except subprocess.TimeoutExpired:
                logger.warning("AvRecorder: ffmpeg did not exit, killing")
                self._proc.kill()
                self._proc.communicate()
            finally:
                self._proc = None

        self._writer.write(
            {"type": "av_stopped"},
            t_ms=self._session.session_time_ms(),
        )
        logger.info("Recording saved → %s", self._session.recording_path)

    def _capture_loop(self) -> None:
        interval = 1.0 / self._FPS
        geo      = self._session.screen.geometry()
        region   = {
            "left": geo.x(), "top": geo.y(),
            "width": geo.width(), "height": geo.height(),
        }
        with mss.mss() as sct:
            while not self._stop_event.is_set():
                t_frame    = time.monotonic()
                screenshot = sct.grab(region)

                # Erster Grab: t0 atomar setzen → PTS 0 == t_ms 0
                if not self._session.start_monotonic_ns:
                    self._session.arm(time.monotonic_ns())

                # mss → BGRA; ffmpeg rawvideo pipe erwartet BGR24
                bgra = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
                    screenshot.height, screenshot.width, 4
                )
                try:
                    self._proc.stdin.write(bgra[:, :, :3].tobytes())
                except (BrokenPipeError, OSError):
                    break  # ffmpeg wurde bereits beendet

                elapsed = time.monotonic() - t_frame
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    def _build_cmd(self) -> list[str]:
        """
        Baut den ffmpeg-Befehl: stdin für Video, OS-API für Audio.

        Windows  → dshow   für Mikrofon
        macOS    → avfoundation
        Linux    → pulse
        """
        mic = self._session.microphone.description()
        out = str(self._session.recording_path)
        fps = str(self._FPS)
        sys = platform.system()

        video_input = [
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{self._w}x{self._h}",
            "-r", fps, "-i", "pipe:0",
        ]
        video_codec = [
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-pix_fmt", "yuv420p",
        ]
        audio_codec = ["-c:a", "aac", "-b:a", "128k"]

        if sys == "Windows":
            audio_input = ["-f", "dshow", "-i", f"audio={mic}"]
        elif sys == "Darwin":
            audio_input = ["-f", "avfoundation", "-i", f"none:{mic}"]
        else:
            audio_input = ["-f", "pulse", "-i", mic]

        return (
            ["ffmpeg", "-loglevel", "error", "-y"]
            + video_input
            + audio_input
            + video_codec
            + audio_codec
            + [out]
        )


# ── InputRecorder ─────────────────────────────────────────────────────────────

class InputRecorder:
    """
    Globale Maus- und Tastatur-Events via pynput.
    Mouse-Move auf 1 Event / 50 ms gedrosselt.
    """

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

            # AvRecorder.start() blockiert bis arm() gesetzt ist.
            # Danach startet InputRecorder — beide laufen auf derselben Uhr.
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
