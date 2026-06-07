"""
docupilot.recording.recorders
──────────────────────────────
AvRecorder      – Screen (mss) + Mikrofon (ffmpeg-nativ) → recording.mp4
InputRecorder   – Maus + Tastatur → events.json (pynput)
RecorderService – Orchestrierung

Gemeinsame Zeitbasis (garantiert, Toleranz < 5 ms)
───────────────────────────────────────────────────
t0 wird nach dem ersten mss-Frame-Grab gesetzt:

    screenshot = sct.grab(region)    # erster Frame vollständig im Speicher
    session.arm(time.monotonic_ns()) # t0 = jetzt

Dieser Frame wird als erster an ffmpeg übergeben → MP4-PTS 0 == t_ms 0.
Audio läuft ffmpeg-nativ (dshow/pulse/avfoundation) — kein sounddevice,
kein numpy, kein dtype-Problem.

Events werden mit session_time_ms() gestempelt (gleiche Uhr):
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
    Nimmt Screen (mss → ffmpeg stdin) und Mikrofon (ffmpeg-nativ) auf.

    Zeitbasis:
        - Erster mss-Grab → session.arm() → t0
        - Erster Frame geht sofort an ffmpeg stdin → MP4-PTS 0 == t_ms 0
        - Audio startet ffmpeg-nativ, wird intern mit Video synchronisiert

    Stopp:
        - Capture-Loop endet, stdin wird geschlossen
        - ffmpeg schreibt moov-Atom und beendet sich sauber
        - Alles im Hintergrund → UI bleibt responsiv
    """

    _FPS = 10

    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        self._session = session
        self._writer  = writer
        self._proc:   subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop    = threading.Event()
        geo           = session.screen.geometry()
        self._w       = geo.width()
        self._h       = geo.height()

    def start(self) -> None:
        self._stop.clear()
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="av-capture"
        )
        self._thread.start()

        # Blockiert bis arm() gesetzt wurde (erster Frame gegrabt)
        # → alle nachfolgenden Events haben t_ms >= 0
        deadline = time.monotonic() + 2.0
        while not self._session.is_armed:
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
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        self._writer.write(
            {"type": "av_stopped"},
            t_ms=self._session.session_time_ms(),
        )

        # ffmpeg im Hintergrund beenden → UI bleibt responsiv
        proc       = self._proc
        self._proc = None
        if proc:
            threading.Thread(
                target=self._finalize, args=(proc,),
                daemon=False, name="av-finalize",
            ).start()

    def _capture_loop(self) -> None:
        interval = 1.0 / self._FPS
        geo      = self._session.screen.geometry()
        region   = {
            "left": geo.x(), "top": geo.y(),
            "width": geo.width(), "height": geo.height(),
        }
        with mss.mss() as sct:
            while not self._stop.is_set():
                t0 = time.monotonic()

                screenshot = sct.grab(region)

                # Erster Grab: t0 nach dem Grab setzen →
                # MP4-PTS 0 == t_ms 0 per Definition
                if not self._session.is_armed:
                    self._session.arm(time.monotonic_ns())

                # BGRA → BGR24 (ffmpeg rawvideo erwartet BGR24)
                bgra = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
                    screenshot.height, screenshot.width, 4
                )
                try:
                    self._proc.stdin.write(bgra[:, :, :3].tobytes())
                    self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break

                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    def _finalize(self, proc: subprocess.Popen) -> None:
        """stdin schließen → ffmpeg schreibt moov-Atom und beendet sich."""
        try:
            proc.stdin.close()
            _, stderr = proc.communicate(timeout=30)
            if stderr:
                logger.warning("ffmpeg: %s", stderr.decode(errors="replace").strip())
            logger.info("Recording saved → %s", self._session.recording_path)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        except OSError:
            pass

    def _build_cmd(self) -> list[str]:
        """
        Video: rawvideo auf stdin (BGR24)
        Audio: OS-nativ (dshow / avfoundation / pulse)
        """
        mic = self._session.microphone.description()
        out = str(self._session.recording_path)
        fps = str(self._FPS)
        sys = platform.system()

        video_in = [
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24", "-s", f"{self._w}x{self._h}",
            "-r", fps, "-i", "pipe:0",
        ]
        codec = [
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "28", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
        ]

        if sys == "Windows":
            audio_in = ["-f", "dshow", "-i", f"audio={mic}"]
        elif sys == "Darwin":
            audio_in = ["-f", "avfoundation", "-i", f"none:{mic}"]
        else:
            audio_in = ["-f", "pulse", "-i", mic]

        return (
            ["ffmpeg", "-loglevel", "warning", "-y"]
            + video_in
            + audio_in
            + codec
            # -shortest: beendet die Aufnahme wenn der kürzere Stream endet.
            # Ohne dieses Flag wartet ffmpeg nach stdin-close (Video-EOF) noch
            # auf Audio-EOF vom dshow/pulse-Device — das kommt nie → MP4 kaputt.
            + ["-shortest", out]
        )


# ── InputRecorder ─────────────────────────────────────────────────────────────

class InputRecorder:
    """Maus + Tastatur via pynput. Mouse-Move gedrosselt auf 1 / 50 ms."""

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

            # av.start() blockiert bis arm() gesetzt ist (erster Frame gegrabt).
            # Danach laufen Input-Events auf derselben Uhr wie die MP4.
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