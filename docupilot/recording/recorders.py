from __future__ import annotations

import platform
import subprocess
import threading
import time
from datetime import datetime, timezone

import mss
import numpy as np
import pynput
from PySide6.QtCore import QObject, Signal

from docupilot.recording.session import EventWriter, Microphone, RecordingSession, Screen


class AvRecorder:
    """
    Handles audiovisual recording by capturing screen and microphone data simultaneously
    and encoding them into a video file. Primarily relies on ffmpeg for the audio-visual
    encoding tasks. Manages threading and subprocess operations for smooth functionality.

    This class ensures efficient capturing and writing of audiovisual content by creating
    video streams from the screen and audio input from the microphone. It also ensures
    proper encoding and resource cleanup post-recording lifecycle.

    :ivar _w: Width of the screen's recording area in pixels.
    :type _w: Int
    :ivar _h: Height of the screen's recording area in pixels.
    :type _h: Int
    """

    _FPS = 10

    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        """
        Initializes the AvRecorder with the given session and writer.
        :param session: The session object containing screen, microphone and events.
        :param writer: The event writer to write recording events to.
        """
        self._session = session
        self._writer = writer
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._on_done_callback = None  # set by RecorderService
        geo = session.screen.geometry()
        self._w = geo.width()
        self._h = geo.height()
        # Letzte bekannte Cursor-Position (Bildschirmkoordinaten).
        # Wird von InputRecorder.update_cursor_pos() thread-safe aktualisiert.
        self._cursor_x: int = 0
        self._cursor_y: int = 0
        self._cursor_lock = threading.Lock()

    def start(self) -> None:
        """
        Starts the audiovisual recording process by starting the capture loop and
        """

        self._stop.clear()
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="av-capture")

        if self._thread:
            self._thread.start()

        deadline = time.monotonic() + 2.0
        while not self._session.is_armed:
            if time.monotonic() > deadline:
                raise RuntimeError("AvRecorder: arm() timed out")
            time.sleep(0.001)

        self._writer.write(
            {"type": "av_started", "fps": self._FPS, "width": self._w, "height": self._h},
            t_ms=self._session.session_time_ms(),
        )

    def stop(self) -> None:
        """
        Ends the audiovisual recording process by stopping the capture loop
        """

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        self._writer.write(
            {"type": "av_stopped"},
            t_ms=self._session.session_time_ms(),
        )

        proc = self._proc
        self._proc = None
        if proc:
            threading.Thread(
                target=self._finalize,
                args=(proc, self._on_done_callback),
                daemon=False,
                name="av-finalize",
            ).start()

    def _capture_loop(self) -> None:
        """
        The loop that captures the screen data
        """

        interval = 1.0 / self._FPS
        geo = self._session.screen.geometry()
        region = {
            "left": geo.x(),
            "top": geo.y(),
            "width": geo.width(),
            "height": geo.height(),
        }
        with mss.MSS() as sct:
            while not self._stop.is_set():
                t0 = time.monotonic()

                screenshot = sct.grab(region)
                if not self._session.is_armed:
                    self._session.arm(time.monotonic_ns())

                bgra = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
                    screenshot.height, screenshot.width, 4
                )

                # Cursor als weisses Dreieck einzeichnen (BGR, ohne Alpha).
                bgr = bgra[:, :, :3].copy()
                self._draw_cursor(bgr, geo.x(), geo.y())

                try:
                    if self._proc and self._proc.stdin:
                        self._proc.stdin.write(bgr.tobytes())
                        self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break

                elapsed = time.monotonic() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

    def update_cursor_pos(self, x: int, y: int) -> None:
        """Aktualisiert die Cursor-Position thread-sicher (gerufen von InputRecorder)."""
        with self._cursor_lock:
            self._cursor_x = x
            self._cursor_y = y

    def _draw_cursor(self, bgr: np.ndarray, screen_x: int, screen_y: int) -> None:
        """
        Zeichnet einen stilisierten Mauszeiger (weisses Dreieck mit schwarzem
        Rand) in das BGR-Frame ein.

        Koordinaten: screen_x/screen_y ist der Ursprung des aufgenommenen
        Bildschirmbereichs. Die gespeicherte Cursor-Position ist in absoluten
        Bildschirmkoordinaten -- wir subtrahieren den Bereichs-Offset um Frame-
        relative Pixelkoordinaten zu erhalten.
        """
        import cv2
        with self._cursor_lock:
            cx = self._cursor_x - screen_x
            cy = self._cursor_y - screen_y

        h, w = bgr.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return  # Cursor ausserhalb des aufgenommenen Bereichs

        # Pfeilspitze oben links, 3 Punkte des Dreiecks (skaliert mit Frame-Hoehe).
        size = max(8, int(h * 0.018))  # ~15 px bei 864 px Hoehe
        pts = np.array([
            [cx,          cy         ],
            [cx,          cy + size  ],
            [cx + size//2, cy + int(size * 0.65)],
        ], dtype=np.int32)

        # Schwarzer Rand (1 px breiter)
        cv2.fillPoly(bgr, [pts + 1], (0, 0, 0))
        # Weisser Pfeil
        cv2.fillPoly(bgr, [pts], (255, 255, 255))

    @staticmethod
    def _finalize(proc: subprocess.Popen, on_done=None) -> None:
        """
        Finalizes the audiovisual recording process by closing the stdin pipe and waiting for
        :param proc: The subprocess object representing the recording process.
        """

        try:
            if proc.stdin:
                proc.stdin.close()
            _, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        except OSError:
            pass
        if on_done is not None:
            on_done()

    def _build_cmd(self) -> list[str]:
        """
        Build the command line arguments for the ffmpeg process.
        :return: A list of strings representing the command line arguments.
        """

        mic = self._session.microphone.description()
        out = str(self._session.recording_path)
        fps = str(self._FPS)
        sys = platform.system()

        video_in = [
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self._w}x{self._h}",
            "-r",
            fps,
            "-i",
            "pipe:0",
        ]
        codec = [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
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
            + ["-shortest", out]
        )


class InputRecorder:
    """
    Handles the recording of input events such as mouse movements, clicks, scrolls,
    and keyboard key presses/releases.

    The InputRecorder class is responsible for managing the recording session of user
    input activities and writing these events using the provided EventWriter. This class
    uses the pynput library to listen to mouse and keyboard events. Events are logged
    and throttled to avoid overloading the writer with excessive messages.
    """

    _THROTTLE_MS = 50.0

    def __init__(self, session: RecordingSession, writer: EventWriter,
                 av_recorder: "AvRecorder | None" = None) -> None:
        self._session = session
        self._writer = writer
        self._av_recorder = av_recorder  # fuer Cursor-Position-Updates
        self._last_move_t_ms = 0.0
        self._mouse: pynput.mouse.Listener | None = None
        self._keyboard: pynput.keyboard.Listener | None = None

    def start(self) -> None:
        self._last_move_t_ms = 0.0
        self._writer.write({"type": "input_started"}, t_ms=self._session.session_time_ms())
        self._mouse = pynput.mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._keyboard = pynput.keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

        if self._mouse and self._keyboard:
            self._mouse.start()
            self._keyboard.start()

    def stop(self) -> None:
        if self._mouse:
            self._mouse.stop()
            self._mouse = None
        if self._keyboard:
            self._keyboard.stop()
            self._keyboard = None
        self._writer.write({"type": "input_stopped"}, t_ms=self._session.session_time_ms())

    def _on_move(self, x: int, y: int) -> None:
        if self._av_recorder is not None:
            self._av_recorder.update_cursor_pos(x, y)

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        self._writer.write(
            {"type": "mouse_click", "x": x, "y": y, "button": str(button), "pressed": pressed},
            t_ms=self._session.session_time_ms(),
        )

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._writer.write(
            {"type": "mouse_scroll", "x": x, "y": y, "dx": dx, "dy": dy},
            t_ms=self._session.session_time_ms(),
        )

    def _on_press(self, key) -> None:
        self._writer.write(
            {"type": "key_press", "key": _key_str(key)}, t_ms=self._session.session_time_ms()
        )

    def _on_release(self, key) -> None:
        self._writer.write(
            {"type": "key_release", "key": _key_str(key)}, t_ms=self._session.session_time_ms()
        )


def _key_str(key) -> str:
    try:
        return str(key.char)
    except AttributeError:
        return str(key)


class RecorderService(QObject):
    """
    Manages the recording process, including starting, stopping, and handling recording sessions.

    The RecorderService class is designed to handle recording sessions by managing associated
    resources such as screen, microphone, and file writing. It provides signals for observing
    the recording state and ensures proper cleanup of resources in case of failures or when
    recording is stopped.
    """

    recording_started = Signal(object)
    recording_stopped = Signal(object)
    recording_error = Signal(str)
    recording_finalized = Signal(object)  # emitted when ffmpeg has finished writing

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session: RecordingSession | None = None
        self._writer: EventWriter | None = None
        self._av: AvRecorder | None = None
        self._input: InputRecorder | None = None

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

            if self._writer:
                self._writer.open()
                self._av = AvRecorder(session, self._writer)
                self._av._on_done_callback = lambda: self.recording_finalized.emit(session)
                self._input = InputRecorder(session, self._writer, self._av)

            if self._av and self._input:
                self._av.start()
                self._input.start()

            if self._writer:
                self._writer.write(
                    {
                        "type": "recording_started",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "metadata": session.to_metadata_dict(),
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
            if self._writer:
                self._writer.write(
                    {
                        "type": "recording_stopping",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    t_ms=session.session_time_ms(),
                )
            if self._input:
                self._input.stop()
            if self._av:
                self._av.stop()

            if self._writer:
                self._writer.write(
                    {
                        "type": "recording_stopped",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    t_ms=session.session_time_ms(),
                )
            self.recording_stopped.emit(session)
            return session

        finally:
            self._close_writer()
            self._session = None
            self._av = None
            self._input = None

    def _cleanup(self) -> None:
        for r in [self._input, self._av]:
            if r:
                try:
                    r.stop()
                except Exception:
                    pass
        self._close_writer()
        self._session = None
        self._av = None
        self._input = None

    def _close_writer(self) -> None:
        if self._writer:
            try:
                self._writer.close()
            finally:
                self._writer = None