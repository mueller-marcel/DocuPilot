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
    _FPS = 10

    def __init__(self, session: RecordingSession, writer: EventWriter, on_done=None) -> None:
        self._session = session
        self._writer = writer
        self._on_done = on_done
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        geo = session.screen.geometry()
        self._w = geo.width()
        self._h = geo.height()

    def start(self) -> None:
        self._stop.clear()
        self._proc = subprocess.Popen(
            self._build_cmd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name="av-capture")
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
                args=(proc, self._on_done),
                daemon=False,
                name="av-finalize",
            ).start()

    def _capture_loop(self) -> None:
        interval_ns = int(1_000_000_000 / self._FPS)
        geo = self._session.screen.geometry()
        region = {
            "left":   geo.x(),
            "top":    geo.y(),
            "width":  geo.width(),
            "height": geo.height(),
        }
        with mss.mss() as sct:
            # Arm before the first grab: t_ms=0 is the logical start of frame 0.
            frame_deadline_ns = time.monotonic_ns()
            self._session.arm(frame_deadline_ns)

            while not self._stop.is_set():
                screenshot = sct.grab(region)
                bgra = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
                    screenshot.height, screenshot.width, 4
                )
                bgr = bgra[:, :, :3].copy()

                try:
                    if self._proc and self._proc.stdin:
                        self._proc.stdin.write(bgr.tobytes())
                        self._proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    break

                # Advance absolute deadline to prevent cumulative drift.
                frame_deadline_ns += interval_ns
                remaining_ns = frame_deadline_ns - time.monotonic_ns()
                if remaining_ns > 0:
                    time.sleep(remaining_ns / 1e9)

    # The mouse cursor is deliberately NOT drawn into the frames. mss does not
    # capture it, and painting it in would (a) inject the event stream into the
    # video modality — the two must stay independent for the Shapley ablation —
    # and (b) make every settled frame differ from the next by a moving arrow,
    # which is exactly the noise the pHash dwell detection has to see through.

    @staticmethod
    def _finalize(proc: subprocess.Popen, on_done=None) -> None:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
        except OSError:
            pass
        if on_done is not None:
            on_done()

    def _build_cmd(self) -> list[str]:
        mic = self._session.microphone.description()
        out = str(self._session.recording_path)
        fps = str(self._FPS)
        sys = platform.system()

        video_in = [
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._w}x{self._h}",
            "-r", fps,
            "-i", "pipe:0",
        ]
        codec = [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
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
    def __init__(self, session: RecordingSession, writer: EventWriter) -> None:
        self._session = session
        self._writer = writer
        self._mouse: pynput.mouse.Listener | None = None
        self._keyboard: pynput.keyboard.Listener | None = None

    def start(self) -> None:
        self._writer.write({"type": "input_started"}, t_ms=self._session.session_time_ms())
        # No on_move handler: pointer motion is not an event we record, and the
        # AvRecorder no longer needs the cursor position (it does not draw it).
        self._mouse = pynput.mouse.Listener(
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
        self._writer.write({"type": "input_stopped"}, t_ms=self._session.session_time_ms())

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
    # ffmpeg keeps muxing after stop_recording() returns, so the file is only
    # complete once _finalize joins the process — this signal is what tells the UI
    # the recording is actually readable.
    recording_finalized = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session: RecordingSession | None = None
        self._writer:  EventWriter      | None = None
        self._av:      AvRecorder       | None = None
        self._input:   InputRecorder    | None = None

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
            self._av = AvRecorder(
                session, self._writer,
                on_done=lambda: self.recording_finalized.emit(session),
            )
            self._input = InputRecorder(session, self._writer)
            self._av.start()
            self._input.start()
            self._writer.write(
                {
                    "type": "recording_started",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "metadata": session.to_metadata_dict(),
                },
                t_ms=session.session_time_ms(),
            )
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
                {
                    "type": "recording_stopping",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                t_ms=session.session_time_ms(),
            )
            self._input.stop()
            self._av.stop()
            self._writer.write(
                {
                    "type": "recording_stopped",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
                t_ms=session.session_time_ms(),
            )
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
