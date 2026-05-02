from __future__ import annotations
from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import QAudioInput, QMediaCaptureSession, QMediaRecorder, QScreenCapture
from docupilot.recording.event_writer import EventWriter
from docupilot.recording.recording_session import RecordingSession

import pynput


class ScreenModalityRecorder:
    """
    Captures the selected screen and writes the output to a video file.
    """

    _FPS = 10.0

    def __init__(self, session: RecordingSession, event_writer: EventWriter) -> None:
        """
        :param session: Active recording session providing file paths and screen reference.
        :param event_writer: Shared writer for appending structured events.
        """

        self._session = session
        self._event_writer = event_writer

        self._capture_session = QMediaCaptureSession()
        self._screen_capture = QScreenCapture()
        self._recorder = QMediaRecorder()

        self._screen_capture.setScreen(session.screen)
        self._recorder.setOutputLocation(QUrl.fromLocalFile(str(session.screen_path)))
        self._recorder.setQuality(QMediaRecorder.Quality.HighQuality)
        # Moderate frame rate for better performance and smaller files.
        self._recorder.setVideoFrameRate(self._FPS)
        self._recorder.errorOccurred.connect(self._on_recorder_error)
        self._screen_capture.errorOccurred.connect(self._on_capture_error)
        self._capture_session.setScreenCapture(self._screen_capture)
        self._capture_session.setRecorder(self._recorder)

    def start(self) -> None:
        """
        Start screen capture and begin recording to the output file.
        """

        geo = self._session.screen.geometry()
        self._event_writer.write({
            "type": "screen_started",
            "file": self._session.screen_file_name,
            "screen_name": self._session.screen.name(),
            "width": geo.width(),
            "height": geo.height(),
            "fps": self._FPS,
        }, t_ms=self._session.session_time_ms())
        self._screen_capture.start()
        self._recorder.record()

    def stop(self) -> None:
        """
        Stop recording and release Qt resources.
        """

        self._recorder.stop()
        self._screen_capture.stop()
        self._event_writer.write({"type": "screen_stopped"}, t_ms=self._session.session_time_ms())

    def _on_recorder_error(self, error, message: str) -> None:
        """
        Handle errors that occur during screen capture or recording.
        :param error: The error object.
        :param message: The message associated with the error.
        """

        self._event_writer.write({
            "type": "screen_recorder_error",
            "error": str(error),
            "message": message,
        }, t_ms=self._session.session_time_ms())

    def _on_capture_error(self, error, message: str) -> None:
        """
        Handle errors that occur during screen capture.
        :param error: The error object.
        :param message: The message associated with the error.
        """
        self._event_writer.write({
            "type": "screen_capture_error",
            "error": str(error),
            "message": message,
        }, t_ms=self._session.session_time_ms())


class AudioModalityRecorder:
    """
    Captures audio from the selected microphone and writes it to an audio file.

    SRP: Only responsible for audio capture lifecycle and its error events.
    DIP: Receives RecordingSession and EventWriter by injection.
    """

    def __init__(self, session: RecordingSession, event_writer: EventWriter) -> None:
        """
        :param session: Active recording session providing file paths and microphone reference.
        :param event_writer: Shared writer for appending structured events.
        """
        self._session = session
        self._event_writer = event_writer

        self._capture_session = QMediaCaptureSession()
        self._audio_input = QAudioInput(session.microphone)
        self._recorder = QMediaRecorder()

        self._recorder.setOutputLocation(QUrl.fromLocalFile(str(session.audio_path)))
        self._recorder.setQuality(QMediaRecorder.Quality.HighQuality)
        self._recorder.errorOccurred.connect(self._on_recorder_error)
        self._capture_session.setAudioInput(self._audio_input)
        self._capture_session.setRecorder(self._recorder)

    def start(self) -> None:
        """Start audio capture and begin recording to the output file."""
        self._event_writer.write({
            "type": "audio_started",
            "file": self._session.audio_file_name,
            "microphone": self._session.microphone.description(),
        }, t_ms=self._session.session_time_ms())
        self._recorder.record()

    def stop(self) -> None:
        """Stop recording and release Qt resources."""
        self._recorder.stop()
        self._event_writer.write({"type": "audio_stopped"}, t_ms=self._session.session_time_ms())

    def _on_recorder_error(self, error, message: str) -> None:
        self._event_writer.write({
            "type": "audio_recorder_error",
            "error": str(error),
            "message": message,
        }, t_ms=self._session.session_time_ms())


class InputModalityRecorder:
    """
    Records global mouse and keyboard events using pynput.

    SRP: Only responsible for input event capture and throttling logic.
    DIP: Receives RecordingSession and EventWriter by injection.

    Mouse-move events are throttled to ~20 events/second (every 50 ms) to keep
    the events file from growing unnecessarily large while still providing
    sufficient resolution for later session segmentation.
    """

    _MOUSE_MOVE_INTERVAL_MS = 50.0

    def __init__(self, session: RecordingSession, event_writer: EventWriter) -> None:
        """
        :param session: Active recording session used for timestamp generation.
        :param event_writer: Shared writer for appending structured events.
        """
        self._session = session
        self._event_writer = event_writer
        self._last_mouse_move_t_ms = 0.0
        self._mouse_listener: pynput.mouse.Listener | None = None
        self._keyboard_listener: pynput.keyboard.Listener | None = None

    def start(self) -> None:
        """Start listening for global mouse and keyboard events."""
        self._last_mouse_move_t_ms = 0.0
        self._event_writer.write({"type": "events_started"}, t_ms=self._session.session_time_ms())

        self._mouse_listener = pynput.mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
            on_scroll=self._on_mouse_scroll,
        )
        self._keyboard_listener = pynput.keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener.start()
        self._keyboard_listener.start()

    def stop(self) -> None:
        """Stop all input listeners."""
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        self._event_writer.write({"type": "events_stopped"}, t_ms=self._session.session_time_ms())

    def _on_mouse_move(self, x: int, y: int) -> None:
        current_t_ms = self._session.session_time_ms()
        if current_t_ms - self._last_mouse_move_t_ms < self._MOUSE_MOVE_INTERVAL_MS:
            return
        self._last_mouse_move_t_ms = current_t_ms
        self._event_writer.write({"type": "mouse_move", "x": x, "y": y}, t_ms=current_t_ms)

    def _on_mouse_click(self, x: int, y: int, button, pressed: bool) -> None:
        self._event_writer.write({
            "type": "mouse_click",
            "x": x,
            "y": y,
            "button": str(button),
            "pressed": pressed,
        }, t_ms=self._session.session_time_ms())

    def _on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._event_writer.write({
            "type": "mouse_scroll",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
        }, t_ms=self._session.session_time_ms())

    def _on_key_press(self, key) -> None:
        self._event_writer.write({"type": "key_press", "key": self._safe_key(key)}, t_ms=self._session.session_time_ms())

    def _on_key_release(self, key) -> None:
        self._event_writer.write({"type": "key_release", "key": self._safe_key(key)}, t_ms=self._session.session_time_ms())

    @staticmethod
    def _safe_key(key) -> str:
        """
        Returns a string representation of a pynput key.

        Printable characters are returned as-is; special keys (e.g., Key.shift)
        are converted via str().
        """
        try:
            return str(key.char)
        except AttributeError:
            return str(key)
