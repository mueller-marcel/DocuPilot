from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QObject, Signal

from docupilot.recording.event_writer import EventWriter
from docupilot.recording.modality_recorders import AudioModalityRecorder, InputModalityRecorder, ScreenModalityRecorder
from docupilot.recording.protocols import Microphone, ModalityRecorder, Screen
from docupilot.recording.recording_session import RecordingSession, RecordingSessionSerializer


class RecorderService(QObject):
    """
    Orchestrates the lifecycle of a multimodal recording session.
    """

    recording_started = Signal(object)
    recording_stopped = Signal(object)
    recording_error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """
        Initialize the RecorderService.
        :param parent:
        """

        super().__init__(parent)

        self.current_session: RecordingSession | None = None
        self._event_writer: EventWriter | None = None
        self._modalities: list[ModalityRecorder] = []

    def start_recording(self, screen: Screen, microphone: Microphone) -> RecordingSession:
        """
        Create a new RecordingSession and start all modalities.

        :param screen: The screen device to capture (must satisfy the Screen protocol).
        :param microphone: The audio input device (must satisfy the Microphone protocol).
        :return: The newly created and active RecordingSession.
        :raises RuntimeError: If a recording is already in progress.
        """

        if self.current_session is not None:
            raise RuntimeError("A recording is already in progress.")

        session = RecordingSession(screen=screen, microphone=microphone)
        session.session_dir.mkdir(parents=True, exist_ok=False)
        self.current_session = session

        try:
            self._event_writer = EventWriter(session.events_path)
            self._event_writer.open()
            self._event_writer.write({
                "type": "recording_started",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "metadata": RecordingSessionSerializer.to_metadata_dict(session),
            }, t_ms=session.session_time_ms())

            self._modalities = self._build_modalities(session, self._event_writer)
            for modality in self._modalities:
                modality.start()

            self.recording_started.emit(session)
            return session

        except Exception:
            self._cleanup_after_failed_start()
            raise

    def stop_recording(self) -> RecordingSession:
        """
        Stop the active recording and finalize all output files.

        :return: The completed RecordingSession.
        :raises RuntimeError: If no recording is currently in progress.
        """
        if self.current_session is None:
            raise RuntimeError("No recording is currently in progress.")

        session = self.current_session

        self._event_writer.write({
            "type": "recording_stopping",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, t_ms=session.session_time_ms())

        # Stop in reverse order to mirror the start sequence.
        for modality in reversed(self._modalities):
            modality.stop()

        self._event_writer.write({
            "type": "recording_stopped",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }, t_ms=session.session_time_ms())
        self._event_writer.close()

        self.current_session = None
        self._modalities = []
        self.recording_stopped.emit(session)
        return session

    def is_recording(self) -> bool:
        """Return True if a recording session is currently active."""
        return self.current_session is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_modalities(
        self,
        session: RecordingSession,
        event_writer: EventWriter,
    ) -> list[ModalityRecorder]:
        """
        Instantiate all modality recorders for the given session.

        OCP: To add a new modality, append it here — no other method needs
        to change.

        :param session: The active recording session.
        :param event_writer: Shared event writer passed to each modality.
        :return: Ordered list of modality recorders.
        """
        return [
            ScreenModalityRecorder(session, event_writer),
            AudioModalityRecorder(session, event_writer),
            InputModalityRecorder(session, event_writer),
        ]

    def _cleanup_after_failed_start(self) -> None:
        """
        Best-effort teardown when start_recording() raises an exception.

        Each modality and the event writer are stopped independently so that
        one failure does not prevent the others from being cleaned up.
        """
        for modality in reversed(self._modalities):
            try:
                modality.stop()
            except Exception:
                pass

        if self._event_writer is not None:
            try:
                self._event_writer.close()
            except Exception:
                pass

        self.current_session = None
        self._modalities = []
