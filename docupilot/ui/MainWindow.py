from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from docupilot.recording.recorder_service import RecorderService
from docupilot.ui.widgets.MicrophoneSelectorWidget import MicrophoneSelectorWidget
from docupilot.ui.widgets.RecordButtonWidget import RecordButtonWidget
from docupilot.ui.widgets.ScreenSelectorWidget import ScreenSelectorWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        """
        Initializes the main window.
        """

        super().__init__()

        self.setWindowTitle("DocuPilot")
        self.resize(1000, 750)

        self.selected_screen = None
        self.selected_microphone = None

        self.record_button_widget: RecordButtonWidget | None = None

        self.recorder_service = RecorderService(parent=self)
        self.recorder_service.recording_error.connect(self.on_recording_error)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """
        Set up the user interface.
        """

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        box_layout = QVBoxLayout(central_widget)

        self.screen_selector = ScreenSelectorWidget()
        self.screen_selector.screen_selected.connect(self.on_screen_selected)

        self.microphone_selector = MicrophoneSelectorWidget()
        self.microphone_selector.microphone_selected.connect(
            self.on_microphone_selected
        )

        self.record_button_widget = RecordButtonWidget()
        self.record_button_widget.record_started.connect(self.on_record_started)
        self.record_button_widget.record_stopped.connect(self.on_record_stopped)

        box_layout.addWidget(self.screen_selector)
        box_layout.addWidget(self.microphone_selector)
        box_layout.addStretch()
        box_layout.addWidget(self.record_button_widget)

    def on_screen_selected(self, screen) -> None:
        """
        Handles the screen-selected event.

        :param screen: The screen that has been selected.
        """

        self.selected_screen = screen

    def on_microphone_selected(self, microphone) -> None:
        """
        Handles the microphone-selected event.

        :param microphone: The microphone that has been selected.
        """

        self.selected_microphone = microphone

    def on_record_started(self) -> None:
        """
        Triggered when the recording has started.
        """

        if self.record_button_widget is None:
            return

        if self.selected_screen is None:
            QMessageBox.warning(
                self,
                "Kein Bildschirm ausgewählt",
                "Bitte wähle zuerst einen Bildschirm aus.",
            )
            self.record_button_widget.stop_recording()
            return

        if self.selected_microphone is None:
            QMessageBox.warning(
                self,
                "Kein Mikrofon ausgewählt",
                "Bitte wähle zuerst ein Mikrofon aus.",
            )
            self.record_button_widget.stop_recording()
            return

        try:
            session = self.recorder_service.start_recording(
                screen=self.selected_screen,
                microphone=self.selected_microphone,
            )

            print(f"Aufzeichnung gestartet: {session.session_dir}")
            print(f"Bildschirm-Datei: {session.screen_path}")
            print(f"Audio-Datei: {session.audio_path}")
            print(f"Events-Datei: {session.events_path}")

        except Exception as exc:
            self.record_button_widget.stop_recording()

            QMessageBox.critical(
                self,
                "Aufnahme konnte nicht gestartet werden",
                str(exc),
            )

    def on_record_stopped(self) -> None:
        """
        Triggered when the recording has stopped.
        """

        try:
            session = self.recorder_service.stop_recording()

            print(f"Aufzeichnung gestoppt: {session.session_dir}")
            print(f"Bildschirm-Datei: {session.screen_path}")
            print(f"Audio-Datei: {session.audio_path}")
            print(f"Events-Datei: {session.events_path}")

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Aufnahme konnte nicht gestoppt werden",
                str(exc),
            )

    def on_recording_error(self, message: str) -> None:
        """
        Handles errors emitted by the RecorderService.

        :param message: Error message.
        """

        QMessageBox.warning(
            self,
            "Aufnahmefehler",
            message,
        )