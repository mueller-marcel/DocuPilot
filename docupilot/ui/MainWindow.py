from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from docupilot.recording.recorder_service import RecorderService
from docupilot.ui.widgets.MicrophoneSelectorWidget import MicrophoneSelectorWidget
from docupilot.ui.widgets.RecordButtonWidget import RecordButtonWidget
from docupilot.ui.widgets.ScreenSelectorWidget import ScreenSelectorWidget


class MainWindow(QMainWindow):
    """
    The main window of the application.
    """

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
        self.on_screen_selected(self.screen_selector.get_selected_screen())

        self.microphone_selector = MicrophoneSelectorWidget()
        self.microphone_selector.microphone_selected.connect(self.on_microphone_selected)
        self.on_microphone_selected(self.microphone_selector.get_selected_microphone())

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
        Triggered when the user requests to start recording.
        """
        if self.record_button_widget is None:
            return

        if self.selected_screen is None:
            QMessageBox.warning(
                self,
                "No Screen Selected",
                "Please select a screen before starting the recording.",
            )
            self.record_button_widget.stop_recording()
            return

        if self.selected_microphone is None:
            QMessageBox.warning(
                self,
                "No Microphone Selected",
                "Please select a microphone before starting the recording.",
            )
            self.record_button_widget.stop_recording()
            return

        try:
            self.recorder_service.start_recording(
                screen=self.selected_screen,
                microphone=self.selected_microphone,
            )
        except Exception as exc:
            self.record_button_widget.stop_recording()
            QMessageBox.critical(
                self,
                "Recording Could Not Be Started",
                str(exc),
            )

    def on_record_stopped(self) -> None:
        """
        Triggered when the user requests to stop recording.
        """
        if not self.recorder_service.is_recording():
            return

        try:
            self.recorder_service.stop_recording()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Recording Could Not Be Stopped",
                str(exc),
            )

    def on_recording_error(self, message: str) -> None:
        """
        Handles errors emitted by the RecorderService.

        Resets the record button to the stopped state so the UI
        stays consistent when an error occurs mid-recording.

        :param message: Human-readable error message from the recorder.
        """
        if self.record_button_widget is not None:
            self.record_button_widget.stop_recording()

        QMessageBox.warning(
            self,
            "Recording Error",
            message,
        )