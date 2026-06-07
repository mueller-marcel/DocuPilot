from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from docupilot.recording.recorders import RecorderService
from docupilot.ui.widgets.MicrophoneSelectorWidget import MicrophoneSelectorWidget
from docupilot.ui.widgets.RecordButtonWidget import RecordButtonWidget
from docupilot.ui.widgets.ScreenSelectorWidget import ScreenSelectorWidget


class MainWindow(QMainWindow):
    """
    Main application window.

    This class wires widgets to the RecorderService.
    Recording lifecycle logic remains inside RecorderService.
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("DocuPilot")
        self.resize(1000, 750)

        self.selected_screen: Any | None = None
        self.selected_microphone: Any | None = None

        self.recorder_service = RecorderService(parent=self)
        self.recorder_service.recording_error.connect(self.on_recording_error)

        self.screen_selector: ScreenSelectorWidget | None = None
        self.microphone_selector: MicrophoneSelectorWidget | None = None
        self.record_button_widget: RecordButtonWidget | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        box_layout = QVBoxLayout(central_widget)

        self.screen_selector = ScreenSelectorWidget()
        self.screen_selector.screen_selected.connect(self.on_screen_selected)
        self.on_screen_selected(self.screen_selector.get_selected_screen())

        self.microphone_selector = MicrophoneSelectorWidget()
        self.microphone_selector.microphone_selected.connect(
            self.on_microphone_selected
        )
        self.on_microphone_selected(
            self.microphone_selector.get_selected_microphone()
        )

        self.record_button_widget = RecordButtonWidget()
        self.record_button_widget.record_started.connect(self.on_record_started)
        self.record_button_widget.record_stopped.connect(self.on_record_stopped)

        box_layout.addWidget(self.screen_selector)
        box_layout.addWidget(self.microphone_selector)
        box_layout.addStretch()
        box_layout.addWidget(self.record_button_widget)

    def on_screen_selected(self, screen: Any) -> None:
        self.selected_screen = screen

    def on_microphone_selected(self, microphone: Any) -> None:
        self.selected_microphone = microphone

    def on_record_started(self) -> None:
        if self.record_button_widget is None:
            return

        if not self._has_valid_selection():
            self.record_button_widget.stop_recording()
            return

        try:
            self.recorder_service.start_recording(
                screen=self.selected_screen,
                microphone=self.selected_microphone,
            )
        except Exception as exc:
            self.record_button_widget.stop_recording()
            self._show_error(
                title="Recording Could Not Be Started",
                message=str(exc),
            )

    def on_record_stopped(self) -> None:
        if not self.recorder_service.is_recording():
            return

        try:
            self.recorder_service.stop_recording()
        except Exception as exc:
            self._show_error(
                title="Recording Could Not Be Stopped",
                message=str(exc),
            )

    def on_recording_error(self, message: str) -> None:
        if self.record_button_widget is not None:
            self.record_button_widget.stop_recording()

        self._show_warning(
            title="Recording Error",
            message=message,
        )

    def _has_valid_selection(self) -> bool:
        if self.selected_screen is None:
            self._show_warning(
                title="No Screen Selected",
                message="Please select a screen before starting the recording.",
            )
            return False

        if self.selected_microphone is None:
            self._show_warning(
                title="No Microphone Selected",
                message="Please select a microphone before starting the recording.",
            )
            return False

        return True

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(
            self,
            title,
            message,
        )

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(
            self,
            title,
            message,
        )