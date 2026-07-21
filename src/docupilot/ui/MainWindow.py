from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from docupilot.recording.recorders import RecorderService
from docupilot.recording.session import RecordingSession
from docupilot.ui.AnnotationWindow import AnnotationWindow
from docupilot.ui.ExperimentWindow import ExperimentWindow
from docupilot.ui.widgets.MicrophoneSelectorWidget import MicrophoneSelectorWidget
from docupilot.ui.widgets.RecordButtonWidget import RecordButtonWidget
from docupilot.ui.widgets.ScreenSelectorWidget import ScreenSelectorWidget

_PAGE_RECORDER = 0
_PAGE_ANNOTATION = 1
_PAGE_WAITING = 2


class MainWindow(QMainWindow):
    """
    Main application window.

    Wires the selector widgets to the RecorderService and uses a QStackedWidget
    to switch between the recorder page and the annotation page. After a
    recording is stopped, the app navigates to the annotation page for the
    just-finished session. Existing sessions on disk can also be reopened via
    "Datei > Öffnen".
    """

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("DocuPilot")
        self.resize(1200, 800)

        self.selected_screen: Any | None = None
        self.selected_microphone: Any | None = None

        self.recorder_service = RecorderService(parent=self)
        self.recorder_service.recording_finalized.connect(self._on_recording_finalized)

        self.screen_selector: ScreenSelectorWidget | None = None
        self.microphone_selector: MicrophoneSelectorWidget | None = None
        self.record_button_widget: RecordButtonWidget | None = None
        self.annotation_window: AnnotationWindow | None = None
        self.experiment_window: ExperimentWindow | None = None

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._setup_recorder_page()
        self._setup_annotation_page()
        self._setup_waiting_page()
        self._setup_menu_bar()

    def _setup_recorder_page(self) -> None:
        """
        Set up the recorder page.
        """

        page = QWidget()
        box_layout = QVBoxLayout(page)

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

        self._stack.insertWidget(_PAGE_RECORDER, page)

    def _setup_annotation_page(self) -> None:
        """
        Set up the annotation page.
        """

        self.annotation_window = AnnotationWindow()
        self.annotation_window.back_requested.connect(self._show_recorder_page)
        self._stack.insertWidget(_PAGE_ANNOTATION, self.annotation_window)


    def _setup_waiting_page(self) -> None:
        """
        Set up the waiting page.
        """

        page = QWidget()
        page.setStyleSheet("background:#111;")
        layout = QVBoxLayout(page)
        label = QLabel("Aufnahme wird finalisiert\nBitte warten…")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color:#aaa; font-size:16px; line-height:1.8;")
        layout.addStretch()
        layout.addWidget(label)
        layout.addStretch()
        self._stack.insertWidget(_PAGE_WAITING, page)

    def _setup_menu_bar(self) -> None:
        """
        Erstellt die Menüleiste mit "Datei > Öffnen", um eine bestehende
        Session (recording.mp4 + events.json, optional ground_truth.json)
        aus einem Verzeichnis zu laden.

        :return: None
        """

        file_menu = self.menuBar().addMenu("&Datei")

        open_action = QAction("&Öffnen…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._on_open_session)
        file_menu.addAction(open_action)

        analysis_menu = self.menuBar().addMenu("&Auswertung")

        experiment_action = QAction("&Informationsbeitrag der Modalitäten…", self)
        experiment_action.triggered.connect(self._on_open_experiment)
        analysis_menu.addAction(experiment_action)

    def _on_open_experiment(self) -> None:
        """
        Öffnet das Auswertungsfenster für einen ganzen Korpus. Nicht-modal, damit
        der stundenlange Lauf die restliche Anwendung nicht blockiert.

        :return: None
        """

        if self.experiment_window is None:
            self.experiment_window = ExperimentWindow(self)
        self.experiment_window.show()
        self.experiment_window.raise_()
        self.experiment_window.activateWindow()

    def _show_recorder_page(self) -> None:
        """
        Show the recorder page.
        """

        if self.record_button_widget is not None:
            self.record_button_widget.stop_recording()
        self._stack.setCurrentIndex(_PAGE_RECORDER)

    def _show_annotation_page(self, session: RecordingSession) -> None:
        """
        Show the annotation page.
        :param session: The recorder session to show.
        """

        if self.annotation_window is not None:
            self.annotation_window.load_session(session)
        self._stack.setCurrentIndex(_PAGE_ANNOTATION)

    def on_screen_selected(self, screen: Any) -> None:
        """
        Handles the selection of a screen.
        :param screen: The screen that was selected.
        """

        self.selected_screen = screen

    def on_microphone_selected(self, microphone: Any) -> None:
        """
        Handles the selection of a microphone.
        :param microphone: The microphone that was selected.
        """

        self.selected_microphone = microphone

    def on_record_started(self) -> None:
        """
        Handles the start of a recording.
        :return: None
        """

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
        """
        Handles the stop of a recording.
        :return: None
        """

        if not self.recorder_service.is_recording():
            return

        try:
            self.recorder_service.stop_recording()
            self._stack.setCurrentIndex(_PAGE_WAITING)
        except Exception as exc:
            self._show_error(
                title="Recording Could Not Be Stopped",
                message=str(exc),
            )

    def _on_recording_finalized(self, session) -> None:
        """
        Handles the finalization of a recording. This ensures that the recording has stopped and the files are not corrupt.
        :param session: The recording session that was just finalized.
        """

        self._show_annotation_page(session)

    def _on_open_session(self) -> None:
        """
        Öffnet einen Verzeichnis-Dialog und lädt die dort enthaltene Session
        (recording.mp4, events.json, optional ground_truth.json) über
        RecordingSession.from_directory(). Bei Erfolg wird direkt die
        Annotation-Seite mit der geladenen Session angezeigt.

        :return: None
        """

        directory = QFileDialog.getExistingDirectory(
            self, "Session-Verzeichnis öffnen"
        )
        if not directory:
            return

        try:
            session = RecordingSession.from_directory(Path(directory))
        except FileNotFoundError as exc:
            self._show_error(
                title="Session konnte nicht geöffnet werden",
                message=str(exc),
            )
            return

        self._show_annotation_page(session)

    def _has_valid_selection(self) -> bool:
        """
        Checks whether a screen and a microphone have been selected.
        :return: True if both are selected, False otherwise.
        """

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
        """
        Shows a warning message.
        :param title: The title of the message.
        :param message: The message to show.
        """

        QMessageBox.warning(self, title, message)

    def _show_error(self, title: str, message: str) -> None:
        """
        Show an error message.
        :param title: The title of the message.
        :param message: The message to show.
        """

        QMessageBox.critical(self, title, message)