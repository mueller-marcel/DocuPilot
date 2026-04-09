from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout

from docuPilot.ui.widgets.ScreenSelectorWidget import ScreenSelectorWidget
from docuPilot.ui.widgets.MicrophoneSelectorWidget import MicrophoneSelectorWidget


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
        self.record_button = None

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
        self.microphone_selector.microphone_selected.connect(self.on_microphone_selected)

        self.record_button = QPushButton("● Aufnahme starten")
        self.record_button.setCheckable(True)
        self.record_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_button.setFixedHeight(40)
        self.record_button.toggled.connect(self.toggle_recording)
        self.record_button.setStyleSheet("""
        QPushButton {
            background-color: #d32f2f;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            padding: 8px 16px;
        }

        QPushButton:hover {
            background-color: #e53935;
        }

        QPushButton:checked {
            background-color: #2e7d32;
        }

        QPushButton:checked:hover {
            background-color: #388e3c;
        }
        """)

        box_layout.addWidget(self.screen_selector)
        box_layout.addWidget(self.microphone_selector)
        box_layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.record_button)
        button_layout.addStretch()

        box_layout.addLayout(button_layout)

    def on_screen_selected(self, screen) -> None:
        """
        Handles the screen-selected event
        :param screen: The screen that has been selected
        """

        self.selected_screen = screen

    def on_microphone_selected(self, microphone) -> None:
        """
        Handles the microphone-selected event
        :param microphone: The microphone that has been selected
        """

        self.selected_microphone = microphone

    def toggle_recording(self, checked):
        """
        Triggered when the recording button is clicked
        :param checked: True if the record is started, otherwise false
        """

        if checked:
            self.record_button.setText("■ Aufnahme stoppen")
        else:
            self.record_button.setText("● Aufnahme starten")