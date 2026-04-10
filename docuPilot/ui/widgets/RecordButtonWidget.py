from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QPushButton, QSizePolicy, QWidget, QHBoxLayout


class RecordButtonWidget(QWidget):
    record_started = Signal()
    record_stopped = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the record button widget.
        :param parent: The parent widget.
        """

        super().__init__(parent)

        self._button = QPushButton("Aufnahme starten")
        self._button.setCheckable(True)
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._button.setFixedSize(240, 42)
        self._button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._button.setIconSize(QSize(12, 12))

        self._button.toggled.connect(self._on_toggled)

        self._apply_start_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self._button)
        layout.addStretch()

    def _apply_start_style(self) -> None:
        """
        Applies the visual style for the 'start recording' state.
        :return: None
        """

        self._button.setText("Aufnahme starten")

        self._button.setStyleSheet(
            """
            QPushButton {
                background-color: #d32f2f;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #e53935;
            }

            QPushButton:pressed {
                background-color: #b71c1c;
            }
            """
        )

    def _apply_stop_style(self) -> None:
        """
        Applies the visual style for the 'stop recording' state.
        :return: None
        """

        self._button.setText("Aufnahme stoppen")

        self._button.setStyleSheet(
            """
            QPushButton {
                background-color: #2e7d32;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #388e3c;
            }

            QPushButton:pressed {
                background-color: #1b5e20;
            }
            """
        )

    def _on_toggled(self, checked: bool) -> None:
        """
        Handles the toggle state of the button.
        :param checked: True if the recording is active, otherwise False.
        :return: None
        """

        if checked:
            self._apply_stop_style()
            self.record_started.emit()
        else:
            self._apply_start_style()
            self.record_stopped.emit()

    def is_recording(self) -> bool:
        """
        Returns whether the recording is currently active.
        :return: True if the recording is active, otherwise False.
        """

        return self._button.isChecked()

    def start_recording(self) -> None:
        """
        Activates the recording state programmatically.
        :return: None
        """

        if not self._button.isChecked():
            self._button.setChecked(True)

    def stop_recording(self) -> None:
        """
        Deactivates the recording state programmatically.
        :return: None
        """

        if self._button.isChecked():
            self._button.setChecked(False)