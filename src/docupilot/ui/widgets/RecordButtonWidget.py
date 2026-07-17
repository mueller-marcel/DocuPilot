from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSizePolicy, QWidget


class RecordButtonWidget(QWidget):
    """
    Start/stop recording button.

    User clicks emit record_started or record_stopped.
    Programmatic state changes can update the UI without emitting signals.
    """

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
        self._button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self._button.toggled.connect(self._on_toggled)

        self._apply_start_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self._button)
        layout.addStretch()

    def is_recording(self) -> bool:
        """
        Indicates whether the button is currently recording.
        :return: True if recording, False otherwise.
        """

        return self._button.isChecked()

    def start_recording(self) -> None:
        """
        Set the button to recording state without emitting record_started.
        """
        self.set_recording_state(True, emit_signal=False)

    def stop_recording(self) -> None:
        """
        Set the button to stopped state without emitting record_stopped.
        """

        self.set_recording_state(False, emit_signal=False)

    def set_recording_state(self, recording: bool, emit_signal: bool = False) -> None:
        """
        Set the visual recording state.

        :param recording: True for active recording UI state.
        :param emit_signal: Whether the state change should emit Qt signals.
        """

        if self._button.isChecked() == recording:
            return

        if emit_signal:
            self._button.setChecked(recording)
            return

        blocker = QSignalBlocker(self._button)
        self._button.setChecked(recording)
        del blocker

        if recording:
            self._apply_stop_style()
        else:
            self._apply_start_style()

    def _on_toggled(self, checked: bool) -> None:
        """
        Handle real user-triggered toggle changes.
        """

        if checked:
            self._apply_stop_style()
            self.record_started.emit()
        else:
            self._apply_start_style()
            self.record_stopped.emit()

    def _apply_start_style(self) -> None:
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