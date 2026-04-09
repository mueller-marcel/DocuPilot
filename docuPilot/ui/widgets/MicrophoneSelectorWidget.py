from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import QAbstractButton, QButtonGroup, QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from docuPilot.ui.widgets.MicrophoneTileWidget import MicrophoneTileWidget


class MicrophoneSelectorWidget(QWidget):
    microphone_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the microphone selector widget.
        :param parent: The reference to the parent widget.
        """

        super().__init__(parent)

        self._selected_microphone = None
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._build_ui()
        self.reload_microphones()

    def _build_ui(self) -> None:
        """
        Build the user interface for the microphone selector widget.
        """

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)

        title = QLabel("Mikrofon auswählen")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        root_layout.addWidget(title)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._content_widget = QWidget()
        self._grid = QGridLayout(self._content_widget)
        self._grid.setContentsMargins(4, 4, 4, 4)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(12)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        self._scroll_area.setWidget(self._content_widget)
        root_layout.addWidget(self._scroll_area)

    def reload_microphones(self) -> None:
        """
        Loads the available microphones of the computer.
        :return: The available microphones.
        """

        self._clear_grid()

        microphones = QMediaDevices.audioInputs()
        if not microphones:
            self._selected_microphone = None
            return

        columns = 3
        default_microphone = QMediaDevices.defaultAudioInput()

        default_button = None

        for index, microphone in enumerate(microphones):
            tile = MicrophoneTileWidget(microphone)
            tile.clicked.connect(self._on_tile_clicked)

            row = index // columns
            col = index % columns

            self._grid.addWidget(tile, row, col)
            self._button_group.addButton(tile, index)

            if microphone.id() == default_microphone.id():
                default_button = tile

        buttons = self._button_group.buttons()
        if buttons:
            first_button = buttons[0]
            selected_button = default_button or first_button

            selected_button.setChecked(True)
            self._selected_microphone = selected_button.microphone
            self.microphone_selected.emit(self._selected_microphone)

    def _on_tile_clicked(self) -> None:
        """
        Handles the click event on a microphone tile.
        :return: None
        """

        button = self.sender()
        if button is None or not hasattr(button, "microphone"):
            return

        self._selected_microphone = button.microphone
        self.microphone_selected.emit(self._selected_microphone)

    def _clear_grid(self) -> None:
        """
        Clears the grid layout of the microphone selector widget.
        :return: None
        """

        while self._grid.count():
            if (item := self._grid.takeAt(0)) is None:
                continue

            if (button_widget := item.widget()) and isinstance(button_widget, QAbstractButton):
                self._button_group.removeButton(button_widget)
                button_widget.deleteLater()

    def get_selected_microphone(self):
        """
        Get the selected microphone and return it.
        :return: The selected microphone.
        """

        return self._selected_microphone