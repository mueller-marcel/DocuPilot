from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QApplication, QButtonGroup, QFrame, QGridLayout, QLabel, QScrollArea, QVBoxLayout, \
    QWidget, QAbstractButton

from docuPilot.ui.widgets.ScreenTileWidget import ScreenTileWidget


class ScreenSelectorWidget(QWidget):
    screen_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the screen selector widget.
        :param parent: The reference to the parent widget.
        """

        super().__init__(parent)

        self._selected_screen = None
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._build_ui()
        self.reload_screens()

    def _build_ui(self) -> None:
        """
        Build the user interface for the screen selector widget.
        """

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)

        title = QLabel("Bildschirm auswählen")
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

    def reload_screens(self) -> None:
        """
        Loads the available screens of the computer
        :return: The available screens.
        """

        self._clear_grid()

        screens = QApplication.screens()
        if not screens:
            self._selected_screen = None
            return

        columns = 3

        for index, screen in enumerate(screens):
            tile = ScreenTileWidget(screen)
            tile.clicked.connect(self._on_tile_clicked)

            row = index // columns
            col = index % columns

            self._grid.addWidget(tile, row, col)
            self._button_group.addButton(tile, index)

        buttons = self._button_group.buttons()
        if buttons:
            first_button = buttons[0]
            first_button.setChecked(True)
            self._selected_screen = first_button.screen
            self.screen_selected.emit(self._selected_screen)

    def _on_tile_clicked(self) -> None:
        """
        Handles the click event on a screen tile.
        :return: None
        """

        button = self.sender()
        if button is None or not hasattr(button, "screen"):
            return

        self._selected_screen = button.screen
        self.screen_selected.emit(self._selected_screen)

    def _clear_grid(self) -> None:
        """
        Clears the grid layout of the screen selector widget.
        :return: None
        """

        while self._grid.count():
            if (item := self._grid.takeAt(0)) is None:
                continue

            if (button_widget := item.widget()) and isinstance(button_widget, QAbstractButton):
                self._button_group.removeButton(button_widget)
                button_widget.deleteLater()

    def get_selected_screen(self):
        """
        Get the selected screen and return it.
        :return: The selected screen.
        """

        return self._selected_screen