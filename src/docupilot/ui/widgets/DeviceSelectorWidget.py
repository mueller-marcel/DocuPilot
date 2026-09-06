from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton, QButtonGroup, QFrame, QGridLayout, QLabel, QScrollArea,
    QVBoxLayout, QWidget,
)


class DeviceSelectorWidget(QWidget):
    """
    Generic widget for selecting a device from a grid of tiles.

    The device kind is not known here: a factory builds the tile for each
    device and a loader lists them, so the same grid serves screens and
    microphones.

    :ivar device_selected: emitted with the selected device whenever the
        selection changes, including the initial default.
    """

    device_selected = Signal(object)

    def __init__(
        self,
        title: str,
        load_devices: Callable[[], list[Any]],
        tile_factory: Callable[[Any], QAbstractButton],
        selected_attr_name: str,
        default_device_resolver: Callable[[list[Any]], Any | None] | None = None,
        columns: int = 3,
        parent: QWidget | None = None,
    ) -> None:
        """
        :param title: Title shown above the selector.
        :param load_devices: Function that returns all available devices.
        :param tile_factory: Factory that creates a tile widget for a given device.
        :param selected_attr_name: Name of the tile attribute that contains the
            represented device, for tiles that do not expose `.device`.
        :param default_device_resolver: Optional function that returns the preferred
            default device.
        :param columns: Number of columns in the grid.
        :param parent: Parent widget.
        """
        super().__init__(parent)

        self._title = title
        self._load_devices = load_devices
        self._tile_factory = tile_factory
        self._selected_attr_name = selected_attr_name
        self._default_device_resolver = default_device_resolver
        self._columns = columns

        self._selected_device = None
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)

        self._build_ui()
        self.reload_devices()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(10)

        title_label = QLabel(self._title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        root_layout.addWidget(title_label)

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

    def _device_of(self, tile: Any) -> Any:
        """The device a tile represents; None when the tile carries none."""
        if hasattr(tile, "device"):
            return tile.device
        return getattr(tile, self._selected_attr_name, None)

    def reload_devices(self) -> None:
        """Load available devices and rebuild the grid."""
        self._clear_grid()

        devices = self._load_devices()
        if not devices:
            self._selected_device = None
            return

        default_device = None
        if self._default_device_resolver is not None:
            default_device = self._default_device_resolver(devices)

        default_button = None

        for index, device in enumerate(devices):
            tile = self._tile_factory(device)
            tile.clicked.connect(self._on_tile_clicked)

            self._grid.addWidget(tile, index // self._columns, index % self._columns)
            self._button_group.addButton(tile, index)

            if default_device is not None and self._devices_equal(device, default_device):
                default_button = tile

        buttons = self._button_group.buttons()
        if buttons:
            selected_button = default_button or buttons[0]
            selected_button.setChecked(True)

            self._selected_device = self._device_of(selected_button)
            self.device_selected.emit(self._selected_device)

    def _on_tile_clicked(self) -> None:
        button = self.sender()
        if button is None or not (
            hasattr(button, "device") or hasattr(button, self._selected_attr_name)
        ):
            return

        self._selected_device = self._device_of(button)
        self.device_selected.emit(self._selected_device)

    def _clear_grid(self) -> None:
        while self._grid.count():
            if (item := self._grid.takeAt(0)) is None:
                continue

            if (button_widget := item.widget()) and isinstance(button_widget, QAbstractButton):
                self._button_group.removeButton(button_widget)
                button_widget.deleteLater()

    @staticmethod
    def _devices_equal(left: Any, right: Any) -> bool:
        """Compare by id() when both sides offer one — Qt device objects are
        value types whose identity is their id, not their Python object."""
        if left is None or right is None:
            return False

        left_has_id = hasattr(left, "id") and callable(left.id)
        right_has_id = hasattr(right, "id") and callable(right.id)

        if left_has_id and right_has_id:
            return left.id() == right.id()

        return left == right

    def get_selected_device(self):
        """The currently selected device, or None when there is none."""
        return self._selected_device
