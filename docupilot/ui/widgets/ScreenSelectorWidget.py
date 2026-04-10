from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication, QWidget

from docupilot.ui.widgets.DeviceSelectorWidget import DeviceSelectorWidget
from docupilot.ui.widgets.ScreenTileWidget import ScreenTileWidget


class ScreenSelectorWidget(DeviceSelectorWidget):
    screen_selected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the screen selector widget.
        :param parent: The reference to the parent widget.
        """

        super().__init__(
            title="Bildschirm auswählen",
            load_devices=lambda: QApplication.screens(),
            tile_factory=lambda screen: ScreenTileWidget(screen),
            selected_attr_name="screen",
            default_device_resolver=None,
            columns=3,
            parent=parent,
        )

        self.device_selected.connect(self.screen_selected.emit)

    def reload_screens(self) -> None:
        """
        Reload the available screens.
        """

        self.reload_devices()

    def get_selected_screen(self):
        """
        Get the selected screen and return it.
        """

        return self.get_selected_device()