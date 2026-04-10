from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QWidget

from docuPilot.ui.widgets.DeviceTileWidget import DeviceTileWidget


class ScreenTileWidget(DeviceTileWidget):
    def __init__(self, screen, parent: QWidget | None = None) -> None:
        """
        Initialize the screen tile widget.
        :param screen: The screen object to display.
        :param parent: The parent widget.
        """

        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        screenshot = screen.grabWindow(0)
        thumbnail = screenshot.scaled(
            QSize(220, 120),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        preview_label.setPixmap(thumbnail)

        geometry = screen.geometry()

        super().__init__(
            device=screen,
            device_attr_name="screen",
            preview_widget=preview_label,
            name_text=screen.name(),
            details_text=(
                f"{geometry.width()} x {geometry.height()}  |  "
                f"Position: {geometry.x()}, {geometry.y()}"
            ),
            parent=parent,
        )