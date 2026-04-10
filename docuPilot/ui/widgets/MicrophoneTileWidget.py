from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from docuPilot.ui.widgets.DeviceTileWidget import DeviceTileWidget


class MicrophoneTileWidget(DeviceTileWidget):
    def __init__(self, microphone, parent: QWidget | None = None) -> None:
        """
        Initialize the microphone tile widget.
        :param microphone: The microphone object to display.
        :param parent: The parent widget.
        """

        icon_label = QLabel("🎤")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 64px; color: white;")

        super().__init__(
            device=microphone,
            device_attr_name="microphone",
            preview_widget=icon_label,
            name_text=microphone.description(),
            details_text="Audio-Eingabegerät",
            parent=parent,
        )