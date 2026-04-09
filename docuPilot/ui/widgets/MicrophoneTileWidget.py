from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QPushButton, QSizePolicy, QVBoxLayout, QFrame, QLabel


class MicrophoneTileWidget(QPushButton):
    def __init__(self, microphone, parent: QWidget | None = None) -> None:
        """
        Initialize the microphone tile widget.
        :param microphone: The microphone object to display.
        :param parent: The parent widget.
        """

        super().__init__(parent)
        self.microphone = microphone

        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(260, 190)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        icon_frame = QFrame()
        icon_frame.setObjectName("iconFrame")
        icon_layout = QVBoxLayout(icon_frame)
        icon_layout.setContentsMargins(6, 6, 6, 6)

        icon_label = QLabel("🎤")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setObjectName("microphoneIconLabel")

        icon_layout.addWidget(icon_label)

        name_label = QLabel(microphone.description())
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setObjectName("microphoneNameLabel")

        is_default = microphone == microphone.__class__()  # placeholder, not used
        details_text = "Audio-Eingabegerät"

        details_label = QLabel(details_text)
        details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details_label.setWordWrap(True)
        details_label.setObjectName("microphoneDetailsLabel")

        root_layout.addWidget(icon_frame)
        root_layout.addWidget(name_label)
        root_layout.addWidget(details_label)

        self.setStyleSheet(
            """
            QPushButton {
                background-color: #2b2b2b;
                border: 2px solid #555555;
                border-radius: 12px;
                color: white;
                text-align: left;
            }

            QPushButton:hover {
                border: 2px solid #888888;
                background-color: #333333;
            }

            QPushButton:checked {
                border: 3px solid #4da3ff;
                background-color: #3a3f46;
            }

            QFrame#iconFrame {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 8px;
            }

            QLabel#microphoneIconLabel {
                font-size: 64px;
                color: white;
            }

            QLabel#microphoneNameLabel {
                font-size: 14px;
                font-weight: 600;
                color: white;
            }

            QLabel#microphoneDetailsLabel {
                font-size: 12px;
                color: #d0d0d0;
            }
            """
        )