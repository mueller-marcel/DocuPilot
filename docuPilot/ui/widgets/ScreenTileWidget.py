from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QWidget, QPushButton, QSizePolicy, QVBoxLayout, QFrame, QLabel

class ScreenTileWidget(QPushButton):
    def __init__(self, screen, parent: QWidget | None = None) -> None:
        """
        Initialize the screen tile widget.
        :param screen: The screen object to display.
        :param parent: The parent widget.
        """

        super().__init__(parent)
        self.screen = screen

        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumSize(260, 190)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(6, 6, 6, 6)

        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        screenshot = screen.grabWindow(0)
        thumbnail = screenshot.scaled(
            QSize(220, 120),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        preview_label.setPixmap(thumbnail)

        preview_layout.addWidget(preview_label)

        geometry = screen.geometry()
        name_label = QLabel(screen.name())
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setObjectName("screenNameLabel")

        details_label = QLabel(
            f"{geometry.width()} x {geometry.height()}  |  "
            f"Position: {geometry.x()}, {geometry.y()}"
        )
        details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details_label.setWordWrap(True)
        details_label.setObjectName("screenDetailsLabel")

        root_layout.addWidget(preview_frame)
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

            QFrame#previewFrame {
                background-color: #1e1e1e;
                border: 1px solid #444444;
                border-radius: 8px;
            }

            QLabel#screenNameLabel {
                font-size: 14px;
                font-weight: 600;
                color: white;
            }

            QLabel#screenDetailsLabel {
                font-size: 12px;
                color: #d0d0d0;
            }
            """
        )
