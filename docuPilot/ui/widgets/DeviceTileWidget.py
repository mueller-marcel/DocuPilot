from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget


class DeviceTileWidget(QPushButton):
    def __init__(
        self,
        device,
        device_attr_name: str,
        preview_widget: QWidget,
        name_text: str,
        details_text: str,
        parent: QWidget | None = None,
    ) -> None:
        """
        Initialize the generic device tile widget.

        :param device: The represented device object.
        :param device_attr_name: Name of the public attribute that stores the device (for example, "screen" or "microphone").
        :param preview_widget: Widget shown in the preview area.
        :param name_text: Main title text of the tile.
        :param details_text: Secondary detail text of the tile.
        :param parent: Parent widget.
        """

        super().__init__(parent)

        setattr(self, device_attr_name, device)

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
        preview_layout.addWidget(preview_widget)

        name_label = QLabel(name_text)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setWordWrap(True)
        name_label.setObjectName("deviceNameLabel")

        details_label = QLabel(details_text)
        details_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details_label.setWordWrap(True)
        details_label.setObjectName("deviceDetailsLabel")

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

            QLabel#deviceNameLabel {
                font-size: 14px;
                font-weight: 600;
                color: white;
            }

            QLabel#deviceDetailsLabel {
                font-size: 12px;
                color: #d0d0d0;
            }
            """
        )