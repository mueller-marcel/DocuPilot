from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from docupilot.recording.session import RecordingSession
from docupilot.ui.formatting import format_ms


class BoundaryDialog(QDialog):
    """Dialog zum Anzeigen und Löschen gesetzter Ground-Truth-Grenzen."""

    def __init__(self, boundaries: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gesetzte Grenzen")
        self.setMinimumSize(420, 320)
        self.setModal(True)

        self._boundaries = list(boundaries)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        self._header = QLabel()
        self._header.setStyleSheet("font-size:13px; font-weight:600; color:#222;")
        layout.addWidget(self._header)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget{border:1px solid #ddd; border-radius:6px; font-size:12px;}"
            "QListWidget::item{padding:6px 10px;}"
            "QListWidget::item:selected{background:#dbeafe; color:#222;}"
        )
        layout.addWidget(self._list)

        delete_btn = QPushButton("Ausgewählte löschen")
        delete_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#c0392b;border:1px solid #e74c3c;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#fdf0ef;}"
        )
        delete_btn.clicked.connect(self._delete_selected)

        close_btn = QPushButton("Schließen")
        close_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#f0f0f0;}"
        )
        close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._refresh_list()

    def _refresh_list(self) -> None:
        self._header.setText(f"{len(self._boundaries)} Grenze(n) gesetzt")
        self._list.clear()
        for i, b in enumerate(self._boundaries):
            t_ms = b.get("t_ms", 0.0)
            created = b.get("created_at_utc", "")[:19].replace("T", "  ")
            kind = "Beginn" if RecordingSession.boundary_kind(b) == "start" else "Ende  "
            item = QListWidgetItem(f"#{i + 1}   {kind}   {format_ms(t_ms)}   —   {created} UTC")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)

    def _delete_selected(self) -> None:
        selected = self._list.selectedItems()
        if not selected:
            return
        for idx in sorted({item.data(Qt.ItemDataRole.UserRole) for item in selected}, reverse=True):
            self._boundaries.pop(idx)
        self._refresh_list()

    def get_boundaries(self) -> list[dict]:
        return self._boundaries
