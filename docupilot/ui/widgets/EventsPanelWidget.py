from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


def format_ms(ms: float) -> str:
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}.{int(ms) % 1000:03d}"


_PROMINENT_TYPES = {
    "av_started", "av_stopped",
    "input_started", "input_stopped",
    "recording_started", "recording_stopping", "recording_stopped",
    "mouse_click", "key_press", "key_release", "mouse_scroll",
}

_DOT_COLOR: dict[str, str] = {
    "av_started":         "#534AB7",
    "av_stopped":         "#534AB7",
    "input_started":      "#1D9E75",
    "input_stopped":      "#1D9E75",
    "recording_started":  "#e24b4a",
    "recording_stopping": "#e24b4a",
    "recording_stopped":  "#e24b4a",
    "mouse_click":        "#D85A30",
    "mouse_scroll":       "#D85A30",
    "mouse_move":         "#aaaaaa",
    "key_press":          "#BA7517",
    "key_release":        "#BA7517",
}

_DEFAULT_DOT_COLOR = "#aaaaaa"


class _EventRow(QWidget):
    jumped = Signal(float)

    def __init__(self, ev: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.event_data = ev
        self._active = False
        self._prominent = ev.get("type") in _PROMINENT_TYPES

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Springe zu {format_ms(ev.get('t_ms', 0.0))}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        color = _DOT_COLOR.get(ev.get("type", ""), _DEFAULT_DOT_COLOR)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{color}; border-radius:4px;")
        layout.addWidget(dot)

        type_label = QLabel(ev.get("type", "?"))
        if self._prominent:
            type_label.setStyleSheet("font-size:12px; font-weight:600; color:#222;")
        else:
            type_label.setStyleSheet("font-size:11px; color:#666;")
        layout.addWidget(type_label)

        if self._prominent:
            t = ev.get("type", "")
            detail = ""
            if t == "mouse_click":
                action = "↓" if ev.get("pressed") else "↑"
                detail = f"{ev.get('button', '').replace('Button.', '')}{action}"
            elif t in ("key_press", "key_release"):
                detail = str(ev.get("key", ""))
            elif t == "mouse_scroll":
                detail = f"({ev.get('x', '?')},{ev.get('y', '?')})"
            if detail:
                detail_label = QLabel(detail)
                detail_label.setStyleSheet("font-size:11px; color:#888;")
                layout.addWidget(detail_label)

        layout.addStretch()

        time_label = QLabel(format_ms(ev.get("t_ms", 0.0)))
        time_label.setStyleSheet("font-size:10px; color:#bbb; font-family:monospace;")
        layout.addWidget(time_label)

        self._update_style(active=False)

    def mousePressEvent(self, mouse_event) -> None:
        self.jumped.emit(float(self.event_data.get("t_ms", 0.0)))

    def set_active(self, active: bool, past: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self._update_style(active=active, past=past)

    def _update_style(self, *, active: bool, past: bool = False) -> None:
        if active:
            self.setStyleSheet(
                "background:#dbeafe; border-radius:5px;border-left:3px solid #4da3ff;"
            )
        elif past and not self._prominent:
            self.setStyleSheet("background:transparent; border:none; opacity:0.5;")
        else:
            self.setStyleSheet("background:transparent; border:none;")
        self.setProperty("active", active)


class EventsPanelWidget(QWidget):
    """Scrollbare Event-Sidebar mit Live-Highlighting der aktuellen Abspielposition."""

    jumped = Signal(float)

    TOLERANCE_MS: float = 250.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._events: list[dict] = []
        self._event_rows: list[_EventRow] = []

        self.setStyleSheet("EventsPanelWidget{background:#fafafa;}")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel(f"  Events  ·  ±{int(self.TOLERANCE_MS)} ms  ·  klicken zum Springen")
        header.setFixedHeight(32)
        header.setStyleSheet(
            "background:#f0f0f0; color:#777; font-size:11px;"
            "border-bottom:1px solid #e0e0e0; padding-left:4px;"
        )
        layout.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container.setStyleSheet("background:#fafafa;")
        self._vbox = QVBoxLayout(self._container)
        self._vbox.setContentsMargins(4, 4, 4, 4)
        self._vbox.setSpacing(1)
        self._vbox.addStretch()

        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

    def set_events(self, events: list[dict]) -> None:
        self._events = events

        for row in self._event_rows:
            self._vbox.removeWidget(row)
            row.deleteLater()
        self._event_rows.clear()

        for ev in self._events:
            row = _EventRow(ev, self._container)
            row.jumped.connect(self.jumped)
            self._vbox.insertWidget(self._vbox.count() - 1, row)
            self._event_rows.append(row)

    def highlight(self, pos_ms: float) -> None:
        tol = self.TOLERANCE_MS
        first_active: _EventRow | None = None

        for row in self._event_rows:
            ev_ms = row.event_data.get("t_ms", 0.0)
            active = abs(ev_ms - pos_ms) <= tol
            past = ev_ms < pos_ms - tol
            row.set_active(active, past)

            if active and first_active is None:
                first_active = row

        if first_active is not None:
            self._scroll.ensureWidgetVisible(first_active)
