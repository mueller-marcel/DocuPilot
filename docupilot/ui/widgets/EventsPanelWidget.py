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

from docupilot.ui.event_formatting import DEFAULT_DOT_COLOR, DOT_COLOR, PROMINENT_TYPES, format_ms


class _EventRow(QWidget):
    """
    A single row in EventsPanelWidget, representing one event.
    """

    jumped = Signal(float)

    def __init__(self, ev: dict, parent: QWidget | None = None) -> None:
        """
        Initialize the event row widget.

        :param ev: The event data.
        :param parent: The parent widget.
        :return: None
        """
        super().__init__(parent)

        self.event_data = ev
        self._active = False
        self._prominent = ev.get("type") in PROMINENT_TYPES

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Springe zu {format_ms(ev.get('t_ms', 0.0))}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        color = DOT_COLOR.get(ev.get("type", ""), DEFAULT_DOT_COLOR)
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
            elif t in ("mouse_scroll",):
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
        """
        Emit jumped with this row's timestamp.

        :param mouse_event: The mouse press event.
        :return: None
        """
        self.jumped.emit(float(self.event_data.get("t_ms", 0.0)))

    def set_active(self, active: bool, past: bool) -> None:
        """
        Update the active/past state and restyle if changed.

        :param active: Whether the row is currently active.
        :param past: Whether the row's event lies in the past.
        :return: None
        """
        if self._active == active:
            return

        self._active = active
        self._update_style(active=active, past=past)

    def _update_style(self, *, active: bool, past: bool = False) -> None:
        """
        Apply the stylesheet matching the row's current state.

        :param active: Whether the row is currently active.
        :param past: Whether the row's event lies in the past.
        :return: None
        """
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
    """
    Self-contained event sidebar with header, scrollable event rows, and
    live highlighting around the current playback position.
    """

    jumped = Signal(float)

    TOLERANCE_MS: float = 250.0

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the events panel widget.

        :param parent: The parent widget.
        :return: None
        """
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
        """
        Replace the displayed events and rebuild the row list.

        :param events: The full list of raw event dicts for the session.
        :return: None
        """
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
        """
        Mark rows within TOLERANCE_MS as active, fade older rows, and
        scroll the first active row into view.

        :param pos_ms: The current playback position in milliseconds.
        :return: None
        """
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