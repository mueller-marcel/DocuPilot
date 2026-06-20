from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class FeatureTimelineWidget(QWidget):
    """
    Reusable timeline widget for one lane: draws a feature curve and/or
    event markers. Data-agnostic, fed via set_data()/set_events()/set_boundaries().
    """

    seek_requested = Signal(float)

    DEFAULT_EVENT_COLOR = "#38bdf8"

    _PAD_L = 52
    _PAD_R = 16
    _PAD_T = 16
    _PAD_B = 32

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        event_color: str = DEFAULT_EVENT_COLOR,
    ) -> None:
        """
        Initialize the timeline widget.

        :param parent: Parent widget.
        :param event_color: Hex color used for event marker dots.
        :return: None
        """
        super().__init__(parent)
        self._tracks: list[tuple[str, list[float], str, bool]] = []
        self._duration_ms: float = 0.0
        self._cursor_ms: float = 0.0
        self._boundaries: list[float] = []
        self._events: list[tuple[float, str]] = []
        self._event_color = event_color
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_data(self, tracks: list[tuple[str, list[float], str, bool]], duration_ms: float) -> None:
        """
        Load feature tracks to display.

        :param tracks: List of (label, normalised_values, hex_color, fill).
        :param duration_ms: Total recording duration in milliseconds.
        :return: None
        """
        self._tracks = tracks
        self._duration_ms = duration_ms
        self.update()

    def set_cursor(self, pos_ms: float) -> None:
        """
        Move the playback cursor line.

        :param pos_ms: Cursor position in milliseconds.
        :return: None
        """
        self._cursor_ms = pos_ms
        self.update()

    def set_boundaries(self, boundaries_ms: list[float]) -> None:
        """
        Set ground-truth boundary line positions.

        :param boundaries_ms: List of boundary timestamps in milliseconds.
        :return: None
        """
        self._boundaries = boundaries_ms
        self.update()

    def set_events(self, events: list[tuple[float, str]]) -> None:
        """
        Set event markers to draw as small dots.

        :param events: List of (t_ms, event_type).
        :return: None
        """
        self._events = events
        self.update()

    def mousePressEvent(self, event) -> None:
        """
        Seek the player to the clicked position on the timeline.

        :param event: The mouse press event.
        :return: None
        """
        if self._duration_ms <= 0:
            return
        x = event.position().x()
        frac = (x - self._PAD_L) / max(1, self.width() - self._PAD_L - self._PAD_R)
        frac = max(0.0, min(1.0, frac))
        self.seek_requested.emit(frac * self._duration_ms)

    def paintEvent(self, _event) -> None:
        """
        Render the grid, curves, boundaries, cursor, and event markers.

        :param _event: The paint event.
        :return: None
        """
        if not self._tracks and not self._events:
            self._paint_empty()
            return

        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint

        w = self.width()
        h = self.height()
        pl, pr, pt, pb = self._PAD_L, self._PAD_R, self._PAD_T, self._PAD_B
        plot_w = w - pl - pr
        plot_h = h - pt - pb

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        p.fillRect(0, 0, w, h, QColor("#1e1e2e"))
        p.fillRect(pl, pt, plot_w, plot_h, QColor("#12121a"))

        label_font = QFont("monospace", 8)
        p.setFont(label_font)
        label_color = QColor("#666688")
        grid_pen = QPen(QColor("#2a2a40"))
        grid_pen.setWidth(1)

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = pt + plot_h - int(frac * plot_h)
            p.setPen(grid_pen)
            p.drawLine(pl, y, pl + plot_w, y)
            p.setPen(label_color)
            p.drawText(2, y + 4, pl - 6, 12, Qt.AlignmentFlag.AlignRight, f"{frac:.2f}")

        n_ticks = 6
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            x = pl + int(frac * plot_w)
            t_s = frac * self._duration_ms / 1000
            p.setPen(label_color)
            label = f"{int(t_s // 60):02d}:{t_s % 60:04.1f}"
            p.drawText(x - 24, h - pb + 4, 48, pb - 4, Qt.AlignmentFlag.AlignHCenter, label)
            tick_pen = QPen(QColor("#333355"))
            tick_pen.setWidth(1)
            p.setPen(tick_pen)
            p.drawLine(x, pt, x, pt + plot_h)

        if self._duration_ms > 0:
            for b_ms in self._boundaries:
                bx = pl + int(b_ms / self._duration_ms * plot_w)
                boundary_pen = QPen(QColor("#D85A30"))
                boundary_pen.setWidth(1)
                boundary_pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(boundary_pen)
                p.drawLine(bx, pt, bx, pt + plot_h)

        for label, values, hex_color, do_fill in self._tracks:
            n = len(values)
            if n < 2:
                continue
            xs = [pl + int(i / (n - 1) * plot_w) for i in range(n)]
            ys = [pt + plot_h - int(v * plot_h) for v in values]

            if do_fill:
                fill_color = QColor(hex_color)
                fill_color.setAlpha(35)
                poly_points = [QPoint(xs[0], pt + plot_h)]
                for x, y in zip(xs, ys):
                    poly_points.append(QPoint(x, y))
                poly_points.append(QPoint(xs[-1], pt + plot_h))
                p.setBrush(QBrush(fill_color))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawPolygon(QPolygon(poly_points))

            curve_pen = QPen(QColor(hex_color))
            curve_pen.setWidth(2)
            p.setPen(curve_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(n - 1):
                p.drawLine(xs[i], ys[i], xs[i + 1], ys[i + 1])

            if label == "RMS":
                min_pen = QPen(QColor(hex_color))
                min_pen.setWidth(1)
                p.setPen(min_pen)
                p.setBrush(QBrush(QColor(hex_color)))
                for i in range(1, n - 1):
                    if values[i] < values[i - 1] and values[i] < values[i + 1] and values[i] < 0.30:
                        p.drawEllipse(xs[i] - 3, ys[i] - 3, 6, 6)

        if self._duration_ms > 0:
            cx = pl + int(self._cursor_ms / self._duration_ms * plot_w)
            cursor_pen = QPen(QColor("#ffffff"))
            cursor_pen.setWidth(2)
            p.setPen(cursor_pen)
            p.drawLine(cx, pt, cx, pt + plot_h)

        if self._duration_ms > 0 and self._events:
            is_event_only_lane = not self._tracks
            marker_y = pt + plot_h / 2 if is_event_only_lane else pt + 6
            radius = 4 if is_event_only_lane else 3
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(self._event_color)))
            for ev_ms, _ev_type in self._events:
                ex = pl + int(ev_ms / self._duration_ms * plot_w)
                p.drawEllipse(int(ex - radius), int(marker_y - radius), radius * 2, radius * 2)

        p.end()

    def _paint_empty(self) -> None:
        """
        Render a placeholder when no data or events are available.

        :return: None
        """
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), QColor("#1e1e2e"))
        p.setPen(QColor("#555"))
        p.setFont(QFont("sans-serif", 11))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Keine Audio-Daten verfügbar")
        p.end()