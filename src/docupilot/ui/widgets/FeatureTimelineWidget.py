from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QSizePolicy, QWidget


class FeatureTimelineWidget(QWidget):
    """
    Reusable timeline lane — a pure VIEW: it maps value → pixel and time → pixel
    and derives nothing. Curve values must already be normalised to [0, 1].
    """

    seek_requested = Signal(float)

    EVENT_COLOR = "#38bdf8"

    _PAD_L = 52
    _PAD_R = 16
    _PAD_T = 16
    _PAD_B = 32

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the timeline widget.

        :param parent: Parent widget.
        """
        super().__init__(parent)
        self._curve: list[float] = []
        self._curve_times_ms: list[float] = []
        self._curve_color: str = "#a78bfa"
        self._duration_ms: float = 0.0
        self._cursor_ms: float = 0.0
        self._boundaries: list[float] = []
        self._events: list[tuple[float, str]] = []
        self._detected_boundaries: list[float] = []
        # The curve's pixel polygon, valid for one (width, height, duration):
        # the cursor timer repaints every lane several times a second, and
        # re-projecting thousands of samples each time is what made the dialog
        # sluggish. Only the cursor moves between ticks; the curve does not.
        self._polygon_key: tuple[int, int, float] | None = None
        self._line: QPolygon | None = None
        self._area: QPolygon | None = None
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_curve(self, times_s: list[float], values: list[float], hex_color: str) -> None:
        """
        Load the evidence curve to display. Pass empty lists to clear it.

        :param times_s: Timestamp of each value, in seconds.
        :param values: Curve values, already normalised to [0, 1].
        :param hex_color: Curve colour; the area under it is filled translucently.
        """
        self._curve_times_ms = [t * 1000.0 for t in times_s]
        self._curve = values
        self._curve_color = hex_color
        self._polygon_key = None
        self.update()

    def set_duration(self, duration_ms: float) -> None:
        """
        Update the total duration without touching curve, boundaries or events —
        the player often only knows it after the lane was built.

        :param duration_ms: the current, correct total duration in milliseconds.
        """
        if duration_ms == self._duration_ms:
            return
        self._duration_ms = duration_ms
        self.update()

    def set_cursor(self, pos_ms: float) -> None:
        """
        Move the playback cursor line.

        :param pos_ms: Cursor position in milliseconds.
        """
        self._cursor_ms = pos_ms
        self.update()

    def set_boundaries(self, boundaries_ms: list[float]) -> None:
        """
        Set ground-truth boundary line positions.

        :param boundaries_ms: List of boundary timestamps in milliseconds.
        """
        self._boundaries = boundaries_ms
        self.update()

    def set_events(self, events: list[tuple[float, str]]) -> None:
        """
        Set event markers to draw as small dots.

        :param events: List of (t_ms, event_type).
        """
        self._events = events
        self.update()

    def set_detected_boundaries(self, boundaries_ms: list[float]) -> None:
        """
        Set the boundaries this modality committed to. Distinct from
        set_boundaries(), which holds the annotated ground truth.

        :param boundaries_ms: Boundary timestamps in milliseconds.
        """
        self._detected_boundaries = boundaries_ms
        self.update()

    def mousePressEvent(self, event) -> None:
        """Seek the player to the clicked position on the timeline."""
        if self._duration_ms <= 0:
            return
        x = event.position().x()
        frac = (x - self._PAD_L) / max(1, self.width() - self._PAD_L - self._PAD_R)
        frac = max(0.0, min(1.0, frac))
        self.seek_requested.emit(frac * self._duration_ms)

    # ── Painting ─────────────────────────────────────────────────────────

    def _x_of(self, t_ms: float, plot_w: int) -> int:
        return self._PAD_L + int(t_ms / self._duration_ms * plot_w)

    def _curve_polygons(self, plot_w: int, plot_h: int) -> tuple[QPolygon, QPolygon]:
        """The line and the filled area under it, projected for the current size."""
        key = (plot_w, plot_h, self._duration_ms)
        if self._polygon_key != key or self._line is None or self._area is None:
            pt = self._PAD_T
            xs = [self._x_of(t, plot_w) for t in self._curve_times_ms]
            ys = [pt + plot_h - int(v * plot_h) for v in self._curve]
            points = [QPoint(x, y) for x, y in zip(xs, ys)]
            self._line = QPolygon(points)
            self._area = QPolygon(
                [QPoint(xs[0], pt + plot_h), *points, QPoint(xs[-1], pt + plot_h)]
            )
            self._polygon_key = key
        return self._line, self._area

    def paintEvent(self, _event) -> None:
        """Render the grid, curves, boundaries, cursor, and event markers."""
        if not self._curve and not self._events:
            self._paint_empty()
            return

        w = self.width()
        h = self.height()
        pl, pr, pt, pb = self._PAD_L, self._PAD_R, self._PAD_T, self._PAD_B
        plot_w = w - pl - pr
        plot_h = h - pt - pb

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        p.fillRect(0, 0, w, h, QColor("#ffffff"))
        p.fillRect(pl, pt, plot_w, plot_h, QColor("#f7f7f7"))

        label_font = QFont("monospace", 8)
        p.setFont(label_font)
        label_color = QColor("#666666")
        grid_pen = QPen(QColor("#e0e0e0"))
        grid_pen.setWidth(1)

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = pt + plot_h - int(frac * plot_h)
            p.setPen(grid_pen)
            p.drawLine(pl, y, pl + plot_w, y)
            p.setPen(label_color)
            p.drawText(2, y + 4, pl - 6, 12, Qt.AlignmentFlag.AlignRight, f"{frac:.2f}")

        n_ticks = 6
        tick_pen = QPen(QColor("#e6e6e6"))
        tick_pen.setWidth(1)
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            x = pl + int(frac * plot_w)
            t_s = frac * self._duration_ms / 1000
            p.setPen(label_color)
            label = f"{int(t_s // 60):02d}:{t_s % 60:04.1f}"
            p.drawText(x - 24, h - pb + 4, 48, pb - 4, Qt.AlignmentFlag.AlignHCenter, label)
            p.setPen(tick_pen)
            p.drawLine(x, pt, x, pt + plot_h)

        if self._duration_ms > 0:
            boundary_pen = QPen(QColor("#D85A30"))
            boundary_pen.setWidth(1)
            boundary_pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(boundary_pen)
            for b_ms in self._boundaries:
                bx = self._x_of(b_ms, plot_w)
                p.drawLine(bx, pt, bx, pt + plot_h)

        if len(self._curve) >= 2 and self._duration_ms > 0:
            line, area = self._curve_polygons(plot_w, plot_h)

            # A curve may run a little past the recording's end (the modalities
            # measure their own length); clip instead of painting over the axis.
            p.setClipRect(pl, pt, plot_w, plot_h)

            fill_color = QColor(self._curve_color)
            fill_color.setAlpha(35)
            p.setBrush(QBrush(fill_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(area)

            curve_pen = QPen(QColor(self._curve_color))
            curve_pen.setWidth(2)
            p.setPen(curve_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(line)

            p.setClipping(False)

        if self._duration_ms > 0:
            cx = self._x_of(self._cursor_ms, plot_w)
            cursor_pen = QPen(QColor("#222222"))
            cursor_pen.setWidth(2)
            p.setPen(cursor_pen)
            p.drawLine(cx, pt, cx, pt + plot_h)

        if self._duration_ms > 0 and self._events:
            no_curve = not self._curve
            marker_y = pt + plot_h / 2 if no_curve else pt + 6
            radius = 4 if no_curve else 3
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(self.EVENT_COLOR)))
            for ev_ms, _ev_type in self._events:
                ex = self._x_of(ev_ms, plot_w)
                p.drawEllipse(int(ex - radius), int(marker_y - radius), radius * 2, radius * 2)

        # Fuchsia, to stay distinct from the curve, the ground truth and the cursor.
        if self._duration_ms > 0 and self._detected_boundaries:
            marker_color = QColor("#c026d3")
            line_pen = QPen(marker_color)
            line_pen.setWidth(2)
            s = 5   # triangle half-width
            for t_ms in self._detected_boundaries:
                bx = self._x_of(t_ms, plot_w)
                p.setPen(line_pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawLine(bx, pt, bx, pt + plot_h)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(marker_color))
                p.drawPolygon(QPolygon([
                    QPoint(bx - s, pt),
                    QPoint(bx + s, pt),
                    QPoint(bx, pt + s + 3),
                ]))

        p.end()

    def _paint_empty(self) -> None:
        """Render a placeholder when no data or events are available."""
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), QColor("#ffffff"))
        p.setPen(QColor("#888"))
        p.setFont(QFont("sans-serif", 11))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Keine Daten verfügbar")
        p.end()
