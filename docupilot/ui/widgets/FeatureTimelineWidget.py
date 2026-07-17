from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class FeatureTimelineWidget(QWidget):
    """
    Reusable timeline lane — a pure VIEW.

    It draws exactly what it is handed and derives nothing. Everything on screen
    is fed in through a setter:
      set_curve               — a modality's evidence curve, with its timestamps
      set_detected_boundaries — the boundaries that modality committed to
      set_boundaries          — the annotated ground truth (from the session)
      set_events              — input-event markers
      set_cursor / set_duration — playback state

    No thresholding, peak picking, smoothing or normalisation happens here: those
    are decisions, and they belong to the modality that owns the evidence. The
    widget maps value → pixel and time → pixel, nothing more. Curve values are
    expected already normalised to [0, 1].

    The curve is positioned by its TIMESTAMPS, not by index. The modalities
    sample on different grids and their curves do not all end at the same second;
    stretching each one across the full width would silently shift every lane by
    its own error.
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
        :return: None
        """
        super().__init__(parent)
        self._curve: list[float] = []
        self._curve_times_ms: list[float] = []
        self._curve_color: str = "#a78bfa"
        self._duration_ms: float = 0.0
        self._cursor_ms: float = 0.0
        self._boundaries: list[float] = []
        self._events: list[tuple[float, str]] = []
        # Determined boundaries: timestamps in ms, one per detected boundary,
        # drawn as a single prominent marker each. Replaces the old two-layer
        # scheme (a dashed line at every judged onset plus a mid-height diamond at
        # every flag), which buried the result under the candidates.
        self._detected_boundaries: list[float] = []
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_curve(self, times_s: list[float], values: list[float], hex_color: str) -> None:
        """
        Load the evidence curve to display. Pass empty lists to clear it.

        :param times_s: Timestamp of each value, in seconds.
        :param values: Curve values, already normalised to [0, 1].
        :param hex_color: Curve colour; the area under it is filled translucently.
        :return: None
        """
        self._curve_times_ms = [t * 1000.0 for t in times_s]
        self._curve = values
        self._curve_color = hex_color
        self.update()

    def set_duration(self, duration_ms: float) -> None:
        """
        Update the total recording duration without touching loaded tracks,
        boundaries, or events.

        Wird benötigt, weil die tatsächliche Dauer manchmal erst NACH
        set_data()/set_boundaries()/set_events() bekannt wird (z. B. wenn
        der Media-Player die Dauer noch nicht ermittelt hatte, als der
        Dialog aufgebaut wurde). Ohne diese Methode bliebe self._duration_ms
        dauerhaft auf dem ursprünglichen — möglicherweise 0 — Wert stehen,
        und JEDE Marker-Zeichnung (Grenzen, Events, Cursor, Semantik-Marker)
        ist an "self._duration_ms > 0" gekoppelt und würde für immer
        unsichtbar bleiben.

        :param duration_ms: Die aktuelle, korrekte Gesamtdauer in Millisekunden.
        :return: None
        """
        if duration_ms == self._duration_ms:
            return
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

    def set_detected_boundaries(self, boundaries_ms: list[float]) -> None:
        """
        Set the boundaries the model determined for this lane.

        Each is drawn as one prominent marker (a solid line capped with a
        triangle). Distinct from set_boundaries(), which holds the annotated
        ground truth.

        :param boundaries_ms: Boundary timestamps in milliseconds.
        :return: None
        """
        self._detected_boundaries = boundaries_ms
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
        if not self._curve and not self._events:
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

        if len(self._curve) >= 2 and self._duration_ms > 0:
            xs = [pl + int(t / self._duration_ms * plot_w) for t in self._curve_times_ms]
            ys = [pt + plot_h - int(v * plot_h) for v in self._curve]

            # A curve may run a little past the recording's end (the modalities
            # measure their own length); clip instead of painting over the axis.
            p.setClipRect(pl, pt, plot_w, plot_h)

            fill_color = QColor(self._curve_color)
            fill_color.setAlpha(35)
            poly_points = [QPoint(xs[0], pt + plot_h)]
            poly_points += [QPoint(x, y) for x, y in zip(xs, ys)]
            poly_points.append(QPoint(xs[-1], pt + plot_h))
            p.setBrush(QBrush(fill_color))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygon(poly_points))

            curve_pen = QPen(QColor(self._curve_color))
            curve_pen.setWidth(2)
            p.setPen(curve_pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for i in range(len(xs) - 1):
                p.drawLine(xs[i], ys[i], xs[i + 1], ys[i + 1])

            p.setClipping(False)

        if self._duration_ms > 0:
            cx = pl + int(self._cursor_ms / self._duration_ms * plot_w)
            cursor_pen = QPen(QColor("#ffffff"))
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
                ex = pl + int(ev_ms / self._duration_ms * plot_w)
                p.drawEllipse(int(ex - radius), int(marker_y - radius), radius * 2, radius * 2)

        # ── Determined boundaries: one prominent marker each ──────────────────
        # A solid vertical line capped with a triangle at the top edge. Fuchsia,
        # so it stays distinct from the curve (lane colour), the ground-truth
        # boundaries (dashed dark orange) and the playback cursor (white).
        if self._duration_ms > 0 and self._detected_boundaries:
            marker_color = QColor("#f0abfc")
            line_pen = QPen(marker_color)
            line_pen.setWidth(2)
            s = 5   # triangle half-width
            for t_ms in self._detected_boundaries:
                bx = pl + int(t_ms / self._duration_ms * plot_w)
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
        """
        Render a placeholder when no data or events are available.

        :return: None
        """
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), QColor("#1e1e2e"))
        p.setPen(QColor("#555"))
        p.setFont(QFont("sans-serif", 11))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Keine Daten verfügbar")
        p.end()