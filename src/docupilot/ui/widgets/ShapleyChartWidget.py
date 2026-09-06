from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from docupilot.ui.widgets.chart_style import (
    AXIS, BACKGROUND, MUTED, TEXT, paint_placeholder, small_fonts,
)

_LABEL_WIDTH = 78
_VALUE_WIDTH = 168
_ROW_HEIGHT = 46


class ShapleyChartWidget(QWidget):
    """
    One bar per modality with its confidence interval, against a zero line.

    The zero line carries the argument: a whisker crossing it means the corpus
    cannot tell that modality's contribution from nothing. Reading that off a
    table of numbers takes effort; here it is the first thing visible.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[tuple[str, float, float, float, str]] = []
        self._total: float | None = None
        self.setMinimumHeight(180)
        self.setStyleSheet(f"background:{BACKGROUND};")

    def set_values(
        self,
        rows: list[tuple[str, float, float, float, str]],
        total: float | None = None,
    ) -> None:
        """
        :param rows: (modality, value, ci_low, ci_high, hex colour), strongest first.
        :param total: v(all modalities); the Shapley values must sum to it.
        """
        self._rows = rows
        self._total = total
        self.setMinimumHeight(max(180, _ROW_HEIGHT * len(rows) + 62))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._rows:
            paint_placeholder(painter, self.rect())
            return
        painter.fillRect(self.rect(), QColor(BACKGROUND))

        plot_left = _LABEL_WIDTH
        plot_right = max(plot_left + 40, self.width() - _VALUE_WIDTH)
        span = plot_right - plot_left

        low = min(0.0, min(r[2] for r in self._rows))
        high = max(0.0, max(r[3] for r in self._rows))
        pad = max((high - low) * 0.08, 0.01)
        low, high = low - pad, high + pad

        def x_of(value: float) -> float:
            return plot_left + span * (value - low) / (high - low)

        zero_x = x_of(0.0)

        # Zero line first, so the bars sit on top of it.
        painter.setPen(QPen(QColor(AXIS), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(zero_x), 6, int(zero_x), _ROW_HEIGHT * len(self._rows) + 6)

        small, _ = small_fonts(self.font())
        bold = QFont(small)
        bold.setBold(True)

        for index, (name, value, ci_low, ci_high, colour) in enumerate(self._rows):
            top = 6 + index * _ROW_HEIGHT
            middle = top + _ROW_HEIGHT / 2 - 4

            painter.setFont(bold)
            painter.setPen(QColor(colour))
            painter.drawText(
                QRectF(4, top, _LABEL_WIDTH - 10, _ROW_HEIGHT - 8),
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                name,
            )

            bar_left, bar_right = sorted((zero_x, x_of(value)))
            # A whisker touching zero is the finding, so it must not be hidden
            # behind the bar: the bar is drawn muted, the whisker at full weight.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(colour).lighter(135))
            painter.drawRoundedRect(
                QRectF(bar_left, middle - 9, max(bar_right - bar_left, 1.0), 18), 3, 3
            )

            painter.setPen(QPen(QColor(colour), 2))
            painter.drawLine(int(x_of(ci_low)), int(middle), int(x_of(ci_high)), int(middle))
            for edge in (ci_low, ci_high):
                painter.drawLine(int(x_of(edge)), int(middle - 6),
                                 int(x_of(edge)), int(middle + 6))

            crosses_zero = ci_low <= 0.0 <= ci_high
            painter.setFont(small)
            painter.setPen(QColor("#dc2626") if crosses_zero else QColor(TEXT))
            painter.drawText(
                QRectF(plot_right + 8, top, _VALUE_WIDTH - 12, _ROW_HEIGHT - 8),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{value:+.3f}  [{ci_low:+.3f}, {ci_high:+.3f}]"
                + ("  ~0" if crosses_zero else ""),
            )

        if self._total is not None:
            painter.setFont(small)
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(4, _ROW_HEIGHT * len(self._rows) + 10, self.width() - 8, 20),
                int(Qt.AlignmentFlag.AlignLeft),
                f"   Summe der Beiträge = {sum(r[1] for r in self._rows):.3f}"
                f"   ·   v(alle Modalitäten) = {self._total:.3f}"
                f"   ·   Effizienzeigenschaft erfüllt",
            )
