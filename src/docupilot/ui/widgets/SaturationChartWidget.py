from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from docupilot.ui.widgets.chart_style import (
    AXIS, BACKGROUND, MUTED, TEXT, paint_placeholder, small_fonts,
)

_LINE = "#059669"

_MARGIN_LEFT = 46
_MARGIN_RIGHT = 18
_MARGIN_TOP = 18
_MARGIN_BOTTOM = 42


class SaturationChartWidget(QWidget):
    """
    Mean F1 against the number of modalities, with the gain of each added one.

    The answer to the saturation question is the shape of this line: the gains
    written between the points are what has to shrink, and seeing them next to
    the curve is what distinguishes "still climbing" from "flat".
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._curve: dict[int, float] = {}
        self._gains: dict[int, float] = {}
        self._threshold: float | None = None
        self.setMinimumHeight(190)
        self.setStyleSheet(f"background:{BACKGROUND};")

    def set_values(
        self,
        curve: dict[int, float],
        gains: dict[int, float],
        threshold: float | None = None,
    ) -> None:
        """
        :param curve: mean F1 per number of modalities.
        :param gains: gain from one count to the next.
        :param threshold: relevance threshold a gain must exceed to count.
        """
        self._curve = curve
        self._gains = gains
        self._threshold = threshold
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._curve:
            paint_placeholder(painter, self.rect())
            return
        painter.fillRect(self.rect(), QColor(BACKGROUND))

        left, top = _MARGIN_LEFT, _MARGIN_TOP
        right = self.width() - _MARGIN_RIGHT
        bottom = self.height() - _MARGIN_BOTTOM
        counts = sorted(self._curve)
        steps = max(len(counts) - 1, 1)

        def point_of(k: int) -> QPointF:
            x = left + (right - left) * counts.index(k) / steps
            return QPointF(x, bottom - (bottom - top) * self._curve[k])

        small, tiny = small_fonts(self.font())

        painter.setPen(QPen(QColor(AXIS), 1))
        painter.drawLine(left, top, left, bottom)
        painter.drawLine(left, bottom, right, bottom)

        painter.setFont(tiny)
        for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - (bottom - top) * tick
            painter.setPen(QPen(QColor(AXIS), 1, Qt.PenStyle.DotLine))
            painter.drawLine(left, int(y), right, int(y))
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(2, y - 8, _MARGIN_LEFT - 8, 16),
                             int(Qt.AlignmentFlag.AlignRight), f"{tick:.2f}")

        painter.setPen(QPen(QColor(_LINE), 2))
        for previous, current in zip(counts, counts[1:]):
            painter.drawLine(point_of(previous), point_of(current))

        for k in counts:
            centre = point_of(k)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(_LINE))
            painter.drawEllipse(centre, 4.5, 4.5)

            painter.setFont(small)
            painter.setPen(QColor(TEXT))
            painter.drawText(QRectF(centre.x() - 34, centre.y() - 26, 68, 16),
                             int(Qt.AlignmentFlag.AlignHCenter), f"{self._curve[k]:.3f}")
            painter.setFont(tiny)
            painter.setPen(QColor(MUTED))
            painter.drawText(QRectF(centre.x() - 34, bottom + 6, 68, 16),
                             int(Qt.AlignmentFlag.AlignHCenter),
                             f"{k} Modalität" + ("" if k == 1 else "en"))

        # The gains sit between the points they belong to; below the relevance
        # threshold they are greyed, which is the saturation statement itself.
        painter.setFont(tiny)
        for k, gain in self._gains.items():
            if k not in counts or k - 1 not in counts:
                continue
            start, end = point_of(k - 1), point_of(k)
            relevant = self._threshold is None or abs(gain) >= self._threshold
            painter.setPen(QColor(_LINE) if relevant else QColor(MUTED))
            painter.drawText(
                QRectF((start.x() + end.x()) / 2 - 40,
                       (start.y() + end.y()) / 2 + 6, 80, 16),
                int(Qt.AlignmentFlag.AlignHCenter),
                f"{gain:+.3f}",
            )

        if self._threshold is not None:
            painter.setFont(tiny)
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(left, bottom + 22, right - left, 16),
                int(Qt.AlignmentFlag.AlignRight),
                f"grau = Zugewinn unter der Relevanzschwelle {self._threshold:.3f}",
            )
