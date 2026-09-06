from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from docupilot.ui.widgets.chart_style import (
    BACKGROUND, TEXT, paint_placeholder, small_fonts,
)

_ABSENT = "#d0d0d8"
_CHANCE = "#dc2626"

_DOT_SPACING = 26
_DOT_RADIUS = 6
_VALUE_WIDTH = 168
_ROW_HEIGHT = 30


class SubsetChartWidget(QWidget):
    """
    All 8 modality combinations as one picture: which modalities are in, and how
    well that combination did.

    Membership is shown as filled or hollow dots rather than a name like
    "video+events" — the eye reads the factorial pattern directly, and rows stay
    comparable however long the modality names are (an UpSet-style plot).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._modalities: list[str] = []
        self._colors: dict[str, str] = {}
        self._rows: list[tuple[frozenset[str], float, float, float]] = []
        self._chance: float | None = None
        self.setMinimumHeight(200)
        self.setStyleSheet(f"background:{BACKGROUND};")

    def set_values(
        self,
        modalities: list[str],
        colors: dict[str, str],
        rows: list[tuple[frozenset[str], float, float, float]],
        chance: float | None = None,
    ) -> None:
        """
        :param modalities: column order of the membership dots.
        :param colors: hex colour per modality.
        :param rows: (subset, F1, ci_low, ci_high), best first.
        :param chance: F1 reached by guessing; drawn as the floor.
        """
        self._modalities = modalities
        self._colors = colors
        self._rows = rows
        self._chance = chance
        self.setMinimumHeight(max(200, _ROW_HEIGHT * len(rows) + 54))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._rows:
            paint_placeholder(painter, self.rect())
            return
        painter.fillRect(self.rect(), QColor(BACKGROUND))

        dots_width = _DOT_SPACING * len(self._modalities) + 14
        plot_left = dots_width + 10
        plot_right = max(plot_left + 40, self.width() - _VALUE_WIDTH)
        span = plot_right - plot_left
        top_offset = 24

        small, tiny = small_fonts(self.font())

        # Column header: which dot belongs to which modality.
        painter.setFont(tiny)
        for column, modality in enumerate(self._modalities):
            painter.setPen(QColor(self._colors.get(modality, TEXT)))
            painter.drawText(
                QRectF(7 + column * _DOT_SPACING - 10, 2, _DOT_SPACING + 20, 16),
                int(Qt.AlignmentFlag.AlignHCenter),
                modality[:3],
            )

        for index, (subset, value, ci_low, ci_high) in enumerate(self._rows):
            top = top_offset + index * _ROW_HEIGHT
            middle = top + _ROW_HEIGHT / 2

            for column, modality in enumerate(self._modalities):
                centre_x = 7 + column * _DOT_SPACING + _DOT_RADIUS
                present = modality in subset
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(
                    QColor(self._colors.get(modality, TEXT)) if present
                    else QColor(_ABSENT)
                )
                painter.drawEllipse(
                    QRectF(centre_x - _DOT_RADIUS, middle - _DOT_RADIUS,
                           _DOT_RADIUS * 2, _DOT_RADIUS * 2)
                )

            # A subset's bar is tinted by its strongest member, so the colour
            # says which modality is carrying it.
            tint = (
                QColor(self._colors[max(subset, key=lambda m: self._modalities.index(m))])
                if subset else QColor(_ABSENT)
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tint.lighter(135))
            painter.drawRoundedRect(
                QRectF(plot_left, middle - 7, max(span * value, 1.0), 14), 3, 3
            )

            painter.setPen(QPen(tint, 2))
            painter.drawLine(int(plot_left + span * ci_low), int(middle),
                             int(plot_left + span * ci_high), int(middle))

            painter.setFont(small)
            painter.setPen(QColor(TEXT))
            painter.drawText(
                QRectF(plot_right + 8, top, _VALUE_WIDTH - 12, _ROW_HEIGHT),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"{value:.3f}  [{ci_low:.3f}, {ci_high:.3f}]",
            )

        if self._chance is not None:
            x = plot_left + span * self._chance
            painter.setPen(QPen(QColor(_CHANCE), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(x), top_offset - 4,
                             int(x), top_offset + _ROW_HEIGHT * len(self._rows))
            painter.setFont(tiny)
            painter.setPen(QColor(_CHANCE))
            painter.drawText(
                QRectF(x + 4, top_offset + _ROW_HEIGHT * len(self._rows) + 2, 160, 16),
                int(Qt.AlignmentFlag.AlignLeft),
                f"Zufallsniveau {self._chance:.3f}",
            )
