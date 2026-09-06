"""
The few visual constants the three result charts share, and the one placeholder
they all paint before a result exists.

Kept together so the charts look like one figure set in the report.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter

BACKGROUND = "#ffffff"
AXIS = "#cccccc"
TEXT = "#222222"
MUTED = "#777777"

PLACEHOLDER = "Noch keine Auswertung"


def paint_placeholder(painter: QPainter, rect: QRect) -> None:
    """Fill the widget and say that nothing has been computed yet."""
    painter.fillRect(rect, QColor(BACKGROUND))
    painter.setPen(QColor(MUTED))
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, PLACEHOLDER)


def small_fonts(base: QFont) -> tuple[QFont, QFont]:
    """(9 pt, 8 pt) variants of the widget font — labels and annotations."""
    small = QFont(base)
    small.setPointSize(9)
    tiny = QFont(small)
    tiny.setPointSize(8)
    return small, tiny
