"""
The evaluation report as a paginated A4 PDF.

Charts are the same widgets the window shows, rendered off-screen through a
supersampled painter, so the PDF carries the exact figures — no re-run. Text
is measured before it is drawn, so nothing overlaps and a heading never sits
alone at the bottom of a page.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QDateTime, QMarginsF, QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPageSize, QPainter, QPdfWriter, QPixmap
from PySide6.QtWidgets import QWidget

from docupilot.evaluation import report as rep
from docupilot.ui.report_view import build_charts

_DPI = 200
_CHART_NATURAL_WIDTH = 780
_CHART_SUPERSAMPLE = 3


class _Page:
    """Cursor state of the page being written: where the next block goes."""

    def __init__(self, writer: QPdfWriter, painter: QPainter, margin: int) -> None:
        self.writer = writer
        self.painter = painter
        self.margin = margin
        self.page_w, self.page_h = writer.width(), writer.height()
        self.content_w = self.page_w - 2 * margin
        self.top, self.bottom = margin, self.page_h - margin
        self.y = float(self.top)
        self.number = 1
        self._wrap = int(Qt.TextFlag.TextWordWrap)
        self._f_hint = _font(8, italic=True)

    def footer(self) -> None:
        self.painter.save()
        self.painter.setFont(self._f_hint)
        self.painter.setPen(QColor("#999999"))
        self.painter.drawText(
            QRectF(self.margin, self.page_h - self.margin + 20,
                   self.content_w, self.margin - 30),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            f"DocuPilot · Auswertung · Seite {self.number}",
        )
        self.painter.restore()

    def new_page(self) -> None:
        self.writer.newPage()
        self.y = float(self.top)
        self.number += 1
        self.footer()

    def ensure(self, height: float) -> None:
        """Start a new page when `height` more would run past the bottom."""
        if self.y + height > self.bottom:
            self.new_page()

    def measure(self, text: str, font: QFont) -> float:
        self.painter.setFont(font)
        return self.painter.boundingRect(
            QRectF(0, 0, self.content_w, 1_000_000), self._wrap, text).height()

    def para(self, text: str, font: QFont, color: str,
             *, gap_before: float = 0, gap_after: float = 0) -> None:
        if not text:
            return
        self.y += gap_before
        h = self.measure(text, font)
        self.ensure(h)
        self.painter.setFont(font)
        self.painter.setPen(QColor(color))
        self.painter.drawText(QRectF(self.margin, self.y, self.content_w, h), self._wrap, text)
        self.y += h + gap_after

    def chart(self, widget: QWidget) -> None:
        """Render a widget off-screen at 3x and place it at content width."""
        natural_w = _CHART_NATURAL_WIDTH
        natural_h = max(widget.minimumHeight(), 160)
        widget.resize(natural_w, natural_h)
        widget.ensurePolished()
        ss = _CHART_SUPERSAMPLE
        pixmap = QPixmap(natural_w * ss, natural_h * ss)
        pixmap.fill(Qt.GlobalColor.white)
        pp = QPainter(pixmap)
        pp.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pp.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        pp.scale(ss, ss)
        widget.render(pp, QPoint(0, 0))
        pp.end()
        target_h = self.content_w * natural_h / natural_w
        self.ensure(target_h)
        self.painter.drawPixmap(
            QRectF(self.margin, self.y, self.content_w, target_h), pixmap,
            QRectF(pixmap.rect()))
        self.y += target_h

    def rule(self) -> None:
        self.painter.setPen(QColor("#cccccc"))
        self.painter.drawLine(self.margin, int(self.y), self.margin + self.content_w, int(self.y))
        self.y += 6


def _font(point_size: int, *, bold: bool = False, italic: bool = False,
          family: str | None = None, monospace: bool = False) -> QFont:
    font = QFont(family) if family else QFont()
    if monospace:
        font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(point_size)
    font.setBold(bold)
    font.setItalic(italic)
    return font


def write_report_pdf(path: Path, report: rep.Report, corpus_root: Path | None) -> None:
    """
    Write the full report to `path`.

    :param report: the finished evaluation.
    :param corpus_root: shown in the header; None when unknown.
    """
    writer = QPdfWriter(str(path))
    writer.setResolution(_DPI)
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0))

    f_title = _font(17, bold=True)
    f_meta = _font(9)
    f_head = _font(12, bold=True)
    f_hint = _font(8, italic=True)
    f_body = _font(9, family="Consolas", monospace=True)

    painter = QPainter(writer)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    page = _Page(writer, painter, margin=int(_DPI * 0.7))
    charts = build_charts(report)

    def section(title: str, hint: str) -> None:
        page.y += 16
        page.ensure(page.measure(title, f_head) + page.measure(hint, f_hint) + _DPI * 0.9)
        page.para(title, f_head, "#111111", gap_after=2)
        page.para(hint, f_hint, "#666666", gap_after=6)

    page.footer()
    page.para("DocuPilot – Informationsbeitrag der Modalitäten zur Segmentierung",
              f_title, "#111111", gap_after=6)
    stamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm")
    page.para("\n".join(rep.header_lines(report, corpus_root, stamp)),
              f_meta, "#444444", gap_after=6)
    page.rule()

    for block in rep.sections(report):
        section(block.title, block.hint)
        if block.chart is not None:
            page.chart(charts.by_name(block.chart))
        if block.text is not None:
            # After a chart the text needs air; directly under a heading the
            # heading's own gap is enough.
            page.para(block.text, f_body, "#222222",
                      gap_before=6 if block.chart is not None else 0)

    painter.end()
