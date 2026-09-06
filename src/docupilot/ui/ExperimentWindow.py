"""
The window that runs the factorial experiment over a corpus and shows the result.

A shell, not a calculator: every number comes from the evaluation package, which
stays runnable without any of this. The sections it shows are the report's own
(`report.sections`), so window and PDF cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from docupilot.evaluation import corpus, dataset, experiment
from docupilot.evaluation import report as rep
from docupilot.recording.session import RecordingSession
from docupilot.ui.report_view import ReportCharts, apply_report
from docupilot.ui.ReportPdfWriter import write_report_pdf
from docupilot.ui.widgets.SaturationChartWidget import SaturationChartWidget
from docupilot.ui.widgets.ShapleyChartWidget import ShapleyChartWidget
from docupilot.ui.widgets.SubsetChartWidget import SubsetChartWidget

_BACKGROUND = "#ffffff"

# Wait-spinner frames (Braille); Segoe UI on Windows renders these.
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_BUTTON_STYLE = (
    "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
    "border-radius:6px;padding:6px 16px;font-size:12px;}"
    "QPushButton:hover{background:#f0f0f0;}"
    "QPushButton:disabled{background:#f0f0f0;color:#aaa;border-color:#e0e0e0;}"
)

_NOTE_STYLE = ("color:#222; font-size:11px; padding:4px 6px; "
               "font-family:Consolas,'Courier New',monospace;")


class _ExperimentWorker(QObject):
    """Runs the whole evaluation off the UI thread."""

    progress = Signal(str, int, int)
    failed = Signal(str)
    done = Signal(object)

    def __init__(self, directories: list[Path], start_definition: bool) -> None:
        """
        :param start_definition: also score against the "start" boundaries —
            only meaningful when every session carries them.
        """
        super().__init__()
        self._directories = directories
        self._start_definition = start_definition
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _progress(self, phase: str):
        return lambda label, i, total: self.progress.emit(f"{phase}: {label}", i, total)

    def run(self) -> None:
        try:
            data = experiment.load(
                self._directories,
                on_progress=self._progress("Extraktion"),
                is_cancelled=lambda: self._cancelled,
            )
            if self._cancelled or not data:
                self.failed.emit(
                    "Abgebrochen." if self._cancelled
                    else "Keine auswertbare Session gefunden."
                )
                return

            rows = experiment.run(data, pool="union",
                                  on_progress=self._progress("Teilmenge (fixierter Pool)"))
            rows_isolated = experiment.run(data, pool="own",
                                           on_progress=self._progress("Teilmenge (isoliert)"))
            rows_start = None
            if self._start_definition:
                starts = {
                    d.name: dataset.ground_truth_s(RecordingSession.from_directory(d), "start")
                    for d in self._directories
                }
                rows_start = experiment.run(
                    experiment.with_ground_truth(data, starts), pool="union",
                    on_progress=self._progress("Teilmenge (Definition Beginn)"),
                    extra={"definition": "start"},
                )
            result = rep.analyse(data, rows, rows_isolated, rows_start)
            # τ justification runs last and cancellable: the main result is
            # already computed, and a failure here must never discard it.
            try:
                result.sync = rep.measure_sync(
                    self._directories,
                    on_progress=self._progress("Synchronisation"),
                    is_cancelled=lambda: self._cancelled,
                )
            except Exception:                     # noqa: BLE001 — never discard the result
                pass
            self.done.emit(result)
        except Exception as exc:                      # noqa: BLE001 — shown, not hidden
            self.failed.emit(str(exc))


class ExperimentWindow(QDialog):
    """Runs the factorial experiment over a corpus and shows what it found."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auswertung · Informationsbeitrag der Modalitäten")
        self.setMinimumSize(900, 700)
        self.resize(1180, 950)
        self.setModal(False)
        self.setStyleSheet(f"background:{_BACKGROUND};")

        self._directories: list[Path] = []
        self._corpus_root: Path | None = None
        self._start_definition = False
        self._result: rep.Report | None = None
        self._thread: QThread | None = None
        self._worker: _ExperimentWorker | None = None

        self._build_ui()

    # ── Construction ─────────────────────────────────────────────────────

    def _button(self, text: str, slot, enabled: bool = True) -> QPushButton:
        button = QPushButton(text)
        button.setStyleSheet(_BUTTON_STYLE)
        button.setEnabled(enabled)
        button.clicked.connect(slot)
        return button

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        self._choose_button = self._button("Korpus wählen …", self._on_choose_corpus)
        self._run_button = self._button("Auswertung starten", self._on_run, enabled=False)
        self._cancel_button = self._button("Abbrechen", self._on_cancel, enabled=False)
        self._export_button = self._button("CSV exportieren …", self._on_export, enabled=False)
        self._pdf_button = self._button("Bericht (PDF) …", self._on_export_pdf, enabled=False)

        top = QHBoxLayout()
        for button in (self._choose_button, self._run_button,
                       self._cancel_button, self._export_button, self._pdf_button):
            top.addWidget(button)
        top.addStretch()
        root.addLayout(top)

        self._corpus_label = QLabel("Kein Korpus gewählt.")
        self._corpus_label.setWordWrap(True)
        self._corpus_label.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(self._corpus_label)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["Session", "GT-Grenzen (Ende)", "GT Beginn", "Video-Cache", "Audio-Cache"])
        self._table.setMaximumHeight(150)
        self._table.setStyleSheet(
            "QTableWidget{background:#fff;color:#333;gridline-color:#ddd;"
            "font-size:11px;border:1px solid #ccc;}"
            "QHeaderView::section{background:#f0f0f0;color:#555;border:0;padding:4px;}"
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        root.addWidget(self._table)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet(
            "QProgressBar{background:#e5e7eb;border:0;border-radius:3px;}"
            "QProgressBar::chunk{background:#059669;border-radius:3px;}")
        root.addWidget(self._progress)

        self._spinner = QLabel("")
        self._spinner.setStyleSheet("color:#059669; font-size:15px; font-weight:bold;")
        self._status = QLabel("Bereit.")
        self._status.setStyleSheet("color:#555; font-size:11px;")
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        status_row.addWidget(self._spinner)
        status_row.addWidget(self._status)
        status_row.addStretch()
        root.addLayout(status_row)

        self._spinner_index = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(110)
        self._spinner_timer.timeout.connect(self._tick_spinner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{_BACKGROUND}; border:0;")
        content = QWidget()
        content.setStyleSheet(f"background:{_BACKGROUND};")
        self._sections = QVBoxLayout(content)
        self._sections.setContentsMargins(0, 0, 0, 0)
        self._sections.setSpacing(4)

        self._charts = ReportCharts(
            ShapleyChartWidget(), SubsetChartWidget(), SaturationChartWidget()
        )
        self._placeholder = QLabel("Noch keine Auswertung.")
        self._placeholder.setStyleSheet("color:#777; font-size:12px; padding:12px 6px;")
        self._sections.addWidget(self._placeholder)
        self._sections.addStretch()

        scroll.setWidget(content)
        root.addWidget(scroll, stretch=1)

    @staticmethod
    def _heading(title: str, subtitle: str) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 10, 4, 2)
        layout.setSpacing(1)

        heading = QLabel(title)
        font = QFont(heading.font())
        font.setBold(True)
        font.setPointSize(10)
        heading.setFont(font)
        heading.setStyleSheet("color:#222;")

        hint = QLabel(subtitle)
        hint.setStyleSheet("color:#777; font-size:10px;")
        hint.setWordWrap(True)

        layout.addWidget(heading)
        layout.addWidget(hint)
        return box

    # ── Corpus ───────────────────────────────────────────────────────────

    def _on_choose_corpus(self) -> None:
        """Pick a folder of session directories and report what is usable."""
        chosen = QFileDialog.getExistingDirectory(self, "Korpus-Verzeichnis wählen")
        if chosen:
            self._load_corpus(Path(chosen))

    def run_directory(self, root: Path) -> None:
        """
        Load a corpus directory and start the full workflow at once.

        The menu entry point: select a directory, then segment and evaluate
        without a second click. Ignored while a run is already in progress.
        """
        if self._thread is not None and self._thread.isRunning():
            return
        if self._load_corpus(root) >= 2:
            self._on_run()

    def _load_corpus(self, root: Path) -> int:
        """Scan a corpus directory, fill the table, and report the usable count."""
        self._corpus_root = root
        scanned = corpus.scan(root)
        self._directories = scanned.usable
        self._start_definition = scanned.start_definition_available

        self._table.setRowCount(len(scanned.sessions))
        for row, info in enumerate(scanned.sessions):
            for column, text in enumerate((
                info.name,
                str(info.n_boundaries) if info.annotated else "— keine GT —",
                str(info.n_start_boundaries) if info.n_start_boundaries else "–",
                "ja" if info.has_video_cache else "nein",
                "ja" if info.has_audio_cache else "nein",
            )):
                item = QTableWidgetItem(text)
                if not info.annotated:
                    item.setForeground(Qt.GlobalColor.red)
                self._table.setItem(row, column, item)

        self._corpus_label.setText(corpus.describe(scanned))
        self._run_button.setEnabled(scanned.can_evaluate)
        return len(self._directories)

    # ── Run ──────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        self._run_button.setEnabled(False)
        self._choose_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._progress.setRange(0, 0)
        self._status.setText("Starte …")
        self._start_spinner()

        self._thread = QThread(self)
        self._worker = _ExperimentWorker(self._directories, self._start_definition)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)

        self._thread.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._status.setText("Abbruch angefordert — laufender Schritt wird beendet …")

    def _on_progress(self, label: str, done: int, total: int) -> None:
        self._progress.setRange(0, total)
        self._progress.setValue(done)
        self._status.setText(f"{label}  ({done}/{total})")

    def _start_spinner(self) -> None:
        self._spinner_index = 0
        self._spinner.setText(_SPINNER_FRAMES[0])
        self._spinner_timer.start()

    def _stop_spinner(self) -> None:
        self._spinner_timer.stop()
        self._spinner.setText("")

    def _tick_spinner(self) -> None:
        self._spinner_index = (self._spinner_index + 1) % len(_SPINNER_FRAMES)
        self._spinner.setText(_SPINNER_FRAMES[self._spinner_index])

    def _on_failed(self, message: str) -> None:
        self._stop_spinner()
        self._reset_buttons()
        self._status.setText(f"Fehlgeschlagen: {message.splitlines()[0]}")
        QMessageBox.warning(self, "Auswertung fehlgeschlagen", message)

    def _on_done(self, result: rep.Report) -> None:
        self._stop_spinner()
        self._result = result
        self._reset_buttons()
        self._export_button.setEnabled(True)
        self._pdf_button.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._status.setText(rep.status_line(result))
        self._render(result)

    def _reset_buttons(self) -> None:
        self._run_button.setEnabled(len(self._directories) >= 2)
        self._choose_button.setEnabled(True)
        self._cancel_button.setEnabled(False)

    # ── Rendering ────────────────────────────────────────────────────────

    def _clear_sections(self) -> None:
        """Remove everything but the chart widgets, which are reused."""
        while self._sections.count():
            item = self._sections.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget in (self._charts.shapley, self._charts.subsets, self._charts.saturation):
                widget.setParent(None)
            else:
                widget.deleteLater()

    def _render(self, result: rep.Report) -> None:
        apply_report(self._charts, result)
        self._clear_sections()
        for block in rep.sections(result):
            self._sections.addWidget(self._heading(block.title, block.hint))
            if block.chart is not None:
                self._sections.addWidget(self._charts.by_name(block.chart))
            if block.text is not None:
                note = QLabel(block.text)
                note.setStyleSheet(_NOTE_STYLE)
                note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                self._sections.addWidget(note)
        self._sections.addStretch()

    # ── Export ───────────────────────────────────────────────────────────

    def _on_export_pdf(self) -> None:
        """Write a paginated PDF report of the current result: metrics + charts."""
        if self._result is None:
            return
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Bericht speichern", "auswertung.pdf", "PDF (*.pdf)")
        if not chosen:
            return
        if not chosen.lower().endswith(".pdf"):
            chosen += ".pdf"
        try:
            write_report_pdf(Path(chosen), self._result, self._corpus_root)
        except Exception as exc:                       # noqa: BLE001 — shown, not hidden
            QMessageBox.warning(self, "PDF-Export fehlgeschlagen", str(exc))
            return
        self._status.setText(f"Bericht exportiert nach {chosen}")

    def _on_export(self) -> None:
        """Write the tidy tables of every design run; the thesis figures are
        built from this file."""
        if self._result is None:
            return
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Ergebnistabelle speichern", "ergebnisse.csv", "CSV (*.csv)")
        if not chosen:
            return
        r = self._result
        experiment.write_csv([*r.rows, *r.rows_isolated, *r.rows_start], Path(chosen))
        self._status.setText(f"Exportiert nach {chosen}")

    def closeEvent(self, event) -> None:
        """Stop a running evaluation; cached model verdicts survive."""
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(120_000)
        super().closeEvent(event)
