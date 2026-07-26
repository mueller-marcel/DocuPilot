from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
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

from docupilot.evaluation import analysis, experiment, fusion, metrics, statistics
from docupilot.segmentation import MODALITIES
from docupilot.ui.widgets.SaturationChartWidget import SaturationChartWidget
from docupilot.ui.widgets.ShapleyChartWidget import ShapleyChartWidget
from docupilot.ui.widgets.SubsetChartWidget import SubsetChartWidget

_BACKGROUND = "#1e1e2e"

# Same colours the feature lanes use, so a modality looks the same everywhere.
_COLORS = {"events": "#38bdf8", "audio": "#a78bfa", "video": "#fb923c"}

# Tolerance the headline numbers are reported at. The sweep is in the CSV.
_PRIMARY_TAU_S = 1.0

# Below this a difference counts as practically irrelevant. Provisional: it is
# meant to be replaced by the measured intra-rater consistency, which grounds it
# in data instead of in a judgement call.
_RELEVANCE_THRESHOLD = 0.05

_BUTTON_STYLE = (
    "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
    "border-radius:6px;padding:6px 16px;font-size:12px;}"
    "QPushButton:hover{background:#f0f0f0;}"
    "QPushButton:disabled{background:#3a3a4e;color:#777;border-color:#4a4a5e;}"
)


@dataclass
class _Result:
    """Everything the charts need, computed off the UI thread."""

    rows: list[dict] = field(default_factory=list)
    subset_ci: dict[frozenset[str], statistics.Interval] = field(default_factory=dict)
    shapley: dict[str, statistics.Interval] = field(default_factory=dict)
    total: float = 0.0
    curve: dict[int, float] = field(default_factory=dict)
    gains: dict[int, float] = field(default_factory=dict)
    interactions: dict[tuple[str, str], float] = field(default_factory=dict)
    chance: float = 0.0
    n_sessions: int = 0


class _ExperimentWorker(QObject):
    """Runs the whole evaluation off the UI thread."""

    progress = Signal(str, int, int)
    failed = Signal(str)
    done = Signal(object)

    def __init__(self, directories: list[Path], use_forest: bool) -> None:
        super().__init__()
        self._directories = directories
        self._use_forest = use_forest
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            data = experiment.load(
                self._directories,
                on_progress=lambda name, i, total: self.progress.emit(
                    f"Extraktion: {name}", i, total),
                is_cancelled=lambda: self._cancelled,
            )
            if self._cancelled or not data:
                self.failed.emit(
                    "Abgebrochen." if self._cancelled
                    else "Keine auswertbare Session gefunden."
                )
                return

            rows = experiment.run(
                data,
                fuser_factory=(
                    fusion.ForestFuser if self._use_forest else fusion.RuleFuser
                ),
                on_progress=lambda label, i, total: self.progress.emit(
                    f"Teilmenge {label}", i, total),
            )
            self.done.emit(self._analyse(data, rows))
        except Exception as exc:                      # noqa: BLE001 — shown, not hidden
            self.failed.emit(str(exc))

    @staticmethod
    def _analyse(data, rows) -> _Result:
        paired = experiment.paired_f1(rows, _PRIMARY_TAU_S)
        values = experiment.subset_values(rows, _PRIMARY_TAU_S)
        players = tuple(MODALITIES)

        phi = statistics.shapley_ci(paired, players)
        curve = analysis.saturation(values, players)

        return _Result(
            rows=rows,
            subset_ci={s: statistics.subset_ci(scores) for s, scores in paired.items()},
            shapley=phi,
            total=values[frozenset(players)],
            curve=curve,
            gains=analysis.marginal_gain(curve),
            interactions=analysis.interaction(values, players),
            chance=float(np.mean([
                metrics.chance_level(d.gt_s, d.duration_s, _PRIMARY_TAU_S) for d in data
            ])),
            n_sessions=len(data),
        )


class ExperimentWindow(QDialog):
    """
    Runs the factorial experiment over a corpus and shows what it found.

    A shell, not a calculator: every number comes from the evaluation package,
    which stays runnable without any of this. The charts are for reading the
    result; the figures for the thesis are generated from the exported CSV,
    where they are reproducible.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Auswertung · Informationsbeitrag der Modalitäten")
        self.setMinimumSize(900, 700)
        self.resize(1180, 950)
        self.setModal(False)
        self.setStyleSheet(f"background:{_BACKGROUND};")

        self._directories: list[Path] = []
        self._result: _Result | None = None
        self._thread: QThread | None = None
        self._worker: _ExperimentWorker | None = None

        self._build_ui()

    # ── Construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        self._choose_button = QPushButton("Korpus wählen …")
        self._choose_button.setStyleSheet(_BUTTON_STYLE)
        self._choose_button.clicked.connect(self._on_choose_corpus)

        self._run_button = QPushButton("Auswertung starten")
        self._run_button.setStyleSheet(_BUTTON_STYLE)
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._on_run)

        self._cancel_button = QPushButton("Abbrechen")
        self._cancel_button.setStyleSheet(_BUTTON_STYLE)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._on_cancel)

        self._export_button = QPushButton("CSV exportieren …")
        self._export_button.setStyleSheet(_BUTTON_STYLE)
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._on_export)

        self._method = QComboBox()
        self._method.addItem("Random Forest (LOSO)", userData=True)
        self._method.addItem("Regel: Maximum (ohne Training)", userData=False)
        self._method.setStyleSheet(
            "QComboBox{background:#252538;color:#ddd;border:1px solid #3a3a4e;"
            "border-radius:6px;padding:5px 10px;font-size:12px;}"
        )

        top = QHBoxLayout()
        for button in (self._choose_button, self._run_button,
                       self._cancel_button, self._export_button):
            top.addWidget(button)
        top.addWidget(QLabel(" Verfahren:", styleSheet="color:#888;font-size:11px;"))
        top.addWidget(self._method)
        top.addStretch()
        root.addLayout(top)

        self._corpus_label = QLabel("Kein Korpus gewählt.")
        self._corpus_label.setStyleSheet("color:#aaa; font-size:11px;")
        root.addWidget(self._corpus_label)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(
            ["Session", "GT-Grenzen", "Video-Cache", "Audio-Cache"])
        self._table.setMaximumHeight(150)
        self._table.setStyleSheet(
            "QTableWidget{background:#252538;color:#ddd;gridline-color:#3a3a4e;"
            "font-size:11px;border:1px solid #3a3a4e;}"
            "QHeaderView::section{background:#2e2e44;color:#aaa;border:0;padding:4px;}"
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        root.addWidget(self._table)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet(
            "QProgressBar{background:#252538;border:0;border-radius:3px;}"
            "QProgressBar::chunk{background:#34d399;border-radius:3px;}")
        root.addWidget(self._progress)

        self._status = QLabel("Bereit.")
        self._status.setStyleSheet("color:#888; font-size:11px;")
        root.addWidget(self._status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"background:{_BACKGROUND}; border:0;")
        content = QWidget()
        content.setStyleSheet(f"background:{_BACKGROUND};")
        charts = QVBoxLayout(content)
        charts.setContentsMargins(0, 0, 0, 0)
        charts.setSpacing(4)

        self._shapley_chart = ShapleyChartWidget()
        self._subset_chart = SubsetChartWidget()
        self._saturation_chart = SaturationChartWidget()
        self._interaction_label = QLabel()
        self._interaction_label.setStyleSheet("color:#ccc; font-size:11px; padding:4px 6px;")

        charts.addWidget(self._heading(
            "Shapley-Werte · marginaler Beitrag je Modalität  (TF2)",
            "Balken = Beitrag, Whisker = 95 %-Konfidenzintervall. Ein Whisker über "
            "der Nulllinie bedeutet: von null nicht unterscheidbar."))
        charts.addWidget(self._shapley_chart)

        charts.addWidget(self._heading(
            "Alle 8 Modalitätskombinationen  (TF1)",
            "Gefüllter Punkt = Modalität enthalten. Rote Linie = Zufallsniveau."))
        charts.addWidget(self._subset_chart)

        charts.addWidget(self._heading(
            "Sättigungskurve  (TF3)",
            "Mittlerer F1 je Anzahl Modalitäten; Zahlen dazwischen = Zugewinn."))
        charts.addWidget(self._saturation_chart)

        charts.addWidget(self._heading(
            "Interaktionsindex · Synergie und Redundanz  (TF3)",
            "Negativ = die beiden Modalitäten sagen teilweise dasselbe."))
        charts.addWidget(self._interaction_label)
        charts.addStretch()

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
        heading.setStyleSheet("color:#eee;")

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
        if not chosen:
            return

        root = Path(chosen)
        # Either a folder of sessions, or one session picked directly — the
        # latter is how a single recording gets looked at without moving it.
        candidates = (
            [root] if (root / "recording.mp4").exists()
            else sorted(
                d for d in root.iterdir()
                if d.is_dir() and (d / "recording.mp4").exists()
            )
        )
        usable = [d for d in candidates if (d / "ground_truth.json").exists()]
        self._directories = usable

        self._table.setRowCount(len(candidates))
        without_cache = 0
        for row, directory in enumerate(candidates):
            annotated = (directory / "ground_truth.json").exists()
            boundaries = (
                len(json.loads(
                    (directory / "ground_truth.json").read_text(encoding="utf-8")))
                if annotated else 0
            )
            has_video = (directory / "gui_vlm_cache.json").exists()
            has_audio = (directory / "audio_llm_cache.json").exists()
            without_cache += 0 if has_video else 1

            for column, text in enumerate((
                directory.name,
                str(boundaries) if annotated else "— keine GT —",
                "ja" if has_video else "nein",
                "ja" if has_audio else "nein",
            )):
                item = QTableWidgetItem(text)
                if not annotated:
                    item.setForeground(Qt.GlobalColor.red)
                self._table.setItem(row, column, item)

        # Leave-one-session-out needs something to train on. With a single
        # session there is no such thing, so only the untrained rule is offered
        # — it still yields a Shapley chart, just without cross-validation.
        trainable = len(usable) >= 2
        self._method.model().item(0).setEnabled(trainable)
        if not trainable:
            self._method.setCurrentIndex(1)

        skipped = len(candidates) - len(usable)
        self._corpus_label.setText(
            f"{root}  ·  {len(usable)} Sessions verwendbar"
            + (f"  ·  {skipped} ohne Ground Truth übersprungen" if skipped else "")
            + (f"  ·  ⚠ {without_cache} ohne Video-Cache — dieser Lauf erzeugt "
               f"neue VLM-Aufrufe (Größenordnung ~50 je Session)"
               if without_cache else "  ·  vollständig gecacht, keine Modellkosten")
            + ("  ·  ⚠ nur eine Session: kein LOSO, keine Konfidenzintervalle — "
               "Rauchtest, kein Ergebnis" if len(usable) == 1 else "")
        )
        self._run_button.setEnabled(len(usable) >= 1)

    # ── Run ──────────────────────────────────────────────────────────────

    def _on_run(self) -> None:
        self._run_button.setEnabled(False)
        self._choose_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._progress.setRange(0, 0)
        self._status.setText("Starte …")

        self._thread = QThread(self)
        self._worker = _ExperimentWorker(
            self._directories, use_forest=bool(self._method.currentData())
        )
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

    def _on_failed(self, message: str) -> None:
        self._reset_buttons()
        self._status.setText(f"Fehlgeschlagen: {message.splitlines()[0]}")
        QMessageBox.warning(self, "Auswertung fehlgeschlagen", message)

    def _on_done(self, result: _Result) -> None:
        self._result = result
        self._reset_buttons()
        self._export_button.setEnabled(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        self._status.setText(
            f"Fertig · {result.n_sessions} Session"
            + ("" if result.n_sessions == 1 else "s")
            + f" · {self._method.currentText()} · τ = {_PRIMARY_TAU_S:.2f} s · "
            f"Zufallsniveau {result.chance:.3f}"
            + ("  ·  ⚠ n = 1: Konfidenzintervalle sind entartet"
               if result.n_sessions < 2 else "")
        )
        self._render(result)

    def _reset_buttons(self) -> None:
        self._run_button.setEnabled(len(self._directories) >= 1)
        self._choose_button.setEnabled(True)
        self._cancel_button.setEnabled(False)

    # ── Rendering ────────────────────────────────────────────────────────

    def _render(self, result: _Result) -> None:
        self._shapley_chart.set_values(
            sorted(
                ((m, ci.point, ci.lo, ci.hi, _COLORS.get(m, "#ddd"))
                 for m, ci in result.shapley.items()),
                key=lambda row: -row[1],
            ),
            total=result.total,
        )

        self._subset_chart.set_values(
            modalities=list(MODALITIES),
            colors=_COLORS,
            rows=sorted(
                ((subset, ci.point, ci.lo, ci.hi)
                 for subset, ci in result.subset_ci.items()),
                key=lambda row: -row[1],
            ),
            chance=result.chance,
        )

        self._saturation_chart.set_values(
            result.curve, result.gains, threshold=_RELEVANCE_THRESHOLD
        )

        self._interaction_label.setText("\n".join(
            f"   {a} ↔ {b}:   {value:+.3f}   "
            + ("Synergie — gemeinsam mehr als einzeln"
               if value > 0 else "Redundanz — teilweise dieselbe Information")
            for (a, b), value in result.interactions.items()
        ))

    def _on_export(self) -> None:
        """Write the tidy table; the thesis figures are built from this file."""
        if self._result is None:
            return
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Ergebnistabelle speichern", "ergebnisse.csv", "CSV (*.csv)")
        if not chosen:
            return
        experiment.write_csv(self._result.rows, Path(chosen))
        self._status.setText(f"Exportiert nach {chosen}")

    def closeEvent(self, event) -> None:
        """Stop a running evaluation; cached model verdicts survive."""
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(120_000)
        super().closeEvent(event)
