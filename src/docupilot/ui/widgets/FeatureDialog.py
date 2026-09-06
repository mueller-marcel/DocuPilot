from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from docupilot.recording.session import RecordingSession
from docupilot.segmentation import MODALITIES, BoundaryEvidence, segment
from docupilot.ui.widgets.FeatureTimelineWidget import FeatureTimelineWidget

_GROUND_TRUTH_COLOR = "#059669"

# How each modality is presented. The keys are the segmentation package's
# modality names; everything in the value is a UI decision and lives only here.
_LANES: dict[str, dict[str, str]] = {
    "events": {
        "section": "  ►  Events (Leerlauf + Wechsel der Eingabeart)",
        "label":   "Event-Score  ·  Klick / Taste / Scroll",
        "name":    "Events",
        "color":   FeatureTimelineWidget.EVENT_COLOR,
    },
    "audio": {
        "section": "  ►  Audio (Ansage-Fenster, LLM-bewertet)",
        "label":   "Audio-Score  ·  Grenze im Fenster erwartet",
        "name":    "Audio",
        "color":   "#a78bfa",
    },
    "video": {
        "section": "  ►  Video (pHash-Zustände + VLM: neuer Zustand?)",
        "label":   "Video-Boundary-Score  ·  neuer Zustand?",
        "name":    "Video",
        "color":   "#fb923c",
    },
}


class _FeatureWorker(QObject):
    """Runs segmentation OFF the UI thread and turns its callbacks into signals."""

    # The evidence stays HERE and the signal carries only the modality that is
    # done. A queued cross-thread signal marshals every argument, and an `object`
    # payload has to go through a wrapper whose construction fails on this stack
    # ("__init__() should return None, not 'NoneType'") — reported once per lane,
    # because segment() runs the callback inside its own try. A key the receiver
    # looks up needs no wrapper.
    lane_ready  = Signal(str)           # modality; the evidence via evidence()
    lane_failed = Signal(str, str)      # (modality, message)
    progress    = Signal(str, int, int)  # (modality, done, total)
    finished    = Signal()

    def __init__(self, session: RecordingSession) -> None:
        super().__init__()
        self._session = session
        self._cancelled = False
        self._evidence: dict[str, BoundaryEvidence] = {}

    def cancel(self) -> None:
        """Ask the run to stop at the next point segmentation checks."""
        self._cancelled = True

    def evidence(self, modality: str) -> BoundaryEvidence | None:
        """One finished lane's evidence, or None while it has not arrived."""
        return self._evidence.get(modality)

    def _keep(self, modality: str, evidence: BoundaryEvidence) -> None:
        """Store first, announce second — the receiver reads only after the emit."""
        self._evidence[modality] = evidence
        self.lane_ready.emit(modality)

    def run(self) -> None:
        segment(
            self._session,
            on_result=self._keep,
            on_error=self.lane_failed.emit,
            on_progress=self.progress.emit,
            is_cancelled=lambda: self._cancelled,
        )
        self.finished.emit()


class FeatureDialog(QDialog):
    """
    Non-modal dialog with one timeline lane per modality, plus a ground-truth lane
    that appears only when boundaries were annotated.

    A pure view: it starts segmentation on a worker thread and draws what comes
    back. Every boundary decision was made in the segmentation package.
    """

    def __init__(
        self,
        player: QMediaPlayer,
        session: RecordingSession,
        duration_ms: float,
        parent: QWidget | None = None,
    ) -> None:
        """
        Build the dialog and start segmentation.

        :param player: Media player providing playback position and seeking.
        :param session: Recording session to segment and display.
        :param duration_ms: Total recording duration in milliseconds.
        :param parent: Parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle("Feature-Verläufe · Handlungsgrenzen je Modalität")
        self.setMinimumSize(820, 700)
        self.resize(1100, 950)
        self.setModal(False)

        self._player = player
        self._session = session
        # Player, lanes, extractors and GT share the recording's real timeline
        # (the MP4 carries real timestamps). The player may still report 0 until it
        # has probed the file; _sync_cursor corrects that.
        self._duration_ms = float(player.duration()) or duration_ms

        self._ground_truth_markers = session.ground_truth_markers()
        self._boundaries_ms = [t_ms for t_ms, _ in self._ground_truth_markers]

        self._canvases: list[FeatureTimelineWidget] = []
        self._lanes: dict[str, FeatureTimelineWidget] = {}
        self._lane_status = {m: f"{_LANES[m]['name']}: wartet …" for m in MODALITIES}

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._sync_cursor)
        self._timer.start()

        self._thread: QThread | None = None
        self._worker: _FeatureWorker | None = None
        QTimer.singleShot(50, self._start_worker)

    # ── Construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(8)

        header = QLabel(
            "Audio (violett)  ·  Video (orange)  ·  Events (hellblau)  ·  "
            "gestrichelt = gesetzte Grenzen  ·  Klick auf Timeline → Sprung im Video"
            + ("  ·  Ground Truth (grün)" if self._ground_truth_markers else "")
        )
        header.setStyleSheet("color:#555; font-size:11px;")
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:#ffffff;")

        lanes_widget = QWidget()
        lanes_widget.setStyleSheet("background:#ffffff;")
        self._lanes_layout = QVBoxLayout(lanes_widget)
        self._lanes_layout.setContentsMargins(0, 0, 0, 0)
        self._lanes_layout.setSpacing(2)

        if self._ground_truth_markers:
            gt = self._add_lane(
                f"  ►  Ground Truth ({len(self._ground_truth_markers)} gesetzte Grenzen)",
                "Ground Truth", _GROUND_TRUTH_COLOR, height=100,
            )
            gt.set_events(self._ground_truth_markers)

        for modality in MODALITIES:
            lane = _LANES[modality]
            self._lanes[modality] = self._add_lane(
                lane["section"], lane["label"], lane["color"], height=180
            )
        # A JSON parse, so the dots are up before the worker starts.
        self._lanes["events"].set_events(
            [(float(e.get("t_ms", 0.0)), str(e.get("type", "")))
             for e in self._session.input_events()]
        )

        self._lanes_layout.addStretch()
        scroll.setWidget(lanes_widget)
        root.addWidget(scroll, stretch=1)

        self._status = QLabel("Wird berechnet …")
        self._status.setStyleSheet("color:#666; font-size:10px;")  # gut lesbar auf Weiß
        root.addWidget(self._status)

        close_btn = QPushButton("Schließen")
        close_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#f0f0f0;}"
        )
        close_btn.clicked.connect(self.accept)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _add_lane(
        self, section: str, label: str, hex_color: str, *, height: int
    ) -> FeatureTimelineWidget:
        """Add a section header, a lane label and its canvas; return the canvas."""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color:#dddddd; margin:6px 0;")
        self._lanes_layout.addWidget(separator)

        section_label = QLabel(section)
        section_label.setFixedHeight(24)
        section_label.setStyleSheet(
            "color:#555; font-size:11px; font-style:italic;"
            "background:#ffffff; padding-left:6px;"
        )
        self._lanes_layout.addWidget(section_label)

        lane_label = QLabel(f"  {label}")
        lane_label.setFixedHeight(22)
        lane_label.setStyleSheet(
            f"color:{hex_color}; font-size:11px; font-weight:600; "
            f"background:#ffffff; border-left:3px solid {hex_color}; padding-left:6px;"
        )
        self._lanes_layout.addWidget(lane_label)

        canvas = FeatureTimelineWidget()
        canvas.setFixedHeight(height)
        canvas.set_boundaries(self._boundaries_ms)
        canvas.set_duration(self._duration_ms)
        canvas.seek_requested.connect(self._on_seek)
        self._lanes_layout.addWidget(canvas)
        self._canvases.append(canvas)
        return canvas

    # ── Worker lifecycle ─────────────────────────────────────────────────

    def _start_worker(self) -> None:
        """Kick off segmentation on a background thread."""
        self._set_duration(float(self._player.duration()))

        self._thread = QThread(self)
        self._worker = _FeatureWorker(self._session)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.lane_ready.connect(self._on_lane_ready)
        self._worker.lane_failed.connect(self._on_lane_failed)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()
        self._refresh_status()

    def _refresh_status(self) -> None:
        self._status.setText(
            "  " + "  ·  ".join(self._lane_status[m] for m in MODALITIES)
        )

    def _on_lane_ready(self, modality: str) -> None:
        """Draw one modality's curve and the boundaries it committed to."""
        evidence = None if self._worker is None else self._worker.evidence(modality)
        if evidence is None:                      # cancelled between emit and slot
            return
        canvas = self._lanes[modality]
        canvas.set_curve(
            evidence.times_s.tolist(), evidence.score.tolist(), _LANES[modality]["color"]
        )
        canvas.set_detected_boundaries([t * 1000.0 for t in evidence.boundaries_s])
        self._lane_status[modality] = (
            f"{_LANES[modality]['name']}: {len(evidence.boundaries_s)} Grenzen"
        )
        self._refresh_status()

    def _on_lane_failed(self, modality: str, message: str) -> None:
        self._lanes[modality].set_curve([], [], _LANES[modality]["color"])
        self._lane_status[modality] = f"{_LANES[modality]['name']}: Fehler ({message})"
        self._refresh_status()

    def _on_progress(self, modality: str, done: int, total: int) -> None:
        self._lane_status[modality] = f"{_LANES[modality]['name']}: {done}/{total}"
        self._refresh_status()

    # ── Playback ─────────────────────────────────────────────────────────

    def _set_duration(self, duration_ms: float) -> None:
        """
        Push a newly known duration to every lane. Every marker is drawn relative
        to it, so a lane left at 0 stays blank until the player probes the file.
        """
        if duration_ms <= 0 or duration_ms == self._duration_ms:
            return
        self._duration_ms = duration_ms
        for canvas in self._canvases:
            canvas.set_duration(duration_ms)

    def _sync_cursor(self) -> None:
        """Broadcast the playback position, and the duration once it is known."""
        self._set_duration(float(self._player.duration()))
        pos = float(self._player.position())
        for canvas in self._canvases:
            canvas.set_cursor(pos)

    def _on_seek(self, ms: float) -> None:
        """Seek the player and refresh its paused frame."""
        self._player.setPosition(int(ms))
        if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self._player.play()
            self._player.pause()

    def closeEvent(self, event) -> None:
        """
        Stop the cursor timer and the worker before closing, or it would keep
        paying for verdicts nobody is looking at. Cached verdicts survive.
        """
        self._timer.stop()
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            # A request in flight can still be blocked on the model; give it room.
            self._thread.wait(120_000)
        super().closeEvent(event)
