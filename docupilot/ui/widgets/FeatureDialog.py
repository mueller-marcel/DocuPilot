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

from docupilot.ui.widgets.FeatureTimelineWidget import FeatureTimelineWidget

_GROUND_TRUTH_COLOR = "#34d399"
_AUDIO_COLOR        = "#a78bfa"
_GUI_COLOR          = "#fb923c"
# The lane's curve and its event dots must not drift apart into two blues.
_EVENT_COLOR        = FeatureTimelineWidget.EVENT_COLOR

_EVENTS, _AUDIO, _GUI = "events", "audio", "gui"


class _FeatureWorker(QObject):
    """
    Runs the extractors OFF the UI thread.

    This is not a nicety. Whisper takes minutes, and the GUI extractor sends one
    request per settled state pair to a VLM — on a CPU-only machine that is ~80 s
    each and dozens of them, so computing in the UI thread would freeze the dialog
    for the better part of an hour and Windows would mark it as not responding.
    The worker emits results; only the dialog touches widgets.
    """

    lane_ready   = Signal(str, object)   # (lane key, BoundaryEvidence)
    lane_failed  = Signal(str, str)      # (lane key, message)
    gui_progress = Signal(int, int)      # (judged pairs, total pairs)
    finished     = Signal()

    def __init__(self, session: RecordingSession, duration_ms: float) -> None:
        super().__init__()
        self._session = session
        self._duration_ms = duration_ms
        self._cancelled = False

    def cancel(self) -> None:
        """Ask the run to stop after the pair currently in flight."""
        self._cancelled = True

    def run(self) -> None:
        # Cheapest first, slowest last. Events are a JSON read; the GUI stage
        # costs one VLM call per state pair; Whisper takes minutes and blocks
        # nothing the others need — running it first would mean staring at two
        # empty lanes for the entire transcription.
        self._emit(_EVENTS, self._extract_events)
        self._emit(_GUI, self._extract_gui)
        if not self._cancelled:
            self._emit(_AUDIO, self._extract_audio)
        self.finished.emit()

    def _emit(self, lane: str, extract) -> None:
        try:
            self.lane_ready.emit(lane, extract())
        except Exception as exc:
            self.lane_failed.emit(lane, str(exc))

    def _extract_events(self):
        from docupilot.segmentation.feature_extraction import EventBoundaryExtractor

        return EventBoundaryExtractor.extract(self._session, self._duration_ms / 1000.0)

    def _extract_audio(self):
        import librosa

        from docupilot.segmentation.feature_extraction import (
            _HOP_LENGTH,
            AudioBoundaryExtractor,
            TranscriptionExtractor,
        )

        full_text, words = TranscriptionExtractor.extract_transcript(self._session)
        audio_raw, sr = librosa.load(str(self._session.recording_path))
        n_frames = 1 + (len(audio_raw) - 2048) // _HOP_LENGTH
        return AudioBoundaryExtractor.extract(
            self._session, full_text, words, n_frames, float(sr)
        )

    def _extract_gui(self):
        import cv2

        from docupilot.segmentation.feature_extraction import GUIActionBoundaryExtractor

        cap = cv2.VideoCapture(str(self._session.recording_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        return GUIActionBoundaryExtractor.extract(
            self._session,
            fps=fps,
            on_progress=lambda done, total: self.gui_progress.emit(done, total),
            is_cancelled=lambda: self._cancelled,
        )


class FeatureDialog(QDialog):
    """
    Non-modal dialog with one FeatureTimelineWidget lane per modality: events,
    audio and GUI, plus a ground-truth lane. This dialog is the ADAPTER — it runs
    the extractors off the UI thread and maps their BoundaryEvidence onto the
    lanes; the extraction lives entirely in feature_extraction, the drawing
    entirely in FeatureTimelineWidget. Shares cursor and boundary lines across
    all lanes; clicking a lane seeks the player.

    Ground-Truth-Grenzen werden ausschließlich aus
    session.ground_truth_markers() bezogen — sowohl für die gestrichelten
    Grenzlinien in allen Lanes als auch für die dedizierte Ground-Truth-Lane.
    Es gibt dadurch nur EINE Quelle: unabhängig davon, ob die Session gerade
    aufgezeichnet wurde oder über "Datei > Öffnen" geladen wurde, zeigt
    FeatureDialog exakt dieselben Grenzen. Die Lane erscheint nur, wenn
    mindestens eine Grenze gesetzt wurde.
    """

    def __init__(
        self,
        player: QMediaPlayer,
        session: RecordingSession,
        duration_ms: float,
        parent: QWidget | None = None,
    ) -> None:
        """
        Build the dialog and start async feature computation.

        :param player: Media player providing playback position and seeking.
        :param session: Recording session to load features from. Liefert die
            Ground-Truth-Grenzen über session.ground_truth_markers().
        :param duration_ms: Total recording duration in milliseconds.
        :param parent: Parent widget.
        :return: None
        """
        super().__init__(parent)
        self.setWindowTitle("Feature-Verläufe · Handlungsgrenzen je Modalität")
        self.setMinimumSize(820, 700)
        self.resize(1100, 950)
        self.setModal(False)

        self._player = player
        self._session = session
        self._duration_ms = duration_ms

        # Falls der Player die Dauer zum Zeitpunkt der Konstruktion bereits
        # kennt, direkt übernehmen — das erspart einen unnötigen Wartezyklus
        # bis zum ersten _sync_cursor-Tick. Ist sie noch nicht bekannt (0),
        # übernimmt _sync_cursor() bzw. _compute_features() die Korrektur
        # automatisch, sobald der Player sie ermittelt hat.
        live_duration = float(self._player.duration())
        if live_duration > 0:
            self._duration_ms = live_duration

        # Reading the raw input markers is a JSON parse, so the dots are on screen
        # before the worker starts. Their boundary EVIDENCE arrives later, with
        # the other lanes'.
        from docupilot.segmentation.feature_extraction import EventBoundaryExtractor

        self._event_markers = EventBoundaryExtractor.extract_markers(session)

        # Einzige Quelle für alle Grenzen-Darstellungen: die dashed Linien
        # (_boundaries_ms) UND die dedizierte Ground-Truth-Lane
        # (_ground_truth_markers) stammen aus demselben Aufruf, damit sie
        # niemals auseinanderlaufen können.
        self._ground_truth_markers: list[tuple[float, str]] = session.ground_truth_markers()
        self._boundaries_ms: list[float] = [t_ms for t_ms, _ in self._ground_truth_markers]

        self._canvases: list[FeatureTimelineWidget] = []
        self._lanes: dict[str, FeatureTimelineWidget] = {}
        self._lane_status: dict[str, str] = {
            _EVENTS: "Events: wartet …",
            _AUDIO:  "Audio: wartet …",
            _GUI:    "GUI: läuft …",
        }

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(8)

        header_text = (
            "Audio (violett)  ·  GUI-Zustandsänderungen (orange)  ·  "
            "Events (hellblau)  ·  gestrichelt = gesetzte Grenzen  ·  Klick auf Timeline → Sprung im Video"
        )
        if self._ground_truth_markers:
            header_text += "  ·  Ground Truth (grün)"

        header = QLabel(header_text)
        header.setStyleSheet("color:#aaa; font-size:11px;")
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:#1e1e2e;")

        lanes_widget = QWidget()
        lanes_widget.setStyleSheet("background:#1e1e2e;")
        lanes_layout = QVBoxLayout(lanes_widget)
        lanes_layout.setContentsMargins(0, 0, 0, 0)
        lanes_layout.setSpacing(2)

        def _add_separator() -> None:
            """
            Fügt eine dünne horizontale Trennlinie zwischen zwei Lane-Gruppen ein.

            Extrahiert, weil dieselben drei Zeilen vorher vor jeder Lane-Gruppe
            wiederholt wurden (DRY).

            :return: None
            """
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.HLine)
            separator.setStyleSheet("color:#333355; margin:6px 0;")
            lanes_layout.addWidget(separator)

        def _add_section_header(text: str) -> None:
            """
            Fügt eine kursive Sektions-Überschrift oberhalb einer Lane-Gruppe ein.

            :param text: Anzuzeigender Text, z. B. "  ►  Events (...)".
            :return: None
            """
            lbl = QLabel(text)
            lbl.setFixedHeight(24)
            lbl.setStyleSheet(
                "color:#aaa; font-size:11px; font-style:italic; background:#1e1e2e; padding-left:6px;"
            )
            lanes_layout.addWidget(lbl)

        def _add_lane(label: str, hex_color: str, *, height: int = 140) -> FeatureTimelineWidget:
            """
            Create a labeled lane and append its canvas to self._canvases.

            :param label: Lane label text.
            :param hex_color: Lane accent color.
            :param height: Lane height in pixels.
            :return: The created FeatureTimelineWidget.
            """
            lbl = QLabel(f"  {label}")
            lbl.setFixedHeight(22)
            lbl.setStyleSheet(
                f"color:{hex_color}; font-size:11px; font-weight:600; "
                f"background:#1e1e2e; border-left:3px solid {hex_color}; padding-left:6px;"
            )
            lanes_layout.addWidget(lbl)
            c = FeatureTimelineWidget()
            c.setFixedHeight(height)
            c.set_boundaries(self._boundaries_ms)
            c.seek_requested.connect(self._on_seek)
            lanes_layout.addWidget(c)
            self._canvases.append(c)
            return c

        # ── Ground Truth (nur falls Grenzen gesetzt wurden) ────────────────
        if self._ground_truth_markers:
            _add_separator()
            _add_section_header(
                f"  ►  Ground Truth ({len(self._ground_truth_markers)} gesetzte Grenzen)"
            )
            gt_canvas = _add_lane("Ground Truth", _GROUND_TRUTH_COLOR, height=100)
            gt_canvas.set_events(self._ground_truth_markers)

        # ── Events ───────────────────────────────────────────────────────
        _add_separator()
        _add_section_header("  ►  Events (Leerlauf + Wechsel der Eingabeart)")

        self._lanes[_EVENTS] = _add_lane(
            "Event-Score  ·  Klick / Taste / Scroll", _EVENT_COLOR, height=180
        )
        self._lanes[_EVENTS].set_events(self._event_markers)

        # ── Audio ────────────────────────────────────────────────────────
        _add_separator()
        _add_section_header("  ►  Audio (Ansage-Fenster, LLM-bewertet)")

        self._lanes[_AUDIO] = _add_lane(
            "Audio-Score  ·  Grenze im Fenster erwartet", _AUDIO_COLOR, height=180
        )

        # ── GUI ──────────────────────────────────────────────────────────
        _add_separator()
        _add_section_header(
            "  ►  GUI-Handlungsgrenzen (pHash-Zustände + VLM: neuer Zustand?)"
        )

        self._lanes[_GUI] = _add_lane(
            "GUI-Boundary-Score  ·  neuer Zustand?", _GUI_COLOR, height=180
        )

        lanes_layout.addStretch()
        scroll.setWidget(lanes_widget)
        root.addWidget(scroll, stretch=1)

        self._status = QLabel("Wird berechnet …")
        self._status.setStyleSheet("color:#666; font-size:10px;")
        root.addWidget(self._status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Schließen")
        close_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#f0f0f0;}"
        )
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._sync_cursor)
        self._timer.start()

        self._thread: QThread | None = None
        self._worker: _FeatureWorker | None = None
        QTimer.singleShot(50, self._start_worker)

    # ── Worker lifecycle ─────────────────────────────────────────────────

    def _start_worker(self) -> None:
        """Kick off feature extraction on a background thread."""
        # The player only knows the duration once it has probed the file; at
        # dialog-open time it can still report 0, which would put every marker
        # at t=0. Ask once more here.
        live_duration = float(self._player.duration())
        if live_duration > 0:
            self._duration_ms = live_duration
            for canvas in self._canvases:
                canvas.set_duration(live_duration)

        self._thread = QThread(self)
        self._worker = _FeatureWorker(self._session, self._duration_ms)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.lane_ready.connect(self._on_lane_ready)
        self._worker.lane_failed.connect(self._on_lane_failed)
        self._worker.gui_progress.connect(self._on_gui_progress)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()
        self._refresh_status()

    def _refresh_status(self) -> None:
        self._status.setText(
            "  " + "  ·  ".join(
                self._lane_status[k] for k in (_EVENTS, _AUDIO, _GUI)
            )
        )

    # ── Worker results ───────────────────────────────────────────────────

    _LANE_COLORS = {_EVENTS: _EVENT_COLOR, _AUDIO: _AUDIO_COLOR, _GUI: _GUI_COLOR}
    _LANE_NAMES  = {_EVENTS: "Events", _AUDIO: "Audio", _GUI: "GUI"}

    def _on_lane_ready(self, lane: str, evidence) -> None:
        """
        Draw one modality's evidence curve and the boundaries it committed to.

        Every lane is treated identically: the extractor already decided WHERE its
        boundaries are, because only it knows what its own curve means. This used
        to be two divergent rules here in the UI plus a threshold constant copied
        from feature_extraction by hand.
        """
        canvas = self._lanes[lane]
        canvas.set_curve(
            evidence.score.tolist(), self._LANE_COLORS[lane], self._duration_ms
        )
        canvas.set_detected_boundaries([t * 1000.0 for t in evidence.boundaries_s])

        name = self._LANE_NAMES[lane]
        n = len(evidence.boundaries_s)
        if lane == _GUI:
            from docupilot.segmentation import gui_state_scoring as vlm
            self._lane_status[lane] = f"GUI: {n} Grenzen [{vlm.MODEL}]"
            self._lane_status[_AUDIO] = "Audio: läuft …"
        else:
            self._lane_status[lane] = f"{name}: {n} Grenzen"
        self._refresh_status()

    def _on_lane_failed(self, lane: str, message: str) -> None:
        self._lanes[lane].set_curve([], self._LANE_COLORS[lane], self._duration_ms)
        self._lane_status[lane] = f"{self._LANE_NAMES[lane]}: Fehler ({message})"
        self._refresh_status()

    def _on_gui_progress(self, done: int, total: int) -> None:
        from docupilot.segmentation import gui_state_scoring as vlm

        self._lane_status[_GUI] = f"GUI: {done}/{total} Zustandspaare [{vlm.MODEL}]"
        self._refresh_status()

    def _sync_cursor(self) -> None:
        """
        Broadcast the current playback position to all canvases.

        Fragt bei jedem Tick zusätzlich die LIVE-Dauer des Players ab. War
        self._duration_ms beim Aufbau des Dialogs noch 0 (z. B. weil der
        Media-Player die Dauer einer gerade geöffneten Datei noch nicht
        ermittelt hatte), wird sie hier automatisch nachgetragen — an ALLE
        Lanes gleichzeitig, egal ob Aufnahme oder geöffnete Session. Ohne
        das würde jede Marker-Zeichnung (Grenzen, Events, Cursor, Semantik)
        für immer unsichtbar bleiben, da sie in FeatureTimelineWidget an
        "duration_ms > 0" gekoppelt ist.

        :return: None
        """
        pos = float(self._player.position())
        live_duration = float(self._player.duration())

        if live_duration > 0 and live_duration != self._duration_ms:
            self._duration_ms = live_duration
            for canvas in self._canvases:
                canvas.set_duration(live_duration)

        for canvas in self._canvases:
            canvas.set_cursor(pos)

    def _on_seek(self, ms: float) -> None:
        """
        Seek the player and refresh its paused frame.

        :param ms: Target position in milliseconds.
        :return: None
        """
        self._player.setPosition(int(ms))
        if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self._player.play()
            self._player.pause()

    def closeEvent(self, event) -> None:
        """
        Stop the cursor timer and the extraction thread before closing.

        Without this the worker would keep judging state pairs long after the
        dialog is gone — minutes to an hour of a local VLM running for a window
        nobody is looking at. Verdicts already paid for stay in the cache, so a
        cancelled run is not wasted: reopening the dialog resumes from there.

        :param event: The close event.
        :return: None
        """
        self._timer.stop()
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            # A pair in flight can still be blocked on the VLM; give it room.
            self._thread.wait(120_000)
        super().closeEvent(event)