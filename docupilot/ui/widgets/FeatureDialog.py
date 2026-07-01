from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
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
_EVENT_COLOR        = "#38bdf8"
_SEMANTIC_COLOR     = "#a78bfa"
_GUI_COLOR          = "#fb923c"


class FeatureDialog(QDialog):
    """
    Non-modal dialog with one timeline per audio/video feature plus a
    dedicated event timeline, built from FeatureTimelineWidget. Audio
    columns: RMS, pause duration, pitch reset, energy jump, speech rate.
    Shares cursor and boundary lines across all timelines; clicking a
    timeline seeks the player.

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
        event_markers: list[tuple[float, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """
        Build the dialog and start async feature computation.

        :param player: Media player providing playback position and seeking.
        :param session: Recording session to load features from. Liefert die
            Ground-Truth-Grenzen über session.ground_truth_markers().
        :param duration_ms: Total recording duration in milliseconds.
        :param event_markers: Pre-extracted (t_ms, event_type) markers.
        :param parent: Parent widget.
        :return: None
        """
        super().__init__(parent)
        self.setWindowTitle("Feature-Verläufe · Semantik & GUI · Visualisierung")
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

        self._event_markers: list[tuple[float, str]] = event_markers or []

        # Einzige Quelle für alle Grenzen-Darstellungen: die dashed Linien
        # (_boundaries_ms) UND die dedizierte Ground-Truth-Lane
        # (_ground_truth_markers) stammen aus demselben Aufruf, damit sie
        # niemals auseinanderlaufen können.
        self._ground_truth_markers: list[tuple[float, str]] = session.ground_truth_markers()
        self._boundaries_ms: list[float] = [t_ms for t_ms, _ in self._ground_truth_markers]

        self._canvases: list[FeatureTimelineWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(8)

        header_text = (
            "Semantik (violett)  ·  GUI-Zustandsänderungen (orange)  ·  "
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

        def _add_lane(label: str, hex_color: str, do_fill: bool, *, height: int = 140) -> FeatureTimelineWidget:
            """
            Create a labeled lane and append its canvas to self._canvases.

            :param label: Lane label text.
            :param hex_color: Lane accent color.
            :param do_fill: Whether the curve should be area-filled.
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
            c = FeatureTimelineWidget(event_color=_EVENT_COLOR)
            c.setFixedHeight(height)
            c.set_boundaries(self._boundaries_ms)
            c.seek_requested.connect(self._on_seek)
            lanes_layout.addWidget(c)
            self._canvases.append(c)
            return c

        # ── Ground Truth (4. Lane, nur falls Grenzen gesetzt wurden) ───────
        self._ground_truth_canvas: FeatureTimelineWidget | None = None
        if self._ground_truth_markers:
            _add_separator()
            _add_section_header(
                f"  ►  Ground Truth ({len(self._ground_truth_markers)} gesetzte Grenzen)"
            )
            self._ground_truth_canvas = _add_lane(
                "Ground Truth", _GROUND_TRUTH_COLOR, False, height=100
            )
            self._ground_truth_canvas.set_data([], self._duration_ms)
            self._ground_truth_canvas.set_events(self._ground_truth_markers)

        # ── Events ───────────────────────────────────────────────────────
        _add_separator()
        _add_section_header("  ►  Events (Klick / Taste / Scroll)")

        self._event_canvas = _add_lane("Events", _EVENT_COLOR, False)
        self._event_canvas.set_data([], self._duration_ms)
        self._event_canvas.set_events(self._event_markers)

        # ── Semantik ─────────────────────────────────────────────────────
        _add_separator()
        _add_section_header("  ►  Semantische Merkmale (NLI Boundary-Kandidaten)")

        self._semantic_canvas = _add_lane(
            "NLI-Score  +  Verb-Cluster  +  Boundary-Flag", _SEMANTIC_COLOR, False, height=180
        )

        # ── GUI ──────────────────────────────────────────────────────────
        _add_separator()
        _add_section_header("  ►  GUI-Handlungsgrenzen (OmniParser: neue & modifizierte Elemente)")

        self._gui_canvas = _add_lane(
            "GUI-Boundary-Score  +  Onset-Flag  +  Boundary-Flag", _GUI_COLOR, False, height=180
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

        QTimer.singleShot(50, self._compute_features)

    def _compute_features(self) -> None:
        """
        Extract semantic and GUI features and populate all canvas lanes.

        :return: None
        """
        try:
            import numpy as np
            import librosa
            from docupilot.segmentation.feature_extraction import (
                TranscriptionExtractor,
                SemanticAudioFeatureExtractor,
                GUIActionBoundaryExtractor,
            )

            # _compute_features läuft 50ms nach dem Öffnen des Dialogs — noch
            # bevor der erste _sync_cursor-Tick (80ms) die Dauer korrigieren
            # konnte. Deshalb hier zusätzlich direkt beim Player nachfragen,
            # damit hop_s (und damit alle Semantik-/GUI-Markerpositionen)
            # nicht versehentlich mit einer noch unbekannten Dauer von 0
            # berechnet werden.
            live_duration = float(self._player.duration())
            if live_duration > 0:
                self._duration_ms = live_duration
                for canvas in self._canvases:
                    canvas.set_duration(live_duration)

            # ── Semantische Features ──────────────────────────────────────────
            try:
                full_text, words = TranscriptionExtractor.extract_transcript(self._session)
                audio_raw, sr = librosa.load(str(self._session.recording_path))
                _HOP_LENGTH = 512
                T_sem = 1 + (len(audio_raw) - 2048) // _HOP_LENGTH
                semantic_features = SemanticAudioFeatureExtractor.extract_semantic_features(
                    full_text, words, T_sem, float(sr)
                )

                nli_score_signal = semantic_features[:, 1].tolist()
                cluster_frames = np.where(semantic_features[:, 0] > 0.5)[0]
                hop_s = (self._duration_ms / 1000.0) / T_sem if T_sem > 0 else 0.0
                cluster_markers_ms = [
                    (int(f) * hop_s * 1000.0, "verb_cluster") for f in cluster_frames
                ]
                flag_frames = np.where(semantic_features[:, 2] > 0.5)[0]
                flag_markers_ms = [
                    (int(f) * hop_s * 1000.0, "nli_flag") for f in flag_frames
                ]
                self._semantic_canvas.set_data(
                    [("NLI-Score", nli_score_signal, _SEMANTIC_COLOR, True)],
                    self._duration_ms,
                )
                self._semantic_canvas.set_semantic_markers(
                    cluster_markers=cluster_markers_ms,
                    flag_markers=flag_markers_ms,
                )
                self._status.setText(
                    f"  Semantik: {len(flag_frames)} NLI-Boundary-Kandidaten"
                )
            except Exception as sem_exc:
                self._semantic_canvas.set_data([], self._duration_ms)
                self._status.setText(f"  Semantik: Fehler ({sem_exc})")

            # ── GUI-Zustandsänderungen (OmniParser) ───────────────────────────
            try:
                import cv2 as _cv2
                _cap = _cv2.VideoCapture(str(self._session.recording_path))
                fps = _cap.get(_cv2.CAP_PROP_FPS) or 25.0
                T_v = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                _cap.release()

                gui_features = GUIActionBoundaryExtractor.extract_gui_features(
                    self._session,
                    fps=fps,
                )

                gui_score_signal = gui_features[:, 1].tolist()
                keyframe_frames = np.where(gui_features[:, 0] > 0.5)[0]
                hop_s_v = (self._duration_ms / 1000.0) / T_v if T_v > 0 else 0.0
                keyframe_markers_ms = [
                    (int(f) * hop_s_v * 1000.0, "onset") for f in keyframe_frames
                ]
                gui_flag_frames = np.where(gui_features[:, 2] > 0.5)[0]
                gui_flag_markers_ms = [
                    (int(f) * hop_s_v * 1000.0, "gui_boundary") for f in gui_flag_frames
                ]
                self._gui_canvas.set_data(
                    [("GUI-Boundary-Score", gui_score_signal, _GUI_COLOR, True)],
                    self._duration_ms,
                )
                self._gui_canvas.set_semantic_markers(
                    cluster_markers=keyframe_markers_ms,
                    flag_markers=gui_flag_markers_ms,
                )
                self._status.setText(
                    self._status.text()
                    + f"  ·  GUI: {len(gui_flag_frames)} Boundary-Kandidaten"
                )
            except Exception as gui_exc:
                self._gui_canvas.set_data([], self._duration_ms)
                self._status.setText(self._status.text() + f"  ·  GUI: Fehler ({gui_exc})")

        except Exception as exc:
            self._status.setText(f"Fehler beim Berechnen: {exc}")

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
        Stop the cursor timer before closing.

        :param event: The close event.
        :return: None
        """
        self._timer.stop()
        super().closeEvent(event)