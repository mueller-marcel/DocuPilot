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

_AUDIO_COLOR = "#ef4444"
_VIDEO_COLOR = "#22c55e"
_EVENT_COLOR = "#38bdf8"

_FEATURE_TRACKS: list[tuple[str, slice, str, bool]] = [
    ("RMS  (roh)", slice(0, 1), _AUDIO_COLOR, True),
    ("Pausendauer", slice(1, 2), _AUDIO_COLOR, True),
    ("Pitch-Reset", slice(2, 3), _AUDIO_COLOR, True),
    ("Energie-Sprung", slice(3, 4), _AUDIO_COLOR, True),
    ("Sprechtempo", slice(4, 5), _AUDIO_COLOR, True),
]

_VIDEO_FEATURE_TRACKS: list[tuple[str, int, str, bool]] = [
    ("ECR  (Kanten-änderung)",    0, _VIDEO_COLOR, True),
    ("ARR  (Flächen-änderung)",   1, _VIDEO_COLOR, True),
    ("pHash (Struktur-änderung)", 2, _VIDEO_COLOR, True),
    ("SSIM  (Wahrnehmung)",       3, _VIDEO_COLOR, True),
    ("ROI   (Titelleiste)",       4, _VIDEO_COLOR, True),
]


class FeatureDialog(QDialog):
    """
    Non-modal dialog with one timeline per audio/video feature plus a
    dedicated event timeline, built from FeatureTimelineWidget. Audio
    columns: RMS, pause duration, pitch reset, energy jump, speech rate.
    Shares cursor and boundary lines across all timelines; clicking a
    timeline seeks the player.
    """

    def __init__(
        self,
        player: QMediaPlayer,
        session: RecordingSession,
        duration_ms: float,
        boundaries: list[dict],
        event_markers: list[tuple[float, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """
        Build the dialog and start async feature computation.

        :param player: Media player providing playback position and seeking.
        :param session: Recording session to load features from.
        :param duration_ms: Total recording duration in milliseconds.
        :param boundaries: Ground-truth boundary dicts with a t_ms key.
        :param event_markers: Pre-extracted (t_ms, event_type) markers.
        :param parent: Parent widget.
        :return: None
        """
        super().__init__(parent)
        self.setWindowTitle("Feature-Verläufe · Audio & Video · Visualisierung")
        self.setMinimumSize(820, 700)
        self.resize(1100, 950)
        self.setModal(False)

        self._player = player
        self._session = session
        self._duration_ms = duration_ms
        self._boundaries_ms = [b.get("t_ms", 0.0) for b in boundaries]
        self._event_markers: list[tuple[float, str]] = event_markers or []
        self._canvases: list[FeatureTimelineWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(8)

        header = QLabel(
            "Audio-Features (rot) + Video-Features (grün)  ·  gestrichelt = gesetzte Grenzen  ·  "
            "eigene Zeitleiste unten = Events (hellblau)  ·  Klick auf Timeline → Sprung im Video"
        )
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

        self._n_audio_tracks = len(_FEATURE_TRACKS)
        for _label, _col_slice, hex_color, do_fill in _FEATURE_TRACKS:
            _add_lane(_label, hex_color, do_fill)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("color:#333355; margin:6px 0;")
        lanes_layout.addWidget(separator)
        vid_header = QLabel("  ►  Video-Features (ECR / ARR / pHash / SSIM / ROI)")
        vid_header.setFixedHeight(24)
        vid_header.setStyleSheet(
            "color:#aaa; font-size:11px; font-style:italic; background:#1e1e2e; padding-left:6px;"
        )
        lanes_layout.addWidget(vid_header)

        self._n_video_tracks = len(_VIDEO_FEATURE_TRACKS)
        for _label, _col_idx, hex_color, do_fill in _VIDEO_FEATURE_TRACKS:
            _add_lane(_label, hex_color, do_fill)

        separator_events = QFrame()
        separator_events.setFrameShape(QFrame.Shape.HLine)
        separator_events.setStyleSheet("color:#333355; margin:6px 0;")
        lanes_layout.addWidget(separator_events)
        event_header = QLabel("  ►  Events (Klick / Taste / Scroll)")
        event_header.setFixedHeight(24)
        event_header.setStyleSheet(
            "color:#aaa; font-size:11px; font-style:italic; background:#1e1e2e; padding-left:6px;"
        )
        lanes_layout.addWidget(event_header)

        self._event_canvas = _add_lane("Events", _EVENT_COLOR, False)
        self._event_canvas.set_data([], self._duration_ms)
        self._event_canvas.set_events(self._event_markers)

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
        Extract audio/video features and populate all canvas lanes.

        :return: None
        """
        try:
            from docupilot.segmentation.feature_extraction import (
                AudioFeatureExtractor,
                VideoFeatureExtractor,
            )
            import numpy as np

            audio_features: np.ndarray = AudioFeatureExtractor.extract_audio_features(
                self._session
            )
            T_a = audio_features.shape[0]

            audio_canvases = self._canvases[: self._n_audio_tracks]
            for canvas, (label, col_slice, hex_color, do_fill) in zip(
                audio_canvases, _FEATURE_TRACKS
            ):
                signal = audio_features[:, col_slice][:, 0]
                peak = float(signal.max()) if signal.max() > 0 else 1.0
                normalised = (signal / peak).tolist()
                canvas.set_data([(label, normalised, hex_color, do_fill)], self._duration_ms)

            video_features: np.ndarray = VideoFeatureExtractor.extract_video_features(
                self._session
            )
            T_v = video_features.shape[0]

            video_canvases = self._canvases[
                self._n_audio_tracks : self._n_audio_tracks + self._n_video_tracks
            ]
            for canvas, (label, col_idx, hex_color, do_fill) in zip(
                video_canvases, _VIDEO_FEATURE_TRACKS
            ):
                signal_raw = video_features[:, col_idx]

                if T_v >= 2 and T_a >= 2:
                    x_old = np.linspace(0.0, 1.0, T_v)
                    x_new = np.linspace(0.0, 1.0, T_a)
                    signal_resampled = np.interp(x_new, x_old, signal_raw)
                else:
                    signal_resampled = signal_raw

                peak = float(signal_resampled.max()) if signal_resampled.max() > 0 else 1.0
                normalised = (signal_resampled / peak).tolist()
                canvas.set_data([(label, normalised, hex_color, do_fill)], self._duration_ms)

            rms_col = audio_features[:, 0]
            n_minima = sum(
                1 for i in range(1, T_a - 1)
                if rms_col[i] < rms_col[i - 1]
                and rms_col[i] < rms_col[i + 1]
                and rms_col[i] < 0.30 * float(rms_col.max())
            )
            hop_ms = (self._duration_ms / T_a) if T_a > 0 else 0
            self._status.setText(
                f"  Audio: {T_a} Frames  ·  ~{hop_ms:.0f} ms/Frame  ·  "
                f"Ø RMS {float(np.mean(rms_col)):.4f}  ·  "
                f"{n_minima} RMS-Minima (< 30 % Peak)  ·  "
                f"Video: {T_v} Frames  ·  ECR / ARR / pHash / SSIM / ROI"
            )
        except Exception as exc:
            self._status.setText(f"Fehler beim Berechnen: {exc}")

    def _sync_cursor(self) -> None:
        """
        Broadcast the current playback position to all canvases.

        :return: None
        """
        pos = float(self._player.position())
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