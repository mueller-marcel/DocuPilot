from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QBrush
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from docupilot.recording.session import RecordingSession

try:
    import librosa
    import numpy as np
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

_MARKER_TYPES = {"mouse_click", "key_press", "key_release", "mouse_scroll"}

_DOT_COLOR = {
    "av_started": "#534AB7",
    "av_stopped": "#534AB7",
    "input_started": "#1D9E75",
    "input_stopped": "#1D9E75",
    "recording_started": "#e24b4a",
    "recording_stopping": "#e24b4a",
    "recording_stopped": "#e24b4a",
    "mouse_click": "#D85A30",
    "mouse_scroll": "#D85A30",
    "mouse_move": "#aaaaaa",
    "key_press": "#BA7517",
    "key_release": "#BA7517",
}

_DEFAULT_COLOR_DOT = "#aaaaaa"

_PROMINENT_TYPES = {
    "av_started",
    "av_stopped",
    "input_started",
    "input_stopped",
    "recording_started",
    "recording_stopping",
    "recording_stopped",
    "mouse_click",
    "key_press",
    "key_release",
    "mouse_scroll",
}


def _fmt_ms(ms: float) -> str:
    """
    Displays time in the format HH:MM:SS.mmm.
    :param ms: Milliseconds as float.
    :return: A string in the format HH:MM:SS.mmm.
    """

    total_s = int(ms) // 1000
    return f"{total_s // 60:02d}:{total_s % 60:02d}.{int(ms) % 1000:03d}"


def _event_label(ev: dict) -> str:
    """
    Creates a label for an event.
    :param ev: The event to create a label for.
    :return: The label as string.
    """

    t = ev.get("type", "?")
    parts = [t]
    if t == "mouse_click":
        action = "↓" if ev.get("pressed") else "↑"
        parts.append(
            f"{ev.get('button', '').replace('Button.', '')}{action} ({ev.get('x', '?')},{ev.get('y', '?')})"
        )
    elif t in ("mouse_move", "mouse_scroll"):
        parts.append(f"({ev.get('x', '?')}, {ev.get('y', '?')})")
    elif t in ("key_press", "key_release"):
        parts.append(str(ev.get("key", "")))
    return "  ".join(parts)


class _MarkerSlider(QSlider):
    """
    Custom slider widget that supports visual markers.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initializes the slider widget.
        :param parent: The parent widget.
        """

        super().__init__(Qt.Orientation.Horizontal, parent)
        self._markers: list[float] = []

    def set_markers(self, fractions: list[float]) -> None:
        """
        Set the markers for the slider.
        :param fractions: The list of marker fractions.
        """

        self._markers = [f for f in fractions if 0.0 <= f <= 1.0]
        self.update()

    def paintEvent(self, paint_event) -> None:
        """
        Paint the event markers on the slider.
        :param paint_event: The paint event.
        :return: None
        """

        super().paintEvent(paint_event)

        if not self._markers:
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#D85A30"))
        painter.setPen(Qt.PenStyle.NoPen)
        cy = groove.center().y()

        for frac in self._markers:
            cx = groove.left() + int(frac * groove.width())
            painter.drawEllipse(cx - 3, cy - 3, 6, 6)
        painter.end()


class _EventRow(QWidget):
    """
    Class representing a row in an event view widget.
    """

    jumped = Signal(float)

    def __init__(self, ev: dict, parent: QWidget | None = None) -> None:
        """
        Initializes the event row widget.
        :param ev: The event data.
        :param parent: The parent widget.
        """

        super().__init__(parent)

        self.event_data = ev
        self._active = False
        self._prominent = ev.get("type") in _PROMINENT_TYPES

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f"Springe zu {_fmt_ms(ev.get('t_ms', 0.0))}")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        color = _DOT_COLOR.get(ev.get("type", ""), _DEFAULT_COLOR_DOT)
        dot = QLabel()
        dot.setFixedSize(8, 8)
        dot.setStyleSheet(f"background:{color}; border-radius:4px;")
        layout.addWidget(dot)

        type_label = QLabel(ev.get("type", "?"))
        if self._prominent:
            type_label.setStyleSheet("font-size:12px; font-weight:600; color:#222;")
        else:
            type_label.setStyleSheet("font-size:11px; color:#666;")
        layout.addWidget(type_label)

        if self._prominent:
            t = ev.get("type", "")
            detail = ""
            if t == "mouse_click":
                action = "↓" if ev.get("pressed") else "↑"
                detail = f"{ev.get('button', '').replace('Button.', '')}{action}"
            elif t in ("key_press", "key_release"):
                detail = str(ev.get("key", ""))
            elif t in ("mouse_scroll",):
                detail = f"({ev.get('x', '?')},{ev.get('y', '?')})"
            if detail:
                detail_label = QLabel(detail)
                detail_label.setStyleSheet("font-size:11px; color:#888;")
                layout.addWidget(detail_label)

        layout.addStretch()

        time_label = QLabel(_fmt_ms(ev.get("t_ms", 0.0)))
        time_label.setStyleSheet("font-size:10px; color:#bbb; font-family:monospace;")
        layout.addWidget(time_label)

        self._update_style(active=False)

    def mousePressEvent(self, mouse_event) -> None:
        """
        Handles the mouse press event.
        :param mouse_event: The mouse press event.
        :return: None
        """

        self.jumped.emit(float(self.event_data.get("t_ms", 0.0)))

    def set_active(self, active: bool, past: bool) -> None:
        """
        Set the active state of the event row.
        :param active: Whether the event row is active.
        :param past: Whether the event row is in the past.
        :return: None
        """

        if self._active == active:
            return

        self._active = active
        self._update_style(active=active, past=past)

    def _update_style(self, *, active: bool, past: bool = False) -> None:
        """
        Update the style of the event row.
        :param active: Whether the event row is active.
        :param past: Whether the event row is in the past.
        :return: None
        """

        if active:
            self.setStyleSheet(
                "background:#dbeafe; border-radius:5px;border-left:3px solid #4da3ff;"
            )
        elif past and not self._prominent:
            self.setStyleSheet("background:transparent; border:none; opacity:0.5;")
        else:
            self.setStyleSheet("background:transparent; border:none;")

        self.setProperty("active", active)


class _BoundaryDialog(QDialog):
    """
    The dialog for managing boundaries.
    """

    def __init__(self, boundaries: list[dict], parent: QWidget | None = None) -> None:
        """
        Initialize the dialog.
        :param boundaries: The list of boundaries.
        :param parent: The parent widget.
        """

        super().__init__(parent)

        self.setWindowTitle("Gesetzte Grenzen")
        self.setMinimumSize(420, 320)
        self.setModal(True)

        self._boundaries = list(boundaries)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel(f"{len(self._boundaries)} Grenze(n) gesetzt")
        header.setStyleSheet("font-size:13px; font-weight:600; color:#222;")
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setStyleSheet(
            "QListWidget{border:1px solid #ddd; border-radius:6px; font-size:12px;}"
            "QListWidget::item{padding:6px 10px;}"
            "QListWidget::item:selected{background:#dbeafe; color:#222;}"
        )
        layout.addWidget(self._list)
        self._refresh_list()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        delete_btn = QPushButton("Ausgewählte löschen")
        delete_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#c0392b;border:1px solid #e74c3c;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#fdf0ef;}"
        )
        delete_btn.clicked.connect(self._delete_selected)

        close_btn = QPushButton("Schließen")
        close_btn.setStyleSheet(
            "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
            "border-radius:6px;padding:5px 14px;font-size:12px;}"
            "QPushButton:hover{background:#f0f0f0;}"
        )
        close_btn.clicked.connect(self.accept)

        btn_row.addWidget(delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _refresh_list(self) -> None:
        """
        Refresh the list of boundaries.
        :return: None
        """

        self._list.clear()

        for i, b in enumerate(self._boundaries):
            t_ms = b.get("t_ms", 0.0)
            created = b.get("created_at_utc", "")[:19].replace("T", "  ")
            item = QListWidgetItem(f"#{i + 1}   {_fmt_ms(t_ms)}   —   {created} UTC")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)

    def _delete_selected(self) -> None:
        """
        Delete the selected boundaries.
        :return: None
        """
        selected = self._list.selectedItems()

        if not selected:
            return

        indices = sorted(
            {item.data(Qt.ItemDataRole.UserRole) for item in selected},
            reverse=True,
        )

        for index in indices:
            self._boundaries.pop(index)

        self._refresh_list()

    def get_boundaries(self) -> list[dict]:
        """
        Get the list of boundaries.
        :return: The list of boundaries.
        """

        return self._boundaries


class _RmsCanvas(QWidget):
    """
    Custom widget that paints the RMS energy curve on a timeline.
    Clicking on the canvas emits a seek_requested signal (position in ms).
    """

    seek_requested = Signal(float)

    _PAD_L = 52   # left padding (y-axis labels)
    _PAD_R = 16
    _PAD_T = 16
    _PAD_B = 32   # bottom padding (time labels)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rms: list[float] = []          # normalised RMS values [0..1]
        self._duration_ms: float = 0.0
        self._cursor_ms: float = 0.0
        self._boundaries: list[float] = []   # boundary times in ms
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------ data
    def set_data(self, rms_values: list[float], duration_ms: float) -> None:
        """Load normalised RMS values and total duration."""
        self._rms = rms_values
        self._duration_ms = duration_ms
        self.update()

    def set_cursor(self, pos_ms: float) -> None:
        """Move the playback cursor line."""
        self._cursor_ms = pos_ms
        self.update()

    def set_boundaries(self, boundaries_ms: list[float]) -> None:
        """Show ground-truth boundary lines."""
        self._boundaries = boundaries_ms
        self.update()

    # --------------------------------------------------------------- interaction
    def mousePressEvent(self, event) -> None:
        if self._duration_ms <= 0:
            return
        x = event.position().x()
        frac = (x - self._PAD_L) / max(1, self.width() - self._PAD_L - self._PAD_R)
        frac = max(0.0, min(1.0, frac))
        self.seek_requested.emit(frac * self._duration_ms)

    # --------------------------------------------------------------- painting
    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._rms:
            self._paint_empty()
            return

        w = self.width()
        h = self.height()
        pl, pr, pt, pb = self._PAD_L, self._PAD_R, self._PAD_T, self._PAD_B
        plot_w = w - pl - pr
        plot_h = h - pt - pb

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # --- background
        p.fillRect(0, 0, w, h, QColor("#1e1e2e"))
        p.fillRect(pl, pt, plot_w, plot_h, QColor("#12121a"))

        # --- horizontal grid lines + y-axis labels
        grid_pen = QPen(QColor("#2a2a40"))
        grid_pen.setWidth(1)
        p.setPen(grid_pen)
        label_font = QFont("monospace", 8)
        p.setFont(label_font)
        label_color = QColor("#666688")

        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = pt + plot_h - int(frac * plot_h)
            p.setPen(grid_pen)
            p.drawLine(pl, y, pl + plot_w, y)
            p.setPen(label_color)
            p.drawText(2, y + 4, pl - 6, 12, Qt.AlignmentFlag.AlignRight, f"{frac:.2f}")

        # --- time axis labels
        n_ticks = 6
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            x = pl + int(frac * plot_w)
            t_s = frac * self._duration_ms / 1000
            p.setPen(label_color)
            label = f"{int(t_s // 60):02d}:{t_s % 60:04.1f}"
            p.drawText(x - 24, h - pb + 4, 48, pb - 4, Qt.AlignmentFlag.AlignHCenter, label)
            tick_pen = QPen(QColor("#333355"))
            tick_pen.setWidth(1)
            p.setPen(tick_pen)
            p.drawLine(x, pt, x, pt + plot_h)

        # --- boundary lines
        if self._duration_ms > 0:
            for b_ms in self._boundaries:
                bx = pl + int(b_ms / self._duration_ms * plot_w)
                boundary_pen = QPen(QColor("#D85A30"))
                boundary_pen.setWidth(1)
                boundary_pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(boundary_pen)
                p.drawLine(bx, pt, bx, pt + plot_h)

        # --- RMS curve (filled area beneath)
        n = len(self._rms)
        xs = [pl + int(i / (n - 1) * plot_w) for i in range(n)]
        ys = [pt + plot_h - int(v * plot_h) for v in self._rms]

        # filled gradient area
        fill_color = QColor("#4da3ff")
        fill_color.setAlpha(50)
        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint
        poly_points = [QPoint(xs[0], pt + plot_h)]
        for x, y in zip(xs, ys):
            poly_points.append(QPoint(x, y))
        poly_points.append(QPoint(xs[-1], pt + plot_h))
        p.setBrush(QBrush(fill_color))
        p.setPen(Qt.PenStyle.NoPen)
        poly = QPolygon(poly_points)
        p.drawPolygon(poly)

        # curve line
        curve_pen = QPen(QColor("#4da3ff"))
        curve_pen.setWidth(2)
        p.setPen(curve_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(n - 1):
            p.drawLine(xs[i], ys[i], xs[i + 1], ys[i + 1])

        # --- local minima markers (potential boundaries)
        min_pen = QPen(QColor("#1D9E75"))
        min_pen.setWidth(1)
        p.setPen(min_pen)
        p.setBrush(QBrush(QColor("#1D9E75")))
        for i in range(1, n - 1):
            if self._rms[i] < self._rms[i - 1] and self._rms[i] < self._rms[i + 1]:
                # only draw significant minima (below 30 % of peak)
                if self._rms[i] < 0.30:
                    p.drawEllipse(xs[i] - 3, ys[i] - 3, 6, 6)

        # --- playback cursor
        if self._duration_ms > 0:
            cx = pl + int(self._cursor_ms / self._duration_ms * plot_w)
            cursor_pen = QPen(QColor("#ffffff"))
            cursor_pen.setWidth(2)
            p.setPen(cursor_pen)
            p.drawLine(cx, pt, cx, pt + plot_h)

        p.end()

    def _paint_empty(self) -> None:
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), QColor("#1e1e2e"))
        p.setPen(QColor("#555"))
        p.setFont(QFont("sans-serif", 11))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Keine Audio-Daten verfügbar")
        p.end()


class _RmsVisualizerDialog(QDialog):
    """
    Modal dialog showing the RMS energy timeline for the current session.
    The playback cursor is synced live with the video player.
    Clicking on the timeline seeks the video.
    """

    def __init__(
        self,
        player: QMediaPlayer,
        session_path,
        duration_ms: float,
        boundaries: list[dict],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("RMS-Energieverlauf · Audio-Visualisierung")
        self.setMinimumSize(820, 340)
        self.resize(1000, 380)
        self.setModal(False)  # non-modal → bleibt offen beim Scrubben

        self._player = player
        self._duration_ms = duration_ms
        self._rms_values: list[float] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # --- header
        header = QLabel("RMS-Energieverlauf  ·  grüne Punkte = lokale Minima  ·  orange gestrichelt = gesetzte Grenzen")
        header.setStyleSheet("color:#aaa; font-size:11px;")
        layout.addWidget(header)

        # --- legend row
        legend = QHBoxLayout()
        for color, text in [
            ("#4da3ff", "RMS-Energie"),
            ("#1D9E75", "Lokales Minimum (< 30 %)"),
            ("#D85A30", "Gesetzte Grenze"),
            ("#ffffff", "Playback-Cursor"),
        ]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color:{color}; font-size:14px;")
            lbl = QLabel(text)
            lbl.setStyleSheet("color:#888; font-size:11px; margin-right:14px;")
            legend.addWidget(dot)
            legend.addWidget(lbl)
        legend.addStretch()
        layout.addLayout(legend)

        # --- canvas
        self._canvas = _RmsCanvas()
        self._canvas.seek_requested.connect(self._on_seek)
        layout.addWidget(self._canvas, stretch=1)

        # --- status bar
        self._status = QLabel("Wird berechnet …")
        self._status.setStyleSheet("color:#666; font-size:10px;")
        layout.addWidget(self._status)

        # --- close button
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
        layout.addLayout(btn_row)

        # set boundaries
        b_ms = [b.get("t_ms", 0.0) for b in boundaries]
        self._canvas.set_boundaries(b_ms)

        # live cursor timer
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._sync_cursor)
        self._timer.start()

        # load rms asynchronously via QTimer (keeps UI responsive)
        self._session_path = session_path
        QTimer.singleShot(50, lambda: self._compute_rms())

    def _compute_rms(self) -> None:
        """Extract RMS from the audio file and hand it to the canvas."""
        if not _LIBROSA_AVAILABLE:
            self._status.setText("librosa nicht installiert – bitte 'pip install librosa' ausführen.")
            return

        try:
            import numpy as np
            audio, sr = librosa.load(str(self._session_path), mono=True)
            rms = librosa.feature.rms(y=audio, hop_length=512)[0]  # shape (T,)

            # normalise to [0, 1]
            peak = float(rms.max()) if rms.max() > 0 else 1.0
            normalised = (rms / peak).tolist()
            self._rms_values = normalised
            self._canvas.set_data(normalised, self._duration_ms)

            # stats for status bar
            mean_val = float(np.mean(rms))
            n_minima = sum(
                1 for i in range(1, len(normalised) - 1)
                if normalised[i] < normalised[i - 1]
                and normalised[i] < normalised[i + 1]
                and normalised[i] < 0.30
            )
            hop_ms = 512 / (sr / 1000)
            self._status.setText(
                f"  {len(normalised)} Frames  ·  Fenster ~{hop_ms:.0f} ms  ·  "
                f"Ø RMS {mean_val:.4f}  ·  {n_minima} lokale Minima (< 30 % Peak)"
            )
        except Exception as exc:
            self._status.setText(f"Fehler beim Laden: {exc}")

    def _sync_cursor(self) -> None:
        pos = float(self._player.position())
        self._canvas.set_cursor(pos)

    def _on_seek(self, ms: float) -> None:
        self._player.setPosition(int(ms))
        if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self._player.play()
            self._player.pause()

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)


class AnnotationWindow(QWidget):
    """
    Annotation page shown after a recording has been stopped.
    """

    back_requested = Signal()
    TOLERANCE_MS: float = 250.0

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the AnnotationWindow.
        :param parent: The parent widget.
        """

        super().__init__(parent)

        self._session: RecordingSession | None = None
        self._events: list[dict] = []
        self._event_rows: list[_EventRow] = []
        self._boundaries: list[dict] = []
        self._boundary_path: Path | None = None
        self._duration_ms: float = 0.0
        self._rms_dialog: _RmsVisualizerDialog | None = None

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)

        self._build_ui()

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        self._player.mediaStatusChanged.connect(self._on_media_status_changed)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)

    def load_session(self, session: RecordingSession) -> None:
        """
        Load the recording session.
        :param session: The recording session.
        :return: None
        """

        self._session = session
        self._boundaries = []
        self._boundary_path = session.session_dir / "ground_truth.json"
        self._duration_ms = 0.0

        self._events = self._read_events(session.events_path)
        self._populate_event_panel()

        self._player.setSource(QUrl.fromLocalFile(str(session.recording_path)))

        self._dir_label.setText(session.session_dir.name)
        self._update_boundary_count()
        self._timer.start()

    def _build_ui(self) -> None:
        """
        Build the annotation window UI.
        :return: None
        """

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle{background:#ddd; width:1px;}")

        self._video_widget = QVideoWidget()
        self._video_widget.setStyleSheet("background:#111;")
        self._video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._player.setVideoOutput(self._video_widget)
        splitter.addWidget(self._video_widget)

        event_panel = self._build_event_panel()
        event_panel.setFixedWidth(280)
        splitter.addWidget(event_panel)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)

        root.addWidget(splitter, stretch=1)
        root.addWidget(self._build_bottom_bar())

    def _build_toolbar(self) -> QWidget:
        """
        Build the toolbar for the annotation window.
        :return: The widget containing the toolbar.
        """

        toolbar = QWidget()
        toolbar.setFixedHeight(46)
        toolbar.setStyleSheet("background:#f7f7f7; border-bottom:1px solid #ddd;")

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(10, 0, 12, 0)
        layout.setSpacing(8)

        back_btn = QPushButton("← Zurück")
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setStyleSheet(
            "QPushButton{background:none;border:none;color:#555;font-size:13px;}"
            "QPushButton:hover{color:#111;}"
        )
        back_btn.clicked.connect(self._on_back)

        reload_btn = QPushButton("⟳")
        reload_btn.setToolTip("Video neu laden")
        reload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reload_btn.setFixedSize(28, 28)
        reload_btn.setStyleSheet(
            "QPushButton{background:none;border:1px solid #ddd;border-radius:5px;"
            "color:#888;font-size:14px;}"
            "QPushButton:hover{border-color:#aaa;color:#333;}"
        )
        reload_btn.clicked.connect(self._reload_video)

        dir_container = QWidget()
        dir_layout = QHBoxLayout(dir_container)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(4)

        dir_icon = QLabel("📁")
        dir_icon.setStyleSheet("font-size:13px;")
        dir_layout.addWidget(dir_icon)

        self._dir_label = QLabel()
        self._dir_label.setStyleSheet("font-family:monospace; font-size:12px; color:#666;")
        dir_layout.addWidget(self._dir_label)

        self._boundary_count_label = QPushButton("0 Grenzen gesetzt")
        self._boundary_count_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._boundary_count_label.setStyleSheet(
            "QPushButton{background:none;border:none;color:#1a6fc4;"
            "font-size:12px;font-weight:600;}"
            "QPushButton:hover{color:#0a4f9c;text-decoration:underline;}"
        )
        self._boundary_count_label.clicked.connect(self._show_boundary_dialog)

        layout.addWidget(back_btn)
        layout.addWidget(reload_btn)
        layout.addSpacing(8)
        layout.addWidget(dir_container)
        layout.addStretch()

        self._rms_button = QPushButton("♪  RMS-Verlauf")
        self._rms_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._rms_button.setToolTip("Energieverlauf der Stimme auf Zeitleiste anzeigen")
        self._rms_button.setStyleSheet(
            "QPushButton{background:#e8f0fe;color:#1a6fc4;border:1px solid #c5d8f6;"
            "border-radius:5px;font-size:12px;font-weight:600;padding:0 12px;}"
            "QPushButton:hover{background:#d0e4ff;border-color:#4da3ff;}"
            "QPushButton:pressed{background:#bdd4f8;}"
        )
        self._rms_button.clicked.connect(self._show_rms_visualizer)
        layout.addWidget(self._rms_button)
        layout.addSpacing(6)

        layout.addWidget(self._boundary_count_label)
        return toolbar

    def _build_event_panel(self) -> QWidget:
        """
        Build the event panel.
        :return: The widget containing the event panel.
        """

        panel = QFrame()
        panel.setStyleSheet("QFrame{background:#fafafa;}")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QLabel(f"  Events  ·  ±{int(self.TOLERANCE_MS)} ms  ·  klicken zum Springen")
        header.setFixedHeight(32)
        header.setStyleSheet(
            "background:#f0f0f0; color:#777; font-size:11px;"
            "border-bottom:1px solid #e0e0e0; padding-left:4px;"
        )

        layout.addWidget(header)

        self._event_scroll = QScrollArea()
        self._event_scroll.setWidgetResizable(True)
        self._event_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._event_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._event_container = QWidget()
        self._event_container.setStyleSheet("background:#fafafa;")
        self._event_vbox = QVBoxLayout(self._event_container)
        self._event_vbox.setContentsMargins(4, 4, 4, 4)
        self._event_vbox.setSpacing(1)
        self._event_vbox.addStretch()

        self._event_scroll.setWidget(self._event_container)
        layout.addWidget(self._event_scroll)

        return panel

    def _build_bottom_bar(self) -> QWidget:
        """
        Build the bottom bar for the annotation window.
        :return: The widget containing the bottom bar.
        """

        bar = QWidget()
        bar.setFixedHeight(90)
        bar.setStyleSheet("background:#f7f7f7; border-top:1px solid #ddd;")

        layout = QVBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(6)

        progress_row = QHBoxLayout()
        self._time_label = QLabel("00:00.000")
        self._time_label.setStyleSheet(
            "color:#555; font-size:12px; font-family:monospace; min-width:62px;"
        )

        self._slider = _MarkerSlider()
        self._slider.setRange(0, 1000)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#ddd;border-radius:2px;}"
            "QSlider::sub-page:horizontal{background:#4da3ff;border-radius:2px;}"
            "QSlider::handle:horizontal{width:12px;height:12px;margin:-4px 0;"
            "background:#4da3ff;border-radius:6px;border:2px solid #f7f7f7;}"
        )

        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._duration_label = QLabel("00:00.000")
        self._duration_label.setStyleSheet(
            "color:#999; font-size:12px; font-family:monospace; min-width:62px;"
        )

        progress_row.addWidget(self._time_label)
        progress_row.addWidget(self._slider)
        progress_row.addWidget(self._duration_label)
        layout.addLayout(progress_row)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(6)

        btn_start = self._build_multimedia_button("⏮", "Zum Anfang")
        btn_back = self._build_multimedia_button("◀ 5s", "5 s zurück")
        self._play_button = self._build_multimedia_button("▶", "Play / Pause")
        btn_fwd = self._build_multimedia_button("5s ▶", "5 s vor")
        btn_end = self._build_multimedia_button("⏭", "Zum Ende")

        btn_start.clicked.connect(lambda: self._seek_to(0))
        btn_back.clicked.connect(lambda: self._seek_relative(-5000))
        self._play_button.clicked.connect(self._toggle_play)
        btn_fwd.clicked.connect(lambda: self._seek_relative(5000))
        btn_end.clicked.connect(lambda: self._seek_to(self._duration_ms))

        self._boundary_button = QPushButton("✂  Grenze setzen")
        self._boundary_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._boundary_button.setFixedHeight(30)
        self._apply_boundary_idle_style()
        self._boundary_button.clicked.connect(self._set_boundary)

        for b in (btn_start, btn_back, self._play_button, btn_fwd, btn_end):
            transport_row.addWidget(b)
        transport_row.addStretch()
        transport_row.addWidget(self._boundary_button)
        layout.addLayout(transport_row)

        return bar

    @staticmethod
    def _build_multimedia_button(text: str, tooltip: str) -> QPushButton:
        """
        Build the multimedia button (e.g., play/pause, start/end, etc.)
        :param text: The button text.
        :param tooltip: The button tooltip.
        :return: The push button
        """

        b = QPushButton(text)
        b.setToolTip(tooltip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFixedHeight(28)
        b.setStyleSheet(
            "QPushButton{background:#fff;color:#333;border:1px solid #ccc;"
            "border-radius:5px;font-size:12px;padding:0 10px;}"
            "QPushButton:hover{background:#f0f0f0;border-color:#aaa;}"
            "QPushButton:pressed{background:#e8e8e8;}"
        )

        return b

    def _populate_event_panel(self) -> None:
        """
        Populate the event panel with the list of events.
        :return: None
        """

        for row in self._event_rows:
            self._event_vbox.removeWidget(row)
            row.deleteLater()

        self._event_rows.clear()

        for ev in self._events:
            row = _EventRow(ev, self._event_container)
            row.jumped.connect(self._seek_to)
            self._event_vbox.insertWidget(self._event_vbox.count() - 1, row)
            self._event_rows.append(row)

    def _on_tick(self) -> None:
        """
        Handle the tick event.
        :return: None
        """

        pos_ms = self._player.position()
        self._time_label.setText(_fmt_ms(float(pos_ms)))

        if self._duration_ms > 0:
            self._slider.blockSignals(True)
            self._slider.setValue(int(pos_ms / self._duration_ms * 1000))
            self._slider.blockSignals(False)

        self._highlight_events(float(pos_ms))

    def _highlight_events(self, pos_ms: float) -> None:
        """
        Highlight the events at the given position.
        :param pos_ms: The position in milliseconds.
        :return: None
        """

        tol = self.TOLERANCE_MS
        first_active: _EventRow | None = None

        for row in self._event_rows:
            ev_ms = row.event_data.get("t_ms", 0.0)
            active = abs(ev_ms - pos_ms) <= tol
            past = ev_ms < pos_ms - tol
            row.set_active(active, past)

            if active and first_active is None:
                first_active = row

        if first_active is not None:
            self._event_scroll.ensureWidgetVisible(first_active)

    def _on_duration_changed(self, duration_ms: int) -> None:
        """
        Handle the duration change event.
        :param duration_ms: The duration in milliseconds.
        :return: None
        """

        self._duration_ms = float(duration_ms)
        self._duration_label.setText(_fmt_ms(float(duration_ms)))

        if self._duration_ms > 0:
            fractions = [
                ev.get("t_ms", 0.0) / self._duration_ms
                for ev in self._events
                if ev.get("type") in _MARKER_TYPES
            ]

            self._slider.set_markers(fractions)

    def _on_playback_state_changed(self, state) -> None:
        """
        Handle the playback state change event.
        :param state: The playback state.
        :return: None
        """

        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_button.setText("⏸" if playing else "▶")

    def _on_media_status_changed(self, status) -> None:
        """
        Handles the media status change event.
        :param status: The media status.
        :return: None
        """

        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._player.play()
            self._player.pause()

    def _toggle_play(self) -> None:
        """
        Toggle the play/pause state of the video.
        :return: None
        """

        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _seek_to(self, ms: float) -> None:
        """
        Set the playback position to the given time in milliseconds.
        :param ms: The time in milliseconds.
        :return: None
        """

        self._player.setPosition(int(ms))

    def _seek_relative(self, delta_ms: int) -> None:
        """
        Seek relative to the current playback position by the given number of milliseconds.
        :param delta_ms: The number of milliseconds to seek relative to the current position.
        :return: None
        """

        target = max(0, min(self._player.position() + delta_ms, int(self._duration_ms)))
        self._player.setPosition(target)

    def _on_slider_moved(self, value: int) -> None:
        """
        Handles the slider value change event.
        :param value: The new slider value.
        :return: Nonw
        """

        if self._duration_ms > 0:
            self._player.setPosition(int(value / 1000 * self._duration_ms))
            if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._player.play()
                self._player.pause()

    def _reload_video(self) -> None:
        """
        Reload the video.
        :return: None
        """

        if self._session is not None:
            self._player.setSource(QUrl())
            self._player.setSource(QUrl.fromLocalFile(str(self._session.recording_path)))

    def _set_boundary(self) -> None:
        """
        Set a boundary at the current playback position.
        :return: None
        """

        pos_ms = float(self._player.position())
        self._boundaries.append(
            {
                "type": "boundary",
                "t_ms": pos_ms,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        self._write_boundaries()
        self._update_boundary_count()
        self._flash_boundary_button()

    def _show_boundary_dialog(self) -> None:
        """
        Show the boundary dialog.
        :return: None
        """

        dlg = _BoundaryDialog(self._boundaries, parent=self)
        dlg.exec()
        self._boundaries = dlg.get_boundaries()
        self._write_boundaries()
        self._update_boundary_count()

    def _write_boundaries(self) -> None:
        """
        Write the boundaries to the JSON file.
        :return: None
        """

        if self._boundary_path is None:
            return
        with self._boundary_path.open("w", encoding="utf-8") as fh:
            json.dump(self._boundaries, fh, ensure_ascii=False, indent=2)

    def _update_boundary_count(self) -> None:
        """
        Update the boundary count label.
        :return: None
        """

        boundary_count = len(self._boundaries)
        self._boundary_count_label.setText(f"{boundary_count} Grenze{'n' if boundary_count != 1 else ''} gesetzt")

    def _flash_boundary_button(self) -> None:
        """
        Flash the boundary button.
        :return: None
        """

        self._boundary_button.setStyleSheet(
            "QPushButton{background:#e8f5e9;color:#2e7d32;border:1.5px solid #4caf50;"
            "border-radius:5px;font-size:13px;font-weight:600;padding:0 16px;}"
        )

        QTimer.singleShot(600, self._apply_boundary_idle_style)

    def _apply_boundary_idle_style(self) -> None:
        """
        Apply the idle style to the boundary button.
        :return: None
        """
        self._boundary_button.setStyleSheet(
            "QPushButton{background:#fff3ee;color:#D85A30;border:1.5px solid #D85A30;"
            "border-radius:5px;font-size:13px;font-weight:600;padding:0 16px;}"
            "QPushButton:hover{background:#ffe8de;}"
            "QPushButton:pressed{background:#ffd6c4;}"
        )

    def _show_rms_visualizer(self) -> None:
        """
        Open (or re-focus) the RMS energy visualizer dialog for the current session.
        """
        if self._session is None:
            return

        # close stale dialog if session changed
        if self._rms_dialog is not None:
            self._rms_dialog.close()
            self._rms_dialog = None

        self._rms_dialog = _RmsVisualizerDialog(
            player=self._player,
            session_path=self._session.recording_path,
            duration_ms=self._duration_ms,
            boundaries=self._boundaries,
            parent=self,
        )
        self._rms_dialog.show()

    def _on_back(self) -> None:
        """
        Resets the player and timer.
        :return: None
        """

        self._player.stop()
        self._timer.stop()
        self.back_requested.emit()

    @staticmethod
    def _read_events(path: Path) -> list[dict]:
        """
        Read the events from the JSON file.
        :param path: The path to the JSON file.
        :return: A list of event dictionaries.
        """

        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []