from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QPainter
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
