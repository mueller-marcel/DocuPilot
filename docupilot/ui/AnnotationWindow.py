from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from docupilot.recording.session import RecordingSession

from docupilot.ui.event_formatting import MARKER_TYPES, format_ms
from docupilot.ui.widgets.EventsPanelWidget import EventsPanelWidget
from docupilot.ui.widgets.FeatureDialog import FeatureDialog


class _MarkerSlider(QSlider):
    """
    Slider widget that paints small dots over its groove for marker
    positions (used to highlight notable events along the timeline).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the slider widget.

        :param parent: The parent widget.
        :return: None
        """
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._markers: list[float] = []

    def set_markers(self, fractions: list[float]) -> None:
        """
        Set marker positions as fractions of the slider range.

        :param fractions: Marker positions in the range [0.0, 1.0].
        :return: None
        """
        self._markers = [f for f in fractions if 0.0 <= f <= 1.0]
        self.update()

    def paintEvent(self, paint_event) -> None:
        """
        Paint the slider, then overlay marker dots.

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


class _BoundaryDialog(QDialog):
    """
    Dialog for viewing and deleting ground-truth boundaries.
    """

    def __init__(self, boundaries: list[dict], parent: QWidget | None = None) -> None:
        """
        Initialize the dialog.

        :param boundaries: The list of boundaries.
        :param parent: The parent widget.
        :return: None
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
        Rebuild the list widget from self._boundaries.

        :return: None
        """
        self._list.clear()

        for i, b in enumerate(self._boundaries):
            t_ms = b.get("t_ms", 0.0)
            created = b.get("created_at_utc", "")[:19].replace("T", "  ")
            item = QListWidgetItem(f"#{i + 1}   {format_ms(t_ms)}   —   {created} UTC")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self._list.addItem(item)

    def _delete_selected(self) -> None:
        """
        Remove the selected boundaries and refresh the list.

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
        Get the current list of boundaries.

        :return: The list of boundaries.
        """
        return self._boundaries


class AnnotationWindow(QWidget):
    """
    Annotation page shown after a recording has been stopped. Orchestrates
    the player, toolbar, and bottom bar, and delegates to EventsPanelWidget
    and FeatureDialog rather than building either itself.

    Ground-Truth-Grenzen werden NICHT mehr lokal gehalten — sie leben
    ausschließlich in session.ground_truth_data. AnnotationWindow liest und
    schreibt sie über die dafür vorgesehenen RecordingSession-Methoden
    (load_ground_truth, add_ground_truth_boundary, set_ground_truth_boundaries).
    Dadurch verhält sich eine frisch aufgezeichnete Session exakt so wie eine
    über "Datei > Öffnen" geladene: beide teilen sich dieselbe Quelle.
    """

    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """
        Initialize the AnnotationWindow.

        :param parent: The parent widget.
        :return: None
        """
        super().__init__(parent)

        self._session: RecordingSession | None = None
        self._events: list[dict] = []
        self._duration_ms: float = 0.0
        self._feature_dialog: FeatureDialog | None = None

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
        Load a recording session into the player and event panel.

        Lädt zusätzlich vorhandene Ground-Truth-Grenzen aus
        session.ground_truth.json nach, damit bereits gesetzte Grenzen auch
        nach einem erneuten Öffnen der Session sichtbar sind — unabhängig
        davon, ob die Session gerade erst aufgenommen oder von der Platte
        geöffnet wurde.

        :param session: The recording session.
        :return: None
        """
        self._session = session
        self._duration_ms = 0.0

        session.load_ground_truth()

        self._events = self._read_events(session.events_path)
        self._events_panel.set_events(self._events)

        self._player.setSource(QUrl.fromLocalFile(str(session.recording_path)))

        self._dir_label.setText(session.session_dir.name)
        self._update_boundary_count()
        self._timer.start()

    def _build_ui(self) -> None:
        """
        Build the toolbar, video/event-panel splitter, and bottom bar.

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

        self._events_panel = EventsPanelWidget()
        self._events_panel.setFixedWidth(280)
        self._events_panel.jumped.connect(self._seek_to)
        splitter.addWidget(self._events_panel)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(0, 1)

        root.addWidget(splitter, stretch=1)
        root.addWidget(self._build_bottom_bar())

    def _build_toolbar(self) -> QWidget:
        """
        Build the top toolbar.

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

        self._features_button = QPushButton("♪  Features")
        self._features_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._features_button.setToolTip("Energieverlauf der Stimme auf Zeitleiste anzeigen")
        self._features_button.setStyleSheet(
            "QPushButton{background:#e8f0fe;color:#1a6fc4;border:1px solid #c5d8f6;"
            "border-radius:5px;font-size:12px;font-weight:600;padding:0 12px;}"
            "QPushButton:hover{background:#d0e4ff;border-color:#4da3ff;}"
            "QPushButton:pressed{background:#bdd4f8;}"
        )
        self._features_button.clicked.connect(self._show_feature_dialog)
        layout.addWidget(self._features_button)
        layout.addSpacing(6)

        layout.addWidget(self._boundary_count_label)
        return toolbar

    def _build_bottom_bar(self) -> QWidget:
        """
        Build the bottom bar with progress slider and transport controls.

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
        Build a styled transport button.

        :param text: The button text.
        :param tooltip: The button tooltip.
        :return: The push button.
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

    def _on_tick(self) -> None:
        """
        Update time label, progress slider, and event highlighting.

        :return: None
        """
        pos_ms = self._player.position()
        self._time_label.setText(format_ms(float(pos_ms)))

        if self._duration_ms > 0:
            self._slider.blockSignals(True)
            self._slider.setValue(int(pos_ms / self._duration_ms * 1000))
            self._slider.blockSignals(False)

        self._events_panel.highlight(float(pos_ms))

    def _on_duration_changed(self, duration_ms: int) -> None:
        """
        Update duration label and slider markers.

        :param duration_ms: The duration in milliseconds.
        :return: None
        """
        self._duration_ms = float(duration_ms)
        self._duration_label.setText(format_ms(float(duration_ms)))

        if self._duration_ms > 0:
            fractions = [
                ev.get("t_ms", 0.0) / self._duration_ms
                for ev in self._events
                if ev.get("type") in MARKER_TYPES
            ]

            self._slider.set_markers(fractions)

    def _on_playback_state_changed(self, state) -> None:
        """
        Update the play/pause button label.

        :param state: The playback state.
        :return: None
        """
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_button.setText("⏸" if playing else "▶")

    def _on_media_status_changed(self, status) -> None:
        """
        Pre-buffer a paused frame once media has loaded.

        :param status: The media status.
        :return: None
        """
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._player.play()
            self._player.pause()

    def _toggle_play(self) -> None:
        """
        Toggle play/pause.

        :return: None
        """
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _seek_to(self, ms: float) -> None:
        """
        Set the playback position.

        :param ms: The target time in milliseconds.
        :return: None
        """
        self._player.setPosition(int(ms))

    def _seek_relative(self, delta_ms: int) -> None:
        """
        Seek relative to the current playback position, clamped to bounds.

        :param delta_ms: Offset in milliseconds, may be negative.
        :return: None
        """
        target = max(0, min(self._player.position() + delta_ms, int(self._duration_ms)))
        self._player.setPosition(target)

    def _on_slider_moved(self, value: int) -> None:
        """
        Seek the player to match the dragged slider position.

        :param value: The new slider value.
        :return: None
        """
        if self._duration_ms > 0:
            self._player.setPosition(int(value / 1000 * self._duration_ms))
            if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._player.play()
                self._player.pause()

    def _reload_video(self) -> None:
        """
        Reload the current session's video source.

        :return: None
        """
        if self._session is not None:
            self._player.setSource(QUrl())
            self._player.setSource(QUrl.fromLocalFile(str(self._session.recording_path)))

    def _set_boundary(self) -> None:
        """
        Record a boundary at the current playback position.

        Delegiert das Anlegen und Speichern vollständig an
        session.add_ground_truth_boundary(), damit es nur EINE Stelle im
        Code gibt, die das Schema einer Grenze und die Persistenz kennt.
        Als Label wird der formatierte Zeitstempel verwendet, damit die
        Marker in der FeatureDialog-Timeline sofort identifizierbar sind.

        :return: None
        """
        if self._session is None:
            return

        pos_ms = float(self._player.position())
        self._session.add_ground_truth_boundary(pos_ms, label=format_ms(pos_ms))

        self._update_boundary_count()
        self._flash_boundary_button()

    def _show_boundary_dialog(self) -> None:
        """
        Open the boundary management dialog and apply its result.

        :return: None
        """
        if self._session is None:
            return

        dlg = _BoundaryDialog(self._session.ground_truth_data, parent=self)
        dlg.exec()
        self._session.set_ground_truth_boundaries(dlg.get_boundaries())
        self._update_boundary_count()

    def _update_boundary_count(self) -> None:
        """
        Refresh the boundary count label from session.ground_truth_data.

        :return: None
        """
        if self._session is None:
            return

        boundary_count = len(self._session.ground_truth_data)
        self._boundary_count_label.setText(f"{boundary_count} Grenze{'n' if boundary_count != 1 else ''} gesetzt")

    def _flash_boundary_button(self) -> None:
        """
        Briefly highlight the boundary button after setting a boundary.

        :return: None
        """
        self._boundary_button.setStyleSheet(
            "QPushButton{background:#e8f5e9;color:#2e7d32;border:1.5px solid #4caf50;"
            "border-radius:5px;font-size:13px;font-weight:600;padding:0 16px;}"
        )

        QTimer.singleShot(600, self._apply_boundary_idle_style)

    def _apply_boundary_idle_style(self) -> None:
        """
        Restore the boundary button's idle stylesheet.

        :return: None
        """
        self._boundary_button.setStyleSheet(
            "QPushButton{background:#fff3ee;color:#D85A30;border:1.5px solid #D85A30;"
            "border-radius:5px;font-size:13px;font-weight:600;padding:0 16px;}"
            "QPushButton:hover{background:#ffe8de;}"
            "QPushButton:pressed{background:#ffd6c4;}"
        )

    def _show_feature_dialog(self) -> None:
        """
        Open (or re-focus) the feature visualizer dialog.

        boundaries wird nicht mehr übergeben: FeatureDialog liest die
        Ground-Truth-Grenzen direkt aus session.ground_truth_data, damit es
        keine zweite Kopie mehr geben kann, die von der ersten abweicht.

        :return: None
        """
        if self._session is None:
            return

        if self._feature_dialog is not None:
            self._feature_dialog.close()
            self._feature_dialog = None

        from docupilot.segmentation.feature_extraction import EventFeatureExtractor

        event_markers = EventFeatureExtractor.extract_event_markers(self._session)

        self._feature_dialog = FeatureDialog(
            player=self._player,
            session=self._session,
            duration_ms=self._duration_ms,
            event_markers=event_markers,
            parent=self,
        )
        self._feature_dialog.show()

    def _on_back(self) -> None:
        """
        Stop playback and signal that the page should be left.

        :return: None
        """
        self._player.stop()
        self._timer.stop()
        self.back_requested.emit()

    @staticmethod
    def _read_events(path: Path) -> list[dict]:
        """
        Read events from a session's events.json file.

        :param path: Path to the events.json file.
        :return: List of event dictionaries, empty if missing or invalid.
        """
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []