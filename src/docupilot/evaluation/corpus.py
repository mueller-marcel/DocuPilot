"""
What a corpus directory contains, before anything is computed on it.

The window needs this to fill its table and to refuse a run it cannot do;
keeping the scan out of the widget lets a script make the same decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RECORDING_FILE = "recording.mp4"
GROUND_TRUTH_FILE = "ground_truth.json"
VIDEO_CACHE_FILE = "gui_vlm_cache.json"
AUDIO_CACHE_FILE = "audio_llm_cache.json"


@dataclass(frozen=True)
class SessionInfo:
    """One session directory as the corpus scan saw it."""

    directory: Path
    n_boundaries: int | None
    """Annotated "end" boundaries; None when there is no ground truth file."""
    n_start_boundaries: int
    """Annotated "start" boundaries (the exposé's definition), if any."""
    has_video_cache: bool
    has_audio_cache: bool

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def annotated(self) -> bool:
        return self.n_boundaries is not None


@dataclass(frozen=True)
class CorpusScan:
    """Every candidate session under a root, and which of them can be evaluated."""

    root: Path
    sessions: list[SessionInfo]

    @property
    def usable(self) -> list[Path]:
        """Directories that can be scored — a session without ground truth cannot."""
        return [s.directory for s in self.sessions if s.annotated]

    @property
    def without_video_cache(self) -> int:
        """Sessions whose first run will pay for VLM calls."""
        return sum(1 for s in self.sessions if not s.has_video_cache)

    @property
    def start_definition_available(self) -> bool:
        """True when EVERY usable session also carries "start" boundaries, so
        the definition sensitivity can be run on the same corpus."""
        usable = [s for s in self.sessions if s.annotated]
        return bool(usable) and all(s.n_start_boundaries > 0 for s in usable)

    @property
    def can_evaluate(self) -> bool:
        """
        Whether the corpus supports leave-one-session-out at all.

        One session cannot provide a fold that excludes the session being
        scored, so the run is refused rather than faked.
        """
        return len(self.usable) >= 2


def describe(scan: CorpusScan) -> str:
    """
    One line stating what the corpus offers and what it will cost — the text the
    window shows above the session table.

    Here rather than in the widget so the wording is testable without a display,
    and so a script can print the same summary.
    """
    usable = len(scan.usable)
    skipped = len(scan.sessions) - usable
    parts = [f"{scan.root}  ·  {usable} Sessions verwendbar"]
    if skipped:
        parts.append(f"{skipped} ohne Ground Truth übersprungen")
    parts.append(
        f"⚠ {scan.without_video_cache} ohne Video-Cache — dieser Lauf erzeugt neue "
        f"VLM-Aufrufe (Größenordnung ~50 je Session)"
        if scan.without_video_cache else
        "vollständig gecacht, keine Modellkosten"
    )
    parts.append(
        "Definition \"Beginn\" vollständig annotiert — Sensitivitätslauf aktiv"
        if scan.start_definition_available else
        "Definition \"Beginn\" nicht in allen Sessions annotiert — kein Sensitivitätslauf"
    )
    if not scan.can_evaluate:
        parts.append("⚠ mindestens zwei Sessions nötig (Leave-one-session-out)")
    return "  ·  ".join(parts)


def _count_kinds(ground_truth: Path) -> tuple[int, int]:
    """(end, start) boundary counts of one ground_truth.json."""
    from docupilot.recording.session import RecordingSession

    entries = json.loads(ground_truth.read_text(encoding="utf-8"))
    kinds = [RecordingSession.boundary_kind(e) for e in entries]
    return kinds.count("end"), kinds.count("start")


def scan(root: Path) -> CorpusScan:
    """
    List the sessions under `root`.

    Either a folder of session directories, or one session picked directly — the
    latter is how a single recording gets looked at without moving it.

    :param root: the chosen directory.
    :return: the scan; sessions in name order.
    """
    candidates = (
        [root] if (root / RECORDING_FILE).exists()
        else sorted(
            d for d in root.iterdir()
            if d.is_dir() and (d / RECORDING_FILE).exists()
        )
    )
    sessions = []
    for directory in candidates:
        ground_truth = directory / GROUND_TRUTH_FILE
        n_end, n_start = _count_kinds(ground_truth) if ground_truth.exists() else (None, 0)
        sessions.append(SessionInfo(
            directory=directory,
            n_boundaries=n_end,
            n_start_boundaries=n_start,
            has_video_cache=(directory / VIDEO_CACHE_FILE).exists(),
            has_audio_cache=(directory / AUDIO_CACHE_FILE).exists(),
        ))
    return CorpusScan(root=root, sessions=sessions)
