"""
Action boundaries from the screen recording alone: pHash tile activity finds the
frames where the screen stands still, and a VLM judges each settled state against
an ANCHOR holding the last established one.

Reads only the video stream, so the modality stays independent for the ablation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    BoundaryEvidence,
    apply_gaussian,
)

MODALITY = "video"

_PHASH_SIZE = 8
_ACTIVITY_GRID = 8

# Swept on session_07 for the cheapest setting at full recall (41 proposals, 8/8).
# A missed action is unrecoverable — the VLM never sees it. Recall here,
# precision in the VLM.
_ACTIVITY_QUIET = 0.08   # per-tile pHash distance below which a frame is "still"
_MIN_DWELL_S = 0.5       # a still run this long is a settled state

# How far INTO a dwell the frame shown to the model is sampled; the first still
# frame can still show a fading animation.
_SETTLE_OFFSET_S = 0.2

# Mean grey-level difference above which a tile counts as changed, for the
# Set-of-Mark box only. Not tuned — the box only has to point, not to decide.
_PIXEL_CHANGE_EPS = 1.0

_SPREAD_S = 1.0

# Budget cap: a flickering recording could fragment into thousands of dwells.
_MAX_CALLS = 400


def _phash(gray: np.ndarray):
    import imagehash
    from PIL import Image

    return imagehash.phash(Image.fromarray(gray), hash_size=_PHASH_SIZE)


def _tiles(gray: np.ndarray) -> tuple:
    """One pHash per cell of the activity grid."""
    h, w = gray.shape[:2]
    g = _ACTIVITY_GRID
    return tuple(
        _phash(gray[r * h // g:(r + 1) * h // g, c * w // g:(c + 1) * w // g])
        for r in range(g)
        for c in range(g)
    )


def _distance(a: tuple, b: tuple) -> float:
    """Largest per-tile pHash distance, normalised to [0, 1]."""
    return max(float(x - y) for x, y in zip(a, b)) / (_PHASH_SIZE ** 2)


def _changed_region(
    before: np.ndarray, after: np.ndarray
) -> tuple[float, float, float, float] | None:
    """
    Bounding box (x0, y0, x1, y1 in [0,1]) of the tiles whose PIXELS differ — the
    Set-of-Mark hint that tells the model where to look.

    Pixels, not pHash: on a low-texture tile the hash flips on codec noise alone.

    :return: None when no tile differs measurably.
    """
    import cv2

    g = _ACTIVITY_GRID
    a = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.int16)
    h, w = a.shape[:2]

    changed = [
        (r, c)
        for r in range(g)
        for c in range(g)
        if float(np.abs(
            a[r * h // g:(r + 1) * h // g, c * w // g:(c + 1) * w // g]
            - b[r * h // g:(r + 1) * h // g, c * w // g:(c + 1) * w // g]
        ).mean()) > _PIXEL_CHANGE_EPS
    ]
    if not changed:
        return None

    rows = [r for r, _ in changed]
    cols = [c for _, c in changed]
    return (min(cols) / g, min(rows) / g, (max(cols) + 1) / g, (max(rows) + 1) / g)


def _fps(video_path: str) -> float:
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        cap.release()


def _frame_times_s(video_path: str, n_frames: int) -> np.ndarray:
    """
    Real capture time of each frame in seconds, read straight from the MP4.

    The recorder encodes with real wall-clock timestamps (VFR), so each frame's
    presentation time IS the moment it was captured — the same clock the events
    run on. ffprobe reads those times exactly from the container (unlike OpenCV's
    POS_MSEC, which some builds derive from an assumed frame rate). The origin is
    shifted to the first frame so it matches the event clock's zero.

    :raises RuntimeError: when the times cannot be read or do not match the frames.
    """
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    ).stdout
    times = [float(line.split(",")[0]) for line in out.splitlines() if line.strip()]
    if len(times) != n_frames:
        raise RuntimeError(
            f"ffprobe lieferte {len(times)} Frame-Zeiten, dekodiert wurden "
            f"{n_frames} — inkonsistente Aufnahme."
        )
    t = np.asarray(times, dtype=np.float64)
    return t - t[0]


def _scan(video_path: str) -> tuple[int, np.ndarray]:
    """Stream the video once; return the frame count and the per-frame activity."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, np.zeros(0, dtype=np.float32)

    hashes: list[tuple] = []
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            hashes.append(_tiles(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
    finally:
        cap.release()

    activity = np.zeros(len(hashes), dtype=np.float32)
    for i in range(1, len(hashes)):
        activity[i] = _distance(hashes[i - 1], hashes[i])
    return len(hashes), activity


def _dwells(activity: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    """Maximal runs of still frames lasting at least min_frames, inclusive."""
    out: list[tuple[int, int]] = []
    i, n = 0, len(activity)
    while i < n:
        if activity[i] < _ACTIVITY_QUIET:
            j = i
            while j < n and activity[j] < _ACTIVITY_QUIET:
                j += 1
            if j - i >= min_frames:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def _read_frames(video_path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """
    Fetch the given frames in one linear pass, downscaled on read.

    Sequential decoding beats seeking on a long-GOP MP4, and full-size frames
    would hold ~0.5 GB in memory for nothing — the model only sees the downscale.
    """
    import cv2

    from docupilot.segmentation.video_scoring import downscale

    frames: dict[int, np.ndarray] = {}
    if not wanted:
        return frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames
    last = max(wanted)
    try:
        idx = 0
        while idx <= last:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx in wanted:
                frames[idx] = downscale(bgr)
            idx += 1
    finally:
        cap.release()
    return frames


def extract(
    session: RecordingSession,
    *,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> BoundaryEvidence:
    """
    Action boundaries from the video frames alone: dwells -> a VLM judges each
    against the anchor -> the anchor advances only on an accepted boundary.

    Runs for minutes; must not be called on a UI thread.

    :param session: session whose .recording_path points at the MP4.
    :param use_cache: reuse verdicts cached in the session directory.
    :param on_progress: called as (judged_pairs, total_pairs) after each pair.
    :param is_cancelled: polled before each pair; True stops early and keeps the
        evidence gathered so far.
    :raises RuntimeError: when no VLM backend is reachable.
    """
    from docupilot.segmentation import video_scoring as vlm

    if not vlm.is_available():
        raise RuntimeError(
            f"Cloud-Modell '{vlm.MODEL}' nicht nutzbar. Benötigt:\n"
            "  poetry install         (Paket 'anthropic')\n"
            "  ANTHROPIC_API_KEY=...  oder  ant auth login\n\n"
            "Ohne VLM kann die Video-Modalität keine Handlungsgrenzen bestimmen — "
            "ein rein struktureller Score misst Pixelmenge, nicht Bedeutung."
        )

    video_path = str(session.recording_path)
    fps = _fps(video_path)
    n_frames, activity = _scan(video_path)
    times_s = _frame_times_s(video_path, n_frames)
    score = np.zeros(n_frames, dtype=np.float32)
    boundaries_s: list[float] = []
    if n_frames < 2:
        return BoundaryEvidence(times_s, score, boundaries_s)

    dwells = _dwells(activity, max(1, round(_MIN_DWELL_S * fps)))
    if len(dwells) < 2:
        return BoundaryEvidence(times_s, score, boundaries_s)

    # One settled frame per dwell, sampled _SETTLE_OFFSET_S in and downscaled on
    # read. The model sees them stitched into a BEFORE|AFTER composite.
    offset = max(0, int(_SETTLE_OFFSET_S * fps))
    settled = [min(end, start + offset) for start, end in dwells]
    halves = _read_frames(video_path, set(settled))

    cache = (
        vlm.Cache(session.session_dir / "gui_vlm_cache.json") if use_cache else None
    )

    anchor: int | None = None
    calls = 0
    spread = max(1, int(_SPREAD_S * fps))

    for (dwell_start, _), frame_idx in zip(dwells, settled):
        if frame_idx not in halves:
            continue
        if anchor is None:                       # the workflow's starting point
            anchor = frame_idx
            continue
        if calls >= _MAX_CALLS or (is_cancelled is not None and is_cancelled()):
            break

        # No changed region means the two states are pixel-identical: nothing to
        # judge, no call to pay for.
        region = _changed_region(halves[anchor], halves[frame_idx])
        if region is None:
            continue

        judgement = vlm.judge(
            vlm.encode_pair(halves[anchor], halves[frame_idx], region), cache=cache
        )
        calls += 1
        if on_progress is not None:
            on_progress(calls, max(1, len(dwells) - 1))
        if judgement is None:                    # unusable answer: leave no
            continue                             # evidence rather than guess

        apply_gaussian(score, dwell_start, judgement.p_boundary, spread)
        if judgement.p_boundary >= BOUNDARY_THRESHOLD:
            # Accepting the boundary and advancing the anchor are ONE decision.
            boundaries_s.append(float(times_s[dwell_start]))
            anchor = frame_idx

    if cache is not None:
        cache.flush()                # also on cancel: keep what we paid for

    return BoundaryEvidence(times_s, score, boundaries_s)
