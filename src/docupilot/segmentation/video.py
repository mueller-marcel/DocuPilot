"""
Action boundaries from the screen recording alone: pHash tile activity finds the
frames where the screen stands still, and a VLM judges each settled state against
an ANCHOR holding the last established one.

Reads only the video stream, so the modality stays independent for the ablation.

The module is layered so the decisions are separable from the decoding: the
activity signal is a function of a frame SEQUENCE, the dwell segmentation a
function of that signal, and the anchor walk a function of the dwells plus two
callables. Only `_scan`, `_read_frames` and `_frame_times_s` touch OpenCV and
ffprobe.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation import store
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    BoundaryEvidence,
    apply_gaussian,
)
from docupilot.segmentation.video_scoring import Judgement

MODALITY = "video"

_PHASH_SIZE = 8
_ACTIVITY_GRID = 8

# Swept on the development session (session_30) for the cheapest setting at full
# recall. A missed action is unrecoverable — the VLM never sees it. Recall here,
# precision in the VLM.
ACTIVITY_QUIET = 0.08    # per-tile pHash distance below which a frame is "still"
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

# The activity scan is the expensive, model-free half of this modality and is
# also what the synchronisation report reads. It is a pure function of the
# recording bytes and these two constants, so it is kept beside the recording
# and read back instead of decoding every frame again.
_ACTIVITY_FILE = "video_activity.npz"
_ACTIVITY_FORMAT = 1

Region = tuple[float, float, float, float]


@dataclass(frozen=True)
class ActivityScan:
    """Per-frame stillness of one recording, plus what is needed to place it in time."""

    n_frames: int
    activity: np.ndarray     # (n_frames,) float32 — largest per-tile pHash distance to the previous frame
    fps: float
    times_s: np.ndarray      # (n_frames,) float64 — capture time of each frame, first frame = 0


@dataclass(frozen=True)
class DwellVerdict:
    """What the model said about one dwell, and where that dwell begins."""

    dwell_start: int         # frame index the screen settled at — where the evidence goes
    judgement: Judgement


# ── Activity signal ───────────────────────────────────────────────────────────

def _phash(gray: np.ndarray):
    import imagehash
    from PIL import Image

    return imagehash.phash(Image.fromarray(gray), hash_size=_PHASH_SIZE)


def _tile_bounds(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """(row0, row1, col0, col1) of every cell of the activity grid, row-major."""
    g = _ACTIVITY_GRID
    return [
        (r * h // g, (r + 1) * h // g, c * w // g, (c + 1) * w // g)
        for r in range(g)
        for c in range(g)
    ]


def _tiles(gray: np.ndarray) -> tuple:
    """One pHash per cell of the activity grid."""
    h, w = gray.shape[:2]
    return tuple(_phash(gray[r0:r1, c0:c1]) for r0, r1, c0, c1 in _tile_bounds(h, w))


def _distance(a: tuple, b: tuple) -> float:
    """Largest per-tile pHash distance, normalised to [0, 1]."""
    return max(float(x - y) for x, y in zip(a, b)) / (_PHASH_SIZE ** 2)


def frame_activity(gray_frames: Iterable[np.ndarray]) -> np.ndarray:
    """
    The activity signal of a sequence of GREYSCALE frames: per frame, the largest
    per-tile pHash distance to its predecessor, normalised to [0, 1].

    Tiles rather than the whole frame because a meaningful change often covers a
    small region (one cell recoloured, an arrow appearing in a header) and would
    average away; the MAXIMUM over tiles rather than the mean for the same
    reason. The first frame has no predecessor and scores 0.

    Consumes the frames one at a time, so only two frames' hashes are alive at
    once rather than one 64-hash tuple per frame of the whole recording.

    :param gray_frames: the frames, in order.
    :return: (n_frames,) float32.
    """
    activity: list[float] = []
    previous: tuple | None = None
    for gray in gray_frames:
        current = _tiles(gray)
        activity.append(0.0 if previous is None else _distance(previous, current))
        previous = current
    return np.asarray(activity, dtype=np.float32)


def dwells(
    activity: np.ndarray, min_frames: int, quiet: float = ACTIVITY_QUIET
) -> list[tuple[int, int]]:
    """
    Maximal runs of still frames lasting at least `min_frames`, inclusive bounds.

    The reading direction is inverted on purpose: what is searched is the REST,
    not the change. On a screen recording the surface stands still between two
    operations, and that stillness is more robustly measured than the transition
    that ends it. The minimum duration is the first guard against
    over-segmentation — it discards rests too short to be a settled state.
    """
    out: list[tuple[int, int]] = []
    i, n = 0, len(activity)
    while i < n:
        if activity[i] < quiet:
            j = i
            while j < n and activity[j] < quiet:
                j += 1
            if j - i >= min_frames:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def settled_frames(dwell_list: Sequence[tuple[int, int]], fps: float) -> list[int]:
    """The frame shown to the model per dwell: `_SETTLE_OFFSET_S` into it, so a
    fading animation in the first still frame is not what gets judged."""
    offset = max(0, int(_SETTLE_OFFSET_S * fps))
    return [min(end, start + offset) for start, end in dwell_list]


# ── Changed region (the Set-of-Mark hint) ─────────────────────────────────────

def _gray16(bgr: np.ndarray) -> np.ndarray:
    """Grey levels as int16, so a difference cannot wrap around like uint8 would."""
    import cv2

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)


def changed_region(a: np.ndarray, b: np.ndarray) -> Region | None:
    """
    Bounding box (x0, y0, x1, y1 in [0,1]) of the tiles whose PIXELS differ
    between two greyscale frames — the hint that tells the model where to look.

    Pixels, not pHash: on a low-texture tile the hash flips on codec noise alone.

    :return: None when no tile differs measurably, which means the two states are
        effectively identical and there is nothing to pay a model call for.
    """
    g = _ACTIVITY_GRID
    h, w = a.shape[:2]
    rows: list[int] = []
    cols: list[int] = []
    for index, (r0, r1, c0, c1) in enumerate(_tile_bounds(h, w)):
        if float(np.abs(a[r0:r1, c0:c1] - b[r0:r1, c0:c1]).mean()) > _PIXEL_CHANGE_EPS:
            rows.append(index // g)
            cols.append(index % g)
    if not rows:
        return None
    return (min(cols) / g, min(rows) / g, (max(cols) + 1) / g, (max(rows) + 1) / g)


# ── The anchor walk: the modality's actual decision logic ─────────────────────

def walk_dwells(
    steps: Sequence[tuple[int, int]],
    has_change: Callable[[int, int], bool],
    judge: Callable[[int, int], Judgement | None],
    *,
    threshold: float = BOUNDARY_THRESHOLD,
    max_calls: int = _MAX_CALLS,
    on_call: Callable[[int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    on_limit: Callable[[], None] | None = None,
) -> list[DwellVerdict]:
    """
    Judge each settled state against an ANCHOR holding the last ESTABLISHED one.

    This is the core of the video modality. Comparing dwell *i* against dwell
    *i−1* would make every judgement local: opening a menu, stepping into a
    submenu and finally clicking the entry that changes something would produce
    three unrelated comparisons. Against an anchor, the intermediate states are
    rejected AND leave the anchor where it was, so the final question is "has
    anything been completed since the state the user started from?" — the
    question that matters.

    Setting a boundary and advancing the anchor are therefore ONE decision.

    :param steps: (dwell_start_frame, settled_frame) per dwell, in order. The
        first settled frame becomes the initial anchor and is never judged.
    :param has_change: (anchor, current) -> is there anything to compare at all.
        A False costs no model call, which is why it is separate from `judge`.
    :param judge: (anchor, current) -> the verdict, or None when the model gave
        no usable answer; that pair then gets no evidence rather than a guess.
    :param threshold: at or above this p_boundary the anchor advances.
    :param max_calls: budget for `judge` calls; `on_limit` is called when hit.
    :param on_call: called with the running call count after each judged pair.
    :param is_cancelled: polled before each pair; True stops and keeps what was
        gathered so far.
    :return: one entry per JUDGED pair that produced a verdict, in order.
    """
    verdicts: list[DwellVerdict] = []
    anchor: int | None = None
    calls = 0

    for dwell_start, current in steps:
        if anchor is None:                       # the workflow's starting point
            anchor = current
            continue
        if is_cancelled is not None and is_cancelled():
            break
        if calls >= max_calls:
            if on_limit is not None:
                on_limit()
            break
        if not has_change(anchor, current):
            continue

        judgement = judge(anchor, current)
        calls += 1
        if on_call is not None:
            on_call(calls)
        if judgement is None:
            continue

        verdicts.append(DwellVerdict(dwell_start, judgement))
        if judgement.p_boundary >= threshold:
            anchor = current

    return verdicts


def evidence_from_verdicts(
    verdicts: Sequence[DwellVerdict],
    times_s: np.ndarray,
    fps: float,
    threshold: float = BOUNDARY_THRESHOLD,
) -> BoundaryEvidence:
    """
    Draw the verdicts onto the score curve.

    The peak goes at the dwell's BEGINNING, not at the frame the model saw: the
    dwell begins when the screen settles, and that is the instant the annotation
    means. The sampled frame is only what the model was shown.
    """
    score = np.zeros(len(times_s), dtype=np.float32)
    spread = max(1, int(_SPREAD_S * fps))
    boundaries_s: list[float] = []
    for verdict in verdicts:
        apply_gaussian(score, verdict.dwell_start, verdict.judgement.p_boundary, spread)
        if verdict.judgement.p_boundary >= threshold:
            boundaries_s.append(float(times_s[verdict.dwell_start]))
    return BoundaryEvidence(times_s, score, boundaries_s)


# ── Decoding (the only part that touches OpenCV and ffprobe) ──────────────────

def parse_frame_times(csv_text: str, n_frames: int) -> np.ndarray:
    """
    Frame presentation times from ffprobe's CSV output, zeroed on the first frame.

    :raises RuntimeError: when the count does not match the decoded frames — a
        mismatch means the container and the decoder disagree, and silently
        trusting either would put the whole modality on a wrong time axis.
    """
    times = [float(line.split(",")[0]) for line in csv_text.splitlines() if line.strip()]
    if len(times) != n_frames:
        raise RuntimeError(
            f"ffprobe lieferte {len(times)} Frame-Zeiten, dekodiert wurden "
            f"{n_frames} — inkonsistente Aufnahme."
        )
    t = np.asarray(times, dtype=np.float64)
    return t - t[0] if len(t) else t


def _frame_times_s(video_path: str, n_frames: int) -> np.ndarray:
    """
    Real capture time of each frame in seconds, read straight from the MP4.

    The recorder encodes with ffmpeg's wall clock at read time (VFR), so a frame's
    presentation time is essentially its capture time. Events run on
    time.monotonic() zeroed at arm(); the two share an origin to within the pipe
    start-up latency (~40 ms measured, see evaluation.synchronization), not
    exactly. ffprobe reads the times from the container (unlike OpenCV's POS_MSEC,
    which some builds derive from an assumed rate).
    """
    import subprocess

    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    ).stdout
    return parse_frame_times(out, n_frames)


def _scan(video_path: str) -> tuple[int, np.ndarray, float]:
    """Stream the video once; return frame count, per-frame activity and the fps."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, np.zeros(0, dtype=np.float32), 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    def frames():
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        finally:
            cap.release()

    activity = frame_activity(frames())
    return len(activity), activity, fps


def _read_frames(video_path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """
    Fetch the given frames in one linear pass, downscaled on read.

    Sequential decoding beats seeking on a long-GOP MP4, and full-size frames
    would hold ~0.5 GB in memory for nothing — the model only sees the downscale.
    """
    frames: dict[int, np.ndarray] = {}
    if not wanted:
        return frames                      # nothing to decode, nothing to import

    import cv2

    from docupilot.segmentation.video_scoring import downscale

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


def _activity_key(session: RecordingSession) -> str:
    """What the stored scan depends on: the recording bytes and the grid."""
    return (
        f"{_ACTIVITY_FORMAT}|{store.file_digest(session.recording_path)}"
        f"|{_PHASH_SIZE}|{_ACTIVITY_GRID}"
    )


def scan_activity(session: RecordingSession, use_cache: bool = True) -> ActivityScan:
    """
    The per-frame activity of a recording, read from the session directory when
    it was scanned before, decoded otherwise.

    :param session: the recording; its .recording_path points at the MP4.
    :param use_cache: read a stored scan and store a fresh one. False decodes
        and stores nothing.
    :raises RuntimeError: when the frame times cannot be read (see parse_frame_times).
    """
    path = session.session_dir / _ACTIVITY_FILE
    key = _activity_key(session) if use_cache else None

    if key is not None and path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["key"]) == key:
                    return ActivityScan(
                        n_frames=int(data["n_frames"]),
                        activity=np.asarray(data["activity"], dtype=np.float32),
                        fps=float(data["fps"]),
                        times_s=np.asarray(data["times_s"], dtype=np.float64),
                    )
        except Exception:                # noqa: BLE001 — a stale or damaged file is a miss
            pass

    video_path = str(session.recording_path)
    n_frames, activity, fps = _scan(video_path)
    times_s = _frame_times_s(video_path, n_frames)
    scan = ActivityScan(n_frames=n_frames, activity=activity, fps=fps, times_s=times_s)

    if key is not None:
        store.write_npz(path, key=np.array(key), n_frames=np.array(n_frames),
                        activity=activity, fps=np.array(fps), times_s=times_s)
    return scan


# ── Entry point ───────────────────────────────────────────────────────────────

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
    :param use_cache: reuse verdicts and the activity scan cached in the
        session directory.
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

    scan = scan_activity(session, use_cache=use_cache)
    if scan.n_frames < 2:
        return BoundaryEvidence(scan.times_s, np.zeros(scan.n_frames, dtype=np.float32), [])

    dwell_list = dwells(scan.activity, max(1, round(_MIN_DWELL_S * scan.fps)))
    if len(dwell_list) < 2:
        return BoundaryEvidence(scan.times_s, np.zeros(scan.n_frames, dtype=np.float32), [])

    settled = settled_frames(dwell_list, scan.fps)
    halves = _read_frames(str(session.recording_path), set(settled))
    steps = [
        (start, frame) for (start, _), frame in zip(dwell_list, settled) if frame in halves
    ]

    # The anchor is compared against every following dwell, so its grey version
    # is computed once and kept, not rebuilt per comparison.
    grays: dict[int, np.ndarray] = {}

    def gray(index: int) -> np.ndarray:
        if index not in grays:
            grays[index] = _gray16(halves[index])
        return grays[index]

    regions: dict[tuple[int, int], Region] = {}

    def has_change(anchor: int, current: int) -> bool:
        region = changed_region(gray(anchor), gray(current))
        if region is None:
            return False
        regions[(anchor, current)] = region
        return True

    cache = (
        vlm.Cache(session.session_dir / "gui_vlm_cache.json") if use_cache else None
    )

    def judge(anchor: int, current: int) -> Judgement | None:
        composite = vlm.encode_pair(
            halves[anchor], halves[current], regions[(anchor, current)]
        )
        return vlm.judge(composite, cache=cache)

    def limit_reached() -> None:
        # A silent cap reads as "everything covered". Say what was dropped so
        # the missing tail is visible, not mistaken for "no more boundaries".
        warnings.warn(
            f"Video: _MAX_CALLS={_MAX_CALLS} erreicht in {session.session_dir.name} "
            f"— restliche Dwells ungeprüft, spätere Grenzen fehlen.",
            stacklevel=2,
        )

    total = max(1, len(steps) - 1)
    verdicts = walk_dwells(
        steps, has_change, judge,
        on_call=(None if on_progress is None else lambda done: on_progress(done, total)),
        is_cancelled=is_cancelled,
        on_limit=limit_reached,
    )

    if cache is not None:
        cache.flush()                # also on cancel: keep what we paid for

    return evidence_from_verdicts(verdicts, scan.times_s, scan.fps)
