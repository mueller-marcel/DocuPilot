"""
The video modality: action boundaries from the screen recording ALONE.

No events, no audio — the modality stays independent for the 2^3 Shapley
ablation. This module reads exactly one stream: the recording's video frames.

ACTION (see docs/annotationsleitfaden.md for the full definition)
  A boundary marks the COMPLETION of a user-triggered operation — the moment its
  RESULT becomes visible and settles into a state that persists. What counts is
  the finished result, not "data vs. view": a deliberate view/mode change
  (reading view, details view) counts; a menu opened on the way, a selection, or
  navigation to reach the next target does not. The delayed result of a user
  operation (a build/test/filter finishing) counts too, even though it appears on
  its own.

  This modality sees only the screen, so it cannot verify the trigger — that is
  the events modality's job, kept separate for the ablation. It judges, per
  settled-state pair, whether a finished result is visible; the VLM prompt in
  video_scoring.py encodes exactly this distinction.

The structure answers exactly ONE question — "where does the screen stand still?"
— and the VLM answers every other:

  1. Activity signal.  pHash on an 8x8 TILE grid; activity = the largest per-tile
     change. The grid is not a refinement, it is the difference between seeing
     the actions and not seeing them. Measured on session_07 (Excel, 2560x1440,
     8 annotated actions), peak activity per action:

                             full+halves   8x8 grid
       Autofilter aktiviert      0.000       0.125    <- invisible before
       Bereich kopiert           0.031       0.344    <- invisible before
       Sortiert                  0.125       0.500
       Filter angewandt          0.156       0.469

     On a 2560x1440 screen the table covers ~7% of the pixels, and a whole-frame
     pHash (a 32x32 DCT) averages a row of filter arrows away completely. Max over
     tiles, never mean: an action changes ONE region, and a mean would divide it
     by the 63 tiles where nothing happened.

  2. Dwells.  Maximal runs of still frames are the settled states the UI rests in
     — GIFdroid's "steady state" (Feng et al., ICSE 2022), the cut points of
     SeeAction (arXiv:2503.12873). Parameters swept against session_07's eight
     actions; see _ACTIVITY_QUIET.

  3. Anchored VLM state tracking.  An ANCHOR holds the last ESTABLISHED state.
     Every following dwell is compared BY THE MODEL against that anchor — not
     against its immediate predecessor, because "new state" is meaningless except
     relative to the last state the user actually established:

       anchor := first dwell
       for each following dwell d:
           p := P(ACTION_COMPLETED | anchor, d)          <- the VLM
           write p as graded evidence at d's settle frame
           if p >= BOUNDARY_THRESHOLD:  anchor := d      <- a new state is set

     Opening the File menu is rejected AND leaves the anchor at the pre-menu
     state, so the action that follows is judged against the state the user
     started from. That is what makes the menu case fall out by itself.

DELIBERATELY ABSENT — do not reintroduce:
  - Any magnitude score. Integrated burst activity measures HOW MUCH the screen
    changed, which is not the question: it rated a window activating above a cell
    being reformatted, and a dropdown above the filter it applies.
  - Non-maximum suppression. "Within N seconds the higher score wins" is a
    structural rule that can delete a real boundary behind a mis-scored menu. The
    anchor makes it unnecessary: a second boundary must beat the state the first
    one established.
  - A structural fallback. Without the VLM this extractor has nothing to say, and
    a lane full of magnitude peaks reads as evidence when it is noise. It raises
    instead.
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

# Swept on session_07 against its eight video-derived actions, keeping the
# cheapest setting at FULL recall (every action proposed within +-1 s of its
# annotated boundary):
#
#   grid 2 (== the old full+halves signal): NO setting ever reaches 8/8
#   grid 8, quiet 0.05, dwell 0.3 s: 65 proposals, 8/8
#   grid 8, quiet 0.08, dwell 0.5 s: 41 proposals, 8/8   <- chosen
#   grid 8, quiet 0.08, dwell 0.8 s: 37 proposals, 8/8
#
# Below ~37 proposals real actions start to be missed, and a missed action is
# unrecoverable — the VLM never sees it. Recall here, precision in the VLM.
_ACTIVITY_QUIET = 0.08   # per-tile pHash distance below which a frame is "still"
_MIN_DWELL_S = 0.5       # a still run this long is a settled state

# Settled frames are sampled this far INTO a dwell: the first still frame can
# still show a fading animation the pHash no longer reads as motion.
#
# Reasoned, not measured — and it only picks WHICH frame the model is shown, not
# where the boundary lands (that is the dwell's start). Anything from one frame to
# a few tenths would do the same job.
_SETTLE_OFFSET_S = 0.2

# A tile counts as CHANGED (for the Set-of-Mark box only) when its mean absolute
# grey-level difference exceeds this. One grey level out of 255: far above JPEG
# noise, far below anything a human would call a change. Deliberately not tuned —
# the box only has to point, not to decide.
_PIXEL_CHANGE_EPS = 1.0

_SPREAD_S = 1.0

# A pathological recording (constant flicker) could fragment into thousands of
# dwells. Anchoring is sequential, so the budget is spent from the start.
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
    Bounding box (x0, y0, x1, y1 in [0,1]) of the tiles whose PIXELS differ.

    This is the Set-of-Mark input (Yang et al., arXiv:2310.11441): the box is
    drawn onto both halves of the composite so the model is told where to look and
    stops inventing changes elsewhere.

    Pixels, not pHash — and that is the whole point. A perceptual hash is a DCT
    sign pattern, and on a low-texture tile (the empty half of a spreadsheet)
    those coefficients sit near zero, so codec noise flips them: measured on this
    dataset, tiles with almost no texture reported a pHash distance of 0.34 while
    their pixels had moved by 0.08 of one grey level. Roughly 7 % of tiles lie that
    way. A box built on the hash therefore covered the whole screen on pairs that
    were visually identical — it marked nothing.

    The mean absolute grey-level difference has no such failure mode. It is used
    ONLY here, to point the model at a region; the activity signal that finds the
    settled states keeps using pHash, where its codec robustness is what we want
    and the max-over-tiles noise is already absorbed by the calibrated quiet
    threshold.

    The box says WHERE something differs, never WHETHER it is an action — a
    dropdown opening gets a box too. That judgement stays with the model.

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
    Fetch the given frames in one linear pass, already downscaled.

    Sequential decoding beats seeking: CAP_PROP_POS_FRAMES on a long-GOP MP4
    re-decodes from the preceding keyframe every time, so dozens of seeks cost
    more than reading the file straight through.

    Downscaling happens HERE, not afterwards: at 2560x1440 a single frame is
    11 MB, and keeping four dozen of them at full size would hold ~0.5 GB while a
    local VLM already occupies ~7 GB of a 14 GB machine.
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
    Action boundaries for one recording, from the video frames alone.

    On a CPU-only machine each state pair costs the VLM ~80 s, and a typical
    recording has dozens — so this runs for minutes and MUST NOT be called on a UI
    thread. on_progress / is_cancelled let a worker thread report and stop; both
    are plain callbacks (no Qt in here).

    :param session: session whose .recording_path points at the MP4.
    :param use_cache: reuse verdicts cached in the session directory.
    :param on_progress: called as (judged_pairs, total_pairs) after each pair.
    :param is_cancelled: polled before each pair; True stops early and keeps the
        evidence gathered so far.
    :raises RuntimeError: when no VLM backend is reachable.
    """
    from docupilot.segmentation import video_scoring as vlm

    if not vlm.is_available():
        hint = (
            f"Cloud-Backend '{vlm.MODEL}' nicht nutzbar. Benötigt:\n"
            "  poetry install         (Paket 'anthropic')\n"
            "  ANTHROPIC_API_KEY=...  oder  ant auth login\n"
            "Alternativ lokal: DOCUPILOT_VLM=ollama"
            if vlm.BACKEND == "anthropic" else
            f"Ollama nicht erreichbar ({vlm.OLLAMA_HOST}) oder Modell "
            f"'{vlm.MODEL}' fehlt:\n"
            f"  ollama serve\n"
            f"  ollama pull {vlm.MODEL}"
        )
        raise RuntimeError(
            hint + "\n\nOhne VLM kann die Video-Modalität keine Handlungsgrenzen "
            "bestimmen — ein rein struktureller Score misst Pixelmenge, nicht "
            "Bedeutung."
        )

    video_path = str(session.recording_path)
    fps = _fps(video_path)
    n_frames, activity = _scan(video_path)
    times_s = np.arange(n_frames, dtype=np.float64) / fps
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

        # No changed region means the two settled states are pixel-identical: the
        # same state, nothing to judge, no call to pay for. This replaces an
        # earlier pHash identity check, which answered the same question but got
        # it wrong — on a low-texture tile the hash flips on codec noise alone
        # (see _changed_region).
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
            # Accepting the boundary and advancing the anchor are ONE decision: if
            # a new state was established, later dwells must be judged against it.
            # The boundary is the settle frame, not the peak of the spread — the
            # spread only exists to make the curve readable.
            boundaries_s.append(float(times_s[dwell_start]))
            anchor = frame_idx

    if cache is not None:
        cache.flush()                # also on cancel: keep what we paid for

    return BoundaryEvidence(times_s, score, boundaries_s)
