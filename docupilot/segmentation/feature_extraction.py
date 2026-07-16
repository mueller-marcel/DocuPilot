from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from docupilot.recording.session import RecordingSession

# Hop length used consistently across all audio feature extractions.
# At sr=22050 (librosa default): 512 samples ≈ 23 ms per frame.
_HOP_LENGTH = 512


# ── Shared module-level helpers ───────────────────────────────────────────────

def _read_events(events_path: Path) -> list[dict]:
    try:
        with events_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _apply_gaussian(
    features: np.ndarray,
    col: int,
    center: int,
    score: float,
    spread_lo: int,
    spread_hi: int,
) -> None:
    """
    Write a (potentially asymmetric) Gaussian evidence peak into features[:, col].

    For symmetric spread: spread_lo == spread_hi.
    Uses np.maximum so overlapping peaks don't cancel — only the stronger wins.
    """
    n = features.shape[0]

    if spread_lo > 0 and center > 0:
        lo = max(0, center - spread_lo)
        offsets = np.arange(lo, center) - center
        g = np.exp(-0.5 * (offsets / max(spread_lo / 2.0, 1.0)) ** 2)
        features[lo:center, col] = np.maximum(
            features[lo:center, col], (score * g).astype(np.float32)
        )

    if 0 <= center < n:
        features[center, col] = max(features[center, col], np.float32(score))

    if spread_hi > 0 and center + 1 < n:
        hi = min(n, center + spread_hi + 1)
        offsets = np.arange(center + 1, hi) - center
        g = np.exp(-0.5 * (offsets / max(spread_hi / 2.0, 1.0)) ** 2)
        features[center + 1:hi, col] = np.maximum(
            features[center + 1:hi, col], (score * g).astype(np.float32)
        )


# ── GUI Action Boundary Extractor ─────────────────────────────────────────────
#
# Graded action-boundary evidence from the screen recording ALONE — no events,
# no audio, so the video modality stays independent for the 2^3 Shapley ablation.
#
# ACTION (see docs/annotationsleitfaden.md for the full definition):
#   A boundary marks the COMPLETION of a user-triggered operation — the moment
#   its RESULT becomes visible and settles into a state that persists. What
#   counts is the finished result, not "data vs. view": a deliberate view/mode
#   change (reading view, details view) counts; a menu opened on the way, a
#   selection, or navigation to reach the next target does not. The delayed
#   result of a user operation (a build/test/filter finishing) counts too, even
#   though it appears on its own.
#
#   The video extractor sees only the screen, so it cannot verify the trigger
#   (that is the events modality's job — kept separate for the Shapley ablation).
#   It judges, per settled-state pair, whether a finished result is visible; the
#   VLM prompt in gui_state_scoring.py encodes exactly this distinction.
#
# The structure answers exactly ONE question — "where does the screen stand
# still?" — and the VLM answers every other:
#
#   1. Activity signal.  pHash on an 8x8 TILE grid; activity = the largest
#      per-tile change. The grid is not a refinement, it is the difference
#      between seeing the actions and not seeing them. Measured on session_07
#      (Excel, 2560x1440, 8 annotated actions), peak activity per action:
#
#                              full+halves   8x8 grid
#        Autofilter aktiviert      0.000       0.125    <- invisible before
#        Bereich kopiert           0.031       0.344    <- invisible before
#        Sortiert                  0.125       0.500
#        Filter angewandt          0.156       0.469
#
#      On a 2560x1440 screen the table covers ~7% of the pixels, and a
#      whole-frame pHash (a 32x32 DCT) averages a row of filter arrows away
#      completely. Max over tiles, never mean: an action changes ONE region, and
#      a mean would divide it by the 63 tiles where nothing happened.
#
#   2. Dwells.  Maximal runs of still frames are the settled states the UI rests
#      in — GIFdroid's "steady state" (Feng et al., ICSE 2022), the cut points
#      of SeeAction (arXiv:2503.12873). Parameters swept against session_07's
#      eight actions; see _ACTIVITY_QUIET.
#
#   3. Anchored VLM state tracking.  An ANCHOR holds the last ESTABLISHED state.
#      Every following dwell is compared BY THE MODEL against that anchor — not
#      against its immediate predecessor, because "new state" is meaningless
#      except relative to the last state the user actually established:
#
#        anchor := first dwell
#        for each following dwell d:
#            p := P(ACTION_COMPLETED | anchor, d)        <- the VLM
#            write p as graded evidence at d's settle frame
#            if p >= _ANCHOR_ADVANCE:  anchor := d       <- a new state is set
#
#      Opening the File menu is rejected AND leaves the anchor at the pre-menu
#      state, so the action that follows is judged against the state the user
#      started from. That is what makes the menu case fall out by itself.
#
# DELIBERATELY ABSENT — do not reintroduce:
#   - Any magnitude score. Integrated burst activity measures HOW MUCH the screen
#     changed, which is not the question: it rated a window activating above a
#     cell being reformatted, and a dropdown above the filter it applies.
#   - Non-maximum suppression. "Within N seconds the higher score wins" is a
#     structural rule that can delete a real boundary behind a mis-scored menu.
#     The anchor makes it unnecessary: a second boundary must beat the state the
#     first one established.
#   - A structural fallback. Without the VLM this extractor has nothing to say,
#     and a lane full of magnitude peaks reads as evidence when it is noise. It
#     raises instead.
#
# Output (T_v, 3), unchanged for dataset_builder and FeatureDialog:
#   col 0 — onset_flag:  1.0 at each judged settle frame
#   col 1 — score:       graded P(action) in [0, 1], Gaussian-spread
#   col 2 — flag:        0/1 at threshold (DISPLAY ONLY; dataset_builder uses
#                        col 1 — thresholding happens once, downstream, for every
#                        modality alike, which is the Shapley prerequisite)

_PHASH_SIZE    = 8
_ACTIVITY_GRID = 8

# Swept on session_07 against its eight video-derived actions, keeping the
# cheapest setting at FULL recall (every action proposed within the +-1 s
# labeling tolerance of dataset_builder):
#
#   grid 2 (== the old full+halves signal): NO setting ever reaches 8/8
#   grid 8, quiet 0.05, dwell 0.3 s: 65 proposals, 8/8
#   grid 8, quiet 0.08, dwell 0.5 s: 41 proposals, 8/8   <- chosen
#   grid 8, quiet 0.08, dwell 0.8 s: 37 proposals, 8/8
#
# Below ~37 proposals real actions start to be missed, and a missed action is
# unrecoverable — the VLM never sees it. Recall here, precision in the VLM.
_ACTIVITY_QUIET = 0.08   # per-tile pHash distance below which a frame is "still"
_MIN_DWELL_S    = 0.5    # a still run this long is a settled state

# Settled frames are sampled this far INTO a dwell: the first still frame can
# still show a fading animation the pHash no longer reads as motion.
_SETTLE_OFFSET_S = 0.2

# A tile counts as CHANGED (for the Set-of-Mark box only) when its mean absolute
# grey-level difference exceeds this. One grey level out of 255: far above JPEG
# noise, far below anything a human would call a change. Deliberately not tuned —
# the box only has to point, not to decide.
_PIXEL_CHANGE_EPS = 1.0

# At or above this probability the system is taken to REST in a new established
# state, and later dwells are compared against it. This selects the reference
# frame; it is not a boundary decision — col 1 keeps the graded score either way.
_ANCHOR_ADVANCE = 0.5

_SPREAD_S       = 1.0
_FLAG_THRESHOLD = 0.5

# A pathological recording (constant flicker) could fragment into thousands of
# dwells. Anchoring is sequential, so the budget is spent from the start.
_MAX_CALLS = 400


class GUIActionBoundaryExtractor:
    """
    Graded GUI action-boundary evidence, from the video file only.

    pHash tile activity -> dwell segmentation -> a local VLM judges each
    (anchor, dwell) pair of settled states. Requires a reachable Ollama server;
    there is no structural fallback, because measuring how much the screen
    changed does not answer whether a new state was reached.
    """

    # ── Activity signal ───────────────────────────────────────────────

    @staticmethod
    def _phash(gray: np.ndarray) -> object:
        import imagehash
        from PIL import Image
        return imagehash.phash(Image.fromarray(gray), hash_size=_PHASH_SIZE)

    @staticmethod
    def _tiles(gray: np.ndarray) -> tuple:
        """One pHash per cell of the activity grid."""
        h, w = gray.shape[:2]
        g = _ACTIVITY_GRID
        ph = GUIActionBoundaryExtractor._phash
        return tuple(
            ph(gray[r * h // g:(r + 1) * h // g, c * w // g:(c + 1) * w // g])
            for r in range(g) for c in range(g)
        )

    @staticmethod
    def _distance(a: tuple, b: tuple) -> float:
        """Largest per-tile pHash distance, normalised to [0, 1]."""
        return max(float(x - y) for x, y in zip(a, b)) / (_PHASH_SIZE ** 2)

    @staticmethod
    def _changed_region(
        before: np.ndarray, after: np.ndarray
    ) -> tuple[float, float, float, float] | None:
        """
        Bounding box (x0, y0, x1, y1 in [0,1]) of the tiles whose PIXELS differ.

        This is the Set-of-Mark input (Yang et al., arXiv:2310.11441): the box is
        drawn onto both halves of the composite so the model is told where to
        look and stops inventing changes elsewhere.

        Pixels, not pHash — and that is the whole point. A perceptual hash is a
        DCT sign pattern, and on a low-texture tile (the empty half of a
        spreadsheet) those coefficients sit near zero, so codec noise flips them:
        measured on this dataset, tiles with almost no texture reported a pHash
        distance of 0.34 while their pixels had moved by 0.08 of one grey level.
        Roughly 7 % of tiles lie that way. A box built on the hash therefore
        covered the whole screen on pairs that were visually identical — it
        marked nothing.

        The mean absolute grey-level difference has no such failure mode. It is
        used ONLY here, to point the model at a region; the activity signal that
        finds the settled states keeps using pHash, where its codec robustness is
        what we want and the max-over-tiles noise is already absorbed by the
        calibrated quiet threshold.

        The box says WHERE something differs, never WHETHER it is an action — a
        dropdown opening gets a box too. That judgement stays with the model.

        :return: None when no tile differs measurably.
        """
        import cv2

        g = _ACTIVITY_GRID
        a = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY).astype(np.int16)
        b = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY).astype(np.int16)
        h, w = a.shape[:2]

        changed = []
        for r in range(g):
            for c in range(g):
                y0, y1 = r * h // g, (r + 1) * h // g
                x0, x1 = c * w // g, (c + 1) * w // g
                diff = float(np.abs(a[y0:y1, x0:x1] - b[y0:y1, x0:x1]).mean())
                if diff > _PIXEL_CHANGE_EPS:
                    changed.append((r, c))
        if not changed:
            return None

        rows = [r for r, _ in changed]
        cols = [c for _, c in changed]
        return (
            min(cols) / g,
            min(rows) / g,
            (max(cols) + 1) / g,
            (max(rows) + 1) / g,
        )

    @staticmethod
    def _scan(video_path: str) -> tuple[list[tuple], np.ndarray]:
        """Stream the video once; return per-frame tile hashes and the activity."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], np.zeros(0, dtype=np.float32)

        hashes: list[tuple] = []
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                hashes.append(GUIActionBoundaryExtractor._tiles(gray))
        finally:
            cap.release()

        activity = np.zeros(len(hashes), dtype=np.float32)
        for i in range(1, len(hashes)):
            activity[i] = GUIActionBoundaryExtractor._distance(hashes[i - 1], hashes[i])
        return hashes, activity

    # ── Dwell segmentation ────────────────────────────────────────────

    @staticmethod
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

    @staticmethod
    def _read_frames(video_path: str, wanted: set[int]) -> dict[int, np.ndarray]:
        """
        Fetch the given frames in one linear pass, already downscaled.

        Sequential decoding beats seeking: CAP_PROP_POS_FRAMES on a long-GOP MP4
        re-decodes from the preceding keyframe every time, so dozens of seeks
        cost more than reading the file straight through.

        Downscaling happens HERE, not afterwards: at 2560x1440 a single frame is
        11 MB, and keeping four dozen of them at full size would hold ~0.5 GB
        while Ollama already occupies ~7 GB of a 14 GB machine.
        """
        import cv2

        from docupilot.segmentation.gui_state_scoring import downscale

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

    # ── Extraction ────────────────────────────────────────────────────

    @staticmethod
    def extract_gui_features(
        recording_session: RecordingSession,
        fps: float,
        use_cache: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """
        Graded action-boundary evidence for one recording.

        On a CPU-only machine each state pair costs the VLM ~80 s, and a typical
        recording has dozens — so this runs for minutes and MUST NOT be called on
        a UI thread. on_progress / is_cancelled exist so a worker thread can
        report and be stopped; both are optional and pure callbacks (no Qt here).

        :param recording_session: session whose .recording_path points at the MP4.
        :param fps: video frame rate.
        :param use_cache: reuse verdicts cached in the session directory.
        :param on_progress: called as (judged_pairs, total_pairs) after each pair.
        :param is_cancelled: polled before each pair; return True to stop early
            (the evidence gathered so far is kept and returned).
        :return: (T_v, 3) float32 — onset flag, graded score, display flag.
        :raises RuntimeError: when no Ollama server with the model is reachable.
        """
        from docupilot.segmentation import gui_state_scoring as vlm

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
                hint + "\n\nOhne VLM kann die GUI-Modalität keine Handlungsgrenzen "
                "bestimmen — ein rein struktureller Score misst Pixelmenge, nicht "
                "Bedeutung."
            )

        video_path = str(recording_session.recording_path)
        hashes, activity = GUIActionBoundaryExtractor._scan(video_path)
        features = np.zeros((len(hashes), 3), dtype=np.float32)
        if len(hashes) < 2:
            return features

        dwells = GUIActionBoundaryExtractor._dwells(
            activity, max(1, round(_MIN_DWELL_S * fps))
        )
        if len(dwells) < 2:
            return features

        # One settled frame per dwell, sampled _SETTLE_OFFSET_S in and downscaled
        # on read. The model sees them stitched into a BEFORE|AFTER composite.
        offset = max(0, int(_SETTLE_OFFSET_S * fps))
        settled = [min(end, start + offset) for start, end in dwells]
        halves = GUIActionBoundaryExtractor._read_frames(video_path, set(settled))

        cache = (
            vlm.Cache(recording_session.session_dir / "gui_vlm_cache.json")
            if use_cache else None
        )

        anchor: int | None = None
        calls = 0
        total = max(0, len(dwells) - 1)               # upper bound on VLM calls
        spread = max(1, int(_SPREAD_S * fps))

        for (dwell_start, _), frame_idx in zip(dwells, settled):
            if frame_idx not in halves:
                continue
            if anchor is None:                       # the workflow's starting point
                anchor = frame_idx
                continue
            if calls >= _MAX_CALLS or (is_cancelled is not None and is_cancelled()):
                break

            # No changed region means the two settled states are pixel-identical:
            # the same state, nothing to judge, no call to pay for. This replaces
            # an earlier pHash identity check, which answered the same question
            # but got it wrong — on a low-texture tile the hash flips on codec
            # noise alone (see _changed_region).
            region = GUIActionBoundaryExtractor._changed_region(
                halves[anchor], halves[frame_idx]
            )
            if region is None:
                continue

            composite = vlm.encode_pair(halves[anchor], halves[frame_idx], region)
            judgement = vlm.judge(composite, cache=cache)
            calls += 1
            if on_progress is not None:
                on_progress(calls, total)
            if judgement is None:                    # unusable answer: leave no
                continue                             # evidence rather than guess

            features[dwell_start, 0] = 1.0
            _apply_gaussian(
                features, 1, dwell_start, judgement.p_boundary, spread, spread
            )
            if judgement.p_boundary >= _ANCHOR_ADVANCE:
                anchor = frame_idx                   # a new state is established

        if cache is not None:
            cache.flush()                            # also on cancel: keep what we paid for

        features[:, 2] = (features[:, 1] >= _FLAG_THRESHOLD).astype(np.float32)
        return features


# ── Transcription Extractor ───────────────────────────────────────────────────

class TranscriptionExtractor:
    """
    Transcribes the audio track of a RecordingSession using OpenAI Whisper.
    Returns the full transcript text and raw word-level timestamp data.
    The whisper import is deferred so the model is only loaded when needed.
    """

    # "base" missed German instructional vocabulary in pilot recordings;
    # "small" is the best quality/latency trade-off for offline extraction.
    _MODEL_SIZE = "small"

    @staticmethod
    def extract_transcript(recording_session: RecordingSession) -> tuple[str, list[dict]]:
        import whisper

        model = whisper.load_model(TranscriptionExtractor._MODEL_SIZE)
        result = model.transcribe(
            str(recording_session.recording_path),
            verbose=False,
            language="de",
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        full_text: str = result.get("text", "").strip()
        words: list[dict] = [
            w for seg in result.get("segments", []) for w in seg.get("words", [])
        ]
        return full_text, words


# ── Sentence Segmenter ────────────────────────────────────────────────────────

class SentenceSegmenter:
    """
    Segments a German transcript into sentences with timestamps using spaCy.

    The former hard "es"-subject filter has been removed. Sentence-type
    discrimination is the NLI classifier's job — a syntactic pre-filter
    deleted evidence the classifier never got to see, and misfired on
    sentences like "Es öffnet sich das Menü, klicken Sie dann auf ...".

    Uses de_core_news_lg; falls back to md / sm when lg is not installed.
    """

    _MODEL_NAME = "de_core_news_lg"
    _MODEL_FALLBACKS = ["de_core_news_md", "de_core_news_sm"]

    @staticmethod
    def segment(full_text: str, words: list[dict]) -> list[tuple[float, str]]:
        if not full_text or not words:
            return []

        import spacy

        nlp = None
        for name in [SentenceSegmenter._MODEL_NAME, *SentenceSegmenter._MODEL_FALLBACKS]:
            try:
                nlp = spacy.load(name)
                break
            except OSError:
                continue
        if nlp is None:
            raise OSError(
                "Kein deutsches spaCy-Modell gefunden. Bitte installieren:\n"
                "  python -m spacy download de_core_news_lg"
            )

        doc = nlp(full_text)

        # Build char-offset → timestamp lookup from the Whisper word list.
        # Cursor advances past each match so repeated words map to their own
        # occurrence rather than always the first one.
        char_to_time: list[tuple[int, float]] = []
        char_cursor = 0
        for w in words:
            surface = w.get("word", "").strip()
            if not surface:
                continue
            pos = full_text.find(surface, char_cursor)
            if pos != -1:
                char_to_time.append((pos, float(w.get("start", 0.0))))
                char_cursor = pos + len(surface)

        def _resolve_time(char_idx: int) -> float:
            if not char_to_time:
                return 0.0
            best_t = char_to_time[0][1]
            for pos, t in char_to_time:
                if pos <= char_idx:
                    best_t = t
                else:
                    break
            return best_t

        return [
            (_resolve_time(sent.start_char) * 1000.0, sent.text.strip())
            for sent in doc.sents
            if sent.text.strip()
        ]


# ── Audio Boundary Extractor ──────────────────────────────────────────────────
#
# Graded action-boundary evidence from the audio track ALONE — no screen, no
# events, so the audio modality stays independent for the 2^3 Shapley ablation.
#
# WHAT AUDIO CAN AND CANNOT KNOW
#   A boundary is the moment a user-triggered change SETTLES INTO A NEW PERSISTENT
#   STATE (docs/annotationsleitfaden.md). That moment is visual. Nothing audible
#   happens when a filter finishes applying — the microphone hears speech and
#   silence, and the participant is usually not speaking at all while the screen
#   settles. So this extractor CANNOT localise the instant, and pretending it can
#   would fake the very quantity the Shapley analysis is supposed to measure.
#
#   What the narration does carry is STRUCTURE. The participant announces each
#   step before or while doing it, in order. From announcement i and announcement
#   i+1 it follows that step i completed BETWEEN them. Audio therefore knows the
#   INTERVAL a boundary falls in, not the point — and that is exactly what gets
#   encoded here.
#
#   Measured on session_30 (7 rule-conform boundaries, Whisper small + spaCy):
#     Ansage -> Grenze              : median 6.84 s, range 1.98 .. 19.10 s
#     Grenze -> naechste Ansage     : median  3.2 s, range 1.44 ..  8.84 s
#   Every one of the 7 boundaries falls inside the interval opened by its own
#   announcement, and no boundary falls inside an interval opened by a
#   means-sentence. The interval claim holds; the point claim never could.
#
# THE SEMANTIC STAGE
#   Announcing a step is not the same as announcing a boundary: "Navigiere zum
#   erstellten Datenblatt" announces a MEANS, and rule C excludes it. Deciding
#   this needs a model that can apply the definition, which is why the old
#   zero-shot NLI stage is gone — see audio_boundary_scoring.py for the
#   measurement that killed it (four hypothesis wordings, separation negative in
#   all four) and for why the replacement is an LLM.
#
# EVIDENCE GEOMETRY
#   Each sentence i opens an execution window [t_i, t_{i+1}) and fills it with a
#   raised-cosine bump scaled by P(OPERATION | sentence i):
#
#     0 at t_i          the step cannot be finished at the instant it is announced
#     peak inside       where the completion is expected
#     0 at t_{i+1}      by the next announcement the step is demonstrably done
#
#   The zeros at the announcements are not cosmetic — they keep adjacent windows
#   from fusing into one plateau, so find_peaks in dataset_builder yields one
#   candidate per announced step instead of one candidate per run of them.
#
# DELIBERATELY ABSENT — do not reintroduce:
#   - A narrow peak on the sentence onset. The old stage put a 1.5 s spread on the
#     announcement; it covered 1 of 7 boundaries, so the ±1 s feature window of
#     dataset_builder saw nothing and audio's Shapley value measured the geometry
#     rather than the modality.
#   - Any prosodic/RMS "pause detector" as a boundary source. A pause marks that
#     the user stopped talking, not that the screen settled; it fires on thinking,
#     reading and breathing alike.
#   - A fallback when the LLM is unavailable. Without a per-sentence judgement
#     every window is equal, the lane tiles the whole timeline, and a uniform lane
#     reads as evidence while carrying none. It raises instead.
#
# Output (T_a, 3):
#   col 0 — sentence_start: 1.0 at each sentence-start frame
#   col 1 — score:          graded window evidence in [0, 1]
#   col 2 — flag:           0/1 at threshold (DISPLAY ONLY; dataset_builder uses
#                           col 1 — thresholding happens once, downstream, for
#                           every modality alike, which is the Shapley prerequisite)

# Where in its window does the completion sit? Derived from the definition's own
# structure, not fitted: executing takes a variable and often long time (rule A
# admits delayed results — builds, filters), while the pause that follows the
# completion is brief (rule C: the user "hält inne", then announces the next
# step). So the completion sits LATE in the interval, but not at its edge.
#
# PROVISIONAL: the direction is structural, the exact fraction is not. The
# measured median on session_30 is 0.77 of the window (n=7, ONE session), which
# corroborates 0.75 but cannot justify it. Calibrate on a dev split before
# freezing — never on the evaluation set.
_COMPLETION_POSITION = 0.75

# The last window is normally closed by this session's own median announcement
# gap (derived from this recording's audio alone, so it adapts to a fast or slow
# speaker without importing a constant from elsewhere — or from another
# modality). This value is only reached when a recording has a SINGLE sentence
# and there is no gap to take a median of.
_LAST_WINDOW_FALLBACK_S = 8.0

_AUDIO_FLAG_THRESHOLD = 0.5


def _apply_window(
    features: np.ndarray,
    col: int,
    lo: int,
    hi: int,
    peak: int,
    score: float,
) -> None:
    """
    Write a raised-cosine bump over [lo, hi] peaking at `peak`, scaled by `score`.

    Zero at both edges, `score` at the peak, asymmetric when the peak is off
    centre. np.maximum so overlapping windows don't cancel — only the stronger
    wins, same rule as _apply_gaussian.
    """
    n = features.shape[0]
    lo, hi = max(0, lo), min(n - 1, hi)
    if hi <= lo or score <= 0.0:
        return
    peak = min(max(peak, lo), hi)

    x = np.arange(lo, hi + 1, dtype=np.float64)
    shape = np.zeros_like(x)

    rise = x <= peak
    if peak > lo:
        shape[rise] = 0.5 * (1.0 - np.cos(np.pi * (x[rise] - lo) / (peak - lo)))
    else:
        shape[rise] = 1.0                      # peak sits on the left edge

    fall = ~rise
    if hi > peak:
        shape[fall] = 0.5 * (1.0 + np.cos(np.pi * (x[fall] - peak) / (hi - peak)))

    features[lo:hi + 1, col] = np.maximum(
        features[lo:hi + 1, col], (score * shape).astype(np.float32)
    )


class AudioBoundaryExtractor:
    """
    Graded action-boundary evidence, from the audio track only.

    Whisper transcript -> spaCy sentences -> an LLM judges each narrated sentence
    against the boundary definition -> each accepted sentence fills its execution
    window with graded evidence.

    Requires a reachable LLM; there is no structural fallback, because "the user
    said something here" does not answer whether a step completed.
    """

    @staticmethod
    def extract_audio_features(
        recording_session: RecordingSession,
        full_text: str,
        words: list[dict],
        n_frames: int,
        sampling_rate: float,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Graded boundary evidence for one recording, on the audio hop grid.

        Transcription happens in the caller: both call sites need the transcript
        anyway, and Whisper is by far the slowest step — keeping it outside lets
        the UI report it as its own phase.

        :param recording_session: session; used ONLY for the verdict cache path.
        :param full_text: Whisper transcript.
        :param words: Whisper word-level timestamps.
        :param n_frames: length of the audio hop grid.
        :param sampling_rate: audio sampling rate (hop grid = sr / _HOP_LENGTH).
        :param use_cache: reuse verdicts cached in the session directory.
        :return: (T_a, 3) float32 — sentence-start flag, graded score, display flag.
        :raises RuntimeError: when no LLM is reachable.
        """
        from docupilot.segmentation import audio_boundary_scoring as llm

        features = np.zeros((n_frames, 3), dtype=np.float32)
        if not full_text or not words or n_frames <= 0:
            return features

        # Checked before spaCy is loaded: the lg model costs seconds, and there is
        # nothing this extractor can deliver without the judge anyway.
        if not llm.is_available():
            raise RuntimeError(
                f"LLM-Backend '{llm.MODEL}' nicht nutzbar. Benötigt:\n"
                "  poetry install         (Paket 'anthropic')\n"
                "  ANTHROPIC_API_KEY=...  oder  ant auth login\n\n"
                "Ohne semantisches Urteil kann die Audio-Modalität keine "
                "Handlungsgrenzen bestimmen — jede Ansage wäre gleich viel wert "
                "und die Spur würde die ganze Zeitachse gleichmäßig füllen."
            )

        sentences = SentenceSegmenter.segment(full_text, words)
        if not sentences:
            return features

        frames_per_second = sampling_rate / _HOP_LENGTH
        starts_s = [t_ms / 1000.0 for t_ms, _ in sentences]

        # Column 0 — sentence onsets. Pure structure, no judgement: this is the
        # one thing audio observes directly.
        for t_s in starts_s:
            idx = int(round(t_s * frames_per_second))
            if 0 <= idx < n_frames:
                features[idx, 0] = 1.0

        cache = (
            llm.Cache(recording_session.session_dir / "audio_llm_cache.json")
            if use_cache else None
        )
        judgements = llm.judge([text for _, text in sentences], cache=cache)
        if cache is not None:
            cache.flush()
        if judgements is None:                   # unusable answer: leave no
            return features                      # evidence rather than guess

        # The last window has no closing announcement. Two things close it anyway:
        # this session's own median announcement gap, and the end of the
        # recording — the step demonstrably finished before the user stopped
        # recording. Whichever comes first wins.
        #
        # Clamping to the recording end is not cosmetic. A window running past the
        # last frame puts its peak ON the array edge, and find_peaks (which
        # generates the candidates in dataset_builder) cannot detect an edge
        # maximum because there is no descent after it. The last boundary of every
        # session would then be missing from the audio candidate set — a
        # systematic hole in exactly the arm whose contribution we are measuring.
        gaps = np.diff(starts_s)
        horizon = float(np.median(gaps)) if len(gaps) else _LAST_WINDOW_FALLBACK_S
        duration_s = (n_frames - 1) / frames_per_second
        window_ends = [*starts_s[1:], min(starts_s[-1] + horizon, duration_s)]

        for t_s, end_s, judgement in zip(starts_s, window_ends, judgements):
            if end_s <= t_s:                     # spaCy split, same Whisper word
                continue
            lo = int(round(t_s * frames_per_second))
            hi = int(round(end_s * frames_per_second))
            peak = int(round(
                (t_s + _COMPLETION_POSITION * (end_s - t_s)) * frames_per_second
            ))
            _apply_window(features, 1, lo, hi, peak, judgement.p_boundary)

        features[:, 2] = (features[:, 1] >= _AUDIO_FLAG_THRESHOLD).astype(np.float32)
        return features


# ── Event Feature Extractor ───────────────────────────────────────────────────

_EVENT_WINDOW_S           = 0.25
_MAX_EVENT_RECENCY_S      = 5.0
_EVENT_IDLE_FULL_S        = 3.0
_EVENT_SPREAD_S           = 1.0
_EVENT_BOUNDARY_THRESHOLD = 0.5
_EVENT_IDLE_WEIGHT        = 0.7
_EVENT_TYPE_CHANGE_WEIGHT = 0.3


class EventFeatureExtractor:
    """
    Extracts event-based features and boundary evidence from events.json.

    extract_event_features  — per-frame descriptive features (count, recency,
                               type flags); interface and semantics unchanged.
    extract_event_boundary_evidence — boundary evidence in the same (n, 3)
                               format as the GUI and semantic extractors.
    extract_event_markers   — (t_ms, type) list for the FeatureDialog timeline.
    """

    _MARKER_EVENT_TYPES = {"mouse_click", "key_press", "key_release", "mouse_scroll"}

    _TYPE_CATEGORY = {
        "mouse_click":  "mouse",
        "mouse_scroll": "scroll",
        "key_press":    "keyboard",
        "key_release":  "keyboard",
    }

    @staticmethod
    def extract_event_markers(recording_session: RecordingSession) -> list[tuple[float, str]]:
        events = _read_events(recording_session.events_path)
        markers = [
            (float(ev.get("t_ms", 0.0)), str(ev.get("type", "")))
            for ev in events
            if ev.get("type") in EventFeatureExtractor._MARKER_EVENT_TYPES
        ]
        markers.sort(key=lambda m: m[0])
        return markers

    @staticmethod
    def extract_event_features(
        recording_session: RecordingSession,
        frame_times_ms: np.ndarray,
    ) -> np.ndarray:
        markers = EventFeatureExtractor.extract_event_markers(recording_session)
        n_frames = len(frame_times_ms)
        features = np.zeros((n_frames, 5), dtype=np.float32)

        if not markers:
            features[:, 1] = _MAX_EVENT_RECENCY_S
            return features

        marker_times_s = np.array([t_ms / 1000.0 for t_ms, _ in markers], dtype=np.float64)
        marker_types   = [ev_type for _, ev_type in markers]

        for i, t_ms in enumerate(frame_times_ms):
            t_s = float(t_ms) / 1000.0
            lo = int(np.searchsorted(marker_times_s, t_s - _EVENT_WINDOW_S, side="left"))
            hi = int(np.searchsorted(marker_times_s, t_s + _EVENT_WINDOW_S, side="right"))
            types_in_window = marker_types[lo:hi]

            features[i, 0] = hi - lo
            features[i, 2] = float(any(t == "mouse_click"                 for t in types_in_window))
            features[i, 3] = float(any(t in ("key_press", "key_release")  for t in types_in_window))
            features[i, 4] = float(any(t == "mouse_scroll"                for t in types_in_window))

            past_idx = int(np.searchsorted(marker_times_s, t_s, side="right")) - 1
            features[i, 1] = (
                min(t_s - marker_times_s[past_idx], _MAX_EVENT_RECENCY_S)
                if past_idx >= 0 else _MAX_EVENT_RECENCY_S
            )

        return features

    @staticmethod
    def extract_event_boundary_evidence(
        recording_session: RecordingSession,
        frame_times_ms: np.ndarray,
    ) -> np.ndarray:
        """
        Boundary evidence from the event stream, in the same (n, 3) format
        as GUIActionBoundaryExtractor and AudioBoundaryExtractor.

        Evidence model for each event e_i with preceding gap g_i:
          idle_component = min(g_i / _EVENT_IDLE_FULL_S, 1)   in [0, 1]
          type_component = 1 if input category changed vs. previous event
          score(e_i)     = 0.7 * idle + 0.3 * type_change

        The very first event gets idle_component = 1.0 (the workflow's
        first step necessarily starts there).

        Output columns:
          Col 0 — onset_flag:     1.0 at frames containing a scored event
          Col 1 — boundary_score: graded evidence [0, 1], Gaussian spread
          Col 2 — boundary_flag:  hard 0/1 at threshold (display only)
        """
        markers  = EventFeatureExtractor.extract_event_markers(recording_session)
        n_frames = len(frame_times_ms)
        features = np.zeros((n_frames, 3), dtype=np.float32)
        if not markers or n_frames == 0:
            return features

        frame_times_s = np.asarray(frame_times_ms, dtype=np.float64) / 1000.0
        duration_s = float(frame_times_s[-1] - frame_times_s[0])
        fps    = (n_frames - 1) / duration_s if duration_s > 0 else 1.0
        spread = max(1, int(_EVENT_SPREAD_S * fps))

        prev_t_s: float | None = None
        prev_cat: str | None   = None

        for t_ms, ev_type in markers:
            t_s = t_ms / 1000.0
            cat = EventFeatureExtractor._TYPE_CATEGORY.get(ev_type, "other")

            if prev_t_s is None:
                idle_component = 1.0
                type_component = 0.0
            else:
                idle_component = min((t_s - prev_t_s) / _EVENT_IDLE_FULL_S, 1.0)
                type_component = 1.0 if cat != prev_cat else 0.0

            score = _EVENT_IDLE_WEIGHT * idle_component + _EVENT_TYPE_CHANGE_WEIGHT * type_component

            center = min(max(int(np.searchsorted(frame_times_s, t_s)), 0), n_frames - 1)
            features[center, 0] = 1.0
            _apply_gaussian(features, 1, center, score, spread, spread)

            prev_t_s = t_s
            prev_cat = cat

        features[:, 2] = (features[:, 1] >= _EVENT_BOUNDARY_THRESHOLD).astype(np.float32)
        return features
