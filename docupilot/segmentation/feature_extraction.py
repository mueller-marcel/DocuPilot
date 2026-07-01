from typing import List, Tuple
from pathlib import Path

from docupilot.recording.session import RecordingSession

import json
import numpy as np

# Hop length used consistently across all feature extractions.
# At sr=22050 (librosa default): 512 samples ≈ 23 ms per frame.
_HOP_LENGTH = 512


# ── GUI Action Boundary Extractor ─────────────────────────────────────────────
#
# Detects action boundary EVIDENCE from screen recordings. A GUI state
# transition is located via pHash (cheap pre-filter only) and semantically
# verified via OmniParser: did new UI elements appear, or were existing
# elements modified in content?
#
# DESIGN PRINCIPLE (experiment-ready):
#   Extractors emit graded, calibrated evidence in [0, 1]. They make NO
#   boundary decisions. Thresholding, minimum-gap suppression and peak
#   picking happen ONCE, downstream, in the dataset builder / fusion head —
#   identically for every modality. Only then is a difference between two
#   modality subsets attributable to the modality itself, not to
#   per-extractor decision logic (required for the Shapley ablation).
#
# Scientific basis:
#   - Lu et al. (2024). OmniParser for Pure Vision Based GUI Agent. arXiv:2408.00203.
#   - Wornow et al. (2024): pixel pre-filter + semantic verification pipeline.
#   - RippleGUItester (2025): pixel diff → bounding boxes → LLM verification.
#   - ActionNet (Zhao et al., 2019): vision-only action segmentation in screencasts.
#
# Pipeline:
#   Phase 1 — pHash scan (streaming, hashes only, no frame storage)
#       Detects the ONSET frame of each potential transition.
#   Phase 2 — Pixel diff on onset pairs → changed-region bounding boxes.
#       For very strong pHash changes (full-screen transition) the pixel
#       diff degenerates; the WHOLE FRAME is then used as the single ROI so
#       OmniParser still verifies semantically. There is NO fast path that
#       bypasses semantic verification.
#   Phase 3 — OmniParser on regions → graded evidence:
#       evidence = max over regions of max(new_element_frac, modified_frac)
#   Phase 4 — Gaussian-spread evidence written into (T_v, 3) array.
#
# Output columns (shape and meaning unchanged):
#   Col 0 — onset_flag:          1.0 at detected transition onset frames
#   Col 1 — gui_boundary_score:  graded semantic evidence [0, 1] (was: constant 1.0)
#   Col 2 — gui_boundary_flag:   hard 0/1 at _GUI_BOUNDARY_THRESHOLD
#                                (backward-compat / visualisation only —
#                                 downstream code must use col 1)

# pHash distance to trigger onset detection (candidate PROPOSAL only,
# never a boundary decision).
_PHASH_ONSET_THRESHOLD = 0.12

# pHash distance above which the pixel diff is unreliable (full-screen
# transition). OmniParser then runs on the whole frame as ROI instead of
# being skipped. (Previously this threshold accepted boundaries blindly —
# that let scrolls / video playback through as false boundaries.)
_PHASH_FULLFRAME_THRESHOLD = 0.25

# Frames to look back/ahead around each onset for before/after comparison.
# At 10fps a click happens ~300ms before the visual response; 5 frames
# (500ms) captures the pre-click GUI state even for fast users.
_BEFORE_FRAME_OFFSET = 5
_AFTER_FRAME_OFFSET  = 5

# Minimum changed region: RESOLUTION-INDEPENDENT.
# The old fraction-based value (1.5% of frame area) was calibrated for one
# resolution and silently drifted on others (cursor ≈ 24×36 px regardless
# of resolution class). We therefore require a region to be larger than a
# cursor-sized box scaled to the recording height.
_MIN_DIFF_REGION_CURSOR_SCALE = 2.5   # region must exceed 2.5 cursor areas
_CURSOR_PX_AT_1080P = (24, 36)        # (w, h) of a typical pointer at 1080p

# OmniParser weights and confidence.
_OMNIPARSER_WEIGHTS = "weights/icon_detect/model.pt"
_OMNIPARSER_CONF    = 0.30

# IoU threshold: boxes above this are considered the "same" element.
_ELEMENT_IOU_THRESHOLD = 0.5

# Normalised MSE threshold inside a matched bounding box crop.
# If pixel content changed more than this, the element is "modified".
_MODIFIED_MSE_THRESHOLD = 0.02

# Backward-compat flag threshold for column 2 (display only).
_GUI_BOUNDARY_THRESHOLD = 0.5
_GUI_SPREAD_S           = 1.0

# Burst de-duplication of raw pHash onsets (an animation produces many
# consecutive onsets for ONE transition — this is signal hygiene, not a
# boundary decision).
_ONSET_DEDUP_WINDOW_S = 0.5


class GUIActionBoundaryExtractor:
    """
    Extracts graded GUI action-boundary EVIDENCE from screen recordings
    using: pHash onset proposal → pixel-diff region extraction →
    OmniParser semantic element comparison.

    The evidence value is derived entirely from OmniParser observations
    (fraction of new elements, fraction of modified elements). pHash and
    pixel-diff are cheap pre-filters that decide WHERE OmniParser runs —
    they never contribute to the evidence value itself and never make
    boundary decisions.
    """

    _PHASH_SIZE = 8

    # ── pHash helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _phash(gray: np.ndarray) -> object:
        import imagehash
        from PIL import Image
        return imagehash.phash(
            Image.fromarray(gray),
            hash_size=GUIActionBoundaryExtractor._PHASH_SIZE,
        )

    @staticmethod
    def _phash_dist(a: object, b: object) -> float:
        return float(a - b) / (GUIActionBoundaryExtractor._PHASH_SIZE ** 2)

    # ── Pixel-diff region extraction ──────────────────────────────────────────

    @staticmethod
    def _min_region_area(h: int, w: int) -> int:
        """
        Resolution-independent minimum region area: a multiple of the
        cursor footprint scaled to the recording height. Filters
        cursor-only movement at ANY resolution without a hand-tuned
        fraction that only holds for one clip.
        """
        scale = h / 1080.0
        cw, ch = _CURSOR_PX_AT_1080P
        cursor_area = (cw * scale) * (ch * scale)
        return int(cursor_area * _MIN_DIFF_REGION_CURSOR_SCALE)

    @staticmethod
    def _diff_regions(
        before_bgr: np.ndarray,
        after_bgr: np.ndarray,
    ) -> list[tuple[int, int, int, int]]:
        """
        Return bounding boxes of connected changed regions between two frames.

        :return: List of (x1, y1, x2, y2) in pixel coordinates.
        """
        import cv2

        h, w = before_bgr.shape[:2]
        min_area = GUIActionBoundaryExtractor._min_region_area(h, w)

        diff = cv2.absdiff(
            cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(after_bgr,  cv2.COLOR_BGR2GRAY),
        )
        _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes: list[tuple[int, int, int, int]] = []
        for lbl in range(1, n_labels):
            if int(stats[lbl, cv2.CC_STAT_AREA]) < min_area:
                continue
            x  = int(stats[lbl, cv2.CC_STAT_LEFT])
            y  = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            boxes.append((x, y, x + bw, y + bh))
        return boxes

    # ── OmniParser element detection ──────────────────────────────────────────

    @staticmethod
    def _detect_elements(
        model,
        frame_bgr: np.ndarray,
        roi: tuple[int, int, int, int],
    ) -> list[tuple[float, float, float, float]]:
        """
        Run OmniParser on a cropped ROI and return detected element boxes
        in normalised coordinates [0, 1] relative to the ROI.
        """
        x1, y1, x2, y2 = roi
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        results = model.predict(crop, verbose=False, conf=_OMNIPARSER_CONF)
        ch, cw = crop.shape[:2]
        boxes: list[tuple[float, float, float, float]] = []
        for result in results:
            for box in result.boxes.xyxy.tolist():
                bx1, by1, bx2, by2 = box
                boxes.append((bx1 / cw, by1 / ch, bx2 / cw, by2 / ch))
        return boxes

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        """Intersection over Union of two normalised boxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1)
        ub = (bx2 - bx1) * (by2 - by1)
        return inter / (ua + ub - inter)

    # ── Graded semantic evidence ──────────────────────────────────────────────

    @staticmethod
    def _new_element_fraction(
        before_boxes: list[tuple],
        after_boxes:  list[tuple],
    ) -> float:
        """
        Fraction of after-elements without a sufficiently overlapping match
        in before_boxes. GRADED replacement for the former boolean
        _has_new_elements: 0.9 (dialog opened, 90% new) is stronger
        evidence than 0.15 (one new list row) — the Random Forest needs
        that gradation to weigh modalities against each other.
        """
        if not after_boxes:
            return 0.0
        if not before_boxes:
            return 1.0
        new_count = sum(
            1 for ab in after_boxes
            if not any(
                GUIActionBoundaryExtractor._iou(ab, bb) >= _ELEMENT_IOU_THRESHOLD
                for bb in before_boxes
            )
        )
        return new_count / len(after_boxes)

    @staticmethod
    def _modified_element_fraction(
        before_boxes: list[tuple],
        after_boxes:  list[tuple],
        before_crop:  np.ndarray,
        after_crop:   np.ndarray,
    ) -> float:
        """
        Fraction of MATCHED elements whose pixel content changed
        significantly inside their bounding box (list got new entry, field
        filled, status changed). Hover highlights stay below the MSE
        threshold. GRADED replacement for the former boolean check.
        """
        import cv2

        if not before_boxes or not after_boxes:
            return 0.0
        ch, cw = before_crop.shape[:2]
        if ch == 0 or cw == 0:
            return 0.0

        modified_count = 0
        matched_count  = 0

        for ab in after_boxes:
            best_iou = 0.0
            best_bb  = None
            for bb in before_boxes:
                iou = GUIActionBoundaryExtractor._iou(ab, bb)
                if iou > best_iou:
                    best_iou = iou
                    best_bb  = bb
            if best_bb is None or best_iou < _ELEMENT_IOU_THRESHOLD:
                continue  # unmatched → contributes to new_element_fraction

            matched_count += 1

            ax1 = int(ab[0] * cw);      ay1 = int(ab[1] * ch)
            ax2 = int(ab[2] * cw);      ay2 = int(ab[3] * ch)
            bx1 = int(best_bb[0] * cw); by1 = int(best_bb[1] * ch)
            bx2 = int(best_bb[2] * cw); by2 = int(best_bb[3] * ch)

            crop_a = after_crop[ay1:ay2,  ax1:ax2]
            crop_b = before_crop[by1:by2, bx1:bx2]
            if crop_a.size == 0 or crop_b.size == 0:
                continue

            target = (max(1, ax2 - ax1), max(1, ay2 - ay1))
            crop_b_resized = cv2.resize(crop_b, target, interpolation=cv2.INTER_AREA)

            ga = cv2.cvtColor(crop_a,         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            gb = cv2.cvtColor(crop_b_resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            mse = float(np.mean((ga - gb) ** 2))
            if mse > _MODIFIED_MSE_THRESHOLD:
                modified_count += 1

        if matched_count == 0:
            return 0.0
        return modified_count / matched_count

    # ── Frame access (streaming, memory-bounded) ──────────────────────────────

    @staticmethod
    def _scan_phashes(video_path: str) -> tuple[list[object], int]:
        """
        Pass 1: stream all frames, keep ONLY the pHashes.
        Memory: O(T) hashes instead of O(T) full frames — a 10-minute
        1080p recording no longer needs ~20 GB of RAM.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0
        hashes: list[object] = []
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                hashes.append(GUIActionBoundaryExtractor._phash(gray))
        finally:
            cap.release()
        return hashes, len(hashes)

    @staticmethod
    def _grab_frames(video_path: str, frame_indices: set[int]) -> dict[int, np.ndarray]:
        """
        Pass 2: stream again and keep only the requested frames
        (before/after pairs around onsets). Sequential read is faster and
        more reliable than per-frame seeking for most codecs.
        """
        import cv2
        wanted = dict.fromkeys(frame_indices)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        grabbed: dict[int, np.ndarray] = {}
        idx = 0
        try:
            while wanted:
                ret, bgr = cap.read()
                if not ret:
                    break
                if idx in wanted:
                    grabbed[idx] = bgr
                    wanted.pop(idx)
                idx += 1
        finally:
            cap.release()
        return grabbed

    # ── Main extraction ───────────────────────────────────────────────────────

    @staticmethod
    def extract_gui_features(
        recording_session: RecordingSession,
        fps: float,
    ) -> np.ndarray:
        """
        Extract graded GUI action-boundary evidence from the video.

        :param recording_session: Recording session (MP4 path).
        :param fps: Video frame rate in frames per second.
        :return: np.ndarray of shape (T_v, 3), dtype float32.
        """
        video_path = str(recording_session.recording_path)

        # ── Phase 1: streaming pHash scan ──────────────────────────────────────
        phash_list, T_v = GUIActionBoundaryExtractor._scan_phashes(video_path)

        features = np.zeros((T_v, 3), dtype=np.float32)
        if T_v < _BEFORE_FRAME_OFFSET + _AFTER_FRAME_OFFSET + 1:
            return features

        phash_dists = [0.0] + [
            GUIActionBoundaryExtractor._phash_dist(phash_list[i - 1], phash_list[i])
            for i in range(1, T_v)
        ]

        onset_frames = [
            i for i in range(1, T_v) if phash_dists[i] >= _PHASH_ONSET_THRESHOLD
        ]
        for i in onset_frames:
            features[i, 0] = 1.0

        if not onset_frames:
            return features

        # Burst de-duplication (one transition = many consecutive onsets).
        dedup_gap = max(1, int(fps * _ONSET_DEDUP_WINDOW_S))
        deduped: list[int] = [onset_frames[0]]
        for f in onset_frames[1:]:
            if f - deduped[-1] > dedup_gap:
                deduped.append(f)
        onset_frames = deduped

        # ── Phase 2 prep: fetch only the needed before/after frames ───────────
        needed: set[int] = set()
        pairs: list[tuple[int, int, int]] = []  # (onset, before_idx, after_idx)
        for onset in onset_frames:
            b = max(0, onset - _BEFORE_FRAME_OFFSET)
            a = min(T_v - 1, onset + _AFTER_FRAME_OFFSET)
            needed.update((b, a))
            pairs.append((onset, b, a))
        frames = GUIActionBoundaryExtractor._grab_frames(video_path, needed)

        # ── Phase 2 + 3: regions → OmniParser → graded evidence ───────────────
        from ultralytics import YOLO
        model = YOLO(_OMNIPARSER_WEIGHTS)

        evidence: list[tuple[int, float]] = []
        for onset, before_idx, after_idx in pairs:
            before_bgr = frames.get(before_idx)
            after_bgr  = frames.get(after_idx)
            if before_bgr is None or after_bgr is None:
                continue

            h, w = before_bgr.shape[:2]

            # Full-screen transition → pixel diff degenerates → whole frame
            # becomes the single ROI. OmniParser STILL verifies semantically
            # (no blind acceptance: scrolls / video playback are rejected
            # because their elements match between frames).
            if phash_dists[onset] >= _PHASH_FULLFRAME_THRESHOLD:
                regions = [(0, 0, w, h)]
            else:
                regions = GUIActionBoundaryExtractor._diff_regions(before_bgr, after_bgr)
                if not regions:
                    continue

            best = 0.0
            for roi in regions:
                x1, y1, x2, y2 = roi
                before_elems = GUIActionBoundaryExtractor._detect_elements(
                    model, before_bgr, roi
                )
                after_elems = GUIActionBoundaryExtractor._detect_elements(
                    model, after_bgr, roi
                )
                new_frac = GUIActionBoundaryExtractor._new_element_fraction(
                    before_elems, after_elems
                )
                mod_frac = GUIActionBoundaryExtractor._modified_element_fraction(
                    before_elems, after_elems,
                    before_bgr[y1:y2, x1:x2], after_bgr[y1:y2, x1:x2],
                )
                best = max(best, new_frac, mod_frac)
                if best >= 1.0:
                    break

            if best > 0.0:
                evidence.append((onset, float(np.clip(best, 0.0, 1.0))))

        # ── Phase 4: Gaussian-spread evidence (NO min-gap suppression here —
        #    candidate merging is the dataset builder's job) ────────────────────
        spread = int(_GUI_SPREAD_S * fps)
        for frame_idx, score in evidence:
            lo = max(0, frame_idx - spread)
            hi = min(T_v, frame_idx + spread + 1)
            offsets  = np.arange(lo, hi) - frame_idx
            gaussian = np.exp(-0.5 * (offsets / max(spread / 2, 1)) ** 2)
            features[lo:hi, 1] = np.maximum(
                features[lo:hi, 1], (score * gaussian).astype(np.float32)
            )

        # Column 2: backward-compat display flag only.
        features[:, 2] = (features[:, 1] >= _GUI_BOUNDARY_THRESHOLD).astype(np.float32)
        return features


class TranscriptionExtractor:
    """
    Transcribes the audio track of a RecordingSession using OpenAI Whisper.
    Returns the full transcript text and the raw word-level timestamp data.
    The whisper import is deferred, so the model is only loaded when needed.
    """

    # "base" missed German instructional vocabulary in pilot recordings;
    # "small" is the best quality/latency trade-off for offline extraction.
    _MODEL_SIZE = "small"

    @staticmethod
    def extract_transcript(recording_session: RecordingSession) -> Tuple[str, List[dict]]:
        """
        Transcribe the audio track of a recording session.

        :param recording_session: The recording session whose MP4 should be transcribed.
        :return: A tuple of (full_text, words) where full_text is the complete
            transcript as a single string and words is a flat list of Whisper
            word dicts, each containing "word", "start", and "end" keys
            (times in seconds).  Returns ("", []) when no speech is detected.
        """
        import whisper  # deferred — only load when transcription is requested

        model = whisper.load_model(TranscriptionExtractor._MODEL_SIZE)
        result = model.transcribe(
            str(recording_session.recording_path),
            verbose=False,
            language="de",
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        full_text: str = result.get("text", "").strip()

        words: List[dict] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append(w)

        return full_text, words


class SentenceSegmenter:
    """
    Segments a German transcript into sentences with timestamps using spaCy.

    NOTE: The former hard "es"-subject filter (dropping suspected system-
    reaction sentences BEFORE classification) has been removed. Sentence-
    type discrimination is the NLI classifier's job — a hard syntactic
    pre-filter deleted evidence the classifier (and later the Random
    Forest) never got to see, and misfired on sentences like
    "Es öffnet sich das Menü, klicken Sie dann auf ...".

    Uses de_core_news_lg for sentence boundary detection; falls back to
    de_core_news_md or de_core_news_sm when lg is not installed.
    """

    _MODEL_NAME = "de_core_news_lg"
    _MODEL_FALLBACKS = ["de_core_news_md", "de_core_news_sm"]

    @staticmethod
    def segment(
        full_text: str,
        words: List[dict],
    ) -> List[Tuple[float, str]]:
        """
        Split the transcript into sentences and return their timestamps.

        :param full_text: Full transcript string as returned by Whisper.
        :param words: Flat list of Whisper word dicts with "word", "start",
            and "end" keys (times in seconds).
        :return: List of (t_ms, sentence_text) tuples sorted by time.
            Returns an empty list when the transcript is empty.
        """
        if not full_text or not words:
            return []

        import spacy  # deferred

        nlp = None
        for model_name in [SentenceSegmenter._MODEL_NAME, *SentenceSegmenter._MODEL_FALLBACKS]:
            try:
                nlp = spacy.load(model_name)
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
        # Cursor advances PAST each match so repeated words ("klicken ...
        # klicken") map to their own occurrence instead of the first one.
        char_to_time: List[Tuple[int, float]] = []
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

        sentences: List[Tuple[float, str]] = []
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            t_s = _resolve_time(sent.start_char)
            sentences.append((t_s * 1000.0, text))

        return sentences


_NLI_HYPOTHESIS_TEMPLATE = (
    "Dieser Satz ist eine direkte Handlungsanweisung an einen menschlichen "
    "Benutzer, eine Aktion in der Software auszuführen: {}."
)
_NLI_CANDIDATE_LABELS = ["wahr", "falsch"]
_ACTION_LABEL = "wahr"

# Backward-compat display threshold for column 2 ONLY. The graded NLI score
# in column 1 is what downstream code (dataset builder, Random Forest)
# consumes — no evidence is deleted below this value anymore.
_NLI_THRESHOLD = 0.80


class SemanticAudioFeatureExtractor:
    """
    Extracts graded semantic boundary evidence from the audio transcript of
    a RecordingSession using a zero-shot NLI classifier.

    CHANGES vs. previous version (interfaces unchanged):
      - EVERY sentence gets its P(ACTION) written as graded evidence —
        the hard threshold and min-gap suppression no longer delete
        evidence before the Random Forest sees it.
      - The "sentence 0 is always a boundary" prior has been removed: it
        injected a label-independent decision that would have leaked into
        the audio modality's measured contribution.
      - NLI calls are batched (one pipeline call for all sentences).

    Output columns (shape and meaning unchanged):
      Col 0 — sentence_boundary: 1.0 at frames where any sentence starts
      Col 1 — action_score:      graded P(ACTION) as Gaussian evidence
      Col 2 — action_flag:       hard 0/1 at _NLI_THRESHOLD (display only)
    """

    _NLI_SPREAD_S = 1.5
    _NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    @staticmethod
    def extract_semantic_features(
        full_text: str,
        words: List[dict],
        n_frames: int,
        sampling_rate: float,
    ) -> np.ndarray:
        """
        Extract semantic boundary features aligned to the audio frame grid.

        :param full_text: Full transcript string from TranscriptionExtractor.
        :param words: Word-level Whisper dicts with "word", "start", "end".
        :param n_frames: Number of frames (must match VideoFeatureExtractor output).
        :param sampling_rate: Audio sampling rate returned by librosa.load.
        :return: np.ndarray of shape (n_frames, 3), dtype float32.
        """
        features = np.zeros((n_frames, 3), dtype=np.float32)

        if not full_text or not words:
            return features

        frames_per_second = sampling_rate / _HOP_LENGTH

        # ── Step 1: sentence segmentation ─────────────────────────────────────
        sentences = SentenceSegmenter.segment(full_text, words)
        if not sentences:
            return features

        # ── Step 2: mark all sentence-start frames (column 0) ─────────────────
        for t_ms, _ in sentences:
            frame_idx = int(round((t_ms / 1000.0) * frames_per_second))
            if 0 <= frame_idx < n_frames:
                features[frame_idx, 0] = 1.0

        # ── Step 3: batched NLI — graded P(ACTION) for EVERY sentence ─────────
        from transformers import pipeline  # deferred

        zero_shot = pipeline(
            "zero-shot-classification",
            model=SemanticAudioFeatureExtractor._NLI_MODEL,
        )
        texts = [text for _, text in sentences]
        results = zero_shot(
            texts,
            candidate_labels=_NLI_CANDIDATE_LABELS,
            hypothesis_template=_NLI_HYPOTHESIS_TEMPLATE,
        )
        if isinstance(results, dict):  # single-sentence transcripts
            results = [results]

        boundary_evidence: List[Tuple[float, float]] = []
        for (t_ms, _), result in zip(sentences, results):
            # P(ACTION) regardless of label ranking — graded, not thresholded.
            idx = result["labels"].index(_ACTION_LABEL)
            p_action = float(result["scores"][idx])
            boundary_evidence.append((t_ms / 1000.0, p_action))

        # ── Step 4: write graded scores as asymmetric Gaussians ───────────────
        # Wide spread before the boundary (uncertain when the action starts),
        # narrow after (avoid bleeding into the following RESULT sentence).
        spread_before = int(SemanticAudioFeatureExtractor._NLI_SPREAD_S * frames_per_second)
        spread_after  = int(SemanticAudioFeatureExtractor._NLI_SPREAD_S * 0.3 * frames_per_second)

        for t_s, score in boundary_evidence:
            center = int(round(t_s * frames_per_second))

            lo = max(0, center - spread_before)
            offsets_l = np.arange(lo, center) - center
            sigma_l = max(spread_before / 2, 1)
            gaussian_l = np.exp(-0.5 * (offsets_l / sigma_l) ** 2)
            features[lo:center, 1] = np.maximum(
                features[lo:center, 1], (score * gaussian_l).astype(np.float32)
            )

            if 0 <= center < n_frames:
                features[center, 1] = max(features[center, 1], np.float32(score))

            hi = min(n_frames, center + spread_after + 1)
            offsets_r = np.arange(center + 1, hi) - center
            sigma_r = max(spread_after / 2, 1)
            gaussian_r = np.exp(-0.5 * (offsets_r / sigma_r) ** 2)
            features[center + 1:hi, 1] = np.maximum(
                features[center + 1:hi, 1], (score * gaussian_r).astype(np.float32)
            )

        # Column 2: backward-compat display flag only.
        features[:, 2] = (features[:, 1] >= _NLI_THRESHOLD).astype(np.float32)

        return features


# Window size to track events in seconds
_EVENT_WINDOW_S = 0.25

# Maximum size for the events
_MAX_EVENT_RECENCY_S = 5.0

# Idle gap after which the NEXT event is considered a step-onset candidate.
_EVENT_IDLE_FULL_S = 3.0     # gap ≥ 3 s → idle component saturates at 1.0

# Gaussian spread for event boundary evidence (matches GUI spread so all
# modalities emit comparable evidence shapes).
_EVENT_SPREAD_S = 1.0

# Backward-compat display threshold for the event evidence flag column.
_EVENT_BOUNDARY_THRESHOLD = 0.5

# Weighting of the two evidence components.
_EVENT_IDLE_WEIGHT = 0.7
_EVENT_TYPE_CHANGE_WEIGHT = 0.3


class EventFeatureExtractor:
    """
    Provides event feature extraction.

    extract_event_features (UNCHANGED interface & semantics) returns the
    descriptive per-frame features (count, recency, type flags).

    extract_event_boundary_evidence (NEW, additive) turns the raw event
    stream into a boundary HYPOTHESIS in the same (n_frames, 3) evidence
    format as GUI and semantic audio — the missing piece that lets the
    events modality participate in candidate generation on equal footing:
      - idle-gap onset: first event after a pause (a new step begins)
      - input-type transition: keyboard → mouse or vice versa
    """

    # Event types that are considered for the event feature extraction.
    _MARKER_EVENT_TYPES = {"mouse_click", "key_press", "key_release", "mouse_scroll"}

    # Coarse input categories for transition detection.
    _TYPE_CATEGORY = {
        "mouse_click":  "mouse",
        "mouse_scroll": "scroll",
        "key_press":    "keyboard",
        "key_release":  "keyboard",
    }

    @staticmethod
    def extract_event_markers(recording_session: RecordingSession) -> List[Tuple[float, str]]:
        """
        Extract the event markers from the events.json file.
        :param recording_session: The recording session contains the path to events.json.
        :return: A list of tuples (t_ms, type) where t_ms is the timestamp of the event
        """
        events = EventFeatureExtractor._read_events(recording_session.events_path)

        markers = [
            (float(ev.get("t_ms", 0.0)), str(ev.get("type", "")))
            for ev in events
            if ev.get("type") in EventFeatureExtractor._MARKER_EVENT_TYPES
        ]
        markers.sort(key=lambda marker: marker[0])

        return markers

    @staticmethod
    def extract_event_features(
        recording_session: RecordingSession,
        frame_times_ms: np.ndarray,
    ) -> np.ndarray:
        """
        Extract the event features
        :param recording_session: The recording session containing the path to events.json.
        :param frame_times_ms: The frame times in milliseconds.
        :return: A vector of event features.
        """
        markers = EventFeatureExtractor.extract_event_markers(recording_session)
        n_frames = len(frame_times_ms)
        features = np.zeros((n_frames, 5), dtype=np.float32)

        if not markers:
            features[:, 1] = _MAX_EVENT_RECENCY_S
            return features

        marker_times_s = np.array([t_ms / 1000.0 for t_ms, _ in markers], dtype=np.float64)
        marker_types = [ev_type for _, ev_type in markers]

        for i, t_ms in enumerate(frame_times_ms):
            t_s = float(t_ms) / 1000.0
            lo = np.searchsorted(marker_times_s, t_s - _EVENT_WINDOW_S, side="left")
            hi = np.searchsorted(marker_times_s, t_s + _EVENT_WINDOW_S, side="right")
            types_in_window = marker_types[lo:hi]

            features[i, 0] = hi - lo
            features[i, 2] = float(any(t == "mouse_click" for t in types_in_window))
            features[i, 3] = float(any(t in ("key_press", "key_release") for t in types_in_window))
            features[i, 4] = float(any(t == "mouse_scroll" for t in types_in_window))
            past_idx = np.searchsorted(marker_times_s, t_s, side="right") - 1
            if past_idx >= 0:
                features[i, 1] = min(t_s - marker_times_s[past_idx], _MAX_EVENT_RECENCY_S)
            else:
                features[i, 1] = _MAX_EVENT_RECENCY_S

        return features

    @staticmethod
    def extract_event_boundary_evidence(
        recording_session: RecordingSession,
        frame_times_ms: np.ndarray,
    ) -> np.ndarray:
        """
        NEW (additive): boundary evidence from the event stream, in the
        same (n_frames, 3) format as the GUI and semantic extractors.

        Evidence model per event e_i with preceding gap g_i:
          idle_component  = min(g_i / _EVENT_IDLE_FULL_S, 1)   in [0, 1]
          type_component  = 1 if input category changed vs. previous event
          score(e_i)      = 0.7 * idle_component + 0.3 * type_component

        The very first event of a session gets idle_component = 1.0
        (the workflow's first step necessarily begins there).

        Output columns:
          Col 0 — onset_flag:      1.0 at frames containing a scored event
          Col 1 — boundary_score:  graded evidence [0, 1], Gaussian spread
          Col 2 — boundary_flag:   hard 0/1 at threshold (display only)
        """
        markers = EventFeatureExtractor.extract_event_markers(recording_session)
        n_frames = len(frame_times_ms)
        features = np.zeros((n_frames, 3), dtype=np.float32)
        if not markers or n_frames == 0:
            return features

        frame_times_s = np.asarray(frame_times_ms, dtype=np.float64) / 1000.0
        duration_s = float(frame_times_s[-1] - frame_times_s[0])
        fps = (n_frames - 1) / duration_s if duration_s > 0 else 1.0
        spread = max(1, int(_EVENT_SPREAD_S * fps))

        prev_t_s: float | None = None
        prev_cat: str | None = None

        for t_ms, ev_type in markers:
            t_s = t_ms / 1000.0
            cat = EventFeatureExtractor._TYPE_CATEGORY.get(ev_type, "other")

            if prev_t_s is None:
                idle_component = 1.0
                type_component = 0.0
            else:
                gap = t_s - prev_t_s
                idle_component = min(gap / _EVENT_IDLE_FULL_S, 1.0)
                type_component = 1.0 if cat != prev_cat else 0.0

            score = (
                _EVENT_IDLE_WEIGHT * idle_component
                + _EVENT_TYPE_CHANGE_WEIGHT * type_component
            )

            center = int(np.searchsorted(frame_times_s, t_s))
            center = min(max(center, 0), n_frames - 1)
            features[center, 0] = 1.0

            lo = max(0, center - spread)
            hi = min(n_frames, center + spread + 1)
            offsets = np.arange(lo, hi) - center
            gaussian = np.exp(-0.5 * (offsets / max(spread / 2, 1)) ** 2)
            features[lo:hi, 1] = np.maximum(
                features[lo:hi, 1], (score * gaussian).astype(np.float32)
            )

            prev_t_s = t_s
            prev_cat = cat

        features[:, 2] = (features[:, 1] >= _EVENT_BOUNDARY_THRESHOLD).astype(np.float32)
        return features

    @staticmethod
    def _read_events(events_path: Path) -> List[dict]:
        """
        Read the events.json file
        :param events_path: The path to the file
        :return: A list of events
        """
        try:
            with events_path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []