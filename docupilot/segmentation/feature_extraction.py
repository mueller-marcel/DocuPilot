from __future__ import annotations

import json
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
    For the semantic extractor: spread_lo > spread_hi (wide before, narrow after).
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
# Detects action-completion EVIDENCE from screen recordings using only the
# video file. Both signals are derived purely from pixel data so the video
# modality stays independent of audio and events for Shapley analysis.
#
#   Signal 1 — CLIP semantic embeddings (primary):
#     Frames are sampled at _CLIP_SAMPLE_FPS and embedded by a pretrained
#     CLIP vision encoder. Cosine distance between consecutive embeddings
#     measures the semantic distance between UI states: high distance means
#     the UI transitioned to a meaningfully different state, which is the
#     visual signature of a completed action. This catches dialog opens/
#     closes, form submissions showing a result page, panel replacements,
#     and navigation — any change that is semantically significant — while
#     ignoring cosmetic cursor movement and animation noise that confuse
#     pixel-diff and even pHash approaches.
#
#   Signal 2 — Multi-scale pHash (secondary):
#     Three perceptual hashes per frame (full, left-half, right-half) detect
#     structural layout changes at full video frame rate. This bridges the
#     temporal gaps between CLIP samples and contributes where CLIP misses
#     fast visual events (tooltip flashes, highlight state changes).
#
# DESIGN PRINCIPLE (unchanged):
#   Both signals emit graded evidence in [0, 1]. No boundary decision is
#   made here. Thresholding and peak picking happen once, downstream, in
#   dataset_builder — identically for every modality (Shapley prerequisite).
#
# Output columns (shape and semantics unchanged):
#   Col 0 — onset_flag:         1.0 at detected transition frames
#   Col 1 — gui_boundary_score: graded evidence [0, 1], Gaussian spread
#   Col 2 — gui_boundary_flag:  hard 0/1 at threshold (display only;
#                                dataset_builder uses col 1, not col 2)

_CLIP_MODEL_NAME     = "openai/clip-vit-base-patch32"
_CLIP_SAMPLE_FPS     = 1.0   # CLIP inference rate; 1 fps limits compute
_CLIP_BATCH_SIZE     = 32    # frames per CLIP forward pass
_CLIP_ONSET_THRESH   = 0.03  # cosine distance [0, 2] to register an onset
_CLIP_SATURATE       = 0.20  # distance at which evidence reaches cap
_CLIP_EVIDENCE_CAP   = 0.90  # primary signal: high cap

_PHASH_SIZE          = 8
_PHASH_ONSET_THRESH  = 0.08  # fraction of differing hash bits
_PHASH_SATURATE      = 0.35
_PHASH_EVIDENCE_CAP  = 0.50  # secondary signal: lower cap
_PHASH_DEDUP_WIN_S   = 0.5   # burst de-dup: one animation = one onset

_GUI_SPREAD_S           = 1.0
_GUI_BOUNDARY_THRESHOLD = 0.5


class GUIActionBoundaryExtractor:
    """
    Extracts graded GUI action-completion evidence from screen recordings.
    Uses only the video file — no events, no audio (Shapley independence).

    Signal 1 (primary): CLIP semantic embeddings detect meaningful UI state
    transitions that signal a completed action (high cosine distance).

    Signal 2 (secondary): Multi-scale pHash detects structural layout
    changes at full frame rate, bridging gaps between CLIP samples.
    """

    # ── pHash helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _phash(gray: np.ndarray) -> object:
        import imagehash
        from PIL import Image
        return imagehash.phash(Image.fromarray(gray), hash_size=_PHASH_SIZE)

    @staticmethod
    def _phash_dist(a: object, b: object) -> float:
        return float(a - b) / (_PHASH_SIZE ** 2)

    @staticmethod
    def _max_phash_dist(a: tuple, b: tuple) -> float:
        d = GUIActionBoundaryExtractor._phash_dist
        return max(d(a[0], b[0]), d(a[1], b[1]), d(a[2], b[2]))

    # ── pHash scan (full video, multi-scale) ──────────────────────────────────

    @staticmethod
    def _scan_phashes(video_path: str) -> tuple[list[tuple], int]:
        """
        Stream all frames; store (full, left-half, right-half) pHash triples.
        Memory: O(T) hash triples — no pixel data retained.
        """
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], 0
        hashes: list[tuple] = []
        ph = GUIActionBoundaryExtractor._phash
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                mid = gray.shape[1] // 2
                hashes.append((ph(gray), ph(gray[:, :mid]), ph(gray[:, mid:])))
        finally:
            cap.release()
        return hashes, len(hashes)

    # ── CLIP embedding extraction ──────────────────────────────────────────────

    @staticmethod
    def _clip_sample_indices(total_frames: int, video_fps: float) -> list[int]:
        """Evenly spaced frame indices at _CLIP_SAMPLE_FPS."""
        step = max(1, round(video_fps / _CLIP_SAMPLE_FPS))
        return list(range(0, total_frames, step))

    @staticmethod
    def _extract_clip_embeddings(video_path: str, indices: list[int]) -> np.ndarray:
        """
        Extract L2-normalized CLIP image embeddings for the requested frames.

        Streams the video in a single pass, collecting only the indexed
        frames, then runs the CLIP vision encoder in batches.

        Returns shape (len(indices), 512), float32.
        """
        if not indices:
            return np.zeros((0, 512), dtype=np.float32)

        import cv2
        import torch
        from PIL import Image
        from transformers import CLIPProcessor, CLIPVisionModelWithProjection

        # CLIPVisionModelWithProjection returns output.image_embeds — a plain
        # tensor of projected features — regardless of transformers version.
        model = CLIPVisionModelWithProjection.from_pretrained(_CLIP_MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
        model.eval()

        wanted = set(indices)
        frames_bgr: dict[int, np.ndarray] = {}
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        try:
            while wanted:
                ret, bgr = cap.read()
                if not ret:
                    break
                if frame_idx in wanted:
                    frames_bgr[frame_idx] = bgr
                    wanted.discard(frame_idx)
                frame_idx += 1
        finally:
            cap.release()

        pil_frames = [
            Image.fromarray(cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB))
            for i in indices
            if i in frames_bgr
        ]
        if not pil_frames:
            return np.zeros((0, 512), dtype=np.float32)

        embeddings: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pil_frames), _CLIP_BATCH_SIZE):
                batch = pil_frames[start:start + _CLIP_BATCH_SIZE]
                pixel_values = processor(images=batch, return_tensors="pt")["pixel_values"]
                feats = model(pixel_values=pixel_values).image_embeds
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embeddings.append(feats.cpu().numpy())

        return np.concatenate(embeddings, axis=0).astype(np.float32)

    # ── Main extraction ───────────────────────────────────────────────────────

    @staticmethod
    def extract_gui_features(
        recording_session: RecordingSession,
        fps: float,
    ) -> np.ndarray:
        """
        Extract graded GUI action-completion evidence from the video.

        :param recording_session: Recording session (.recording_path → MP4).
        :param fps: Video frame rate in frames per second.
        :return: np.ndarray of shape (T_v, 3), dtype float32.
        """
        video_path = str(recording_session.recording_path)
        spread = max(1, int(_GUI_SPREAD_S * fps))

        # ── Pass 1: multi-scale pHash (full video, fast) ──────────────────────
        phash_list, T_v = GUIActionBoundaryExtractor._scan_phashes(video_path)

        features = np.zeros((T_v, 3), dtype=np.float32)
        if T_v < 2:
            return features

        phash_dists = [0.0] + [
            GUIActionBoundaryExtractor._max_phash_dist(phash_list[i - 1], phash_list[i])
            for i in range(1, T_v)
        ]

        onset_raw = [i for i in range(1, T_v) if phash_dists[i] >= _PHASH_ONSET_THRESH]
        for i in onset_raw:
            features[i, 0] = 1.0

        dedup_gap = max(1, int(fps * _PHASH_DEDUP_WIN_S))
        onset_deduped: list[int] = []
        for f in onset_raw:
            if not onset_deduped or f - onset_deduped[-1] > dedup_gap:
                onset_deduped.append(f)

        sat_range_ph = _PHASH_SATURATE - _PHASH_ONSET_THRESH
        for f in onset_deduped:
            raw_score = (phash_dists[f] - _PHASH_ONSET_THRESH) / sat_range_ph
            score = float(np.clip(raw_score, 0.0, 1.0)) * _PHASH_EVIDENCE_CAP
            _apply_gaussian(features, 1, f, score, spread, spread)

        # ── Pass 2: CLIP semantic embeddings (sampled, primary) ───────────────
        clip_indices = GUIActionBoundaryExtractor._clip_sample_indices(T_v, fps)
        embeddings = GUIActionBoundaryExtractor._extract_clip_embeddings(
            video_path, clip_indices
        )

        if embeddings.shape[0] >= 2:
            # Cosine distance of L2-normalized vectors = 1 − dot product.
            dots = np.einsum("nd,nd->n", embeddings[:-1], embeddings[1:])
            cos_dists = (1.0 - np.clip(dots, -1.0, 1.0)).astype(np.float32)

            sat_range_cl = _CLIP_SATURATE - _CLIP_ONSET_THRESH
            for j, dist in enumerate(cos_dists):
                if dist < _CLIP_ONSET_THRESH:
                    continue
                # Assign evidence to the later frame (where the change became visible).
                video_frame = clip_indices[j + 1] if j + 1 < len(clip_indices) else T_v - 1
                if video_frame >= T_v:
                    continue
                raw_score = float(dist - _CLIP_ONSET_THRESH) / sat_range_cl
                score = float(np.clip(raw_score, 0.0, 1.0)) * _CLIP_EVIDENCE_CAP
                features[video_frame, 0] = 1.0
                _apply_gaussian(features, 1, video_frame, score, spread, spread)

        features[:, 2] = (features[:, 1] >= _GUI_BOUNDARY_THRESHOLD).astype(np.float32)
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


# ── Semantic Audio Feature Extractor ──────────────────────────────────────────

_NLI_HYPOTHESIS_TEMPLATE = (
    "Dieser Satz ist eine direkte Handlungsanweisung an einen menschlichen "
    "Benutzer, eine Aktion in der Software auszuführen: {}."
)
_NLI_CANDIDATE_LABELS = ["wahr", "falsch"]
_ACTION_LABEL = "wahr"

# Backward-compat display threshold for column 2 ONLY.
_NLI_THRESHOLD = 0.80


class SemanticAudioFeatureExtractor:
    """
    Extracts graded semantic boundary evidence from the audio transcript using
    a zero-shot NLI classifier.

    Every sentence gets its P(ACTION) written as graded evidence — no
    hard threshold or min-gap suppression deletes evidence before the
    Random Forest sees it. NLI calls are batched (one pipeline call).

    Output columns:
      Col 0 — sentence_boundary: 1.0 at sentence-start frames
      Col 1 — action_score:      graded P(ACTION) as asymmetric Gaussian
      Col 2 — action_flag:       hard 0/1 at _NLI_THRESHOLD (display only)
    """

    _NLI_SPREAD_S = 1.5
    _NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    @staticmethod
    def extract_semantic_features(
        full_text: str,
        words: list[dict],
        n_frames: int,
        sampling_rate: float,
    ) -> np.ndarray:
        features = np.zeros((n_frames, 3), dtype=np.float32)
        if not full_text or not words:
            return features

        frames_per_second = sampling_rate / _HOP_LENGTH

        sentences = SentenceSegmenter.segment(full_text, words)
        if not sentences:
            return features

        # Mark all sentence-start frames (column 0).
        for t_ms, _ in sentences:
            frame_idx = int(round((t_ms / 1000.0) * frames_per_second))
            if 0 <= frame_idx < n_frames:
                features[frame_idx, 0] = 1.0

        # Batched NLI — graded P(ACTION) for every sentence.
        from transformers import pipeline

        zero_shot = pipeline(
            "zero-shot-classification",
            model=SemanticAudioFeatureExtractor._NLI_MODEL,
        )
        results = zero_shot(
            [text for _, text in sentences],
            candidate_labels=_NLI_CANDIDATE_LABELS,
            hypothesis_template=_NLI_HYPOTHESIS_TEMPLATE,
        )
        if isinstance(results, dict):
            results = [results]

        # Asymmetric Gaussian: wide before (uncertain action start),
        # narrow after (avoid bleeding into the following result sentence).
        spread_before = int(SemanticAudioFeatureExtractor._NLI_SPREAD_S * frames_per_second)
        spread_after  = int(SemanticAudioFeatureExtractor._NLI_SPREAD_S * 0.3 * frames_per_second)

        for (t_ms, _), result in zip(sentences, results):
            idx = result["labels"].index(_ACTION_LABEL)
            p_action = float(result["scores"][idx])
            center = int(round((t_ms / 1000.0) * frames_per_second))
            _apply_gaussian(features, 1, center, p_action, spread_before, spread_after)

        features[:, 2] = (features[:, 1] >= _NLI_THRESHOLD).astype(np.float32)
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
        as GUIActionBoundaryExtractor and SemanticAudioFeatureExtractor.

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
