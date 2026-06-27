from typing import List, Tuple
from pathlib import Path
from scipy.signal import find_peaks

from docupilot.recording.session import RecordingSession

import json
import librosa
import numpy as np

# Hop length used consistently across all feature extractions.
# At sr=22050 (librosa default): 512 samples ≈ 23 ms per frame.
_HOP_LENGTH = 512

# Frame length used for both RMS and pitch extraction so that both end up
# with the same number of frames (T) and stay perfectly aligned.
_FRAME_LENGTH = 2048

# Fundamental-frequency range for speech (covers low male to high female
# voices); used by the YIN pitch tracker.
_F0_MIN = 80.0
_F0_MAX = 400.0

# The threshold to detect silence in the RMS energy envelope.
_SILENCE_THRESHOLD_DB = -35.0

# Minimum length (seconds) of a silent run to count as a speech pause
# rather than a brief dip between syllables.
_MIN_PAUSE_DURATION_S = 0.2

# Window (seconds) immediately before/after a pause used to characterize
# the pitch reset and energy jump across that pause.
_BOUNDARY_WINDOW_S = 0.2

# Window (seconds) used to estimate the local speech rate from syllable-nucleus peaks in the energy envelope.
_SPEECH_RATE_WINDOW_S = 1.0

# Minimum spacing (seconds) between two syllable-nucleus peaks.
_MIN_SYLLABLE_SPACING_S = 0.1


class AudioFeatureExtractor:
    """
    Service that extracts the audio features from the mp4 file.
    """

    @staticmethod
    def extract_audio_features(recording_session: RecordingSession) -> np.ndarray:
        """
        Extract the audio features from the recording session.

        :param recording_session: The recording session contains the path to the mp4 file.
        :return: np.ndarray of shape (T, 5) – one 5-dim feature vector per frame.
        """

        audio, sampling_rate = librosa.load(recording_session.recording_path)

        # RMS energy
        rms = librosa.effects.feature.rms(
            y=audio, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH
        )[0]

        # Pitch
        pitch = librosa.yin(
            audio,
            fmin=_F0_MIN,
            fmax=_F0_MAX,
            sr=sampling_rate,
            frame_length=_FRAME_LENGTH,
            hop_length=_HOP_LENGTH,
        )

        # Speech pauses
        pause_segments = AudioFeatureExtractor._detect_pause_segments(rms, sampling_rate)

        # Cues for speech pauses
        pause_duration = AudioFeatureExtractor._compute_pause_duration(
            rms, pause_segments, sampling_rate
        )

        # Pitch reset
        pitch_reset = AudioFeatureExtractor._compute_pitch_reset(
            pitch, pause_segments, sampling_rate
        )

        # Energy jumps
        energy_jump = AudioFeatureExtractor._compute_energy_jump(rms, pause_segments, sampling_rate)

        # Speech rate
        speech_rate = AudioFeatureExtractor._compute_speech_rate(rms, sampling_rate)

        # Number of frames to use for each feature vector.
        n_frames = min(
            len(rms),
            len(pitch),
            len(pause_duration),
            len(pitch_reset),
            len(energy_jump),
            len(speech_rate),
        )

        return np.stack(
            [
                rms[:n_frames],
                pause_duration[:n_frames],
                pitch_reset[:n_frames],
                energy_jump[:n_frames],
                speech_rate[:n_frames],
            ],
            axis=1,
        ).astype(np.float32)

    @staticmethod
    def _detect_pause_segments(
        rms: np.ndarray, sampling_rate: int | float
    ) -> List[Tuple[int, int]]:
        """
        Detect speech pauses
        :param rms: The rms energy vector
        :param sampling_rate: The sampling rate of the audio
        :return: A list of tuples (start, end) of pause segments.
        """

        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        is_silent = rms_db < _SILENCE_THRESHOLD_DB

        frames_per_second = sampling_rate / _HOP_LENGTH
        min_pause_frames = int(_MIN_PAUSE_DURATION_S * frames_per_second)

        segments: List[Tuple[int, int]] = []
        start = None
        for i, silent in enumerate(is_silent):
            if silent and start is None:
                start = i
            elif not silent and start is not None:
                if i - start >= min_pause_frames:
                    segments.append((start, i))
                start = None
        if start is not None and len(is_silent) - start >= min_pause_frames:
            segments.append((start, len(is_silent)))

        return segments

    @staticmethod
    def _compute_pause_duration(
        rms: np.ndarray, pause_segments: List[Tuple[int, int]], sampling_rate: int | float
    ) -> np.ndarray:
        """
        Compute the pause duration in seconds.
        :param rms: The rms energy vector
        :param pause_segments: A list of tuples (start, end) of pause segments.
        :param sampling_rate: The sampling rate of the audio
        :return: The array of pause durations in seconds.
        """

        feature = np.zeros(len(rms), dtype=np.float32)
        frames_per_second = sampling_rate / _HOP_LENGTH
        for start, end in pause_segments:
            feature[start:end] = (end - start) / frames_per_second

        return feature

    @staticmethod
    def _compute_pitch_reset(
        pitch: np.ndarray, pause_segments: List[Tuple[int, int]], sampling_rate: int | float
    ) -> np.ndarray:
        """
        Compute the pitch reset
        :param pitch: The pitch vector
        :param pause_segments: The pause segments
        :param sampling_rate: The sampling rate of the audio
        :return: A vector with the pitch reset for each frame.
        """

        feature = np.zeros(len(pitch), dtype=np.float32)
        frames_per_second = sampling_rate / _HOP_LENGTH
        window_frames = max(1, int(_BOUNDARY_WINDOW_S * frames_per_second))

        for start, end in pause_segments:
            before = pitch[max(0, start - window_frames) : start]
            after = pitch[end : end + window_frames]
            before_voiced = before[before > 0]
            after_voiced = after[after > 0]
            if len(before_voiced) == 0 or len(after_voiced) == 0:
                continue
            feature[start:end] = abs(np.median(after_voiced) - np.median(before_voiced))

        return feature

    @staticmethod
    def _compute_energy_jump(
        rms: np.ndarray, pause_segments: List[Tuple[int, int]], sampling_rate: int | float
    ) -> np.ndarray:
        """
        Compute the energy jump
        :param rms: The rms energy vector
        :param pause_segments: The pause segments
        :param sampling_rate: The sampling rate of the audio
        :return: The vector with the energy jumps
        """

        feature = np.zeros(len(rms), dtype=np.float32)
        frames_per_second = sampling_rate / _HOP_LENGTH
        window_frames = max(1, int(_BOUNDARY_WINDOW_S * frames_per_second))

        for start, end in pause_segments:
            before = rms[max(0, start - window_frames) : start]
            after = rms[end : end + window_frames]
            if len(before) == 0 or len(after) == 0:
                continue
            feature[start:end] = abs(float(np.mean(after)) - float(np.mean(before)))

        return feature

    @staticmethod
    def _compute_speech_rate(rms: np.ndarray, sampling_rate: int | float) -> np.ndarray:
        """
        Compute the speech rate
        :param rms: The rms energy vector
        :param sampling_rate: The sampling rate of the audio
        :return: The vector with the speech rate
        """

        frames_per_second = sampling_rate / _HOP_LENGTH
        peak_distance = max(1, int(_MIN_SYLLABLE_SPACING_S * frames_per_second))
        peaks, _ = find_peaks(rms, distance=peak_distance, prominence=np.std(rms) * 0.5)

        window_frames = max(1, int(_SPEECH_RATE_WINDOW_S * frames_per_second))
        half_window = window_frames // 2

        n_frames = len(rms)
        frame_indices = np.arange(n_frames)
        window_starts = np.clip(frame_indices - half_window, 0, n_frames)
        window_ends = np.clip(frame_indices + half_window, 0, n_frames)
        counts_low = np.searchsorted(peaks, window_starts, side="left")
        counts_high = np.searchsorted(peaks, window_ends, side="left")
        peak_counts = counts_high - counts_low

        window_durations_s = (window_ends - window_starts) / frames_per_second
        with np.errstate(divide="ignore", invalid="ignore"):
            feature = np.where(window_durations_s > 0, peak_counts / window_durations_s, 0.0)

        return feature.astype(np.float32)


class VideoFeatureExtractor:
    """
    The service to extract video features from a recording session.
    """

    _CANNY_LOW = 50
    _CANNY_HIGH = 150

    _SAMPLE_NTH_FRAME = 1

    _GAUSS_KERNEL = (5, 5)

    _SMOOTHING_WINDOW_FRAMES = 5

    _PHASH_SIZE = 8

    _ROI_HEIGHT_FRAC = 0.20

    _PEAK_PROMINENCE = 0.08

    @staticmethod
    def extract_video_features(recording_session: RecordingSession) -> np.ndarray:
        """
        Extracts video features from a recording session.
        :param recording_session: The recording session.
        :return: The video features.
        """

        import cv2

        cap = cv2.VideoCapture(str(recording_session.recording_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {recording_session.recording_path}")

        try:
            ecr_v, arr_v, phash_v, ssim_v, roi_v = VideoFeatureExtractor._extract_raw_features(cap)
        finally:
            cap.release()

        if len(ecr_v) == 0:
            return np.empty((0, 5), dtype=np.float32)

        # Smooth the raw features.
        smooth = VideoFeatureExtractor._smooth
        cols_raw = [
            smooth(np.array(ecr_v, dtype=np.float32)),
            smooth(np.array(arr_v, dtype=np.float32)),
            smooth(np.array(phash_v, dtype=np.float32)),
            smooth(np.array(ssim_v, dtype=np.float32)),
            smooth(np.array(roi_v, dtype=np.float32)),
        ]

        # Prominence-filter the raw features.
        cols_prominent = [VideoFeatureExtractor._prominence_filter(col) for col in cols_raw]

        features = np.stack(cols_prominent, axis=1)

        return features

    @staticmethod
    def _extract_raw_features(cap) -> tuple:
        """
        Extract the raw features
        :param cap: The video capture object
        :return: A tuple of the raw features
        """

        import cv2

        ecr_vector: list[float] = []
        arr_vector: list[float] = []
        phash_vector: list[float] = []
        ssim_vector: list[float] = []
        roi_vector: list[float] = []

        prev_gray: np.ndarray | None = None
        prev_edges: np.ndarray | None = None
        prev_hash: object | None = None

        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_idx % VideoFeatureExtractor._SAMPLE_NTH_FRAME != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(
                gray,
                VideoFeatureExtractor._CANNY_LOW,
                VideoFeatureExtractor._CANNY_HIGH,
            )
            h = VideoFeatureExtractor._compute_phash(gray)

            if prev_gray is not None:
                ecr_vector.append(VideoFeatureExtractor._compute_ecr(prev_edges, edges))
                arr_vector.append(VideoFeatureExtractor._compute_arr(prev_gray, gray))
                phash_vector.append(VideoFeatureExtractor._compute_phash_dist(prev_hash, h))
                ssim_vector.append(VideoFeatureExtractor._compute_ssim(prev_gray, gray))
                roi_vector.append(VideoFeatureExtractor._compute_roi_score(prev_gray, gray))

            prev_gray = gray
            prev_edges = edges
            prev_hash = h
            frame_idx += 1

        return ecr_vector, arr_vector, phash_vector, ssim_vector, roi_vector

    @staticmethod
    def _compute_ecr(edges_prev: np.ndarray, edges_curr: np.ndarray) -> float:
        """
        Compute the Edge Change Ratio (ECR) of the current frame.
        :param edges_prev: The edges of the previous frame.
        :param edges_curr: The edges of the current frame.
        :return: The ECR of the current frame compared to the previous frame.
        """

        import cv2

        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(edges_prev, kernel, iterations=1)
        new_edges = cv2.bitwise_and(edges_curr, cv2.bitwise_not(dilated))
        n_prev = float(np.count_nonzero(edges_prev))
        n_new = float(np.count_nonzero(new_edges))

        return min(n_new / (n_prev + 1e-6), 1.0)

    @staticmethod
    def _compute_arr(gray_prev: np.ndarray, gray_curr: np.ndarray) -> float:
        """
        Compute the Adaptive Ratio of the current frame.
        :param gray_prev: The gray scale of the previous frame.
        :param gray_curr: The gray scale of the current frame.
        :return: The ARR of the current frame compared to the previous frame.
        """

        import cv2

        b_prev = cv2.GaussianBlur(gray_prev, VideoFeatureExtractor._GAUSS_KERNEL, 0)
        b_curr = cv2.GaussianBlur(gray_curr, VideoFeatureExtractor._GAUSS_KERNEL, 0)
        diff = cv2.absdiff(b_prev, b_curr)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return float(np.count_nonzero(mask)) / float(mask.size)

    @staticmethod
    def _compute_phash(gray: np.ndarray) -> object:
        """
        Compute the phash vector
        :param gray: The gray scale of the current frame.
        :return: The phash
        """

        import imagehash
        from PIL import Image

        return imagehash.phash(
            Image.fromarray(gray),
            hash_size=VideoFeatureExtractor._PHASH_SIZE,
        )

    @staticmethod
    def _compute_phash_dist(h_prev: object, h_curr: object) -> float:
        """
        Compute the phash distance
        :param h_prev: The phash of the previous frame.
        :param h_curr: The phash of the current frame.
        :return: The phash distance
        """

        n_bits = VideoFeatureExtractor._PHASH_SIZE**2

        return float(h_prev - h_curr) / n_bits

    @staticmethod
    def _compute_ssim(gray_prev: np.ndarray, gray_curr: np.ndarray) -> float:
        """
        Compute the structural similarity index (SSIM) of the current frame.
        :param gray_prev: The gray scale of the previous frame.
        :param gray_curr: The gray scale of the current frame.
        :return: The SSIM of the current frame compared to the previous frame.
        """

        import cv2
        from skimage.metrics import structural_similarity as ssim

        target_w = 256
        h, w = gray_prev.shape
        if w > target_w:
            scale = target_w / w
            new_sz = (target_w, max(1, int(h * scale)))
            a = cv2.resize(gray_prev, new_sz, interpolation=cv2.INTER_AREA)
            b = cv2.resize(gray_curr, new_sz, interpolation=cv2.INTER_AREA)
        else:
            a, b = gray_prev, gray_curr

        score, _ = ssim(a, b, full=True)

        return float(np.clip(1.0 - score, 0.0, 1.0))

    @staticmethod
    def _compute_roi_score(gray_prev: np.ndarray, gray_curr: np.ndarray) -> float:
        """
        Compute the ROI score of the current frame.
        :param gray_prev: The gray scale of the previous frame.
        :param gray_curr: The gray scale of the current frame.
        :return: The roi score of the current frame compared to the previous frame.
        """

        h = gray_prev.shape[0]
        roi_h = max(1, int(h * VideoFeatureExtractor._ROI_HEIGHT_FRAC))
        roi_p = gray_prev[:roi_h, :].astype(np.float32)
        roi_c = gray_curr[:roi_h, :].astype(np.float32)
        diff = np.abs(roi_p - roi_c)
        row_means = diff.mean(axis=1)

        return float(row_means.max() / 255.0)

    @staticmethod
    def _smooth(values: np.ndarray) -> np.ndarray:
        """
        Smooth the given values using a moving average.
        :param values: The values to smooth.
        :return: The smoothed values.
        """

        w = VideoFeatureExtractor._SMOOTHING_WINDOW_FRAMES
        if len(values) < w:
            return values
        kernel = np.ones(w, dtype=np.float32) / w

        return np.convolve(values, kernel, mode="same").astype(np.float32)

    @staticmethod
    def _prominence_filter(values: np.ndarray) -> np.ndarray:
        """
        Applies the prominence filter to the given values.
        :param values: The values to filter.
        :return: The filtered values.
        """

        from scipy.signal import find_peaks

        if len(values) < 3:
            return values

        peak = float(values.max())
        if peak <= 0:
            return np.zeros_like(values)
        normed = values / peak

        peaks, _ = find_peaks(
            normed,
            prominence=VideoFeatureExtractor._PEAK_PROMINENCE,
        )

        result = np.zeros_like(normed)
        if len(peaks) > 0:
            result[peaks] = normed[peaks]

        return result.astype(np.float32)


class TranscriptionExtractor:
    """
    Transcribes the audio track of a RecordingSession using OpenAI Whisper.
    Returns the full transcript text and the raw word-level timestamp data.
    The whisper import is deferred, so the model is only loaded when needed.
    """

    _MODEL_SIZE = "base"

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

    Replaces the former VerbChangeDetector. Verb-level analysis and ellipsis
    detection have been removed — sentence-type classification is now handled
    entirely by the NLI classifier in SemanticAudioFeatureExtractor, following
    the taxonomy of Vander Linden (1995) and Safa et al. (2026):

      ACTION      → user is instructed to perform an action  → boundary
      RESULT      → software reacts / state change described → no boundary
      PRECONDITION→ prerequisite stated                      → no boundary
      BACKGROUND  → context or explanation given             → no boundary

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

        # Build char-offset → timestamp lookup from Whisper word list.
        char_to_time: List[Tuple[int, float]] = []
        char_cursor = 0
        for w in words:
            surface = w.get("word", "").strip()
            pos = full_text.find(surface, char_cursor)
            if pos != -1:
                char_to_time.append((pos, float(w.get("start", 0.0))))
                char_cursor = pos

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
            # Expletiv-"es"-Filter: Sätze wie "Es öffnet sich..." oder
            # "Es erscheint ein Menü" sind Systemreaktionen — keine Nutzer-Aktion.
            # Erkennbar daran dass das erste inhaltliche Subjekt ein Expletiv-"es"
            # (dep_="sb" oder "nsubj") mit Lemma "es" ist.
            # Diese Sätze werden vor dem NLI-Aufruf ausgeschlossen.
            is_system_reaction = any(
                token.lemma_.lower() == "es"
                and token.dep_ in ("sb", "nsubj", "nsubjpass", "ep")
                for token in sent
            )
            if is_system_reaction:
                continue
            t_s = _resolve_time(sent.start_char)
            sentences.append((t_s * 1000.0, text))

        return sentences


# ── NLI configuration ─────────────────────────────────────────────────────────

# NLI hypothesis template — definiert die semantische Frage die das Modell
# für jeden Satz beantwortet. Das Modell prüft ob der Satz die Hypothese
# impliziert (Entailment). Der Platzhalter {} wird durch das candidate_label
# ersetzt, sodass die vollständige Hypothese lautet:
#   "Dieser Satz ist eine direkte Handlungsanweisung an einen menschlichen
#    Benutzer, eine Aktion in der Software auszuführen."
# Systemreaktionen ("Ein Fenster öffnet sich"), Ergebnisse und Kontext
# implizieren diese Hypothese nicht → niedriger Entailment-Score.
_NLI_HYPOTHESIS_TEMPLATE = (
    "Dieser Satz ist eine direkte Handlungsanweisung an einen menschlichen "
    "Benutzer, eine Aktion in der Software auszuführen: {}."
)
_NLI_CANDIDATE_LABELS = ["wahr", "falsch"]
_ACTION_LABEL = "wahr"

# Minimum NLI confidence score for a sentence to be accepted as ACTION.
_NLI_THRESHOLD = 0.65

# Minimum gap (seconds) between two boundary candidates to avoid duplicates.
_MIN_BOUNDARY_GAP_S = 2.0


class SemanticAudioFeatureExtractor:
    """
    Extracts semantic boundary features from the audio transcript of a
    RecordingSession using a four-class NLI sentence-type classifier.

    Sentence-type taxonomy (Vander Linden 1995; Safa et al. 2026):
      ACTION      → user is instructed to perform an action  → boundary
      RESULT      → software reacts / state change described → no boundary
      PRECONDITION→ prerequisite stated                      → no boundary
      BACKGROUND  → context or explanation given             → no boundary

    Pipeline:
      1. SentenceSegmenter  — splits transcript into (t_ms, sentence) pairs
      2. mDeBERTa Zero-Shot — classifies each sentence into the four types
      3. Frame alignment    — writes ACTION scores as Gaussian into (T, 3) array

    Output columns (aligned to AudioFeatureExtractor frame grid):
      Col 0 — sentence_boundary:  1.0 at frames where any sentence starts
      Col 1 — action_score:       Gaussian-weighted NLI score for ACTION sentences
      Col 2 — action_flag:        Hard 0/1 threshold of col 1 at _NLI_THRESHOLD
    """

    _NLI_SPREAD_S = 1.5

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
        :param n_frames: Number of audio frames (must match AudioFeatureExtractor output).
        :param sampling_rate: Audio sampling rate returned by librosa.load.
        :return: np.ndarray of shape (n_frames, 3), dtype float32.
        """
        features = np.zeros((n_frames, 3), dtype=np.float32)

        if not full_text or not words:
            return features

        frames_per_second = sampling_rate / _HOP_LENGTH

        # ── Step 1: Sentence segmentation ─────────────────────────────────────
        sentences = SentenceSegmenter.segment(full_text, words)
        if not sentences:
            return features

        # ── Step 2: Mark all sentence-start frames (column 0) ─────────────────
        for t_ms, _ in sentences:
            frame_idx = int(round((t_ms / 1000.0) * frames_per_second))
            if 0 <= frame_idx < n_frames:
                features[frame_idx, 0] = 1.0

        # ── Step 3: NLI sentence-type classification ───────────────────────────
        from transformers import pipeline  # deferred

        zero_shot = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        )

        # Sentence 0 is always an ACTION boundary (start of first instruction).
        first_t_s = sentences[0][0] / 1000.0
        boundary_candidates: List[Tuple[float, float]] = [(first_t_s, 1.0)]
        last_boundary_t_s = first_t_s

        for t_ms, sentence_text in sentences[1:]:
            t_s = t_ms / 1000.0

            result = zero_shot(
                sentence_text,
                candidate_labels=_NLI_CANDIDATE_LABELS,
                hypothesis_template=_NLI_HYPOTHESIS_TEMPLATE,
            )
            top_label = result["labels"][0]
            score = float(result["scores"][0])

            # Only ACTION sentences trigger boundaries.
            if top_label != _ACTION_LABEL or score < _NLI_THRESHOLD:
                continue

            # Enforce minimum gap — keep higher-confidence candidate on collision.
            if t_s - last_boundary_t_s < _MIN_BOUNDARY_GAP_S:
                if score > boundary_candidates[-1][1]:
                    boundary_candidates[-1] = (t_s, score)
                continue

            boundary_candidates.append((t_s, score))
            last_boundary_t_s = t_s

        # ── Step 4: Write ACTION scores into frame arrays (columns 1 & 2) ──────
        spread_frames = int(SemanticAudioFeatureExtractor._NLI_SPREAD_S * frames_per_second)

        for t_s, score in boundary_candidates:
            center = int(round(t_s * frames_per_second))
            lo = max(0, center - spread_frames)
            hi = min(n_frames, center + spread_frames + 1)

            offsets = np.arange(lo, hi) - center
            gaussian = np.exp(-0.5 * (offsets / max(spread_frames / 2, 1)) ** 2)

            features[lo:hi, 1] = np.maximum(
                features[lo:hi, 1], (score * gaussian).astype(np.float32)
            )

        # Column 2: hard flag wherever column 1 exceeds threshold.
        features[:, 2] = (features[:, 1] >= _NLI_THRESHOLD).astype(np.float32)

        return features


# Window size to track events in seconds
_EVENT_WINDOW_S = 0.25

# Maximum size for the events
_MAX_EVENT_RECENCY_S = 5.0


class EventFeatureExtractor:
    """
    Provides event feature extraction.
    """

    # Event types that are considered for the event feature extraction.
    _MARKER_EVENT_TYPES = {"mouse_click", "key_press", "key_release", "mouse_scroll"}

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