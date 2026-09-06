"""
Action boundaries from the audio track alone: each narrated sentence opens an
execution window filled with a bump scaled by an LLM's verdict on it.

Audio knows the INTERVAL a boundary falls in, never the instant: the narration
announces steps in order, so step i completed between announcement i and i+1.
"""

from __future__ import annotations

import threading
from bisect import bisect_right
from collections.abc import Callable, Sequence

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation.evidence import (
    BOUNDARY_THRESHOLD,
    GRID_HZ,
    BoundaryEvidence,
    apply_window,
    grid,
)

MODALITY = "audio"

# "base" missed German instructional vocabulary in pilot recordings.
_WHISPER_MODEL = "small"

_SPACY_MODELS = ("de_core_news_lg", "de_core_news_md", "de_core_news_sm")

# Where in its window the completion is expected.
# PROVISIONAL: the direction is structural, the fraction is not (measured median
# 0.77, n=7). Calibrate on a dev split, never on the evaluation set.
_COMPLETION_POSITION = 0.75

# Horizon for the last window, used only when a recording has a single sentence
# and there is no announcement gap to take a median of.
_LAST_WINDOW_FALLBACK_S = 8.0

# Loaded models, one set PER THREAD. Whisper and spaCy take a minute to load
# and a corpus run transcribes 25 recordings, so reloading per session was the
# single largest avoidable cost of this modality. Per thread rather than per
# process because two segmentation runs may overlap (feature dialog and
# experiment window each own a worker) and Whisper's decode installs hooks on
# the model it runs on — sharing one instance across threads would let the
# runs corrupt each other's decode.
_MODELS = threading.local()


def _keep_attention_weights() -> None:
    """
    Make Whisper materialise its cross-attention weights for the whole process.

    Word timestamps are read off those weights, and the fused SDPA kernel never
    returns them (`qk` stays None). Whisper switches SDPA off around the alignment
    pass itself, but the switch is a CLASS attribute — process-wide and not
    thread-safe. Two segmentation runs overlap here (the feature dialog and the
    experiment window each own a worker thread), so one run restores SDPA while
    the other is still aligning, and the alignment then indexes None:
    "'NoneType' object is not subscriptable". Off once, no window; the explicit
    attention path is the one Whisper used before SDPA existed.
    """
    from whisper.model import MultiHeadAttention

    MultiHeadAttention.use_sdpa = False


def _whisper_model():
    """This thread's Whisper model, loaded on first use."""
    model = getattr(_MODELS, "whisper", None)
    if model is None:
        import whisper

        model = whisper.load_model(_WHISPER_MODEL)
        _MODELS.whisper = model
    return model


def _nlp():
    """
    This thread's German spaCy pipeline, loaded on first use.

    :raises OSError: when no German model is installed.
    """
    nlp = getattr(_MODELS, "nlp", None)
    if nlp is not None:
        return nlp

    import spacy

    for name in _SPACY_MODELS:
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
    _MODELS.nlp = nlp
    return nlp


def _transcribe(session: RecordingSession) -> tuple[str, list[dict]]:
    """Whisper transcript plus word-level timestamps. Reads only the audio track."""
    _keep_attention_weights()

    result = _whisper_model().transcribe(
        str(session.recording_path),
        verbose=False,
        language="de",
        word_timestamps=True,
        condition_on_previous_text=False,
        # A SCALAR, not Whisper's default (0.0, 0.2, ... 1.0): the tuple is a
        # fallback chain that re-decodes a segment at rising temperature when it
        # looks repetitive or low-confidence, and every temperature above 0 draws
        # tokens by SAMPLING. Two runs of the same recording then yield different
        # transcripts, different sentences, a different verdict-cache key and
        # therefore different audio evidence — the modality would not be
        # reproducible. At 0 the decoder is a deterministic argmax.
        temperature=0.0,
    )
    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    return result.get("text", "").strip(), words


def _sentences(full_text: str, words: list[dict]) -> list[tuple[float, str]]:
    """Split the transcript into (t_s, text) sentences using spaCy."""
    if not full_text or not words:
        return []

    nlp = _nlp()

    # Char offset -> timestamp, from the Whisper word list. The cursor advances
    # past each match so a repeated word maps to its own occurrence rather than
    # always the first one; as a consequence the offsets are strictly ascending,
    # which is what lets a sentence start be looked up by bisection.
    positions: list[int] = []
    starts: list[float] = []
    cursor = 0
    for w in words:
        surface = w.get("word", "").strip()
        if not surface:
            continue
        pos = full_text.find(surface, cursor)
        if pos != -1:
            positions.append(pos)
            starts.append(float(w.get("start", 0.0)))
            cursor = pos + len(surface)

    def _time_at(char_idx: int) -> float:
        # The last word starting at or before the sentence; a sentence that
        # begins before any matched word inherits the first word's time.
        if not positions:
            return 0.0
        index = bisect_right(positions, char_idx) - 1
        return starts[index] if index >= 0 else starts[0]

    return [
        (_time_at(sent.start_char), sent.text.strip())
        for sent in nlp(full_text).sents
        if sent.text.strip()
    ]


def execution_windows(
    starts_s: Sequence[float], duration_s: float
) -> list[tuple[float, float, float]]:
    """
    The interval each announcement opens, and where inside it the completion is
    expected: (start, end, peak) per sentence, in seconds.

    This is the whole geometric argument of the modality. The speaker announces
    steps IN ORDER and carries them out afterwards, so step *i* completes
    somewhere between announcement *i* and announcement *i+1* — audio knows the
    interval, never the instant. Putting a narrow peak on the sentence onset
    would fake a precision the modality does not have, and the Shapley value
    would then measure that geometry instead of the modality.

    The last window has no following announcement to close it and is closed by
    this session's own MEDIAN announcement gap (a fast speaker gets a shorter
    one), at the latest by the end of the recording.

    :param starts_s: sentence onsets in seconds, ascending.
    :param duration_s: length of the recording.
    :return: one (start, end, peak) per sentence; windows that would be empty
        are still returned and rejected by the caller.
    """
    if not starts_s:
        return []
    gaps = np.diff(starts_s)
    horizon = float(np.median(gaps)) if len(gaps) else _LAST_WINDOW_FALLBACK_S
    ends = [*starts_s[1:], min(starts_s[-1] + horizon, duration_s)]
    return [
        (start, end, start + _COMPLETION_POSITION * (end - start))
        for start, end in zip(starts_s, ends)
    ]


def _duration_s(session: RecordingSession) -> float:
    """Length of the audio track. Decoded, not read off the container header."""
    import librosa

    y, sr = librosa.load(str(session.recording_path))
    return len(y) / float(sr)


def extract(
    session: RecordingSession,
    *,
    use_cache: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> BoundaryEvidence:
    """
    Action boundaries from the audio track alone: Whisper transcript -> sentences
    -> an LLM judges each -> each accepted sentence fills its execution window.

    on_progress reports the two phases; is_cancelled is honoured between them.

    :param session: the recording; only its audio track and cache path are read.
    :raises RuntimeError: when no LLM is reachable.
    """
    from docupilot.segmentation import audio_scoring as judge

    # Before Whisper and spaCy load their models: they cost a minute together.
    if not judge.is_available():
        raise RuntimeError(
            f"LLM-Backend '{judge.MODEL}' nicht nutzbar. Benötigt:\n"
            "  poetry install         (Paket 'anthropic')\n"
            "  ANTHROPIC_API_KEY=...  oder  ant auth login\n\n"
            "Ohne semantisches Urteil kann die Audio-Modalität keine "
            "Handlungsgrenzen bestimmen — jede Ansage wäre gleich viel wert und "
            "die Spur würde die ganze Zeitachse gleichmäßig füllen."
        )

    if on_progress is not None:
        on_progress(0, 2)
    duration_s = _duration_s(session)
    times_s = grid(duration_s)
    if len(times_s) == 0:
        return BoundaryEvidence.empty()
    score = np.zeros(len(times_s), dtype=np.float32)

    full_text, words = _transcribe(session)
    sentences = _sentences(full_text, words)
    if not sentences or (is_cancelled is not None and is_cancelled()):
        return BoundaryEvidence(times_s, score, [])

    if on_progress is not None:
        on_progress(1, 2)
    cache = (
        judge.Cache(session.session_dir / "audio_llm_cache.json") if use_cache else None
    )
    judgements = judge.judge([text for _, text in sentences], cache=cache)
    if cache is not None:
        cache.flush()
    if judgements is None:                    # unusable answer: leave no evidence
        return BoundaryEvidence(times_s, score, [])   # rather than guess
    if on_progress is not None:
        on_progress(2, 2)

    windows = execution_windows([t_s for t_s, _ in sentences], duration_s)

    boundaries_s: list[float] = []
    for (start_s, end_s, peak_s), judgement in zip(windows, judgements):
        if end_s <= start_s:                  # spaCy split, same Whisper word
            continue
        peak = apply_window(
            score,
            lo=int(round(start_s * GRID_HZ)),
            hi=int(round(end_s * GRID_HZ)),
            peak=int(round(peak_s * GRID_HZ)),
            value=judgement.p_boundary,
        )
        if judgement.p_boundary >= BOUNDARY_THRESHOLD:
            # The PEAK, not the start: the completion sits inside the announced
            # step, not at the announcement that opens it.
            boundaries_s.append(float(times_s[peak]))

    return BoundaryEvidence(times_s, score, boundaries_s)
