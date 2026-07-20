"""
Action boundaries from the audio track alone: each narrated sentence opens an
execution window filled with a bump scaled by an LLM's verdict on it.

Audio knows the INTERVAL a boundary falls in, never the instant: the narration
announces steps in order, so step i completed between announcement i and i+1.
"""

from __future__ import annotations

from collections.abc import Callable

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


def _transcribe(session: RecordingSession) -> tuple[str, list[dict]]:
    """Whisper transcript plus word-level timestamps. Reads only the audio track."""
    import whisper

    result = whisper.load_model(_WHISPER_MODEL).transcribe(
        str(session.recording_path),
        verbose=False,
        language="de",
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    return result.get("text", "").strip(), words


def _sentences(full_text: str, words: list[dict]) -> list[tuple[float, str]]:
    """Split the transcript into (t_s, text) sentences using spaCy."""
    if not full_text or not words:
        return []

    import spacy

    nlp = None
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

    # Char offset -> timestamp, from the Whisper word list. The cursor advances
    # past each match so a repeated word maps to its own occurrence rather than
    # always the first one.
    char_to_time: list[tuple[int, float]] = []
    cursor = 0
    for w in words:
        surface = w.get("word", "").strip()
        if not surface:
            continue
        pos = full_text.find(surface, cursor)
        if pos != -1:
            char_to_time.append((pos, float(w.get("start", 0.0))))
            cursor = pos + len(surface)

    def _time_at(char_idx: int) -> float:
        if not char_to_time:
            return 0.0
        best = char_to_time[0][1]
        for pos, t in char_to_time:
            if pos > char_idx:
                break
            best = t
        return best

    return [
        (_time_at(sent.start_char), sent.text.strip())
        for sent in nlp(full_text).sents
        if sent.text.strip()
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

    # The last window is closed by this session's own median announcement gap or
    # by the recording's end, whichever comes first.
    starts_s = [t_s for t_s, _ in sentences]
    gaps = np.diff(starts_s)
    horizon = float(np.median(gaps)) if len(gaps) else _LAST_WINDOW_FALLBACK_S
    window_ends = [*starts_s[1:], min(starts_s[-1] + horizon, duration_s)]

    boundaries_s: list[float] = []
    for start_s, end_s, judgement in zip(starts_s, window_ends, judgements):
        if end_s <= start_s:                  # spaCy split, same Whisper word
            continue
        peak = apply_window(
            score,
            lo=int(round(start_s * GRID_HZ)),
            hi=int(round(end_s * GRID_HZ)),
            peak=int(round((start_s + _COMPLETION_POSITION * (end_s - start_s)) * GRID_HZ)),
            value=judgement.p_boundary,
        )
        if judgement.p_boundary >= BOUNDARY_THRESHOLD:
            # The PEAK, not the start: the completion sits inside the announced
            # step, not at the announcement that opens it.
            boundaries_s.append(float(times_s[peak]))

    return BoundaryEvidence(times_s, score, boundaries_s)
