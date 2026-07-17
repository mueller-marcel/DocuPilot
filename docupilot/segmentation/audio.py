"""
The audio modality: action boundaries from the audio track ALONE.

No screen, no events — the modality stays independent for the 2^3 Shapley
ablation. This module reads exactly one stream: the recording's audio.

WHAT AUDIO CAN AND CANNOT KNOW
  A boundary is the moment a user-triggered change SETTLES INTO A NEW PERSISTENT
  STATE (docs/annotationsleitfaden.md). That moment is visual. Nothing audible
  happens when a filter finishes applying — the microphone hears speech and
  silence, and the participant is usually not speaking at all while the screen
  settles. So this extractor CANNOT localise the instant, and pretending it could
  would fake the very quantity the Shapley analysis is supposed to measure.

  What the narration does carry is STRUCTURE. The participant announces each step
  before or while doing it, in order. From announcement i and announcement i+1 it
  follows that step i completed BETWEEN them. Audio therefore knows the INTERVAL
  a boundary falls in, not the point — and that is exactly what gets encoded.

  Measured on session_30 (7 rule-conform boundaries, Whisper small + spaCy):
    Ansage -> Grenze           : median 6.84 s, range 1.98 .. 19.10 s
    Grenze -> naechste Ansage  : median  3.2 s, range 1.44 ..  8.84 s
  Every one of the 7 boundaries falls inside the interval opened by its own
  announcement, and no boundary falls inside an interval opened by a
  means-sentence. The interval claim holds; the point claim never could.

THE SEMANTIC STAGE
  Announcing a step is not the same as announcing a boundary: "Navigiere zum
  erstellten Datenblatt" announces a MEANS, and rule C excludes it. Deciding this
  needs a model that can apply the definition, which is why the old zero-shot NLI
  stage is gone — see audio_scoring.py for the measurement that killed it (four
  hypothesis wordings, separation negative in all four) and for why the
  replacement is an LLM.

EVIDENCE GEOMETRY
  Each sentence i opens an execution window [t_i, t_{i+1}) and fills it with a
  raised-cosine bump scaled by P(OPERATION | sentence i):

    0 at t_i        the step cannot be finished at the instant it is announced
    peak inside     where the completion is expected
    0 at t_{i+1}    by the next announcement the step is demonstrably done

  The zeros at the announcements are not cosmetic — they keep adjacent windows
  from fusing into one plateau, so each announced step yields ONE peak instead of
  one peak per run of steps.

DELIBERATELY ABSENT — do not reintroduce:
  - A narrow peak on the sentence onset. The old stage put a 1.5 s spread on the
    announcement; it covered 1 of 7 boundaries, so audio evidence never even
    reached the boundary it was supposed to mark.
  - Any prosodic/RMS "pause detector" as a boundary source. A pause marks that
    the user stopped talking, not that the screen settled; it fires on thinking,
    reading and breathing alike.
  - A fallback when the LLM is unavailable. Without a per-sentence judgement every
    window is equal, the lane tiles the whole timeline, and a uniform lane reads
    as evidence while carrying none. It raises instead.
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

# "base" missed German instructional vocabulary in pilot recordings; "small" is
# the best quality/latency trade-off for offline extraction.
_WHISPER_MODEL = "small"

_SPACY_MODELS = ("de_core_news_lg", "de_core_news_md", "de_core_news_sm")

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
# modality). This value is only reached when a recording has a SINGLE sentence and
# there is no gap to take a median of.
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
    """
    Split the transcript into (t_s, text) sentences using spaCy.

    A syntactic pre-filter used to drop sentences here before the classifier saw
    them. It is gone on purpose: deciding what a sentence announces is the
    judge's job, and the filter deleted evidence nobody ever got to weigh.
    """
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
    Action boundaries for one recording, from the audio track alone.

    Transcription happens here rather than in the caller: it is the one thing
    that turns audio into something judgeable, and a caller that had to run
    Whisper first would have to know this modality's internals.

    on_progress reports the two coarse phases (transcription, judgement) — the
    stages are minutes apart and a caller showing nothing would look hung.
    is_cancelled is part of the shared contract; there is no useful mid-Whisper
    stopping point, so it is only honoured between phases.

    :param session: the recording; only its audio track and cache path are read.
    :raises RuntimeError: when no LLM is reachable.
    """
    from docupilot.segmentation import audio_scoring as judge

    # Checked before Whisper and spaCy load their models: together they cost a
    # minute, and there is nothing this extractor can deliver without the judge.
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

    # The last window has no closing announcement. Two things close it anyway:
    # this session's own median announcement gap, and the end of the recording —
    # the step demonstrably finished before the user stopped recording. Whichever
    # comes first wins, which also keeps the peak estimate inside the recording
    # instead of extrapolating past its end.
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
            # The boundary is the window's PEAK, not its start: audio expects the
            # completion inside the announced step, not at the announcement that
            # opens it.
            boundaries_s.append(float(times_s[peak]))

    return BoundaryEvidence(times_s, score, boundaries_s)
