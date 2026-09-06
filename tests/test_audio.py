"""
The audio modality: sentence timing, the execution windows that encode "audio
knows the interval, never the instant", and the LLM answer parsing.

Whisper and the LLM are replaced by fakes — the transport is not under test.
"""

from dataclasses import asdict

import numpy as np
import pytest

from docupilot.segmentation import audio
from docupilot.segmentation import audio_scoring as asc

TEXT = "Ich öffne das Menü Datei. Dann erstelle ich ein neues Dokument. Jetzt speichere ich es ab."


def words_for(text: str, start: float = 0.5, step: float = 0.45) -> list[dict]:
    """Whisper-style word list with evenly spaced onsets."""
    out, t = [], start
    for word in text.replace(".", " .").split():
        if word == ".":
            continue
        out.append({"word": " " + word, "start": round(t, 2), "end": round(t + 0.3, 2)})
        t += step
    return out


class TestSentences:
    def test_sentences_get_the_onset_of_their_first_word(self):
        sentences = audio._sentences(TEXT, words_for(TEXT))
        assert len(sentences) == 3
        assert sentences[0][0] == pytest.approx(0.5)
        assert sentences[0][1].startswith("Ich öffne")
        assert [t for t, _ in sentences] == sorted(t for t, _ in sentences)

    def test_a_repeated_word_maps_to_its_own_occurrence(self):
        # The cursor must advance past each match; otherwise every later
        # occurrence of "ich" would be timed at the first one.
        text = "Ich speichere. Ich schließe."
        sentences = audio._sentences(text, words_for(text))
        assert sentences[1][0] > sentences[0][0]

    def test_missing_input_yields_no_sentences(self):
        assert audio._sentences("", words_for(TEXT)) == []
        assert audio._sentences(TEXT, []) == []


class TestExecutionWindows:
    def test_each_announcement_opens_a_window_up_to_the_next(self):
        windows = audio.execution_windows([0.0, 4.0, 8.0], duration_s=20.0)
        assert [(lo, hi) for lo, hi, _ in windows[:2]] == [(0.0, 4.0), (4.0, 8.0)]

    def test_the_peak_sits_late_in_the_window(self):
        # Structural: announcing takes a moment, executing takes longer, and the
        # completion falls near the end of the announced stretch.
        (lo, hi, peak), = audio.execution_windows([0.0], duration_s=8.0)
        assert lo < peak < hi
        assert peak == pytest.approx(lo + audio._COMPLETION_POSITION * (hi - lo))

    def test_the_last_window_is_closed_by_the_median_announcement_gap(self):
        windows = audio.execution_windows([0.0, 4.0, 8.0], duration_s=100.0)
        assert windows[-1][1] == pytest.approx(12.0)      # median gap of 4 s

    def test_a_fast_speaker_gets_a_shorter_last_window(self):
        fast = audio.execution_windows([0.0, 1.0, 2.0], duration_s=100.0)
        slow = audio.execution_windows([0.0, 10.0, 20.0], duration_s=100.0)
        assert (fast[-1][1] - fast[-1][0]) < (slow[-1][1] - slow[-1][0])

    def test_the_last_window_never_runs_past_the_recording(self):
        # A peak on the array edge cannot be found by find_peaks, so the last
        # boundary of every session would silently vanish.
        windows = audio.execution_windows([0.0, 4.0, 8.0], duration_s=9.0)
        assert windows[-1][1] == pytest.approx(9.0)

    def test_a_single_announcement_falls_back_to_a_fixed_horizon(self):
        (lo, hi, _), = audio.execution_windows([2.0], duration_s=100.0)
        assert hi == pytest.approx(2.0 + audio._LAST_WINDOW_FALLBACK_S)

    def test_no_announcements_no_windows(self):
        assert audio.execution_windows([], duration_s=10.0) == []


class TestExtract:
    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch):
        monkeypatch.setattr(asc, "is_available", lambda: True)
        monkeypatch.setattr(audio, "_duration_s", lambda s: 9.0)
        monkeypatch.setattr(audio, "_transcribe", lambda s: (TEXT, words_for(TEXT)))

    def test_boundaries_land_on_the_window_peak_not_the_announcement(
        self, session, monkeypatch
    ):
        monkeypatch.setattr(asc, "judge", lambda sentences, cache=None, model="m": [
            asc.Judgement("OPERATION", 0.9, ""),
            asc.Judgement("MEANS", 0.3, ""),
            asc.Judgement("OPERATION", 0.7, ""),
        ])
        ev = audio.extract(session, use_cache=False)
        sentences = audio._sentences(TEXT, words_for(TEXT))
        windows = audio.execution_windows([t for t, _ in sentences], 9.0)
        # Two accepted sentences, each reported inside its own window, after the
        # announcement that opened it.
        assert len(ev.boundaries_s) == 2
        for boundary, (lo, hi, _) in zip(ev.boundaries_s, [windows[0], windows[2]]):
            assert lo < boundary <= hi

    def test_a_rejected_sentence_still_leaves_graded_evidence(self, session, monkeypatch):
        monkeypatch.setattr(asc, "judge", lambda sentences, cache=None, model="m": [
            asc.Judgement("MEANS", 0.3, "") for _ in sentences
        ])
        ev = audio.extract(session, use_cache=False)
        assert ev.boundaries_s == []
        assert 0.0 < ev.score.max() < 0.5

    def test_an_unusable_answer_leaves_no_evidence_rather_than_a_guess(
        self, session, monkeypatch
    ):
        monkeypatch.setattr(asc, "judge", lambda sentences, cache=None, model="m": None)
        ev = audio.extract(session, use_cache=False)
        assert not ev.score.any() and ev.boundaries_s == []

    def test_extract_refuses_to_run_without_a_backend(self, session, monkeypatch):
        monkeypatch.setattr(asc, "is_available", lambda: False)
        # Without a per-sentence judgement every window would be equal and the
        # lane would tile the timeline — uniform evidence reads as signal.
        with pytest.raises(RuntimeError, match="nicht nutzbar"):
            audio.extract(session, use_cache=False)


class TestScoring:
    def test_operation_maps_to_its_confidence_and_the_rest_to_the_complement(self):
        raw = ('{"verdicts": [{"i": 0, "category": "OPERATION", "confidence": 0.9, "reason": "a"},'
               '{"i": 1, "category": "MEANS", "confidence": 0.8, "reason": "b"}]}')
        verdicts = asc.parse(raw, 2)
        assert verdicts[0].p_boundary == pytest.approx(0.9)
        assert verdicts[1].p_boundary == pytest.approx(0.2)

    def test_categories_are_case_insensitive_and_confidence_is_clamped(self):
        raw = '{"verdicts": [{"i": 0, "category": "means", "confidence": 1.7}]}'
        assert asc.parse(raw, 1)[0].category == "MEANS"
        assert 0.0 <= asc.parse(raw, 1)[0].p_boundary <= 1.0

    def test_a_skipped_sentence_gets_neutral_evidence(self):
        raw = '{"verdicts": [{"i": 0, "category": "OPERATION", "confidence": 0.9}]}'
        verdicts = asc.parse(raw, 3)
        assert len(verdicts) == 3
        assert verdicts[1].p_boundary == 0.0 and verdicts[2].category == "OTHER"

    def test_unusable_answers_are_none_rather_than_invented(self):
        assert asc.parse("no json here", 2) is None
        assert asc.parse("{}", 2) is None
        assert asc.parse('{"verdicts": [{"i": 0, "category": "NOPE"}]}', 1) is None

    def test_judging_nothing_asks_nothing(self):
        assert asc.judge([]) == []


class TestCache:
    def test_a_verdict_set_survives_a_round_trip(self, tmp_path):
        path = tmp_path / "llm.json"
        cache = asc.Cache(path)
        cache.put("k", [asc.Judgement("OPERATION", 0.9, "r")])
        cache.flush()
        restored = asc.Cache(path).get("k")
        assert [asdict(j) for j in restored] == [
            {"category": "OPERATION", "p_boundary": 0.9, "reason": "r"}
        ]

    def test_the_key_follows_the_transcript_the_model_and_the_prompt(self):
        a = asc.Cache.key(["eins", "zwei"], "model-x")
        assert a != asc.Cache.key(["eins", "drei"], "model-x")
        assert a != asc.Cache.key(["eins", "zwei"], "model-y")
        assert a == asc.Cache.key(["eins", "zwei"], "model-x")

    def test_a_damaged_or_foreign_file_is_a_miss_never_a_crash(self, tmp_path):
        broken = tmp_path / "broken.json"
        broken.write_text("{ not json", encoding="utf-8")
        assert asc.Cache(broken).get("k") is None
        assert asc.Cache(tmp_path / "absent.json").get("k") is None

    def test_a_cached_verdict_set_short_circuits_the_model(self, tmp_path):
        path = tmp_path / "llm.json"
        cache = asc.Cache(path)
        cache.put(asc.Cache.key(["a"], asc.MODEL), [asc.Judgement("OPERATION", 0.8, "")])
        # ask() would raise without credentials; a hit must not reach it.
        assert asc.judge(["a"], cache=cache)[0].p_boundary == pytest.approx(0.8)


def test_evidence_grid_matches_the_declared_rate(session, monkeypatch):
    monkeypatch.setattr(asc, "is_available", lambda: True)
    monkeypatch.setattr(audio, "_duration_s", lambda s: 4.0)
    monkeypatch.setattr(audio, "_transcribe", lambda s: (TEXT, words_for(TEXT)))
    monkeypatch.setattr(asc, "judge", lambda sentences, cache=None, model="m": [
        asc.Judgement("OTHER", 0.0, "") for _ in sentences
    ])
    ev = audio.extract(session, use_cache=False)
    assert len(ev.times_s) == int(4.0 * 50)
    assert np.allclose(np.diff(ev.times_s), 1 / 50.0)
