"""
LLM judgement on narration sentences — the audio modality's semantic stage.

The participant narrates every step aloud ("Kommentiere laut, was du gerade
tust"). This module decides, per narrated sentence, whether it announces an
operation whose COMPLETION is a boundary under our definition (see
docs/annotationsleitfaden.md), or whether it is only a means / filler.

WHY AN LLM AND NOT THE OLD ZERO-SHOT NLI:
  The previous stage asked an mDeBERTa NLI model "is this sentence an action
  instruction?". Measured on real narration (session_30, 10 sentences), four
  different hypothesis wordings ALL failed to separate boundary-steps from
  non-boundary steps — separation was negative in every case ("Navigiere zu den
  Verkaufsdaten zurück" scored higher than "Ich füge die Tabelle ein"). The
  reason is structural: the participant announces EVERY step, so "is this an
  action?" has no variance — the classifier answers a question that is always
  yes. What actually has to be decided is our definition's distinction (an
  operation that completes in a new persistent state vs. a means), and that
  needs a model that can apply a definition.

  It also keeps the Shapley comparison fair: the video modality is judged by a
  modern VLM. Pairing it with a weak 2021 NLI model on the audio side would make
  the decomposition measure classifier quality, not modality information.

THIS MODULE READS THE TRANSCRIPT ONLY — never the screen, never the event
stream. The audio modality has to stay independent for the 2^3 ablation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL = os.environ.get("DOCUPILOT_AUDIO_MODEL", "claude-opus-4-8")

# Bump when the prompt or the score mapping changes — invalidates cached verdicts.
PROMPT_VERSION = "a1"

_BOUNDARY = "OPERATION"
_CATEGORIES = (
    _BOUNDARY,   # announces an operation that completes in a new persistent state
    "MEANS",     # navigation / selection / setup on the way to another operation
    "OTHER",     # filler, commentary, verification — announces nothing
)

_SYSTEM = """\
You are given the spoken narration of a person working through a task in desktop
software, transcribed and split into sentences, in order. They were told to say
each step out loud as they do it.

For EACH sentence decide what it announces, using this definition:

  A BOUNDARY is the completion of an operation the user carried out — the moment
  its result becomes visible and settles into a state that PERSISTS.

Categories:

  OPERATION  The sentence announces an operation whose completion is a boundary:
             something is applied, created, entered, deleted, moved, inserted,
             saved, opened, or a view/mode is deliberately switched to and kept.
             Examples: "Ich aktiviere den Autofilter", "Ich sortiere nach Menge",
             "Ich erstelle ein neues Blatt", "Ich füge die Tabelle ein",
             "Ich wechsle in die Leseansicht".

  MEANS      The sentence announces only a step ON THE WAY to another operation —
             it changes nothing that persists: navigating or switching to where
             the next action will happen, scrolling, selecting/marking, copying
             to the clipboard, opening a menu or dialog.
             Examples: "Navigiere zu den Verkaufsdaten zurück", "Ich scrolle
             nach unten", "Ich markiere die Zeilen", "Nun kopiere ich die
             Tabelle", "Ich öffne das Menü".

  OTHER      Announces no step: filler, thinking aloud, commentary on a result,
             checking or reading something, greetings.
             Examples: "Das sieht gut aus", "Ähm, moment", "Hier sehen wir 19
             Datensätze".

Judge the sentences IN CONTEXT of the sequence: a sentence is MEANS when the
sentence that follows it names the operation it was serving.

The decisive question is NOT whether an action is mentioned — the speaker
narrates every step, so almost every sentence mentions one. The question is
whether what is announced LEAVES A PERSISTENT RESULT (OPERATION) or is only a
way to get there (MEANS).

Answer with ONE JSON object and nothing else:
{"verdicts": [{"i": <sentence index>, "category": "<OPERATION|MEANS|OTHER>",
               "confidence": <0.0-1.0>, "reason": "<max 10 words>"}, ...]}
One entry per sentence, in order, indices starting at 0."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "category": {"type": "string", "enum": list(_CATEGORIES)},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["i", "category", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Judgement:
    """One verdict on one narrated sentence."""
    category: str
    p_boundary: float   # graded evidence in [0, 1]
    reason: str = ""

    @property
    def is_boundary(self) -> bool:
        return self.category == _BOUNDARY


def _load_dotenv() -> None:
    """Read a .env from the project root (see gui_state_scoring for the why)."""
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return
    try:
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        import anthropic
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def is_available() -> bool:
    """True iff the LLM can actually be called right now."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    try:
        client = _client()
    except Exception:
        return False
    return bool(getattr(client, "api_key", None) or getattr(client, "auth_token", None))


def ask(sentences: list[str], model: str = MODEL) -> str:
    """Send the whole narration in ONE call so each sentence is judged in context."""
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    message = _client().messages.create(
        model=model,
        max_tokens=8000,
        system=_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": _SCHEMA}},
        messages=[{"role": "user", "content": (
            f"{len(sentences)} sentences, in order:\n\n{numbered}\n\n"
            "One verdict per sentence. JSON only."
        )}],
    )
    if message.stop_reason == "refusal":
        return ""
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            "Antwort von max_tokens abgeschnitten — max_tokens in "
            "audio_boundary_scoring.ask erhöhen."
        )
    return "".join(b.text for b in message.content if b.type == "text")


def parse(raw: str, n: int) -> list[Judgement] | None:
    """
    Turn the model's answer into one graded judgement per sentence.

    P(boundary) is the confidence when the verdict is OPERATION and its
    complement otherwise — a verdict plus a confidence IS a probability over the
    boundary question, which is what the Random Forest downstream needs.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    by_index: dict[int, Judgement] = {}
    for v in data.get("verdicts", []):
        try:
            i = int(v["i"])
            category = str(v["category"]).strip().upper()
            if category not in _CATEGORIES:
                continue
            conf = float(np.clip(float(v.get("confidence", 0.5)), 0.0, 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        by_index[i] = Judgement(
            category=category,
            p_boundary=conf if category == _BOUNDARY else 1.0 - conf,
            reason=str(v.get("reason", ""))[:120],
        )

    if not by_index:
        return None
    # A sentence the model skipped gets neutral evidence rather than a guess.
    return [by_index.get(i, Judgement("OTHER", 0.0, "kein Urteil")) for i in range(n)]


class Cache:
    """Verdicts cached beside the recording, keyed on the transcript content."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, list] = {}
        if path.exists():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

    @staticmethod
    def key(sentences: list[str], model: str) -> str:
        h = hashlib.sha1("␟".join(sentences).encode("utf-8"))
        h.update(f"|{model}|{PROMPT_VERSION}".encode())
        return h.hexdigest()

    def get(self, key: str) -> list[Judgement] | None:
        raw = self._entries.get(key)
        if not isinstance(raw, list):
            return None
        try:
            return [Judgement(str(e["category"]), float(e["p_boundary"]),
                              str(e.get("reason", ""))) for e in raw]
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, key: str, js: list[Judgement]) -> None:
        self._entries[key] = [
            {"category": j.category, "p_boundary": j.p_boundary, "reason": j.reason}
            for j in js
        ]

    def flush(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError:
            pass


def judge(sentences: list[str], cache: Cache | None = None,
          model: str = MODEL) -> list[Judgement] | None:
    """
    Judge every narrated sentence, reusing a cached verdict set when there is one.

    Transport and API failures are NOT swallowed — they would otherwise fail every
    sentence identically and hand back an empty lane that looks like "no
    boundaries found".
    """
    if not sentences:
        return []
    key = Cache.key(sentences, model)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None and len(hit) == len(sentences):
            return hit

    result = parse(ask(sentences, model), len(sentences))
    if result is not None and cache is not None:
        cache.put(key, result)
    return result
