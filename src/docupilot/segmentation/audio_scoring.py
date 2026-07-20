"""
The audio modality's semantic stage: an LLM decides, per narrated sentence,
whether it announces an operation (a boundary) or only a means.

Reads the transcript only, never the screen or the event stream.
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
# a2: dropped the example sentences; a1's were lifted from the session under
#     evaluation, so its verdicts prove nothing about generalisation.
PROMPT_VERSION = "a2"

_BOUNDARY = "OPERATION"
_CATEGORIES = (
    _BOUNDARY,   # announces an operation that completes in a new persistent state
    "MEANS",     # navigation / selection / setup on the way to another operation
    "OTHER",     # filler, commentary, verification — announces nothing
)

# The annotation guideline restated for narration, and nothing else. No example
# sentences on purpose — see PROMPT_VERSION.
_SYSTEM = """\
You are given the spoken narration of a person working through a task in desktop
software, transcribed and split into sentences, in order. They were told to say
each step out loud as they do it. The narration may be in German.

For EACH sentence decide what it announces, using this definition:

  A BOUNDARY is the completion of an operation the user carried out — the moment
  its result becomes visible and settles into a state that PERSISTS.

Categories:

  OPERATION  The sentence announces an operation whose completion is a boundary:
             something is applied, created, entered, deleted, moved, inserted,
             saved or opened, or a view/mode is deliberately switched to and kept
             because that switch is itself the goal.

  MEANS      The sentence announces only a step ON THE WAY to another operation —
             it leaves nothing that persists: navigating or switching to where
             the next action will happen, scrolling to bring something into view,
             selecting or marking, copying to the clipboard, opening a menu or
             dialog, clicking into a field before typing in it.

  OTHER      Announces no step: filler, thinking aloud, commentary on a result,
             reading or checking something, greetings.

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
    """Read a .env from the project root (see video_scoring for the why)."""
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
            "audio_scoring.ask erhöhen."
        )
    return "".join(b.text for b in message.content if b.type == "text")


def parse(raw: str, n: int) -> list[Judgement] | None:
    """
    Turn the model's answer into one graded judgement per sentence: P(boundary)
    is the confidence for OPERATION and its complement otherwise.
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
    Transport failures are not swallowed — an empty lane would look like a result.
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
