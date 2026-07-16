"""
VLM judgement on GUI state transitions — cloud (Claude) or local (Ollama).

Given two SETTLED screenshots — the UI before a change and after it has come to
rest — a vision-language model answers one question:

    "Eine Handlung ist vom User angestoßen und überführt das System in einen
     neuen Zustand."  Is this transition such an action?

The distinction that carries the whole method is transient vs. persistent state
(WorldGUI, arXiv:2502.08047): opening the File menu repaints a lot of pixels but
leaves the system where it was — it is an affordance the user is still
navigating. Opening the document IS the state change. No appearance metric can
tell those apart (the menu changes MORE pixels), which is why the judgement is
made by a model that answers a question rather than by one that measures a
distance. ViBR (Feng et al., PACMSE/FSE 2026, arXiv:2604.19905) established the
operation and the gap: VLM frame-pair comparison reaches F1 0.87 on GUI state
comparison where SSIM reaches 0.62 and CLIP similarity less.

WHY TWO BACKENDS:
  The local 7B is at the edge of what this task needs. GUI Knowledge Bench (Shi
  et al., arXiv:2510.26098) measures exactly this — infer the action from two
  consecutive screenshots — and reports 50.6 % action-type accuracy for
  Qwen2.5-VL-7B against ~76 % for the strongest frontier models. Both backends
  see the IDENTICAL composite image and the identical prompt, so the two runs are
  directly comparable: the difference measured on the same recording is the
  model's contribution, and a local-vs-cloud gap is a result, not a confound.
  Ollama also remains the "no data leaves the machine" option.

Verdicts are cached per frame-pair content and per model, so re-running the
extractor (every time the FeatureDialog opens) is free, and switching backends
does not overwrite the other backend's cache.

This module reads pixels only. It never touches audio or the event stream — the
video modality has to stay independent for the 2^3 Shapley ablation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

def _load_dotenv() -> None:
    """
    Read KEY=VALUE lines from a .env in the project root into the environment.

    Windows' `setx` writes to the registry, and a process that was already
    running — PyCharm, an open terminal — never re-reads it. The result is an
    app that cannot see a key the user demonstrably set, which is a confusing
    way to lose an afternoon. A .env file is read at import time, so it works
    regardless of when the process started.

    Never overwrites a variable that is already set: the real environment wins.
    """
    root = Path(__file__).resolve().parents[2]
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        # utf-8-sig, not utf-8: PowerShell's Out-File and Notepad write a BOM,
        # which would otherwise stick to the first key ("﻿ANTHROPIC_API_KEY")
        # and make a perfectly correct .env silently do nothing.
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

# Backend: "anthropic" (cloud) or "ollama" (local). Override via DOCUPILOT_VLM.
BACKEND = os.environ.get("DOCUPILOT_VLM", "anthropic").lower()

CLOUD_MODEL = "claude-opus-4-8"
OLLAMA_MODEL = "qwen2.5vl:7b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

MODEL = CLOUD_MODEL if BACKEND == "anthropic" else OLLAMA_MODEL

# Bump when the prompt or the score mapping changes — invalidates cached verdicts.
PROMPT_VERSION = "v7"   # v7: goal-vs-means definition + async results (see
                        # docs/annotationsleitfaden.md); v6: Set-of-Mark box + zoom

# The two frames are stitched into ONE composite image, BEFORE left, AFTER right.
#
# This is not cosmetic. Sent as two separate images, Ollama/qwen2.5vl receives
# both but SWAPS them: given [desktop, excel] it reported "image 1 = Excel,
# image 2 = a bridge". For a before/after question that is fatal — the direction
# decides whether a document was opened or closed. In one labelled composite the
# assignment is stable and correct. ViBR (arXiv:2604.19905) uses the same
# "dual-view composite image" for exactly this comparison.
#
# Each half is downscaled to _HALF_WIDTH: GUI text stays legible, and both
# latency and vision-token count scale with pixel area.
_HALF_WIDTH = 896
_BANNER_H = 34
_GAP = 12
_JPEG_QUALITY = 80

_BOUNDARY = "ACTION_COMPLETED"
_CATEGORIES = (
    _BOUNDARY,        # finished RESULT of a user operation, now persistent —
                      # incl. a deliberate view/mode change and delayed async
                      # results (build/test/filter finishing). See the boundary
                      # definition in docs/annotationsleitfaden.md.
    "TRANSIENT_UI",   # overlay/selection, or navigation the user passes through
    "IN_PROGRESS",    # mid-typing, mid-scroll, spinner: the operation is not done
    "NO_CHANGE",      # caret, clock, codec noise
    "SYSTEM_INITIATED",  # unsolicited system change (notification/toast), NOT a
                         # delayed result of a user action
)

_SYSTEM = """\
The image shows two screenshots from a screen recording of someone using desktop
software, side by side: BEFORE the change on the LEFT, AFTER the screen settled
on the RIGHT.

A RED BOX may be drawn on both halves. It marks the only region where the two
screenshots differ in pixels — this was measured, not guessed. Everything outside
the box is provably identical. If a second row is present, it is that boxed
region magnified, BEFORE on the left and AFTER on the right.

Read the box correctly:
  - It tells you WHERE to look. It does NOT tell you that an action happened —
    a dropdown opening or a tooltip appearing produces a box just the same.
  - Do NOT report any difference outside the box. There is none. If you think you
    see one, you are misreading the image.
  - If the boxed region looks the same in both halves apart from a caret, a
    highlight, or rendering noise, then nothing meaningful changed - say so.
  - If NO box is drawn, the two screenshots are identical.

Work in this order, and do not skip a step:

STEP 1 - OBSERVE. Describe what is actually different inside the box. Look
carefully: the difference may be tiny (small arrows appearing in a table header,
one cell recoloured, fewer rows, a different row order, a status-bar text). If
you find no meaningful difference, say so plainly.

STEP 2 - CLASSIFY, based strictly on what you observed in step 1. Pick exactly
one category:

  NO_CHANGE         Nothing differs, or only a blinking caret / clock / codec
                    noise. If step 1 found no difference, you MUST pick this.
  TRANSIENT_UI      An overlay or intermediate the user is only passing THROUGH,
                    or a move that merely relocates: a menu, dropdown, submenu,
                    dialog, tooltip or context menu that is open; a highlighted
                    selection; scrolling, or switching to another sheet / page /
                    folder the user has not yet done anything to. Nothing was
                    carried out here — it is on the way to an action.
  IN_PROGRESS       The same operation is still running: text half typed into a
                    field, a scroll mid-motion, a spinner, a half-painted frame.
  SYSTEM_INITIATED  A change the user did NOT trigger and did not request: an
                    incoming notification, a chat or mail popup, an autosave
                    toast, a clock tick. NOT the delayed result of something the
                    user just started — that is ACTION_COMPLETED.
  ACTION_COMPLETED  The finished RESULT of an operation the user carried out is
                    now visible and persists. Three cases count:
                    (1) content or structure changed — a filter or sort applied,
                        a typed value committed, formatting applied, a row
                        deleted, a sheet / folder / slide created, a file moved,
                        content pasted, a document opened;
                    (2) a view or mode the user switched to ON PURPOSE that
                        transforms how the content is presented and stays — a
                        reading view, a details view, a persistent zoom;
                    (3) the DELAYED result of a user-triggered operation
                        appearing on its own — a filter finishing, a build or
                        test completing, a page or document finishing loading.

The decisive question is whether this is the finished RESULT of an operation the
user carried out — a goal reached that persists — not how many pixels moved, and
NOT whether it is "data" versus "view". A view or mode the user deliberately
switched to counts; a menu they opened on the way does not; merely scrolling or
switching to where they will act next does not. A dropdown opening repaints far
more of the screen than a filter being applied, yet only the filter is an action.
Size is not importance.

The mouse cursor is not recorded, so you will never see a pointer.

Answer with ONE JSON object and nothing else. Fill "observation" FIRST and let
it decide "category":
{"observation": "<what actually differs, max 20 words>",
 "category": "<one of the five above>",
 "confidence": <0.0-1.0>}"""

_USER = ("LEFT = BEFORE, RIGHT = AFTER the screen settled. The red box marks the "
         "only region where they differ.\n"
         "Observe what changed inside it, then classify it. JSON only.")


@dataclass(frozen=True)
class Judgement:
    """One verdict on one (before, after) pair."""
    category: str
    p_boundary: float   # graded evidence in [0, 1]
    reason: str = ""    # the model's own observation — keep it: it is how you
                        # tell "saw nothing, guessed a label" from a real read

    @property
    def is_boundary(self) -> bool:
        return self.category == _BOUNDARY


# The JSON the model must return. On the cloud backend this is enforced by the
# API (structured outputs), so an unparseable answer cannot happen; on Ollama it
# is only requested, and parse() has to cope with a malformed one.
_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "category": {"type": "string", "enum": list(_CATEGORIES)},
        "confidence": {"type": "number"},
    },
    "required": ["observation", "category", "confidence"],
    "additionalProperties": False,
}


# ── Backends ──────────────────────────────────────────────────────────────────

def is_available(model: str = MODEL) -> bool:
    """True iff the configured backend can actually be called right now."""
    if BACKEND == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        # Constructing the client is NOT enough: it succeeds without credentials
        # and only fails on the first request. That would make the extractor
        # report "available", then silently produce no evidence at all. Check
        # that a credential actually resolved — an unset ANTHROPIC_API_KEY does
        # not mean "none": the SDK also reads an `ant auth login` profile.
        try:
            client = _client()
        except Exception:
            return False
        return bool(getattr(client, "api_key", None) or getattr(client, "auth_token", None))

    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2) as resp:
            tags = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False
    # Exact, not by family: accepting any qwen2.5vl tag would report "available"
    # when only the 3b is present, and the run would then block on Ollama pulling
    # 6 GB for the 7b — halfway through, from inside the UI.
    return model in {m.get("name", "") for m in tags.get("models", [])}


_ANTHROPIC_CLIENT = None


def _client():
    global _ANTHROPIC_CLIENT
    if _ANTHROPIC_CLIENT is None:
        import anthropic
        _ANTHROPIC_CLIENT = anthropic.Anthropic()
    return _ANTHROPIC_CLIENT


def ask(composite: bytes, model: str = MODEL, timeout: float = 600) -> str:
    """Send one BEFORE|AFTER composite to the model and return its raw answer."""
    if BACKEND == "anthropic":
        return _ask_anthropic(composite, model)
    return _ask_ollama(composite, model, timeout)


def _ask_anthropic(composite: bytes, model: str) -> str:
    # Adaptive thinking, because the hard part of this task IS the fine-grained
    # visual comparison — DiffSpot (arXiv:2605.29615) puts the best models at
    # ~23 % on GUI-level differences, so the model needs room to look properly.
    # Effort stays low: the OUTPUT is one small JSON object, and this runs once
    # per settled-state pair across a whole recording.
    #
    # max_tokens covers thinking AND the answer. 4000 is deliberate headroom: a
    # budget that runs out mid-JSON would truncate the answer, parse() would
    # return None, and the pair would silently get no evidence — a wrong result
    # dressed as a missing one.
    message = _client().messages.create(
        model=model,
        max_tokens=4000,
        system=_SYSTEM,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(composite).decode(),
                    },
                },
                {"type": "text", "text": _USER},
            ],
        }],
    )

    if message.stop_reason == "refusal":
        return ""                     # a verdict on this pair: no evidence
    if message.stop_reason == "max_tokens":
        # Not a per-pair problem — the budget is wrong and every pair will hit
        # it. Fail loudly instead of quietly emitting an empty lane.
        raise RuntimeError(
            "Antwort wurde von max_tokens abgeschnitten — Thinking-Budget zu klein. "
            "max_tokens in _ask_anthropic erhöhen."
        )
    return "".join(b.text for b in message.content if b.type == "text")


def _ask_ollama(composite: bytes, model: str, timeout: float) -> str:
    payload = json.dumps({
        "model": model,
        "system": _SYSTEM,
        "prompt": _USER,
        "images": [base64.b64encode(composite).decode()],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "keep_alive": "30m",   # otherwise every call reloads 6 GB from disk
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()).get("response", "")


def parse(raw: str) -> Judgement | None:
    """
    Turn the model's answer into graded evidence.

    P(boundary) is the model's confidence when it says ACTION_COMPLETED and the
    complement otherwise — a verdict plus a confidence IS a probability over the
    boundary question, which is what the Random Forest downstream needs. The
    extractor must never hand it a decision.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    category = str(data.get("category", "")).strip().upper()
    if category not in _CATEGORIES:
        return None
    try:
        confidence = float(np.clip(float(data.get("confidence", 0.5)), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.5

    return Judgement(
        category=category,
        p_boundary=confidence if category == _BOUNDARY else 1.0 - confidence,
        reason=str(data.get("observation", data.get("reason", "")))[:120],
    )


def downscale(bgr: np.ndarray) -> np.ndarray:
    """Shrink one frame to _HALF_WIDTH — done once per settled state, then reused."""
    import cv2

    h, w = bgr.shape[:2]
    if w <= _HALF_WIDTH:
        return bgr
    return cv2.resize(bgr, (_HALF_WIDTH, round(h * _HALF_WIDTH / w)),
                      interpolation=cv2.INTER_AREA)


def _crop(frame: np.ndarray, region: tuple[float, float, float, float],
          pad: float = 0.02) -> np.ndarray:
    """Cut the marked region out of a frame, with a little context around it."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    x0 = max(0, int((x0 - pad) * w))
    y0 = max(0, int((y0 - pad) * h))
    x1 = min(w, int((x1 + pad) * w))
    y1 = min(h, int((y1 + pad) * h))
    if x1 <= x0 or y1 <= y0:
        return frame
    return frame[y0:y1, x0:x1]


def encode_pair(before: np.ndarray, after: np.ndarray,
                region: tuple[float, float, float, float] | None = None) -> bytes:
    """
    Stitch two downscaled frames into one labelled BEFORE|AFTER composite.

    When `region` is given (the tiles whose pHash differs), it is drawn as a box
    on BOTH halves and a magnified crop of it is appended as a second row:

      +-----------------+-----------------+
      | BEFORE  [ box ] | AFTER   [ box ] |    full context
      +-----------------+-----------------+
      | ZOOM: BEFORE    | ZOOM: AFTER     |    the box, enlarged
      +-----------------+-----------------+

    The box is Set-of-Mark prompting (Yang et al., arXiv:2310.11441) — marking a
    region measurably improves what a VLM grounds its answer in. The zoom row is
    the V*/CropVLM finding (Wu & Xie, CVPR 2024; arXiv:2511.19820): MLLMs simply
    miss small details in high-resolution images, and this recording's weakest
    real action (filter arrows appearing) is 0.005 % of the pixels.

    Both are free: one image, one request. Only the picture gets better.
    """
    import cv2

    before, after = before.copy(), after.copy()
    box = (0, 0, 255)   # red — BGR

    if region is not None:
        for half in (before, after):
            h, w = half.shape[:2]
            x0, y0, x1, y1 = region
            cv2.rectangle(
                half,
                (int(x0 * w), int(y0 * h)), (int(x1 * w) - 1, int(y1 * h) - 1),
                box, 3,
            )

    h = max(before.shape[0], after.shape[0])
    half_w = before.shape[1]
    rows = [(before, after)]

    # A zoom row only helps when the change is SMALL. When the box covers most of
    # the screen (a window opening) the crop is the frame again — pure token cost.
    if region is not None:
        x0, y0, x1, y1 = region
        if (x1 - x0) * (y1 - y0) < 0.35:
            zoom_b = _crop(before, region)
            zoom_a = _crop(after, region)
            scale = half_w / max(zoom_b.shape[1], 1)
            size = (half_w, max(1, int(zoom_b.shape[0] * scale)))
            rows.append((
                cv2.resize(zoom_b, size, interpolation=cv2.INTER_CUBIC),
                cv2.resize(zoom_a, size, interpolation=cv2.INTER_CUBIC),
            ))

    total_h = _BANNER_H + h + sum(_BANNER_H + r[0].shape[0] for r in rows[1:])
    canvas = np.full((total_h, half_w + _GAP + half_w, 3), 255, dtype=np.uint8)

    font, scale_txt, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    labels = [("BEFORE", "AFTER"), ("ZOOM: BEFORE", "ZOOM: AFTER")]
    y = 0
    for (left, right), (label_l, label_r) in zip(rows, labels):
        cv2.putText(canvas, label_l, (8, y + 24), font, scale_txt, (0, 0, 0), thick)
        cv2.putText(canvas, label_r, (half_w + _GAP + 8, y + 24),
                    font, scale_txt, (0, 0, 0), thick)
        y += _BANNER_H
        canvas[y:y + left.shape[0], :left.shape[1]] = left
        canvas[y:y + right.shape[0], half_w + _GAP:half_w + _GAP + right.shape[1]] = right
        y += left.shape[0]

    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG-Encoding des Frame-Paars fehlgeschlagen")
    return buf.tobytes()


# ── Cache ─────────────────────────────────────────────────────────────────────

class Cache:
    """
    Verdicts cached beside the recording, keyed by the CONTENT of the frame pair
    (plus model and prompt version). Re-encoding the video or re-tuning the
    proposal stage therefore costs nothing: the same two frames reuse the same
    verdict.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, dict] = {}
        if path.exists():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

    @staticmethod
    def key(composite: bytes, model: str) -> str:
        h = hashlib.sha1(composite)
        h.update(f"|{model}|{PROMPT_VERSION}".encode())
        return h.hexdigest()

    def get(self, key: str) -> Judgement | None:
        """A malformed entry is a cache miss, never a crash — the file is on disk
        and may predate a change to Judgement."""
        e = self._entries.get(key)
        if not isinstance(e, dict):
            return None
        try:
            return Judgement(
                category=str(e["category"]),
                p_boundary=float(e["p_boundary"]),
                reason=str(e.get("reason", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def put(self, key: str, j: Judgement) -> None:
        self._entries[key] = {
            "category": j.category, "p_boundary": j.p_boundary, "reason": j.reason
        }

    def flush(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError:
            pass


def judge(composite: bytes, cache: Cache | None = None,
          model: str = MODEL) -> Judgement | None:
    """
    Score one BEFORE|AFTER composite, reusing a cached verdict when there is one.

    Returns None only when the model ANSWERED but the answer was unusable — that
    is a verdict about one pair, and the pair simply gets no evidence.

    Transport and API failures are NOT swallowed. The SDK already retries 429s
    and 5xx; anything that still escapes (exhausted credit, revoked key, dead
    network) would otherwise fail every single pair identically and hand back an
    empty lane that LOOKS like "no boundaries found". Let it raise.
    """
    key = Cache.key(composite, model)
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    result = parse(ask(composite, model))
    if result is not None and cache is not None:
        cache.put(key, result)
    return result
