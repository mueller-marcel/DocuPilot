"""
The video modality's semantic stage: given two SETTLED screenshots, a VLM decides
whether a user-triggered action completed between them. The model runs in the
cloud (Anthropic); which one is env-driven so it can be A/B-tested.

Reads pixels only, never audio or the event stream.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from docupilot.segmentation import backend

# The model is env-driven so a faster/cheaper one (e.g. Haiku 4.5,
# claude-haiku-4-5-20251001) can be A/B-tested without a code edit. Different
# models get separate cache entries, so opus verdicts stay intact.
MODEL = os.environ.get("DOCUPILOT_CLOUD_MODEL", "claude-opus-4-8")

# Bump when the prompt or the score mapping changes — invalidates cached verdicts.
PROMPT_VERSION = "v7"   # v7: goal-vs-means definition + async results;
                        # v6: Set-of-Mark box + zoom

# ONE composite image, BEFORE left, AFTER right — not cosmetic: sent as two
# separate images the model swaps them, which is fatal for a before/after question.
_HALF_WIDTH = 896
_BANNER_H = 34
_GAP = 12
_JPEG_QUALITY = 80

# A zoom row only helps when the change is SMALL. Above this share of the
# screen the crop is the frame again — pure token cost.
_ZOOM_MAX_AREA = 0.35

_BOUNDARY = "ACTION_COMPLETED"
_CATEGORIES = (
    _BOUNDARY,        # finished RESULT of a user operation, now persistent —
                      # incl. a deliberate view/mode change and delayed async
                      # results (build/test/filter finishing)
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


# The one boundary decision rule lives downstream (BOUNDARY_THRESHOLD on
# p_boundary) — deliberately no category-based second rule here.


# The JSON the model must return. The API enforces it (structured outputs), so an
# unparseable answer cannot happen; parse() stays defensive regardless.
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


# ── Cloud model ───────────────────────────────────────────────────────────────

is_available = backend.is_available


def ask(composite: bytes, model: str = MODEL) -> str:
    """Send one BEFORE|AFTER composite to the model and return its raw answer."""
    # Thinking room for the fine-grained visual comparison; low effort because the
    # output is one small JSON object. max_tokens covers thinking AND the answer.
    message = backend.client().messages.create(
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
            "max_tokens in video_scoring.ask erhöhen."
        )
    return "".join(b.text for b in message.content if b.type == "text")


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


# ── Composite image ───────────────────────────────────────────────────────────

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

    `region` is drawn as a box on both halves and appended enlarged as a second
    row — Set-of-Mark plus zoom, both free: one image, one request.
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

    if region is not None:
        x0, y0, x1, y1 = region
        if (x1 - x0) * (y1 - y0) < _ZOOM_MAX_AREA:
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

class Cache(backend.JsonVerdictCache):
    """
    Verdicts keyed by the CONTENT of the frame pair (plus model and prompt
    version). Re-encoding the video or re-tuning the proposal stage therefore
    costs nothing: the same two frames reuse the same verdict.
    """

    @staticmethod
    def key(composite: bytes, model: str) -> str:
        h = hashlib.sha1(composite)
        h.update(f"|{model}|{PROMPT_VERSION}".encode())
        return h.hexdigest()

    def get(self, key: str) -> Judgement | None:
        e = self._raw(key)
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
        self._store(key, {
            "category": j.category, "p_boundary": j.p_boundary, "reason": j.reason
        })


def judge(composite: bytes, cache: Cache | None = None,
          model: str = MODEL) -> Judgement | None:
    """
    Score one BEFORE|AFTER composite, reusing a cached verdict when there is one.

    None means the model answered unusably — that pair gets no evidence. Transport
    failures are not swallowed: an empty lane would look like a result.
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
