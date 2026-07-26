"""
The experiment itself: every modality subset scored on every session, with the
model that predicts a session never having seen it.

Produces one tidy table. Every later analysis — Shapley, saturation, significance
— is derived from it, so the expensive part runs once.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from docupilot.evaluation import fusion, metrics
from docupilot.evaluation.analysis import subsets
from docupilot.evaluation.dataset import duration_s, ground_truth_s
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import MODALITIES, BoundaryEvidence, segment

# Tolerance the training labels are built with. Held fixed across the tolerance
# sweep so the sweep varies only how predictions are *scored*, not what the model
# was taught — otherwise the two effects could not be told apart.
LABEL_TAU_S = 1.0

TAUS_S: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class SessionData:
    """One recording with everything the experiment needs, extracted once."""

    name: str
    gt_s: list[float]
    duration_s: float
    evidence: dict[str, BoundaryEvidence]


def load(
    directories: Sequence[Path],
    use_cache: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> list[SessionData]:
    """
    Run the segmentation once per session and keep the evidence in memory.

    The costly model calls happen here — once per session and modality, not once
    per subset. All 8 subsets reuse these curves, so the factorial design costs
    three extractions per session rather than eight.

    :param directories: session directories, each with recording.mp4 and
        ground_truth.json.
    :param use_cache: reuse verdicts cached beside the recordings.
    :param on_progress: called as (session name, done, total).
    :param is_cancelled: polled between and inside sessions; True stops early
        and returns what was extracted so far.
    :return: one entry per session, in the given order.
    :raises RuntimeError: when a modality fails — an empty lane would silently
        look like a modality that found nothing.
    """
    out: list[SessionData] = []
    for i, directory in enumerate(directories, start=1):
        if is_cancelled is not None and is_cancelled():
            return out
        session = RecordingSession.from_directory(directory)
        evidence: dict[str, BoundaryEvidence] = {}
        errors: dict[str, str] = {}
        segment(
            session,
            on_result=evidence.__setitem__,
            on_error=errors.__setitem__,
            use_cache=use_cache,
            is_cancelled=is_cancelled,
        )
        if is_cancelled is not None and is_cancelled():
            return out
        if errors:
            raise RuntimeError(
                f"{directory.name}: Modalität(en) fehlgeschlagen — "
                + "; ".join(f"{m}: {msg.splitlines()[0]}" for m, msg in errors.items())
            )
        out.append(SessionData(
            name=directory.name,
            gt_s=ground_truth_s(session),
            duration_s=duration_s(session),
            evidence=evidence,
        ))
        if on_progress is not None:
            on_progress(directory.name, i, len(directories))
    return out


def _prepare(data: Sequence[SessionData], subset: tuple[str, ...]) -> list[dict]:
    """Candidates, features and labels per session — independent of the fold."""
    columns = sorted(subset)
    prepared = []
    for session in data:
        times = fusion.candidate_times(session.evidence, columns)
        prepared.append({
            "times": times,
            "features": fusion.feature_matrix(times, session.evidence, columns),
            "labels": fusion.label_candidates(times, session.gt_s, LABEL_TAU_S),
        })
    return prepared


def run(
    data: Sequence[SessionData],
    fuser_factory: Callable[[], object] = fusion.ForestFuser,
    players: Sequence[str] = MODALITIES,
    taus_s: Sequence[float] = TAUS_S,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> list[dict]:
    """
    Score every subset on every session, leaving that session out of training.

    :param data: the corpus.
    :param fuser_factory: builds an untrained decider; swap to compare methods.
    :param players: the modalities the factorial design runs over.
    :param taus_s: tolerances every prediction is scored at.
    :param on_progress: called as (subset label, done, total).
    :return: one row per (session, subset, tolerance).
    """
    all_subsets = subsets(players)
    rows: list[dict] = []

    for index, coalition in enumerate(all_subsets, start=1):
        subset = tuple(sorted(coalition))
        label = "+".join(subset) if subset else "{}"

        prepared = _prepare(data, subset) if subset else None

        for held_out, session in enumerate(data):
            if not subset:
                predicted: list[float] = []          # no modality, no candidates
            else:
                train = [p for i, p in enumerate(prepared) if i != held_out]
                fuser = fuser_factory()
                if train:
                    fuser.fit(
                        np.vstack([p["features"] for p in train]),
                        np.concatenate([p["labels"] for p in train]),
                    )
                # A single-session corpus leaves nothing to train on. The rule
                # needs no training and still works; a trained fuser would have
                # to see the session it predicts, so it is not offered there.
                test = prepared[held_out]
                scores = fuser.predict_proba(test["features"])
                keep = scores >= 0.5
                predicted = fusion.suppress(test["times"][keep], scores[keep])

            for tau in taus_s:
                m = metrics.match(session.gt_s, predicted, tau)
                rows.append({
                    "session": session.name,
                    "subset": label,
                    "n_modalities": len(subset),
                    "tau_s": tau,
                    "tp": m.tp, "fp": m.fp, "fn": m.fn,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                })

        if on_progress is not None:
            on_progress(label, index, len(all_subsets))

    return rows


def subset_values(rows: Sequence[dict], tau_s: float) -> dict[frozenset[str], float]:
    """
    Macro-averaged F1 per subset — the characteristic function the Shapley
    values are computed from.

    Macro because sessions are the unit of the experiment: each counts once,
    regardless of length.
    """
    per_subset: dict[frozenset[str], list[float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        key = frozenset() if row["subset"] == "{}" else frozenset(row["subset"].split("+"))
        per_subset.setdefault(key, []).append(row["f1"])
    return {k: float(np.mean(v)) for k, v in per_subset.items()}


def paired_f1(rows: Sequence[dict], tau_s: float) -> dict[frozenset[str], dict[str, float]]:
    """
    Per-session F1 per subset — what the paired statistics later need.

    The mean alone cannot answer whether a difference is real; that question is
    asked of the per-session differences.
    """
    out: dict[frozenset[str], dict[str, float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        key = frozenset() if row["subset"] == "{}" else frozenset(row["subset"].split("+"))
        out.setdefault(key, {})[row["session"]] = row["f1"]
    return out


def chance_rows(
    data: Sequence[SessionData], taus_s: Sequence[float] = TAUS_S, seed: int = 0
) -> list[dict]:
    """The floor: F1 reached by guessing, one row per (session, tolerance)."""
    return [
        {
            "session": session.name,
            "subset": "chance",
            "n_modalities": 0,
            "tau_s": tau,
            "f1": metrics.chance_level(session.gt_s, session.duration_s, tau, seed=seed),
        }
        for session in data
        for tau in taus_s
    ]


def write_csv(rows: Sequence[dict], path: Path) -> None:
    """Write the tidy table; figures for the thesis are generated from this file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
