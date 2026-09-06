"""
The experiment itself: every modality subset scored on every session, with the
model that predicts a session never having seen it.

Produces one tidy table per candidate-pool policy. Every later analysis —
Shapley, saturation, significance — is derived from these tables, so the
expensive part runs once per policy.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from docupilot.evaluation import fusion, metrics
from docupilot.evaluation.analysis import subsets
from docupilot.evaluation.dataset import duration_s, ground_truth_s
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import MODALITIES, BoundaryEvidence, segment

# Tolerance the training labels are built with and the threshold is calibrated
# at. Held fixed across the tolerance sweep so the sweep varies only how
# predictions are *scored*, not what the model was taught — otherwise the two
# effects could not be told apart.
LABEL_TAU_S = 1.0

TAUS_S: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

# Rows are keyed by this label; the empty coalition has no modality to name.
EMPTY_SUBSET_LABEL = "{}"

# Where a coalition's candidates come from.
#   "union": the pool is fixed to ALL players for every coalition, so the
#            factorial design varies only the information available for
#            scoring, never the search space — the ablation the research
#            question's "Informationsbeitrag" asks for. The primary design.
#   "own":   each coalition proposes its own candidates and sees nothing of
#            the others — what "isoliert" (sub-question 1) literally means.
#            The difference to "union" for a single modality is the credit
#            that modality receives for the other modalities' TIMING alone.
CandidatePool = Literal["union", "own"]


@dataclass(frozen=True)
class SessionData:
    """One recording with everything the experiment needs, extracted once."""

    name: str
    gt_s: list[float]
    duration_s: float
    evidence: dict[str, BoundaryEvidence]


@dataclass(frozen=True)
class _Prepared:
    """One session's candidates, described by the given modalities, labelled once."""

    times: np.ndarray
    features: np.ndarray      # (n_candidates, FEATURES_PER_MODALITY * n_modalities)
    labels: np.ndarray
    gt_s: list[float]


def load(
    directories: Sequence[Path],
    use_cache: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    kind: str = "end",
) -> list[SessionData]:
    """
    Run the segmentation once per session and keep the evidence in memory.

    The costly model calls happen here — once per session and modality, not once
    per subset. All 8 subsets reuse these curves, so the factorial design costs
    three extractions per session rather than eight.

    :param directories: session directories, each with recording.mp4 and
        ground_truth.json.
    :param use_cache: reuse the lanes and verdicts cached beside the recordings —
        a session extracted once is loaded from its own directory afterwards.
    :param on_progress: called as (session name, done, total).
    :param is_cancelled: polled between and inside sessions; True stops early
        and returns what was extracted so far.
    :param kind: which boundary definition to load as ground truth — "end"
        (the action's result has settled) or "start" (the next action's first
        input); see RecordingSession.
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
            gt_s=ground_truth_s(session, kind),
            duration_s=duration_s(session),
            evidence=evidence,
        ))
        if on_progress is not None:
            on_progress(directory.name, i, len(directories))
    return out


def with_ground_truth(data: Sequence[SessionData], gt_by_name: Mapping[str, list[float]]) -> list[SessionData]:
    """The same corpus scored against another set of boundaries — the evidence
    is untouched, only the reference changes."""
    return [
        SessionData(d.name, list(gt_by_name[d.name]), d.duration_s, d.evidence)
        for d in data
    ]


def subset_label(subset: Sequence[str] | frozenset[str]) -> str:
    """The row key for a coalition: its modalities sorted and joined."""
    return "+".join(sorted(subset)) or EMPTY_SUBSET_LABEL


def subset_from_label(label: str) -> frozenset[str]:
    """Inverse of `subset_label`."""
    return frozenset() if label == EMPTY_SUBSET_LABEL else frozenset(label.split("+"))


def _prepare(
    data: Sequence[SessionData], pool: Sequence[str], columns: Sequence[str]
) -> list[_Prepared]:
    """
    Candidates from `pool`, features for `columns`, labels — per session.

    :param pool: modalities whose score peaks become the candidates.
    :param columns: modalities whose features describe them, block order.
    """
    prepared = []
    for session in data:
        times = fusion.candidate_times(session.evidence, pool)
        prepared.append(_Prepared(
            times=times,
            features=fusion.feature_matrix(times, session.evidence, columns),
            labels=fusion.label_candidates(times, session.gt_s, LABEL_TAU_S),
            gt_s=session.gt_s,
        ))
    return prepared


def _predict_held_out(
    prepared: Sequence[_Prepared], column_index: Sequence[int], held_out: int
) -> tuple[list[float], float]:
    """
    Leave-one-session-out: fit on every other session, calibrate the threshold
    on those sessions' out-of-bag predictions, predict the held-out one.

    :param column_index: which feature columns the coalition may read.
    :return: (predicted boundaries, threshold used).
    """
    train = [p for i, p in enumerate(prepared) if i != held_out]
    fuser = fusion.ForestFuser()
    fuser.fit(
        np.vstack([p.features[:, column_index] for p in train]),
        np.concatenate([p.labels for p in train]),
    )

    # The OOB probabilities come back as one vector over the stacked training
    # rows; cut it back into sessions so the threshold is chosen on the same
    # per-session F1 the experiment attributes.
    oob = fuser.oob_proba()
    folds, offset = [], 0
    for p in train:
        folds.append((p.times, oob[offset:offset + len(p.times)], p.gt_s))
        offset += len(p.times)
    threshold = fusion.choose_threshold(folds, LABEL_TAU_S)

    test = prepared[held_out]
    scores = fuser.predict_proba(test.features[:, column_index])
    return fusion.decide(test.times, scores, threshold), threshold


def run(
    data: Sequence[SessionData],
    players: Sequence[str] = MODALITIES,
    taus_s: Sequence[float] = TAUS_S,
    pool: CandidatePool = "union",
    on_progress: Callable[[str, int, int], None] | None = None,
    extra: Mapping[str, object] | None = None,
) -> list[dict]:
    """
    Score every subset on every session, leaving that session out of training.

    The empty coalition predicts nothing under either policy: with no feature
    columns there is no basis on which any candidate could be accepted, so
    v(∅) = 0.

    :param data: the corpus; at least two sessions (leave-one-session-out).
    :param players: the modalities the factorial design runs over.
    :param taus_s: tolerances every prediction is scored at.
    :param pool: candidate-pool policy, see `CandidatePool`.
    :param on_progress: called as (subset label, done, total).
    :param extra: columns added to every row, e.g. the boundary definition.
    :return: one row per (session, subset, tolerance).
    :raises ValueError: when fewer than two sessions are given.
    """
    if len(data) < 2:
        raise ValueError(
            "Leave-one-session-out braucht mindestens zwei Sessions — mit einer "
            "gäbe es keine Trainingsdaten, die die bewertete Session nicht "
            "enthalten."
        )

    all_columns = sorted(players)
    union = _prepare(data, all_columns, all_columns) if pool == "union" else None
    all_subsets = subsets(players)
    rows: list[dict] = []

    for index, coalition in enumerate(all_subsets, start=1):
        subset = tuple(sorted(coalition))
        label = subset_label(subset)

        if not subset:
            prepared, column_index = None, []
        elif pool == "union":
            prepared = union
            column_index = [
                c for m in subset for c in fusion.block_columns(all_columns.index(m))
            ]
        else:
            prepared = _prepare(data, subset, subset)
            column_index = list(range(fusion.FEATURES_PER_MODALITY * len(subset)))

        for held_out, session in enumerate(data):
            if prepared is None:
                predicted, threshold = [], float("nan")
            else:
                predicted, threshold = _predict_held_out(prepared, column_index, held_out)
            for tau in taus_s:
                m = metrics.match(session.gt_s, predicted, tau)
                rows.append({
                    "session": session.name,
                    "pool": pool,
                    "subset": label,
                    "n_modalities": len(subset),
                    "tau_s": tau,
                    "threshold": threshold,
                    "n_predicted": len(predicted),
                    "tp": m.tp, "fp": m.fp, "fn": m.fn,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                    **(dict(extra) if extra else {}),
                })

        if on_progress is not None:
            on_progress(label, index, len(all_subsets))

    return rows


def subset_values(
    rows: Sequence[dict], tau_s: float, metric: str = "f1"
) -> dict[frozenset[str], float]:
    """
    Macro-averaged metric per subset — the characteristic function the Shapley
    values are computed from.

    Macro because sessions are the unit of the experiment: each counts once,
    regardless of length.
    """
    per_subset: dict[frozenset[str], list[float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        per_subset.setdefault(subset_from_label(row["subset"]), []).append(row[metric])
    return {k: float(np.mean(v)) for k, v in per_subset.items()}


def paired_metric(
    rows: Sequence[dict], tau_s: float, metric: str = "f1"
) -> dict[frozenset[str], dict[str, float]]:
    """
    Per-session value of one metric per subset — what the paired statistics need.

    The mean alone cannot answer whether a difference is real; that question is
    asked of the per-session values.

    :param metric: which column to pull, e.g. "f1", "recall", "precision".
    """
    out: dict[frozenset[str], dict[str, float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s:
            continue
        out.setdefault(subset_from_label(row["subset"]), {})[row["session"]] = row[metric]
    return out


def taus_in(rows: Sequence[dict]) -> list[float]:
    """The tolerances a result table was scored at, ascending."""
    return sorted({row["tau_s"] for row in rows})


def write_csv(rows: Sequence[dict], path: Path) -> None:
    """Write the tidy table; figures for the thesis are generated from this file."""
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
