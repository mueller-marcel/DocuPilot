"""
Dataset builder: turns the per-modality evidence arrays from
feature_extraction.py into ONE labeled candidate table for the Random
Forest and the subsequent factorial (2^3) modality ablation.

Design principles (see thesis methodology):
  - Extractors emit graded evidence only; ALL decision logic (peak
    picking, candidate merging, labeling tolerance) lives HERE, once,
    identically for every modality — a prerequisite for attributing
    subset differences to the modalities themselves (Shapley).
  - The labeling tolerance MUST equal the evaluation tolerance
    (boundary-F1 window) later on: train and measure with one yardstick.

Pipeline per session:
  1. resample_to_grid   — bring audio evidence onto the video frame grid
  2. generate_candidates — union of per-modality peaks, merged
  3. window_features     — mean/max/std/slope per channel around each t_k
  4. label_candidates    — 0/1 against annotated ground-truth boundaries
  5. build_session_table — one pandas row per candidate

build_dataset() concatenates all sessions and keeps session_id / app_name
as metadata columns for grouped cross-validation (leave-one-app-out).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import json
import numpy as np
import pandas as pd

# ── Configuration (calibrate ONCE on the pilot set, then freeze) ──────────────

# Window half-size around each candidate for the aggregate statistics.
WINDOW_HALF_S = 1.0

# Minimum distance between peaks WITHIN one modality's evidence signal.
PEAK_MIN_DISTANCE_S = 1.0

# Minimum evidence height for a peak to become a candidate proposal.
# Deliberately LOW: recall matters at this stage — the Random Forest
# rejects weak candidates later; a candidate never proposed can never
# be learned.
PEAK_MIN_HEIGHT = 0.10

# Candidates from different modalities closer than this are merged into
# one (the merged time is the evidence-weighted mean of the group).
CANDIDATE_MERGE_GAP_S = 1.0

# Labeling tolerance: a candidate is positive iff an annotated boundary
# lies within ±LABEL_TOLERANCE_S. MUST match the boundary-F1 evaluation
# tolerance used in the ablation study.
LABEL_TOLERANCE_S = 1.0

# Per-modality channel names, in the column order of the evidence arrays.
# (onset flags are counts, scores are the graded evidence — the display
# flag column 2 is intentionally NOT a feature: it is derived from a
# threshold and would leak extractor-side decision logic into the model.)
MODALITY_CHANNELS: Dict[str, List[str]] = {
    "video":  ["onset", "score"],
    "audio":  ["sentence_start", "score"],
    "events": ["onset", "score"],
}

WINDOW_STATS = ("mean", "max", "std", "slope")


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SessionEvidence:
    """All evidence for one recording session on a COMMON time grid."""
    session_id: str
    app_name: str
    frame_times_s: np.ndarray                    # (T,) video frame times
    evidence: Dict[str, np.ndarray]              # modality -> (T, 2) [flag, score]
    ground_truth_s: List[float] = field(default_factory=list)


# ── Step 1: resampling onto the common (video) grid ──────────────────────────

def resample_to_grid(
    src_values: np.ndarray,
    src_times_s: np.ndarray,
    dst_times_s: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate a (T_src, D) evidence array onto dst_times_s.

    Audio evidence lives on the librosa hop grid (~43 Hz), video/events on
    the video grid (e.g. 10 Hz). The candidate table needs ONE grid;
    the video grid is the natural choice (coarsest, and candidate times
    are reported in video time anyway).

    Flag columns (0/1 spikes) survive interpolation as fractional bumps —
    we re-binarize them afterwards so a spike between two dst frames is
    assigned to the nearest one instead of being halved.
    """
    src_values = np.atleast_2d(src_values.T).T  # ensure (T, D)
    T_dst, D = len(dst_times_s), src_values.shape[1]
    out = np.zeros((T_dst, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(dst_times_s, src_times_s, src_values[:, d])
    return out


def rebinarize_flags(resampled: np.ndarray, flag_columns: Sequence[int]) -> np.ndarray:
    """Snap interpolated flag columns back to 0/1 (any value > 0.5 → 1)."""
    out = resampled.copy()
    for c in flag_columns:
        out[:, c] = (out[:, c] > 0.5).astype(np.float32)
    return out


# ── Step 2: candidate generation ──────────────────────────────────────────────

def generate_candidates(
    session: SessionEvidence,
    min_distance_s: float = PEAK_MIN_DISTANCE_S,
    min_height: float = PEAK_MIN_HEIGHT,
    merge_gap_s: float = CANDIDATE_MERGE_GAP_S,
) -> List[float]:
    """
    Union of per-modality evidence peaks, merged across modalities.

    Every modality proposes candidates from its OWN evidence score channel;
    proposals closer than merge_gap_s are fused into one candidate at the
    evidence-weighted mean time. No accept/reject decision is made here.

    :return: Sorted candidate times in seconds.
    """
    from scipy.signal import find_peaks

    t = session.frame_times_s
    if len(t) < 2:
        return []
    fps = 1.0 / float(np.median(np.diff(t)))
    distance = max(1, int(min_distance_s * fps))

    proposals: List[tuple[float, float]] = []  # (time_s, evidence_height)
    for modality, arr in session.evidence.items():
        score = arr[:, 1]
        peaks, props = find_peaks(score, height=min_height, distance=distance)
        for p, h in zip(peaks, props["peak_heights"]):
            proposals.append((float(t[p]), float(h)))

    if not proposals:
        return []
    proposals.sort(key=lambda x: x[0])

    # Merge nearby cross-modality proposals (evidence-weighted mean time).
    merged: List[float] = []
    group: List[tuple[float, float]] = [proposals[0]]
    for time_s, height in proposals[1:]:
        if time_s - group[-1][0] <= merge_gap_s:
            group.append((time_s, height))
        else:
            merged.append(_weighted_mean_time(group))
            group = [(time_s, height)]
    merged.append(_weighted_mean_time(group))
    return merged


def _weighted_mean_time(group: List[tuple[float, float]]) -> float:
    times = np.array([g[0] for g in group])
    weights = np.array([max(g[1], 1e-6) for g in group])
    return float(np.average(times, weights=weights))


# ── Step 3: window features ───────────────────────────────────────────────────

def _window_stats(values: np.ndarray, times_s: np.ndarray,
                  t_k: float, half_s: float) -> Dict[str, float]:
    """mean / max / std / slope of one channel in [t_k - half_s, t_k + half_s]."""
    mask = (times_s >= t_k - half_s) & (times_s <= t_k + half_s)
    win = values[mask]
    win_t = times_s[mask]
    if len(win) == 0:
        return {s: 0.0 for s in WINDOW_STATS}
    slope = 0.0
    if len(win) >= 2 and float(np.ptp(win_t)) > 0:
        slope = float(np.polyfit(win_t - t_k, win, 1)[0])
    return {
        "mean":  float(np.mean(win)),
        "max":   float(np.max(win)),
        "std":   float(np.std(win)),
        "slope": slope,
    }


def window_features(
    session: SessionEvidence,
    t_k: float,
    half_s: float = WINDOW_HALF_S,
) -> Dict[str, float]:
    """
    Feature vector for one candidate: for every modality channel the four
    window statistics, with `modality__channel_stat` column names — the
    prefixes are what the ablation masking keys on later
    (e.g. video__score_max, audio__score_mean, events__onset_mean).
    """
    row: Dict[str, float] = {}
    for modality, arr in session.evidence.items():
        channels = MODALITY_CHANNELS[modality]
        for c_idx, c_name in enumerate(channels):
            stats = _window_stats(arr[:, c_idx], session.frame_times_s, t_k, half_s)
            for stat_name, value in stats.items():
                row[f"{modality}__{c_name}_{stat_name}"] = value
    return row


# ── Step 4: ground-truth labeling ─────────────────────────────────────────────

def load_ground_truth(annotation_path: Path) -> List[float]:
    """
    Load annotated boundary times (seconds) from an AnnotationWindow
    export. Accepts either a plain list of t_ms values or a list of
    dicts with a "t_ms" key.
    """
    with Path(annotation_path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    times_ms: List[float] = []
    for item in data:
        if isinstance(item, dict):
            times_ms.append(float(item.get("t_ms", 0.0)))
        else:
            times_ms.append(float(item))
    return sorted(t / 1000.0 for t in times_ms)


def label_candidate(
    t_k: float,
    ground_truth_s: Sequence[float],
    tolerance_s: float = LABEL_TOLERANCE_S,
) -> int:
    """1 iff an annotated boundary lies within ±tolerance_s of t_k."""
    return int(any(abs(t_k - gt) <= tolerance_s for gt in ground_truth_s))


def match_report(
    candidates_s: Sequence[float],
    ground_truth_s: Sequence[float],
    tolerance_s: float = LABEL_TOLERANCE_S,
) -> Dict[str, int]:
    """
    Sanity metrics for the candidate stage itself:
      - missed_gt: annotated boundaries with NO candidate in tolerance.
        These are unlearnable — if this is > 0, the candidate generation
        (peak height / merge gap) needs loosening, NOT the classifier.
    """
    missed = sum(
        1 for gt in ground_truth_s
        if not any(abs(gt - c) <= tolerance_s for c in candidates_s)
    )
    positives = sum(label_candidate(c, ground_truth_s, tolerance_s) for c in candidates_s)
    return {
        "n_candidates": len(candidates_s),
        "n_ground_truth": len(ground_truth_s),
        "n_positive_candidates": positives,
        "missed_gt": missed,
    }


# ── Step 5: table assembly ────────────────────────────────────────────────────

def build_session_table(session: SessionEvidence) -> pd.DataFrame:
    """One labeled row per candidate for one session."""
    candidates = generate_candidates(session)
    rows: List[Dict] = []
    for t_k in candidates:
        row: Dict = {
            "session_id": session.session_id,
            "app_name":   session.app_name,
            "t_k":        round(t_k, 3),
        }
        row.update(window_features(session, t_k))
        row["label"] = label_candidate(t_k, session.ground_truth_s)
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset(sessions: List[SessionEvidence]) -> pd.DataFrame:
    """
    Concatenate all session tables into the final training table.
    session_id and app_name are METADATA (grouped CV keys), never features.
    """
    tables = [build_session_table(s) for s in sessions]
    tables = [t for t in tables if not t.empty]
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, ignore_index=True)
    return df


def feature_columns(df: pd.DataFrame) -> List[str]:
    """All model input columns (excludes metadata and label)."""
    return [c for c in df.columns if c not in ("session_id", "app_name", "t_k", "label")]


# ── Convenience: full per-session pipeline ────────────────────────────────────

def build_session_evidence(
    recording_session,
    session_id: str,
    app_name: str,
    fps: float,
    annotation_path: Path,
    audio_sampling_rate: float = 22050.0,
) -> SessionEvidence:
    """
    Run all three extractors for one RecordingSession and assemble a
    SessionEvidence on the video frame grid.
    """
    from docupilot.segmentation.feature_extraction import (
        _HOP_LENGTH,
        EventFeatureExtractor,
        GUIActionBoundaryExtractor,
        SemanticAudioFeatureExtractor,
        TranscriptionExtractor,
    )

    # Video evidence (already on the video grid).
    gui = GUIActionBoundaryExtractor.extract_gui_features(recording_session, fps)
    T_v = gui.shape[0]
    frame_times_s = np.arange(T_v) / fps
    frame_times_ms = frame_times_s * 1000.0

    # Audio evidence (audio hop grid) → resample onto the video grid.
    full_text, words = TranscriptionExtractor.extract_transcript(recording_session)
    audio_fps = audio_sampling_rate / _HOP_LENGTH
    duration_s = T_v / fps
    n_audio_frames = int(round(duration_s * audio_fps))
    sem = SemanticAudioFeatureExtractor.extract_semantic_features(
        full_text, words, n_audio_frames, audio_sampling_rate
    )
    audio_times_s = np.arange(n_audio_frames) / audio_fps
    sem_on_grid = resample_to_grid(sem[:, :2], audio_times_s, frame_times_s)
    sem_on_grid = rebinarize_flags(sem_on_grid, flag_columns=[0])

    # Event evidence (computed directly on the video grid).
    ev = EventFeatureExtractor.extract_event_boundary_evidence(
        recording_session, frame_times_ms
    )

    return SessionEvidence(
        session_id=session_id,
        app_name=app_name,
        frame_times_s=frame_times_s,
        evidence={
            "video":  gui[:, :2].astype(np.float32),
            "audio":  sem_on_grid.astype(np.float32),
            "events": ev[:, :2].astype(np.float32),
        },
        ground_truth_s=load_ground_truth(annotation_path),
    )
