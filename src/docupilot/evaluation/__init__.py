"""
Measure what the segmentation is worth: score each modality combination against
the annotated boundaries, then attribute the result to the modalities.

    metrics.py      match predictions to ground truth, precision/recall/F1
    dataset.py      read ground truth and duration off a recording
    fusion.py       a subset of modalities becomes one prediction
    experiment.py   every subset on every session, leave-one-session-out
    analysis.py       Shapley values, interaction index, saturation curve
    statistics.py     is a difference real, and what can this corpus resolve
    synchronization.py measure the modality clock alignment, don't assume it
"""

from docupilot.evaluation.analysis import (
    efficiency_error,
    interaction,
    marginal_gain,
    saturation,
    shapley,
    subsets,
)
from docupilot.evaluation.dataset import duration_s, ground_truth_s
from docupilot.evaluation.experiment import SessionData, load, run, subset_values
from docupilot.evaluation.metrics import Match, chance_level, macro_f1, match, pooled, sweep
from docupilot.evaluation.statistics import (
    Interval,
    delta_ci,
    minimum_detectable_effect,
    paired_sd,
    required_sessions,
    shapley_ci,
    subset_ci,
)
from docupilot.evaluation.synchronization import (
    click_offsets_s,
    report as sync_report,
    stream_offset_ms,
)

__all__ = [
    "Interval",
    "Match",
    "SessionData",
    "click_offsets_s",
    "sync_report",
    "stream_offset_ms",
    "delta_ci",
    "minimum_detectable_effect",
    "paired_sd",
    "required_sessions",
    "shapley_ci",
    "subset_ci",
    "chance_level",
    "duration_s",
    "efficiency_error",
    "ground_truth_s",
    "interaction",
    "load",
    "macro_f1",
    "marginal_gain",
    "match",
    "pooled",
    "run",
    "saturation",
    "shapley",
    "subset_values",
    "subsets",
    "sweep",
]
