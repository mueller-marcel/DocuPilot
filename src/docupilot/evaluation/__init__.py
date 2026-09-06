"""
Measure what the segmentation is worth: score each modality combination against
the annotated boundaries, then attribute the result to the modalities.

    metrics.py         match predictions to ground truth, precision/recall/F1
    dataset.py         read ground truth and duration off a recording
    media.py           one ffprobe per recording, shared by every consumer
    corpus.py          what a corpus directory contains, before computing anything
    fusion.py          a subset of modalities becomes one prediction
    coupling.py        how much a modality's candidates already coincide with the GT
    experiment.py      every subset on every session, leave-one-session-out
    analysis.py        Shapley values, interaction index, saturation curve
    statistics.py      is a difference real, and what can this corpus resolve
    synchronization.py measure the modality clock alignment, don't assume it
    report.py          the finished evaluation and the sentences that state it
"""

from docupilot.evaluation.analysis import (
    efficiency_error,
    interaction,
    marginal_gain,
    saturation,
    shapley,
    subsets,
)
from docupilot.evaluation.corpus import CorpusScan, SessionInfo, scan
from docupilot.evaluation.coupling import (
    CouplingStats,
    coupling_table,
    modality_coupling,
    union_coupling,
)
from docupilot.evaluation.dataset import duration_s, ground_truth_s
from docupilot.evaluation.experiment import SessionData, load, run, subset_values
from docupilot.evaluation.metrics import Match, chance_level, match
from docupilot.evaluation.report import Report, analyse, measure_sync
from docupilot.evaluation.statistics import (
    Interval,
    delta_ci,
    interaction_ci,
    minimum_detectable_effect,
    paired_sd,
    required_sessions,
    saturation_step_ci,
    saturation_step_required_n,
    shapley_ci,
    subset_ci,
    subset_cis,
)
from docupilot.evaluation.synchronization import (
    click_offsets_s,
    report as sync_report,
    stream_offset_ms,
)

__all__ = [
    "CorpusScan",
    "CouplingStats",
    "union_coupling",
    "Interval",
    "Match",
    "Report",
    "SessionData",
    "SessionInfo",
    "analyse",
    "chance_level",
    "click_offsets_s",
    "coupling_table",
    "delta_ci",
    "duration_s",
    "efficiency_error",
    "ground_truth_s",
    "interaction",
    "interaction_ci",
    "load",
    "marginal_gain",
    "match",
    "measure_sync",
    "minimum_detectable_effect",
    "modality_coupling",
    "paired_sd",
    "required_sessions",
    "run",
    "saturation",
    "saturation_step_ci",
    "saturation_step_required_n",
    "scan",
    "shapley",
    "shapley_ci",
    "stream_offset_ms",
    "subset_ci",
    "subset_cis",
    "subset_values",
    "subsets",
    "sync_report",
]
