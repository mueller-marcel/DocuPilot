"""
Measure what the segmentation is worth: score each modality combination against
the annotated boundaries, then attribute the result to the modalities.

    metrics.py   match predictions to ground truth, precision/recall/F1
    dataset.py   read ground truth and duration off a recording
"""

from docupilot.evaluation.dataset import duration_s, ground_truth_s
from docupilot.evaluation.metrics import Match, chance_level, macro_f1, match, pooled, sweep

__all__ = [
    "Match",
    "chance_level",
    "duration_s",
    "ground_truth_s",
    "macro_f1",
    "match",
    "pooled",
    "sweep",
]
