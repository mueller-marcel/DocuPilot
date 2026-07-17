"""
Segmentation: find the action boundaries in one recording.

A boundary is the completion of a user-triggered operation — the moment its
result settles into a state that persists (docs/annotationsleitfaden.md).

Layout, one module per modality:

    evidence.py       what every modality returns, and how a curve gets drawn
    video.py          boundaries from the screen recording alone
    audio.py          boundaries from the audio track alone
    events.py         boundaries from events.json alone
    video_scoring.py  the VLM that judges a pair of settled screens
    audio_scoring.py  the LLM that judges a narrated sentence
    pipeline.py       run all three over one session

The modality modules never import each other. That is not tidiness: the thesis
decomposes segmentation quality into per-modality contributions with Shapley
values, and a single cross-read would make those numbers measure the leak instead
of the modality.
"""

from docupilot.segmentation.evidence import BoundaryEvidence
from docupilot.segmentation.pipeline import MODALITIES, segment

__all__ = ["MODALITIES", "BoundaryEvidence", "segment"]
