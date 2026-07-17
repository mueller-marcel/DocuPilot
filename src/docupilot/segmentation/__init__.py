"""
Find the action boundaries in one recording — one module per modality, and they
never import each other (see docs/segmentierung.rst).

    evidence.py       what every modality returns, and how a curve gets drawn
    video.py          boundaries from the screen recording alone
    audio.py          boundaries from the audio track alone
    events.py         boundaries from events.json alone
    video_scoring.py  the VLM that judges a pair of settled screens
    audio_scoring.py  the LLM that judges a narrated sentence
    pipeline.py       run all three over one session
"""

from docupilot.segmentation.evidence import BoundaryEvidence
from docupilot.segmentation.pipeline import MODALITIES, segment

__all__ = ["MODALITIES", "BoundaryEvidence", "segment"]
