"""
The synthetic corpus the experiment goldens were generated on. Must stay
byte-for-byte the recipe used by the generator: same seeds, same primitives,
same order of random draws.
"""

from __future__ import annotations

import numpy as np

from docupilot.evaluation.experiment import SessionData
from docupilot.segmentation import evidence as ev
from docupilot.segmentation.evidence import BoundaryEvidence

PLAYERS = ("events", "video", "audio")


def synth_session(name: str, seed: int, duration: float = 120.0) -> SessionData:
    r = np.random.default_rng(seed)
    gt = sorted(r.uniform(5, duration - 5, 8).tolist())
    times = ev.grid(duration)
    evid = {}
    for mod, hit_p, fp_n, jitter in (
        ("video", 0.85, 6, 0.1), ("audio", 0.6, 4, 1.2), ("events", 0.7, 8, 0.5)
    ):
        sc = np.zeros(len(times), dtype=np.float32)
        for gtt in gt:
            if r.uniform() < 0.9:
                c = int(round((gtt + r.normal(0, jitter)) * ev.GRID_HZ))
                ev.apply_gaussian(sc, c, float(np.clip(r.normal(hit_p, 0.15), 0, 1)), 50)
        for _ in range(fp_n):
            c = int(r.integers(0, len(times)))
            ev.apply_gaussian(sc, c, float(np.clip(r.normal(0.35, 0.2), 0, 1)), 50)
        b = [float(times[i]) for i in np.flatnonzero(sc >= 0.5)[:3]]
        evid[mod] = BoundaryEvidence(times, sc, b)
    return SessionData(name=name, gt_s=gt, duration_s=duration, evidence=evid)


def corpus() -> list[SessionData]:
    return [synth_session(f"s{i}", 100 + i) for i in range(5)]
