"""
One finished lane per session, persisted beside the recording.

Opening the feature dialog on a session that was already segmented must not pay
for Whisper and the pHash scan again: both are deterministic, so a lane is a
property of the recording, not of the run that produced it.

The verdict caches next to it (gui_vlm_cache.json, audio_llm_cache.json) keep
what the MODELS said; this keeps what an extractor made of it. That is a much
stronger claim, so the key is correspondingly strict — see fingerprint().
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import numpy as np

from docupilot.recording.session import RecordingSession
from docupilot.segmentation.evidence import BoundaryEvidence

# Bump when the stored layout changes; every existing file then misses.
_FORMAT = 2

_CHUNK = 1 << 20

# Which sources decide whether a stored lane is still valid. Listed per modality
# on purpose: an edit to video.py must not invalidate the audio lane, or an
# unrelated change would make every corpus run pay for Whisper again.
_SOURCES: dict[str, tuple[str, ...]] = {
    "events": ("events.py", "evidence.py"),
    "video":  ("video.py", "video_scoring.py", "evidence.py"),
    "audio":  ("audio.py", "audio_scoring.py", "evidence.py"),
}

# Content digests, keyed on (path, size, mtime): a corpus run asks for the same
# recording's digest several times per session (one lane per modality, plus the
# activity scan), and a 100 MB file should be read once per process, not four
# times. A changed mtime with unchanged bytes — the synced-folder case — only
# costs a re-read, never a wrong answer, because the VALUE is still the content.
_DIGESTS: dict[tuple[str, int, int], str] = {}


def path_for(session: RecordingSession, modality: str) -> Path:
    """Where one modality's lane is kept — in the session, next to its inputs."""
    return session.session_dir / f"{modality}_evidence.npz"


def file_digest(path: Path) -> str:
    """
    SHA-256 of a file's CONTENT, memoised per process; "<missing>" when it
    cannot be read, so a missing input is a stable value rather than an error.
    """
    try:
        st = os.stat(path)
        key = (str(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return "<missing>"
    hit = _DIGESTS.get(key)
    if hit is not None:
        return hit
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return "<missing>"
    _DIGESTS[key] = digest.hexdigest()
    return _DIGESTS[key]


def _sources(modality: str) -> list[Path]:
    """The source files whose content the lane depends on."""
    package = Path(__file__).resolve().parent
    names = _SOURCES.get(modality)
    if names is None:
        # A modality the store was not told about — the package promises that a
        # fourth one needs no caller changes. Hash the whole package rather than
        # guess: over-eager invalidation costs minutes, a stale lane costs a
        # wrong result.
        return sorted(package.glob("*.py"))
    return [package / name for name in names]


def _constants(path: Path) -> str:
    """
    A module's tunables as this run actually resolves them.

    Not covered by hashing the source: MODEL is read from the environment, so
    switching DOCUPILOT_CLOUD_MODEL changes the verdicts without changing a byte
    of code.
    """
    try:
        module = importlib.import_module(f"{__package__}.{path.stem}")
    except Exception:                    # noqa: BLE001 — a module that will not
        return ""                        # import is covered by its bytes alone
    simple = (str, int, float, bool, tuple, frozenset)
    return "\n".join(
        f"{name}={value!r}"
        for name, value in sorted(vars(module).items())
        if name.isupper() and isinstance(value, simple)
    )


def fingerprint(session: RecordingSession, modality: str) -> str:
    """
    Everything the lane depends on, in one digest: the recording and the event
    log byte for byte, the code that reads them, and the constants and model
    names that code resolves at runtime.

    The recording is hashed by CONTENT, not by size and mtime: the corpus lives
    in a synced folder, where mtimes change without the bytes changing — and a
    lane served for the wrong recording is worse than any scan it saves.

    Deliberately NOT the ground truth: annotating a boundary changes what a lane
    is compared against, never what it contains. Re-annotating a session would
    otherwise cost a full re-extraction.
    """
    digest = hashlib.sha256()
    digest.update(f"{_FORMAT}|{modality}\n".encode())
    for path in (session.recording_path, session.events_path):
        digest.update(file_digest(path).encode())
    for path in _sources(modality):
        digest.update(f"\n{path.name}\n".encode())
        digest.update(file_digest(path).encode())
        digest.update(_constants(path).encode())
    return digest.hexdigest()


def load(
    session: RecordingSession, modality: str, fingerprint_: str | None = None
) -> BoundaryEvidence | None:
    """
    The stored lane, or None when there is none for exactly these inputs.

    A file that does not match, cannot be read or predates a change to
    BoundaryEvidence is a miss, never a crash — the same rule the verdict caches
    follow, and the reason a stale file can only ever cost time.

    :param fingerprint_: the expected fingerprint, when the caller already has
        it; computed here otherwise.
    """
    path = path_for(session, modality)
    if not path.exists():
        return None
    expected = fingerprint_ if fingerprint_ is not None else fingerprint(session, modality)
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"]) != expected:
                return None
            return BoundaryEvidence(
                times_s=np.asarray(data["times_s"], dtype=np.float64),
                score=np.asarray(data["score"], dtype=np.float32),
                boundaries_s=[float(t) for t in data["boundaries_s"]],
            )
    except Exception:                    # noqa: BLE001 — see the docstring
        return None


def save(
    session: RecordingSession,
    modality: str,
    evidence: BoundaryEvidence,
    fingerprint_: str | None = None,
) -> None:
    """
    Store one FINISHED lane.

    Never call this for a cancelled extraction: a partial lane is indistinguishable
    from a complete one once it is read back, and would then be served as the
    modality's answer.

    :param fingerprint_: the fingerprint to stamp the lane with, when the caller
        already has it; computed here otherwise.
    """
    path = path_for(session, modality)
    stamp = fingerprint_ if fingerprint_ is not None else fingerprint(session, modality)
    write_npz(path, fingerprint=np.array(stamp),
              times_s=evidence.times_s,
              score=evidence.score,
              boundaries_s=np.asarray(evidence.boundaries_s, dtype=np.float64))


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """
    Write a compressed archive atomically: a run killed mid-write leaves the
    previous file intact rather than a truncated archive. Failures are
    swallowed — a cache that cannot be written only costs the next run time.
    """
    tmp = path.parent / f"{path.name}.tmp"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
