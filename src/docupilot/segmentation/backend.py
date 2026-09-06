"""
Shared plumbing for the two model-judged stages: .env loading, the Anthropic
client, and the on-disk verdict cache they both keep beside the recording.

Utilities only — no modality DATA passes through here, so the extractors'
independence (see the package docstring) is untouched: audio_scoring and
video_scoring share a transport and a cache format, never evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _load_dotenv() -> None:
    """
    Read KEY=VALUE lines from the project's .env into the environment. Never
    overwrites an already-set variable: the real environment wins.

    The root is found by looking upwards for pyproject.toml rather than by a
    fixed number of levels — a fixed one quietly stopped matching when the
    package moved under src/, and a missing key looks exactly like a missing
    .env.
    """
    root = next(
        (d for d in Path(__file__).resolve().parents if (d / "pyproject.toml").exists()),
        None,
    )
    if root is None:
        return
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        # utf-8-sig: PowerShell writes a BOM that would stick to the first key.
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


# At import time, before any module reads its DOCUPILOT_* variable.
_load_dotenv()

_CLIENT = None


def client():
    """The one Anthropic client, created on first use."""
    global _CLIENT
    if _CLIENT is None:
        import anthropic
        _CLIENT = anthropic.Anthropic()
    return _CLIENT


def is_available() -> bool:
    """True iff the model backend can actually be called right now."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    # Constructing the client is not enough — it succeeds without credentials
    # and only fails on the first request. Check that one actually resolved.
    try:
        c = client()
    except Exception:
        return False
    return bool(getattr(c, "api_key", None) or getattr(c, "auth_token", None))


class JsonVerdictCache:
    """
    A JSON map of cache key -> stored verdict, kept beside the recording.

    A file that cannot be read, or an entry that does not fit the current
    verdict shape, is a MISS and never a crash: the file is on disk and may
    predate a change to the verdict type, and a miss only costs one call.
    Subclasses decide the key and the verdict shape; this class only owns the
    file.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._entries: dict[str, Any] = {}
        if path.exists():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

    def _raw(self, key: str) -> Any:
        """The stored entry, whatever shape it has; None when absent."""
        return self._entries.get(key)

    def _store(self, key: str, value: Any) -> None:
        self._entries[key] = value

    def flush(self) -> None:
        """Write everything to disk; a failure is ignored — the verdicts are
        still in memory and only a later re-run would pay for them again."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError:
            pass
