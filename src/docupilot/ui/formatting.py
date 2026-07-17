"""Shared display formatting for the UI layer."""

from __future__ import annotations


def format_ms(ms: float) -> str:
    """Milliseconds as mm:ss.mmm — the one timestamp format the UI shows."""
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}.{int(ms) % 1000:03d}"
