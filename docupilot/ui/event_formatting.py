from __future__ import annotations


MARKER_TYPES = {"mouse_click", "key_press", "key_release", "mouse_scroll"}

PROMINENT_TYPES = {
    "av_started",
    "av_stopped",
    "input_started",
    "input_stopped",
    "recording_started",
    "recording_stopping",
    "recording_stopped",
    "mouse_click",
    "key_press",
    "key_release",
    "mouse_scroll",
}

DOT_COLOR = {
    "av_started": "#534AB7",
    "av_stopped": "#534AB7",
    "input_started": "#1D9E75",
    "input_stopped": "#1D9E75",
    "recording_started": "#e24b4a",
    "recording_stopping": "#e24b4a",
    "recording_stopped": "#e24b4a",
    "mouse_click": "#D85A30",
    "mouse_scroll": "#D85A30",
    "mouse_move": "#aaaaaa",
    "key_press": "#BA7517",
    "key_release": "#BA7517",
}

DEFAULT_DOT_COLOR = "#aaaaaa"


def format_ms(ms: float) -> str:
    """
    Format milliseconds as MM:SS.mmm.

    :param ms: Time in milliseconds.
    :return: Formatted time string.
    """
    total_s = int(ms) // 1000

    return f"{total_s // 60:02d}:{total_s % 60:02d}.{int(ms) % 1000:03d}"