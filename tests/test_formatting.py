"""The one timestamp format the UI shows. Pure string work, no Qt involved."""

from docupilot.ui.formatting import format_ms


def test_milliseconds_become_minutes_seconds_and_thousandths():
    assert format_ms(0) == "00:00.000"
    assert format_ms(1000) == "00:01.000"
    assert format_ms(61234.5) == "01:01.234"
    assert format_ms(3599999) == "59:59.999"


def test_the_fraction_is_truncated_not_rounded():
    # The annotator reads this next to a frame; rounding up would name a
    # timestamp the player never stood on.
    assert format_ms(999.9) == "00:00.999"
