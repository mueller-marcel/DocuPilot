"""
The Qt shell: that the windows construct, wire up to the report API and paint.

These are DEVELOPER-MACHINE tests, not a CI gate. Everything the shell decides
is tested without Qt elsewhere — the corpus summary in test_media_corpus, the
report texts in test_report, the timestamp format in test_formatting — so what
is left here is widget mechanics, which only a real Qt can exercise and which
the running application verifies anyway.

The module skips itself where the Qt platform libraries are absent, which is
the normal state of a build agent. Run the suite with `-ra` to see that stated
in the summary rather than silently passing.
"""

import pytest

# Import, not just presence: PySide6 is always installed as a main dependency,
# but on a headless machine loading the platform plugin fails at import time.
pytest.importorskip("PySide6.QtWidgets", reason="Qt-Plattformbibliotheken fehlen")

from docupilot.evaluation import experiment  # noqa: E402
from docupilot.evaluation import report as rep  # noqa: E402
from conftest import write_session  # noqa: E402
from synthetic import PLAYERS, corpus  # noqa: E402


@pytest.fixture(scope="module")
def report():
    data = corpus()
    return rep.analyse(
        data,
        experiment.run(data, players=PLAYERS),
        experiment.run(data, players=PLAYERS, pool="own"),
        players=PLAYERS,
    )


def test_every_chart_a_section_names_can_be_built(report):
    """The seam between the report and the view: a section may only name a
    chart the view actually has."""
    from docupilot.ui.report_view import build_charts

    charts = build_charts(report)
    for block in rep.sections(report):
        if block.chart:
            assert charts.by_name(block.chart) is not None


class TestExperimentWindow:
    def test_a_corpus_is_listed_and_a_result_fills_the_sections(
        self, app, report, tmp_path
    ):
        from docupilot.ui.ExperimentWindow import ExperimentWindow

        write_session(tmp_path / "session_a", ground_truth=[
            {"t_ms": 1000.0}, {"t_ms": 2000.0, "kind": "start"},
        ])
        write_session(tmp_path / "session_b")

        window = ExperimentWindow()
        try:
            assert window._load_corpus(tmp_path) == 1
            assert window._table.rowCount() == 2
            assert not window._run_button.isEnabled()      # one session is not enough

            window._on_done(report)
            assert window._status.text().startswith("Fertig · 5 Sessions")
            assert window._pdf_button.isEnabled()

            # Rendering again must replace the sections, not append to them.
            count = window._sections.count()
            window._render(report)
            assert window._sections.count() == count
        finally:
            window.close()


def test_the_report_pdf_is_written_and_paginated(app, report, tmp_path):
    from pypdf import PdfReader

    from docupilot.ui.ReportPdfWriter import write_report_pdf

    out = tmp_path / "report.pdf"
    write_report_pdf(out, report, tmp_path)
    # More sections than fit on a page, and the charts are rendered into the
    # file rather than the evaluation being run a second time.
    assert len(PdfReader(out).pages) >= 2
    assert out.stat().st_size > 20_000


def test_charts_paint_with_and_without_a_result(app, report):
    from docupilot.ui.report_view import ReportCharts, build_charts
    from docupilot.ui.widgets.SaturationChartWidget import SaturationChartWidget
    from docupilot.ui.widgets.ShapleyChartWidget import ShapleyChartWidget
    from docupilot.ui.widgets.SubsetChartWidget import SubsetChartWidget

    blank = ReportCharts(ShapleyChartWidget(), SubsetChartWidget(), SaturationChartWidget())
    filled = build_charts(report)
    for charts in (blank, filled):
        for name in ("shapley", "subsets", "saturation"):
            widget = charts.by_name(name)
            widget.resize(700, 300)
            assert not widget.grab().isNull()


def test_a_timeline_lane_reuses_its_curve_polygon(app):
    from docupilot.ui.widgets.FeatureTimelineWidget import FeatureTimelineWidget

    lane = FeatureTimelineWidget()
    lane.resize(600, 180)
    lane.set_duration(10_000.0)
    lane.set_curve([0.0, 1.0, 2.0, 3.0], [0.1, 0.9, 0.2, 0.5], "#fb923c")
    plot = (600 - lane._PAD_L - lane._PAD_R, 180 - lane._PAD_T - lane._PAD_B)

    line, area = lane._curve_polygons(*plot)
    assert line.size() == 4 and area.size() == 6      # closed by two baseline points
    # The cursor timer repaints several times a second; re-projecting thousands
    # of samples each time is what made the dialog sluggish.
    assert lane._curve_polygons(*plot)[0] is line
    lane.set_curve([0.0, 1.0], [0.2, 0.4], "#fb923c")
    assert lane._curve_polygons(*plot)[0] is not line


def test_a_timeline_lane_paints_with_and_without_data(app):
    from docupilot.ui.widgets.FeatureTimelineWidget import FeatureTimelineWidget

    lane = FeatureTimelineWidget()
    lane.resize(600, 180)
    assert not lane.grab().isNull()                   # "Keine Daten verfügbar"
    lane.set_duration(10_000.0)
    lane.set_curve([0.0, 1.0, 2.0], [0.1, 0.9, 0.2], "#fb923c")
    lane.set_boundaries([1500.0])
    lane.set_detected_boundaries([2500.0])
    lane.set_events([(500.0, "mouse_click")])
    assert not lane.grab().isNull()


def test_the_boundary_dialog_shows_which_definition_each_entry_follows(app):
    from docupilot.ui.widgets.BoundaryDialog import BoundaryDialog

    dialog = BoundaryDialog([
        {"t_ms": 1000.0, "kind": "end", "created_at_utc": "2026-01-01T00:00:00"},
        {"t_ms": 2000.0, "kind": "start", "created_at_utc": "2026-01-01T00:00:00"},
        {"t_ms": 3000.0},
    ])
    texts = [dialog._list.item(i).text() for i in range(dialog._list.count())]
    assert "Ende" in texts[0] and "Beginn" in texts[1] and "Ende" in texts[2]

    dialog._list.item(1).setSelected(True)
    dialog._delete_selected()
    assert [b["t_ms"] for b in dialog.get_boundaries()] == [1000.0, 3000.0]
