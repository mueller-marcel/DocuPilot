"""
The UI is driven headless: the point is that the wiring holds together and that
the window and the PDF show the same report, not that pixels look right.
"""

import pytest

from conftest import write_session
from docupilot.evaluation import experiment
from docupilot.evaluation import report as rep
from synthetic import PLAYERS, corpus


@pytest.fixture(scope="module")
def report():
    data = corpus()
    return rep.analyse(
        data,
        experiment.run(data, players=PLAYERS),
        experiment.run(data, players=PLAYERS, pool="own"),
        players=PLAYERS,
    )


def test_format_ms_is_the_one_timestamp_format():
    from docupilot.ui.formatting import format_ms

    assert format_ms(0) == "00:00.000"
    assert format_ms(999.9) == "00:00.999"
    assert format_ms(1000) == "00:01.000"
    assert format_ms(61234.5) == "01:01.234"


class TestExperimentWindow:
    def test_a_corpus_of_one_cannot_be_evaluated(self, app, tmp_path):
        from docupilot.ui.ExperimentWindow import ExperimentWindow

        write_session(tmp_path / "session_a", ground_truth=[{"t_ms": 1000.0}])
        window = ExperimentWindow()
        try:
            assert window._load_corpus(tmp_path) == 1
            # Leave-one-session-out has no fold to train on with one session.
            assert not window._run_button.isEnabled()
            assert "mindestens zwei Sessions" in window._corpus_label.text()
        finally:
            window.close()

    def test_the_table_lists_every_session_and_both_definitions(self, app, tmp_path):
        from docupilot.ui.ExperimentWindow import ExperimentWindow

        write_session(tmp_path / "session_a", ground_truth=[
            {"t_ms": 1000.0}, {"t_ms": 2000.0, "kind": "start"},
        ])
        write_session(tmp_path / "session_b")
        window = ExperimentWindow()
        try:
            assert window._load_corpus(tmp_path) == 1
            assert window._table.rowCount() == 2
            assert window._table.item(0, 1).text() == "1"
            assert window._table.item(0, 2).text() == "1"
            assert window._table.item(1, 1).text() == "— keine GT —"
        finally:
            window.close()

    def test_a_finished_run_fills_the_sections_and_enables_export(self, app, report):
        from docupilot.ui.ExperimentWindow import ExperimentWindow

        window = ExperimentWindow()
        try:
            window._on_done(report)
            assert window._status.text().startswith("Fertig · 5 Sessions")
            assert window._pdf_button.isEnabled() and window._export_button.isEnabled()
            # One widget per section, plus the charts and the trailing stretch.
            assert window._sections.count() > len(rep.sections(report))
        finally:
            window.close()

    def test_rendering_twice_replaces_the_sections_rather_than_appending(self, app, report):
        from docupilot.ui.ExperimentWindow import ExperimentWindow

        window = ExperimentWindow()
        try:
            window._render(report)
            first = window._sections.count()
            window._render(report)
            assert window._sections.count() == first
        finally:
            window.close()


def test_the_pdf_is_paginated_and_carries_the_charts(app, report, tmp_path):
    from pypdf import PdfReader

    from docupilot.ui.ReportPdfWriter import write_report_pdf

    out = tmp_path / "report.pdf"
    write_report_pdf(out, report, tmp_path)
    pages = PdfReader(out).pages
    # The report has more sections than fit on one page, and the charts are
    # rendered into it rather than re-run.
    assert len(pages) >= 2
    assert out.stat().st_size > 20_000
    # Text extraction is not asserted here: under the offscreen Qt platform no
    # real font is loaded and glyphs carry no character map. On a desktop the
    # same call produces selectable text.


def test_charts_paint_a_report_and_a_blank_state(app, report):
    from docupilot.ui.report_view import ReportCharts, apply_report, build_charts
    from docupilot.ui.widgets.SaturationChartWidget import SaturationChartWidget
    from docupilot.ui.widgets.ShapleyChartWidget import ShapleyChartWidget
    from docupilot.ui.widgets.SubsetChartWidget import SubsetChartWidget

    empty = ReportCharts(ShapleyChartWidget(), SubsetChartWidget(), SaturationChartWidget())
    for widget in (empty.shapley, empty.subsets, empty.saturation):
        widget.resize(700, 300)
        assert not widget.grab().isNull()          # placeholder, no crash

    charts = build_charts(report)
    for name in ("shapley", "subsets", "saturation"):
        widget = charts.by_name(name)
        widget.resize(700, 300)
        assert not widget.grab().isNull()
    apply_report(empty, report)


class TestTimelineWidget:
    def test_the_curve_polygon_is_reused_until_the_geometry_changes(self, app):
        from docupilot.ui.widgets.FeatureTimelineWidget import FeatureTimelineWidget

        lane = FeatureTimelineWidget()
        lane.resize(600, 180)
        lane.set_duration(10_000.0)
        lane.set_curve([0.0, 1.0, 2.0, 3.0], [0.1, 0.9, 0.2, 0.5], "#fb923c")
        plot = (600 - lane._PAD_L - lane._PAD_R, 180 - lane._PAD_T - lane._PAD_B)

        line, area = lane._curve_polygons(*plot)
        assert line.size() == 4 and area.size() == 6      # closed by two baseline points
        # The cursor timer repaints several times a second; re-projecting the
        # curve every time is what made the dialog sluggish.
        assert lane._curve_polygons(*plot)[0] is line
        lane.set_curve([0.0, 1.0], [0.2, 0.4], "#fb923c")
        assert lane._curve_polygons(*plot)[0] is not line

    def test_a_lane_paints_with_and_without_data(self, app):
        from docupilot.ui.widgets.FeatureTimelineWidget import FeatureTimelineWidget

        lane = FeatureTimelineWidget()
        lane.resize(600, 180)
        assert not lane.grab().isNull()                    # "Keine Daten verfügbar"
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
    remaining = dialog.get_boundaries()
    assert [b["t_ms"] for b in remaining] == [1000.0, 3000.0]
