"""
Where a finished Report meets the three chart widgets.

The window shows the charts live and the PDF renders them off-screen; both
must display the same figures, so the mapping from report to widget exists
exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

from docupilot.evaluation import report as rep
from docupilot.segmentation import MODALITIES
from docupilot.ui.widgets.SaturationChartWidget import SaturationChartWidget
from docupilot.ui.widgets.ShapleyChartWidget import ShapleyChartWidget
from docupilot.ui.widgets.SubsetChartWidget import SubsetChartWidget


@dataclass(frozen=True)
class ReportCharts:
    """The three figures of a report, as widgets."""

    shapley: ShapleyChartWidget
    subsets: SubsetChartWidget
    saturation: SaturationChartWidget

    def by_name(self, name: str):
        """The widget a `report.Section.chart` name refers to."""
        return getattr(self, name)


def apply_report(charts: ReportCharts, report: rep.Report) -> None:
    """Push a report's numbers into the given chart widgets."""
    charts.shapley.set_values(rep.shapley_rows(report), total=report.total)
    charts.subsets.set_values(
        modalities=list(MODALITIES),
        colors=rep.COLORS,
        rows=rep.subset_rows(report),
        chance=report.chance,
    )
    charts.saturation.set_values(
        report.curve, report.gains, threshold=report.relevance_threshold
    )


def build_charts(report: rep.Report) -> ReportCharts:
    """Fresh, parentless chart widgets showing the report — what the PDF renders."""
    charts = ReportCharts(ShapleyChartWidget(), SubsetChartWidget(), SaturationChartWidget())
    apply_report(charts, report)
    return charts
