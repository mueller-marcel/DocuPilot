import pytest

from docupilot.evaluation import experiment
from docupilot.evaluation import report as rep
from synthetic import PLAYERS, corpus


@pytest.fixture(scope="module")
def full_report():
    data = corpus()
    rows = experiment.run(data, players=PLAYERS, pool="union")
    rows_own = experiment.run(data, players=PLAYERS, pool="own")
    shifted = experiment.with_ground_truth(data, {d.name: [g + 0.9 for g in d.gt_s] for d in data})
    rows_start = experiment.run(shifted, players=PLAYERS, pool="union", extra={"definition": "start"})
    r = rep.analyse(data, rows, rows_own, rows_start, players=PLAYERS)
    r.sync = {"stream_offset_ms": 0.0, "click_median_ms": 138.0, "click_p95_ms": 1418.0,
              "click_absmax_ms": 1492.0, "n": 25.0}
    return r


@pytest.fixture(scope="module")
def minimal_report():
    data = corpus()
    return rep.analyse(data, experiment.run(data, players=PLAYERS), players=PLAYERS)


def test_primary_fields(full_report):
    r = full_report
    assert r.n_sessions == 5 and r.tau_s == rep.PRIMARY_TAU_S
    assert set(r.shapley) == set(PLAYERS)
    assert abs(sum(ci.point for ci in r.shapley.values()) - r.total) < 1e-9
    assert r.efficiency_error < 1e-9
    assert r.f1_all.point == r.subset_ci[frozenset(PLAYERS)].point
    assert 0.0 <= r.chance <= 1.0
    assert set(r.thresholds) == {s for s in r.subset_ci if s}
    assert sorted(r.shapley_by_tau) == list(experiment.TAUS_S)
    assert set(r.shapley_recall) == set(PLAYERS)


def test_bonferroni_intervals_are_wider(full_report):
    r = full_report
    for m in PLAYERS:
        raw, adj = r.shapley[m], r.shapley_bonferroni[m]
        assert adj.lo <= raw.lo + 1e-12 and adj.hi >= raw.hi - 1e-12
    for k in r.gain_ci:
        assert r.gain_ci_bonferroni[k].hi >= r.gain_ci[k].hi - 1e-12


def test_isolated_and_timing_credit(full_report):
    r = full_report
    assert r.isolated is not None and r.isolated.label == "isoliert"
    assert set(r.timing_credit) == set(PLAYERS)
    for m in PLAYERS:
        expected = r.subset_ci[frozenset({m})].point - r.isolated.singles[m].point
        assert r.timing_credit[m] == pytest.approx(expected)


def test_coupling_and_ceiling(full_report):
    r = full_report
    assert set(r.coupling) == set(PLAYERS)
    assert r.ceiling is not None
    assert r.ceiling.coverage >= max(c.coverage for c in r.coupling.values())
    assert r.decision_loss == pytest.approx(r.ceiling.coverage - r.recall_all.point)


def test_start_definition(full_report, minimal_report):
    assert full_report.start_definition is not None
    assert full_report.start_definition.label == "Beginn"
    assert minimal_report.start_definition is None
    assert minimal_report.isolated is None
    assert "nicht annotiert" in rep.definition_text(minimal_report)
    assert "kein isolierter Lauf" in rep.pool_text(minimal_report)


def test_texts_mention_every_modality(full_report):
    for text in (
        rep.adequacy_text(full_report), rep.design_text(full_report), rep.pool_text(full_report),
        rep.robustness_text(full_report), rep.multiplicity_text(full_report),
        rep.coupling_text(full_report), rep.definition_text(full_report),
    ):
        assert all(m in text for m in PLAYERS), text
    assert "Entscheidungsverlust" in rep.ceiling_text(full_report)
    assert "Zufallsniveau" in rep.adequacy_text(full_report)
    assert rep.saturation_text(full_report).count("Modalität:") == len(PLAYERS)
    assert rep.interaction_text(full_report).count("↔") == 3
    assert "Uhrenversatz" in rep.sync_text(full_report.sync)
    assert rep.sync_text({}) == "   (keine Synchronisationsmessung)"


def test_sections_and_chart_rows(full_report):
    blocks = rep.sections(full_report)
    assert [b.chart for b in blocks if b.chart] == ["shapley", "subsets", "saturation"]
    assert all(b.text or b.chart for b in blocks)
    assert len({b.title for b in blocks}) == len(blocks)
    rows = rep.shapley_rows(full_report)
    assert [r[0] for r in rows] == sorted(PLAYERS, key=lambda m: -full_report.shapley[m].point)
    assert rep.subset_rows(full_report)[0][1] == max(ci.point for ci in full_report.subset_ci.values())
    assert rep.status_line(full_report).startswith("Fertig · 5 Sessions")
    assert rep.header_lines(full_report, None, "now")[0] == "Korpus: —"
