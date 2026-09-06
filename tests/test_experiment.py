import csv
import math

import pytest

from docupilot.evaluation import coupling, experiment
from synthetic import PLAYERS, corpus


@pytest.fixture(scope="module")
def data():
    return corpus()


@pytest.fixture(scope="module")
def rows_union(data):
    return experiment.run(data, players=PLAYERS, pool="union")


@pytest.fixture(scope="module")
def rows_own(data):
    return experiment.run(data, players=PLAYERS, pool="own", extra={"definition": "end"})


def test_table_shape(rows_union, rows_own, data):
    n = 8 * len(data) * len(experiment.TAUS_S)
    assert len(rows_union) == n and len(rows_own) == n
    assert {r["pool"] for r in rows_union} == {"union"}
    assert {r["pool"] for r in rows_own} == {"own"}
    assert all(r["definition"] == "end" for r in rows_own)
    assert experiment.taus_in(rows_union) == list(experiment.TAUS_S)
    labels = {r["subset"] for r in rows_union}
    assert experiment.EMPTY_SUBSET_LABEL in labels and "audio+events+video" in labels


def test_empty_coalition_predicts_nothing(rows_union):
    empty = [r for r in rows_union if r["subset"] == experiment.EMPTY_SUBSET_LABEL]
    assert all(r["n_predicted"] == 0 and r["f1"] == 0.0 and math.isnan(r["threshold"]) for r in empty)


def test_thresholds_are_calibrated_from_the_grid(rows_union):
    chosen = {r["threshold"] for r in rows_union if r["subset"] != experiment.EMPTY_SUBSET_LABEL}
    from docupilot.evaluation.fusion import THRESHOLD_GRID

    assert chosen and all(any(abs(t - g) < 1e-9 for g in THRESHOLD_GRID) for t in chosen)


def test_full_model_beats_chance_and_metrics_are_consistent(rows_union, data):
    for r in rows_union:
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        assert r["precision"] == (tp / (tp + fp) if tp + fp else 0.0)
        assert r["recall"] == (tp / (tp + fn) if tp + fn else 0.0)
    values = experiment.subset_values(rows_union, 1.0)
    assert values[frozenset()] == 0.0
    assert values[frozenset(PLAYERS)] > 0.5


def test_isolated_pool_uses_only_its_own_candidates(data, rows_own):
    # A single-modality coalition in the isolated design can never predict
    # more moments than that modality proposes.
    from docupilot.evaluation import fusion

    for session in data:
        for m in PLAYERS:
            n_candidates = fusion.candidate_times(session.evidence, [m]).size
            predicted = {
                r["n_predicted"] for r in rows_own
                if r["session"] == session.name and r["subset"] == m
            }
            assert predicted and max(predicted) <= n_candidates


def test_paired_and_labels(rows_union, data):
    paired = experiment.paired_metric(rows_union, 1.0, "recall")
    assert set(paired[frozenset(PLAYERS)]) == {d.name for d in data}
    for s in (frozenset(), frozenset({"b", "a"})):
        assert experiment.subset_from_label(experiment.subset_label(s)) == s


def test_with_ground_truth_replaces_only_the_reference(data):
    shifted = experiment.with_ground_truth(data, {d.name: [g + 1.0 for g in d.gt_s] for d in data})
    assert shifted[0].evidence is data[0].evidence
    assert shifted[0].gt_s[0] == pytest.approx(data[0].gt_s[0] + 1.0)


def test_run_needs_two_sessions(data):
    with pytest.raises(ValueError):
        experiment.run(data[:1], players=PLAYERS)


def test_coupling_stats(data):
    table = coupling.coupling_table(data, PLAYERS, 1.0)
    union = coupling.union_coupling(data, PLAYERS, 1.0)
    for m, c in table.items():
        assert 0.0 <= c.chance_coverage <= c.coverage <= 1.0 or c.lift < 0
        assert 0.0 <= c.fine_share <= c.coverage
        assert c.offset_p25_s <= c.offset_median_s <= c.offset_p75_s
        assert c.offset_iqr_s >= 0.0
        assert union.coverage >= c.coverage
    assert union.rate >= max(c.rate for c in table.values())


def test_write_csv_unions_columns(rows_union, rows_own, tmp_path):
    path = tmp_path / "out" / "rows.csv"
    experiment.write_csv([*rows_union, *rows_own], path)
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert "definition" in reader.fieldnames and "pool" in reader.fieldnames
        read = list(reader)
    assert len(read) == len(rows_union) + len(rows_own)
    experiment.write_csv([], tmp_path / "none.csv")
    assert not (tmp_path / "none.csv").exists()
