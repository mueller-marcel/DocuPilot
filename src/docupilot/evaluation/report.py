"""
The finished evaluation of one corpus, and the sentences that state it.

Everything the window and the PDF show is computed here, without Qt, so the
same numbers can be produced by a script — the figures for the thesis must be
reproducible without opening a window.

The report answers the three sub-questions on the PRIMARY design (fixed
candidate pool) and surrounds that answer with the controls a reviewer will
ask for: the isolated design for "what does a modality achieve on its own",
the tolerance sweep and the recall-based attribution for "is the ranking an
artefact of τ or of the operating point", family-wise adjusted intervals for
"how many comparisons were made", the coupling statistics for "how much of the
video's lead is the annotation itself", and — once annotated — the second
boundary definition for "does the ranking survive the definition".
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from docupilot.evaluation import (
    analysis, coupling, experiment, metrics, statistics, synchronization,
)
from docupilot.evaluation.coupling import CouplingStats
from docupilot.evaluation.experiment import SessionData
from docupilot.recording.session import RecordingSession
from docupilot.segmentation import MODALITIES

# Tolerance the headline numbers are reported at. The sweep is in the CSV and
# in the robustness section.
PRIMARY_TAU_S = 1.0

# Below this a difference counts as practically irrelevant. Provisional: it is
# meant to be replaced by the measured intra-rater consistency, which grounds it
# in data instead of in a judgement call.
RELEVANCE_THRESHOLD = 0.05

DESIGN = (
    "Random Forest, Leave-one-session-out, Schwelle je Fold auf Out-of-Bag-"
    "Vorhersagen der Trainingssessions kalibriert · Merkmale je Modalität: "
    "Punktwert, Fenstermaximum ±0,5/1/2 s, Sessionrang · Kandidaten: fixierte "
    "Union (primär, Informationsbeitrag) und isoliert je Teilmenge (TF1)"
)

# Same colours the feature lanes use, so a modality looks the same everywhere.
COLORS = {"events": "#38bdf8", "audio": "#a78bfa", "video": "#fb923c"}

Intervals = dict[str, statistics.Interval]


@dataclass(frozen=True)
class Variant:
    """The headline numbers of one alternative design or reference."""

    label: str
    values: dict[frozenset[str], float]
    f1_all: statistics.Interval
    shapley: Intervals
    singles: Intervals
    """F1 with CI of every single-modality coalition."""


@dataclass
class Report:
    """Everything the charts and texts need, computed once off the UI thread."""

    rows: list[dict] = field(default_factory=list)
    rows_isolated: list[dict] = field(default_factory=list)
    rows_start: list[dict] = field(default_factory=list)

    # Primary design at the primary tolerance.
    subset_ci: dict[frozenset[str], statistics.Interval] = field(default_factory=dict)
    shapley: Intervals = field(default_factory=dict)
    total: float = 0.0
    curve: dict[int, float] = field(default_factory=dict)
    gains: dict[int, float] = field(default_factory=dict)
    gain_ci: dict[int, statistics.Interval] = field(default_factory=dict)
    step_required_n: dict[int, int] = field(default_factory=dict)
    interactions: dict[tuple[str, str], statistics.Interval] = field(default_factory=dict)
    curve_no_video: dict[int, float] = field(default_factory=dict)
    gains_no_video: dict[int, float] = field(default_factory=dict)
    shapley_no_video: dict[str, float] = field(default_factory=dict)
    chance: float = 0.0
    efficiency_error: float = 0.0
    f1_all: statistics.Interval | None = None
    recall_all: statistics.Interval | None = None
    precision_all: statistics.Interval | None = None
    mde: float = 0.0
    n_sessions: int = 0
    thresholds: dict[frozenset[str], tuple[float, float, float]] = field(default_factory=dict)
    """(mean, min, max) of the calibrated threshold per coalition over the folds."""

    # Controls.
    shapley_bonferroni: Intervals = field(default_factory=dict)
    interactions_bonferroni: dict[tuple[str, str], statistics.Interval] = field(default_factory=dict)
    gain_ci_bonferroni: dict[int, statistics.Interval] = field(default_factory=dict)
    shapley_by_tau: dict[float, Intervals] = field(default_factory=dict)
    f1_all_by_tau: dict[float, statistics.Interval] = field(default_factory=dict)
    shapley_recall: Intervals = field(default_factory=dict)
    isolated: Variant | None = None
    timing_credit: dict[str, float] = field(default_factory=dict)
    """v_union({m}) − v_isolated({m}): what a modality alone gains from the
    other modalities' candidate TIMING."""
    coupling: dict[str, CouplingStats] = field(default_factory=dict)
    ceiling: CouplingStats | None = None
    start_definition: Variant | None = None
    sync: dict[str, float] = field(default_factory=dict)

    tau_s: float = PRIMARY_TAU_S
    relevance_threshold: float = RELEVANCE_THRESHOLD

    @property
    def decision_loss(self) -> float:
        """Recall the deciders leave on the table: boundaries some modality
        proposed within τ that the full model still did not accept."""
        if self.ceiling is None or self.recall_all is None:
            return float("nan")
        return self.ceiling.coverage - self.recall_all.point


# ── Computation ──────────────────────────────────────────────────────────────

def _variant(
    label: str, rows: Sequence[dict], players: Sequence[str], tau_s: float
) -> Variant:
    paired = experiment.paired_metric(rows, tau_s, "f1")
    values = experiment.subset_values(rows, tau_s)
    cis = statistics.subset_cis(paired)
    return Variant(
        label=label,
        values=values,
        f1_all=cis[frozenset(players)],
        shapley=statistics.shapley_ci(paired, players),
        singles={m: cis[frozenset({m})] for m in players},
    )


def _threshold_summary(rows: Sequence[dict], tau_s: float) -> dict[frozenset[str], tuple[float, float, float]]:
    per_subset: dict[frozenset[str], list[float]] = {}
    for row in rows:
        if row["tau_s"] != tau_s or row.get("threshold") is None:
            continue
        t = float(row["threshold"])
        if t == t:                                    # not NaN (empty coalition)
            per_subset.setdefault(experiment.subset_from_label(row["subset"]), []).append(t)
    return {
        k: (float(np.mean(v)), float(np.min(v)), float(np.max(v)))
        for k, v in per_subset.items()
    }


def analyse(
    data: Sequence[SessionData],
    rows: Sequence[dict],
    rows_isolated: Sequence[dict] | None = None,
    rows_start: Sequence[dict] | None = None,
    players: Sequence[str] = MODALITIES,
    tau_s: float = PRIMARY_TAU_S,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> Report:
    """
    Turn the tidy result tables into the report the thesis reads.

    :param data: the corpus the rows were computed on (for chance level and
        coupling, which need the raw evidence and the durations).
    :param rows: the primary design (`experiment.run(pool="union")`).
    :param rows_isolated: the isolated design (`pool="own"`), for TF1 and the
        timing credit; omitted when not run.
    :param rows_start: the primary design scored against the "start" boundary
        definition; omitted when the corpus is not annotated that way.
    :param players: the modalities, in the order the charts list them.
    :param tau_s: tolerance the headline numbers are taken at.
    :param relevance_threshold: the SESOI the saturation verdict uses.
    """
    players = tuple(players)
    rows = list(rows)
    paired = experiment.paired_metric(rows, tau_s, "f1")
    values = experiment.subset_values(rows, tau_s)
    full, empty = frozenset(players), frozenset()

    phi = statistics.shapley_ci(paired, players)
    curve = analysis.saturation(values, players)
    # Self-test: exact Shapley values must sum to v(all) − v(empty).
    efficiency_error = analysis.efficiency_error(
        values, players, {m: phi[m].point for m in players}
    )

    # Saturation without video separates "immediate saturation because video
    # copies the label" from genuine saturation among the other modalities.
    no_video = tuple(m for m in players if m != "video")
    curve_no_video = analysis.saturation(values, no_video) if no_video else {}

    subset_ci = statistics.subset_cis(paired)
    recall_all = statistics.subset_ci(experiment.paired_metric(rows, tau_s, "recall")[full])
    precision_all = statistics.subset_ci(experiment.paired_metric(rows, tau_s, "precision")[full])
    mde = statistics.minimum_detectable_effect(
        statistics.paired_sd(paired[empty], paired[full]), len(data)
    )

    # One family = the comparisons one sentence of the thesis rests on: the
    # three Shapley values, the three pairwise indices, the three steps. A
    # Bonferroni-adjusted interval says whether a claim survives being one of
    # three — the disclosure the unadjusted intervals cannot give.
    k = len(players)
    alpha_family = 0.05 / max(k, 1)
    k_pairs = len(list(combinations(players, 2)))
    alpha_pairs = 0.05 / max(k_pairs, 1)

    # The ranking must not be an artefact of the tolerance it is read at.
    shapley_by_tau = {
        tau: statistics.shapley_ci(experiment.paired_metric(rows, tau, "f1"), players)
        for tau in experiment.taus_in(rows)
    }
    f1_all_by_tau = {
        tau: statistics.subset_ci(experiment.paired_metric(rows, tau, "f1")[full])
        for tau in experiment.taus_in(rows)
    }

    isolated = (
        _variant("isoliert", rows_isolated, players, tau_s) if rows_isolated else None
    )
    timing_credit = (
        {m: values[frozenset({m})] - isolated.values[frozenset({m})] for m in players}
        if isolated is not None else {}
    )

    return Report(
        rows=rows,
        rows_isolated=list(rows_isolated or []),
        rows_start=list(rows_start or []),
        subset_ci=subset_ci,
        shapley=phi,
        total=values[full],
        curve=curve,
        gains=analysis.marginal_gain(curve),
        gain_ci=statistics.saturation_step_ci(paired, players),
        step_required_n=statistics.saturation_step_required_n(
            paired, players, relevance_threshold
        ),
        interactions=statistics.interaction_ci(paired, players),
        curve_no_video=curve_no_video,
        gains_no_video=analysis.marginal_gain(curve_no_video) if curve_no_video else {},
        shapley_no_video=analysis.shapley(values, no_video) if len(no_video) >= 2 else {},
        chance=float(np.mean([
            metrics.chance_level(d.gt_s, d.duration_s, tau_s) for d in data
        ])),
        efficiency_error=efficiency_error,
        f1_all=subset_ci[full],
        recall_all=recall_all,
        precision_all=precision_all,
        mde=mde,
        n_sessions=len(data),
        thresholds=_threshold_summary(rows, tau_s),
        shapley_bonferroni=statistics.shapley_ci(paired, players, alpha=alpha_family),
        interactions_bonferroni=statistics.interaction_ci(paired, players, alpha=alpha_pairs),
        gain_ci_bonferroni=statistics.saturation_step_ci(paired, players, alpha=alpha_family),
        shapley_by_tau=shapley_by_tau,
        f1_all_by_tau=f1_all_by_tau,
        shapley_recall=statistics.shapley_ci(
            experiment.paired_metric(rows, tau_s, "recall"), players
        ),
        isolated=isolated,
        timing_credit=timing_credit,
        coupling=coupling.coupling_table(data, players, tau_s),
        ceiling=coupling.union_coupling(data, players, tau_s),
        start_definition=(
            _variant("Beginn", rows_start, players, tau_s) if rows_start else None
        ),
        tau_s=tau_s,
        relevance_threshold=relevance_threshold,
    )


def measure_sync(
    directories: Sequence[Path],
    on_progress: Callable[[str, int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, float]:
    """
    Aggregate the measured clock offsets over the corpus — the evidence that
    the chosen tolerance covers the real modality lag rather than assuming it.

    Needs each video's activity scan, so on a corpus without cached scans it
    decodes every recording; callers run it as the final, cancellable phase.

    :param on_progress: called as (session name, done, total).
    :param is_cancelled: polled before each session; True stops the loop and
        returns what was aggregated so far.
    """
    stream: list[float] = []
    click_median: list[float] = []
    click_p95: list[float] = []
    click_absmax: list[float] = []
    total = len(directories)
    for i, directory in enumerate(directories, start=1):
        if is_cancelled is not None and is_cancelled():
            break
        if on_progress is not None:
            on_progress(directory.name, i, total)
        try:
            one = synchronization.report(RecordingSession.from_directory(directory))
        except Exception:                          # noqa: BLE001 — sync is auxiliary
            continue
        offset = one.get("stream_offset_ms")
        if offset is not None and offset == offset:   # not NaN
            stream.append(offset)
        if "click_median_ms" in one:
            click_median.append(one["click_median_ms"])
            click_p95.append(one["click_p95_ms"])
            click_absmax.append(one["click_absmax_ms"])
    return {
        "stream_offset_ms": float(np.median(stream)) if stream else float("nan"),
        "click_median_ms": float(np.median(click_median)) if click_median else float("nan"),
        # Max over sessions, deliberately conservative: "P95 ≤ τ" then means
        # no session has more than 5 % of its reactions beyond this value.
        "click_p95_ms": float(np.max(click_p95)) if click_p95 else float("nan"),
        "click_absmax_ms": float(np.max(click_absmax)) if click_absmax else float("nan"),
        "n": float(len(click_median)),
    }


# ── Rows for the charts ──────────────────────────────────────────────────────

def shapley_rows(report: Report) -> list[tuple[str, float, float, float, str]]:
    """(modality, value, ci_low, ci_high, colour), strongest first."""
    return sorted(
        ((m, ci.point, ci.lo, ci.hi, COLORS.get(m, "#333"))
         for m, ci in report.shapley.items()),
        key=lambda row: -row[1],
    )


def subset_rows(report: Report) -> list[tuple[frozenset[str], float, float, float]]:
    """(subset, F1, ci_low, ci_high), best first."""
    return sorted(
        ((subset, ci.point, ci.lo, ci.hi)
         for subset, ci in report.subset_ci.items()),
        key=lambda row: -row[1],
    )


# ── Texts ────────────────────────────────────────────────────────────────────

def _ci(ci: statistics.Interval | None, signed: bool = False) -> str:
    if ci is None:
        return "–"
    fmt = "+.3f" if signed else ".3f"
    return f"{ci.point:{fmt}} [{ci.lo:{fmt}}, {ci.hi:{fmt}}]"


def _label(subset: frozenset[str]) -> str:
    return experiment.subset_label(subset)


def adequacy_text(report: Report) -> str:
    """Headline metrics of the full modality set, and whether the corpus is big
    enough for the result to be statistically meaningful."""
    significant = [m for m, ci in report.shapley.items() if ci.excludes_zero]
    not_significant = [m for m, ci in report.shapley.items() if not ci.excludes_zero]
    enough = (
        report.n_sessions >= 10
        and report.mde <= report.relevance_threshold
        and bool(significant)
    )
    return "\n".join([
        f"   Alle Modalitäten:   F1 = {_ci(report.f1_all)}   ·   "
        f"Recall = {_ci(report.recall_all)}   ·   Precision = {_ci(report.precision_all)}",
        f"   Zufallsniveau F1 = {report.chance:.3f}",
        "   Signifikante Beiträge (95 %-CI ohne 0): "
        + (", ".join(significant) if significant else "keine")
        + (f"   ·   nicht signifikant: {', '.join(not_significant)}"
           if not_significant else ""),
        f"   n = {report.n_sessions} Sessions   ·   kleinster auflösbarer "
        f"F1-Unterschied ≈ {report.mde:.3f} (80 % Power)",
        "   → Datensatz ausreichend groß; Ergebnis statistisch belastbar."
        if enough else
        "   → eingeschränkte Aussagekraft (kleiner Datensatz / breite Intervalle) — "
        "nur Beiträge mit CI ohne 0 belastbar interpretieren.",
    ])


def design_text(report: Report) -> str:
    """The operating points the calibration chose, per coalition — so the
    reader can see the threshold was neither 0.5 by fiat nor tuned on the
    scored session."""
    lines = [f"   {DESIGN}", "   Kalibrierte Schwelle je Teilmenge (Mittel [min–max] über die Folds):"]
    for subset, (mean, lo, hi) in sorted(report.thresholds.items(), key=lambda kv: (len(kv[0]), _label(kv[0]))):
        lines.append(f"      {_label(subset):22s} {mean:.2f}  [{lo:.2f}–{hi:.2f}]")
    return "\n".join(lines)


def pool_text(report: Report) -> str:
    """
    TF1 as the exposé words it — each modality on its own — next to the same
    modality inside the fixed pool, and the difference: the credit a modality
    receives purely for the other modalities' timing.
    """
    if report.isolated is None:
        return "   (kein isolierter Lauf)"
    iso = report.isolated
    lines = [
        "   Modalität     isoliert (eigene Kandidaten)   im fixierten Pool         Zeitpunkt-Kredit",
    ]
    for m in sorted(report.timing_credit, key=lambda m: -iso.singles[m].point):
        lines.append(
            f"   {m:10s}    {_ci(iso.singles[m]):28s}   {_ci(report.subset_ci[frozenset({m})]):26s}"
            f"{report.timing_credit[m]:+.3f}"
        )
    lines.append(
        "   Alle drei:    isoliert F1 " + _ci(iso.f1_all)
        + "   ·   fixiert F1 " + _ci(report.f1_all)
    )
    lines.append(
        "   Shapley isoliert:  "
        + ", ".join(f"{m} {_ci(ci, signed=True)}" for m, ci in iso.shapley.items())
    )
    strongest_iso = max(iso.singles, key=lambda m: iso.singles[m].point)
    strongest_fix = max(report.subset_ci, key=lambda s: report.subset_ci[s].point if len(s) == 1 else -1)
    lines.append(
        f"   → TF1: isoliert führt {strongest_iso}; im fixierten Pool {_label(strongest_fix)}. "
        "Ein positiver Kredit ist der Anteil, den eine Modalität nur durch die "
        "Zeitpunkte der anderen erhält — Information, die nicht ihre eigene ist."
    )
    return "\n".join(lines)


def robustness_text(report: Report) -> str:
    """Shapley values across the tolerance sweep and by recall: a ranking that
    holds at every τ and without the operating point is a property of the
    modalities, not of the scoring."""
    taus = sorted(report.shapley_by_tau)
    if not taus:
        return "   (kein Toleranz-Sweep)"
    players = list(report.shapley)
    lines = ["   τ [s]      " + "".join(f"{tau:>8.2f}" for tau in taus)]
    for m in players:
        cells = "".join(f"{report.shapley_by_tau[tau][m].point:>8.3f}" for tau in taus)
        lines.append(f"   {m:10s}{cells}")
    lines.append("   F1 alle   " + "".join(f"{report.f1_all_by_tau[tau].point:>8.3f}" for tau in taus))
    lines.append(
        "   Rang je τ: " + "  ·  ".join(
            f"{tau:.2f} s: " + " > ".join(
                sorted(players, key=lambda m: -report.shapley_by_tau[tau][m].point))
            for tau in taus
        )
    )
    lines.append(
        "   Shapley nach Recall (τ = %.2f s): " % report.tau_s
        + ", ".join(f"{m} {_ci(ci, signed=True)}" for m, ci in report.shapley_recall.items())
    )
    ranks = {tuple(sorted(players, key=lambda m: -report.shapley_by_tau[tau][m].point)) for tau in taus}
    lines.append(
        "   → Rangfolge über alle τ stabil." if len(ranks) == 1
        else f"   → Rangfolge wechselt mit τ ({len(ranks)} verschiedene Ordnungen) — "
             "die Antwort auf TF1/TF2 hängt von der Toleranz ab und ist so zu berichten."
    )
    return "\n".join(lines)


def multiplicity_text(report: Report) -> str:
    """Which claims survive a Bonferroni adjustment within their family."""
    def verdicts(raw: Mapping, adjusted: Mapping, name) -> str:
        parts = []
        for key, ci in raw.items():
            adj = adjusted.get(key)
            state = (
                "✓ auch adjustiert" if adj is not None and adj.excludes_zero
                else ("~ nur unadjustiert" if ci.excludes_zero else "✗")
            )
            parts.append(f"{name(key)} {state}")
        return ", ".join(parts)

    k = len(report.shapley)
    return "\n".join([
        f"   Je Familie Bonferroni-adjustiert (α = 0,05/{k}); ✓ = Richtung bleibt gesichert:",
        "   Shapley:      " + verdicts(report.shapley, report.shapley_bonferroni, str),
        "   Interaktion:  " + verdicts(report.interactions, report.interactions_bonferroni,
                                       lambda p: f"{p[0]}↔{p[1]}"),
        "   Sättigung:    " + verdicts(report.gain_ci, report.gain_ci_bonferroni,
                                       lambda s: f"+{s}."),
        "   Hinweis: alle übrigen Intervalle gelten einzeln zu 95 %, nicht simultan.",
    ])


def coupling_text(report: Report) -> str:
    """The coupling table, ordered by fine alignment: the share of boundaries a
    modality hits to within a quarter of the tolerance is the number that
    tells a shared signal from an independent judgement."""
    def s(value: float, fmt: str = "+.2f") -> str:
        return "–" if value != value else f"{value:{fmt}}"

    # Two aligned tables rather than one wide one: the PDF body is ~100
    # monospace columns wide, and a wrapped table row is unreadable.
    ordered = sorted(report.coupling.items(), key=lambda kv: -kv[1].fine_share)
    lines = ["   Modalität    Deckung  Zufall   Lift   Rate   fein (±τ/4)"]
    for m, c in ordered:
        lines.append(
            f"   {m:11s}  {c.coverage:6.2f}  {c.chance_coverage:6.2f}  {c.lift:+5.2f}  "
            f"{c.rate:5.2f}   {c.fine_share:6.0%}"
        )
    if report.ceiling is not None:
        c = report.ceiling
        lines.append(
            f"   Vereinigung  {c.coverage:6.2f}  {c.chance_coverage:6.2f}  {c.lift:+5.2f}  "
            f"{c.rate:5.2f}   {c.fine_share:6.0%}   (Recall-Obergrenze)"
        )
    lines.append("   Versatz Kandidat − Grenze [s]:   Median      IQR [p25, p75]")
    for m, c in ordered:
        lines.append(
            f"   {m:11s}                     {s(c.offset_median_s):>7s}    "
            f"[{s(c.offset_p25_s)}, {s(c.offset_p75_s)}]"
        )
    coupled = [m for m, c in ordered if c.fine_share >= 0.5]
    if coupled:
        lines.append(
            "   ⚠ " + ", ".join(coupled) + ": Feinausrichtung ≥ 50 % — Annotation und "
            "Modalität folgen demselben Signal; der Beitrag ist teils definitorisch."
        )
    lines.append(
        "   Lesart: Deckung = Anteil GT-Grenzen mit Kandidat in ±τ; Zufall = dieselbe "
        "Kandidatenzahl zufällig platziert; Lift = Deckung − Zufall. Ein Versatz-IQR von "
        "wenigen Zehntelsekunden bei hoher Feinausrichtung ist die Signatur eines geteilten "
        "Signals, nicht zweier unabhängiger Urteile."
    )
    return "\n".join(lines)


def ceiling_text(report: Report) -> str:
    """Where the recall is lost: before the decider (no candidate) or in it."""
    if report.ceiling is None or report.recall_all is None:
        return "   (keine Obergrenze berechnet)"
    return "\n".join([
        f"   Recall-Obergrenze (irgendeine Modalität schlägt die Grenze in ±τ vor): "
        f"{report.ceiling.coverage:.3f}",
        f"   Erreichter Recall (alle Modalitäten): {report.recall_all.point:.3f}   ·   "
        f"Entscheidungsverlust: {report.decision_loss:.3f}",
        f"   Vorschlagsverlust (keine Modalität schlägt vor): {1.0 - report.ceiling.coverage:.3f}",
        "   → Der Entscheidungsverlust ist der Anteil, den ein besserer Entscheider auf "
        "denselben Kandidaten noch holen könnte; der Vorschlagsverlust ist für keinen "
        "Entscheider erreichbar.",
    ])


def definition_text(report: Report) -> str:
    """The same evidence scored against the "start" definition — the sensitivity
    of the attribution to the annotation rule the exposé and the thesis differ on."""
    if report.start_definition is None:
        return (
            "   (nicht annotiert — Grenzen der Art \"Beginn\" fehlen in mindestens einer "
            "Session; im Annotationsfenster mit \"Beginn setzen\" nachtragen)"
        )
    alt = report.start_definition
    lines = [
        "   Definition      F1 alle                    Shapley video / audio / events",
        "   Ende (primär)   " + f"{_ci(report.f1_all):26s} "
        + " / ".join(f"{report.shapley[m].point:+.3f}" for m in report.shapley),
        "   Beginn          " + f"{_ci(alt.f1_all):26s} "
        + " / ".join(f"{alt.shapley[m].point:+.3f}" for m in report.shapley),
    ]
    order_end = sorted(report.shapley, key=lambda m: -report.shapley[m].point)
    order_start = sorted(alt.shapley, key=lambda m: -alt.shapley[m].point)
    lines.append(
        "   → Rangfolge unter beiden Definitionen gleich: " + " > ".join(order_end)
        if order_end == order_start else
        f"   → Rangfolge wechselt: Ende {' > '.join(order_end)} · Beginn {' > '.join(order_start)} — "
        "der Beitrag hängt an der Grenzdefinition und ist konditional zu berichten."
    )
    return "\n".join(lines)


def saturation_text(report: Report) -> str:
    """
    Marginal gains with CI, each step classified three-way, the saturation
    verdict, and the video-free variant that separates coupling-driven from
    genuine saturation.

    Three-way because "n.s." is not evidence of absence: a step counts as
    negligible only when its whole CI lies inside ±SESOI (equivalence, Lakens
    2018), it counts as a gain when the CI excludes 0, and anything else is
    undecided — reported with the corpus size the decision would need.
    Saturation is a persistent property, so it can only start after the LAST
    step whose gain is real; a criterion that stops at the first quiet step
    would be falsified by its own table when a later step is significant.
    """
    thr = report.relevance_threshold
    lines: list[str] = []
    kinds: dict[int, str] = {}
    for k in sorted(report.gain_ci):
        ci = report.gain_ci[k]
        gain = report.gains.get(k, 0.0)
        if ci.below(thr):
            kinds[k] = "negligible"
            note = f"nachweislich irrelevant — CI vollständig in ±{thr:.2f}"
        elif ci.excludes_zero:
            kinds[k] = "real"
            note = "Zugewinn, CI ohne 0"
        else:
            kinds[k] = "undecided"
            need = report.step_required_n.get(k, 0)
            note = (
                f"unentscheidbar — CI überdeckt 0 und ±{thr:.2f}"
                + (f"; Äquivalenznachweis bräuchte ≈ {need} Sessions"
                   if need else "")
            )
        lines.append(f"   +{k}. Modalität:  {gain:+.3f}   {ci}   ({note})")

    ks = sorted(kinds)
    real_ks = [k for k in ks if kinds[k] == "real"]
    last_real = max(real_ks) if real_ks else 0
    tail = [k for k in ks if k > last_real]
    if not tail:
        lines.append(
            "   → keine Sättigung im gemessenen Bereich — der letzte Schritt "
            "liefert noch einen Zugewinn mit CI ohne 0."
        )
    elif all(kinds[k] == "negligible" for k in tail):
        lines.append(
            f"   → Sättigung ab {last_real} Modalität(en): jeder weitere "
            f"Schritt ist nachweislich < {thr:.2f} (Äquivalenz, nicht bloß n.s.)."
            if real_ks else
            f"   → alle Schritte nachweislich < {thr:.2f} — keine Modalität "
            "liefert einen praktisch relevanten Beitrag."
        )
    else:
        undecided = [k for k in tail if kinds[k] == "undecided"]
        need = max(
            (report.step_required_n.get(k, 0) for k in undecided), default=0
        )
        lines.append(
            "   → keine belastbare Sättigungsaussage: Schritt "
            + ", ".join(f"+{k}" for k in undecided)
            + f" unentscheidbar (CI überdeckt 0 und ±{thr:.2f})"
            + (f" — Äquivalenznachweis bräuchte ≈ {need} Sessions."
               if need else ".")
        )
    if report.curve_no_video:
        f1 = " → ".join(f"{report.curve_no_video[k]:.3f}"
                        for k in sorted(report.curve_no_video))
        gains = ", ".join(f"{report.gains_no_video[k]:+.3f}"
                          for k in sorted(report.gains_no_video))
        phi = ", ".join(f"{m} {v:+.3f}" for m, v in report.shapley_no_video.items())
        lines.append(
            f"   Ohne Video (Audio+Events):  F1 {f1}   ·   Grenznutzen {gains}"
            + (f"   ·   Shapley {phi}" if phi else "")
        )
    lines.append(
        "   Sättigungskurve = Mittel über alle Teilmengen einer Größe; sie mischt "
        "starke und schwache Paare. Der gierige Pfad (beste Einzelne → bestes Paar → alle) "
        "steht in der Tabelle der 8 Kombinationen."
    )
    return "\n".join(lines)


def sync_text(sync: dict[str, float], tau_s: float = PRIMARY_TAU_S) -> str:
    """
    Two different things are reported here and must not be confused: the
    clock offset between the streams (a recording property, tens of ms) and
    the click-to-reaction latency (a UI property that the boundary definition
    absorbs). Only the first justifies τ; the second is the semantic lag the
    feature windows bridge.
    """
    if not sync:
        return "   (keine Synchronisationsmessung)"

    def ms(key: str) -> str:
        value = sync.get(key, float("nan"))
        return "–" if value != value else f"{value:.0f} ms"

    def known(key: str) -> bool:
        value = sync.get(key, float("nan"))
        return value == value

    tau_ms = tau_s * 1000.0
    lines = [
        f"   Stream-Offset (Audio↔Video): {ms('stream_offset_ms')}   ·   "
        f"Klick→Bildschirmreaktion Median: {ms('click_median_ms')} · "
        f"P95: {ms('click_p95_ms')} "
        f"(max {ms('click_absmax_ms')}, n={int(sync.get('n', 0))})",
    ]
    if known("stream_offset_ms") and abs(sync["stream_offset_ms"]) <= tau_ms * 0.1:
        lines.append(
            f"   →   Uhrenversatz der Streams liegt um mehr als eine Größenordnung unter "
            f"τ = {tau_s:.2f} s: die Modalitäten teilen die Zeitbasis."
        )
    else:
        lines.append(
            f"   →   ⚠ Stream-Offset nicht vernachlässigbar gegenüber τ = {tau_s:.2f} s — "
            "Synchronisation prüfen."
        )
    lines.append(
        "   Die Klick→Reaktion-Latenz ist kein Uhrenfehler, sondern die Zeit vom "
        "Auslösen bis zum sichtbaren Ergebnis; sie begründet die Fenstermerkmale, nicht τ. "
        "τ selbst ist über die Annotationsreliabilität zu begründen."
    )
    return "\n".join(lines)


def interaction_text(report: Report) -> str:
    """A direction is only claimed when the CI commits to one — an index of
    +0.001 whose interval covers 0 is noise, not synergy."""
    def label(ci: statistics.Interval) -> str:
        if not ci.excludes_zero:
            return "von 0 nicht unterscheidbar — keine Richtungsaussage"
        return ("Synergie — gemeinsam mehr als einzeln" if ci.point > 0
                else "Redundanz — teilweise dieselbe Information")

    return "\n".join(
        f"   {a} ↔ {b}:   {ci}   {label(ci)}"
        for (a, b), ci in report.interactions.items()
    )


def status_line(report: Report) -> str:
    """The one-line summary the window shows when a run completes."""
    return (
        f"Fertig · {report.n_sessions} Session"
        + ("" if report.n_sessions == 1 else "s")
        + f" · τ = {report.tau_s:.2f} s · "
        f"F1 {report.f1_all.point:.3f} · Zufallsniveau {report.chance:.3f}"
        + ("  ·  ⚠ n = 1: Konfidenzintervalle sind entartet"
           if report.n_sessions < 2 else "")
        + (f"  ·  ⚠ Shapley-Effizienzfehler {report.efficiency_error:.1e}"
           if report.efficiency_error > 1e-6 else "")
    )


def header_lines(report: Report, corpus_root: Path | None, created: str) -> list[str]:
    """The metadata block at the top of the PDF report."""
    return [
        f"Korpus: {corpus_root if corpus_root else '—'}",
        f"Erstellt: {created}",
        f"Sessions: {report.n_sessions}",
        f"Verfahren: {DESIGN}",
        f"Toleranz τ = {report.tau_s:.2f} s   ·   Zufallsniveau = {report.chance:.3f}"
        + (f"   ·   Shapley-Effizienzfehler = {report.efficiency_error:.1e}"
           if report.efficiency_error > 1e-6
           else "   ·   Effizienzeigenschaft erfüllt"),
    ]


@dataclass(frozen=True)
class Section:
    """One block of the report: a heading, a one-line reading aid, and either a
    chart name or a text body (or both, chart first)."""

    title: str
    hint: str
    chart: str | None = None      # "shapley" | "subsets" | "saturation"
    text: str | None = None


def sections(report: Report) -> list[Section]:
    """
    The report in reading order — shared by the window and the PDF so both
    present the same sections with the same headings.
    """
    return [
        Section("Kennzahlen & statistische Aussagekraft",
                "F1, Recall und Precision der vollen Modalitätsmenge; war der Datensatz groß genug?",
                text=adequacy_text(report)),
        Section("Versuchsdesign",
                "Was gerechnet wurde, und welchen Arbeitspunkt die Kalibrierung je Teilmenge wählte.",
                text=design_text(report)),
        Section("Shapley-Werte · marginaler Beitrag je Modalität (TF2)",
                "Balken = Beitrag, Whisker = 95 %-Konfidenzintervall. Ein Whisker "
                "über der Nulllinie bedeutet: von null nicht unterscheidbar.",
                chart="shapley"),
        Section("Alle 8 Modalitätskombinationen (TF1)",
                "Gefüllter Punkt = Modalität enthalten. Rote Linie = Zufallsniveau.",
                chart="subsets"),
        Section("Kandidatenpool · isoliert gegen fixiert (TF1, Kontrolle)",
                "Isoliert = jede Teilmenge schlägt nur eigene Kandidaten vor (Wortlaut des "
                "Exposés). Fixiert = alle Kandidaten, nur die eigenen Merkmale (Informationsbeitrag).",
                text=pool_text(report)),
        Section("Sättigungskurve (TF3)",
                "Mittlerer F1 je Anzahl Modalitäten; Zahlen dazwischen = Zugewinn.",
                chart="saturation", text=saturation_text(report)),
        Section("Interaktionsindex · Synergie und Redundanz (TF3)",
                "Negativ = die beiden Modalitäten sagen teilweise dasselbe.",
                text=interaction_text(report)),
        Section("Robustheit · Toleranz-Sweep und Recall-Attribution",
                "Shapley-Werte je τ und nach Recall statt F1. Eine Rangfolge, die bei jedem τ "
                "hält, ist eine Eigenschaft der Modalitäten, nicht der Bewertung.",
                text=robustness_text(report)),
        Section("Multiplizität",
                "Je Familie von Vergleichen Bonferroni-adjustierte Intervalle.",
                text=multiplicity_text(report)),
        Section("Kopplung mit der Ground Truth · Confound-Kontrolle",
                "Wie nah die ROHEN Kandidaten einer Modalität an der Annotation liegen, "
                "bevor irgendein Modell entscheidet.",
                text=coupling_text(report)),
        Section("Recall-Obergrenze und Verlustzerlegung",
                "Was kein Entscheider erreichen kann (kein Vorschlag) gegen das, was dieser "
                "Entscheider liegen lässt.",
                text=ceiling_text(report)),
        Section("Grenzdefinition · Ende gegen Beginn (Sensitivität)",
                "Dieselben Kurven, bewertet gegen die zweite Annotation (erster Eingabe des "
                "nächsten Schritts).",
                text=definition_text(report)),
        Section("Zeitsynchronisation",
                "Gemessener Uhrenversatz der Modalitäten, getrennt von der UI-Latenz.",
                text=sync_text(report.sync, report.tau_s)),
    ]
