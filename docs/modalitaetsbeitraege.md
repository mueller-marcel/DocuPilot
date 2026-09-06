# Ermittlung der Modalitätsbeiträge

Dieses Dokument beschreibt, wie aus einem **Set von Aufzeichnungen** der Beitrag
jeder Modalität zur Segmentierungsqualität bestimmt wird. Der Code liegt in
`src/docupilot/evaluation/`. Voraussetzung sind die pro Modalität ermittelten
Grenzkandidaten (siehe [grenzkandidaten.md](grenzkandidaten.md)).

## Idee in einem Satz

Bewerte jede der `2³ = 8` Modalitätskombinationen auf allen Sessions, fasse das zu
einer charakteristischen Funktion `v(S)` zusammen und zerlege diese mit
**Shapley-Werten** in den mittleren marginalen Beitrag jeder Modalität — und
umgib diese Antwort mit den Kontrollen, die ein Gutachter verlangt.

## Ablauf

### 1. Einmal segmentieren je Session (`experiment.load`)
Pro Session werden die drei Modalitäten **einmal** extrahiert und die
`BoundaryEvidence`-Kurven im Speicher gehalten. Alle 8 Kombinationen und beide
Kandidatenpool-Designs nutzen dieselben Kurven.

- Ground Truth (`gt_s`) und Aufnahmedauer werden mitgeladen. Über `kind`
  lässt sich statt der Grenzdefinition „Ende“ die zweite Annotation „Beginn“
  laden (siehe Schritt 7).
- Parameter der Extraktoren wurden ausschließlich auf `session_30` bestimmt,
  die nicht zum Korpus (`session_01`–`25`) gehört.

### 2. Faktorielles Design (`analysis.subsets`)
Es werden **alle** Teilmengen der Modalitäten gebildet — einschließlich der leeren
Menge: `{}`, `{video}`, `{audio}`, `{events}`, `{video,audio}`, …, `{alle drei}`.

### 3. Bewertung je (Subset, Session) (`experiment.run`)
Für jedes Subset und jede Session — mit **Leave-one-session-out**, das Modell sieht
die getestete Session also nie:

1. **Kandidaten** (`fusion.candidate_times`): die lokalen Maxima der
   Score-Kurven. Bewusst die Peaks, nicht die bereits geschwellten
   `boundaries_s` — sonst könnte der Klassifikator Grenzen nur entfernen, nie
   hinzufügen. **Woher** die Kandidaten kommen, ist ein expliziter Faktor
   (`pool`):
   - `"union"` (primär): der Pool ist für **jede** Koalition die Vereinigung
     aller drei Modalitäten; eine Koalition liest aber nur ihre eigenen
     Merkmale. Der Suchraum ist konstant, es variiert nur die verfügbare
     Information — der *Informationsbeitrag*, den TF2 und TF3 fragen.
   - `"own"` (isoliert): jede Koalition schlägt nur eigene Kandidaten vor —
     das, was TF1 mit „isoliert“ wörtlich meint. Die Differenz
     `v_union({m}) − v_own({m})` ist der **Zeitpunkt-Kredit**, den eine
     Modalität allein durch die Kandidaten der anderen erhält.
2. **Merkmale** (`fusion.feature_matrix`): je Modalität **fünf** Spalten, für
   alle Modalitäten nach demselben Rezept — Punktwert am Kandidaten, Maximum im
   Fenster ±0,5 / ±1 / ±2 s, Sessionrang des ±1-s-Maximums
   (`fusion.FEATURE_NAMES`). Mehrere Fensterbreiten machen die zeitliche
   Schärfe einer Modalität (Video: Einrastmoment, Audio: Ausführungsfenster)
   zum Merkmal statt zum Handicap.
3. **Labels** (`fusion.label_candidates`): ein Kandidat ist positiv, wenn er
   ≤ `LABEL_TAU_S` (1 s) an einer Ground-Truth-Grenze liegt. Bewusst *nicht*
   1-zu-1 — das ist das Trainingssignal, nicht die Bewertung.
4. **Entscheider** (`fusion.ForestFuser`): Random Forest über die
   Merkmalsspalten der Koalition, `class_weight="balanced"`.
5. **Schwelle** (`fusion.choose_threshold`): je Fold auf den
   **Out-of-Bag-Vorhersagen der 24 Trainingssessions** kalibriert (Raster
   0,05 … 0,95, Ziel: Makro-F1 bei τ = 1 s). Die Testsession berührt den
   Arbeitspunkt nie; ein fester Wert 0,5 wäre bei balancierter Gewichtung
   nicht F1-optimal.
6. **Vorhersage:** `fusion.decide` → Kandidaten ab der Schwelle,
   `fusion.suppress` behält je Umgebung (Radius 1 s) nur den stärksten.
7. **Matching & Metrik** (`metrics.match`): Vorhersage und Ground Truth werden
   **1-zu-1** zugeordnet (Kuhn-Munkres, Toleranz `τ`); daraus tp/fp/fn und
   Precision/Recall/F1. Ausgewertet über einen Toleranz-Sweep
   `τ ∈ {0.25, 0.5, 1, 1.5, 2, 3}` s.

Ergebnis ist **eine** Tabelle je Pool-Design mit einer Zeile je
`(Session, Subset, τ)`, inklusive der kalibrierten Schwelle
(`experiment.write_csv`). Alle weiteren Analysen leiten sich daraus ab.

### 4. Charakteristische Funktion `v(S)` (`experiment.subset_values`)
Je Subset der **macro-gemittelte F1** bei festem `τ` (jede Session zählt gleich,
unabhängig von ihrer Länge).

### 5. Shapley-Werte (`analysis.shapley`)
Der mittlere marginale Beitrag jeder Modalität über **alle** Beitrittsreihenfolgen.
Bei drei Modalitäten werden alle 8 Koalitionen vollständig ausgewertet — die Werte
sind **exakt**, kein Sampling.

- Eigenschaften: Effizienz, Symmetrie, Dummy, Additivität (Shapley 1953).
- **Selbsttest** (`analysis.efficiency_error`): Die Werte müssen sich zu
  `v(alle) − v(∅)` summieren.

### 6. Statistik (`statistics.py`)
- **BCa-Bootstrap über Sessions** für jede Kennzahl: Subset-F1, Shapley-Werte,
  Interaktionsindizes, Sättigungsschritte. Ein Resampling liefert alle Werte
  einer Familie zugleich (`bootstrap_intervals`).
- **Bonferroni je Familie** (Shapley, Interaktion, Sättigung): zusätzlich
  `α = 0,05/3`-Intervalle, damit sichtbar wird, welche Aussage als eine von
  dreien besteht.
- **MDE und erforderliche Fallzahl** (Lakens 2022): was der Korpus auflösen
  kann, und was ein Äquivalenznachweis bräuchte.

### 7. Kontrollen (`report.analyse`)
- **Kopplung** (`coupling.CouplingStats`): je Modalität Deckung der GT-Grenzen
  durch rohe Kandidaten in ±τ, **Zufallsdeckung** derselben Kandidatenzahl,
  Lift, Rate, **Feinausrichtung** (Anteil in ±τ/4) und die Versatzverteilung
  (Median, IQR). Die Feinausrichtung trennt „informativ“ von „definitorisch
  gekoppelt“: ein geteiltes Signal trifft auf Zehntelsekunden.
- **Recall-Obergrenze** (`coupling.union_coupling`): Deckung der Vereinigung
  aller Kandidaten; daraus Entscheidungs- und Vorschlagsverlust.
- **Toleranz-Sweep** der Shapley-Werte und **Recall-Attribution**: hält die
  Rangfolge bei jedem τ und ohne Arbeitspunkt?
- **Grenzdefinition**: sind alle Sessions zusätzlich mit „Beginn“ annotiert
  (erste Eingabe des nächsten Schritts, Wortlaut des Exposés), werden dieselben
  Kurven gegen diese Referenz bewertet (`experiment.with_ground_truth`).
- **Zeitsynchronisation** (`report.measure_sync`): Stream-Offset und
  Klick→Reaktion, getrennt ausgewiesen.

Der Bericht (`report.sections`) ist die eine Quelle für Fenster und PDF.

## Warum die Beiträge sauber sind

Jedes Subset liest **nur** die Evidenz seiner eigenen Modalitäten
(`fusion`-Docstring). Eine ausgeschlossene Modalität wird nie berührt — im
fixierten Pool entscheidet sie höchstens *wo* gesucht wird, nie *was* eine
Koalition sieht, und genau dieser Anteil wird als Zeitpunkt-Kredit gesondert
ausgewiesen. Zusammen mit Leave-one-session-out, OOB-kalibrierter Schwelle und
der Entwicklungssession außerhalb des Korpus ist sichergestellt, dass die
gemessene Qualität auf die **Variation der Modalitäten** zurückgeht.
