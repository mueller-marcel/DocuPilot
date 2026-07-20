# Evaluationsplan

Wie aus der implementierten Segmentierung die drei Teilfragen der Masterarbeit
beantwortet werden.

**Forschungsfrage:** Wie verändert sich die Qualität der automatisierten
Segmentierung in Abhängigkeit von Art, Anzahl und Kombination der Modalitäten —
und ab welchem Punkt liefert eine weitere Modalität keinen relevanten Beitrag
mehr?

| | Teilfrage | beantwortet durch |
|---|---|---|
| **TF1** | Welche Modalität leistet isoliert am meisten? | Tabelle der 8 Kombinationen |
| **TF2** | Welchen marginalen Beitrag leistet jede Modalität? | Shapley-Werte |
| **TF3** | Ab wann tritt Sättigung / Redundanz ein? | Sättigungskurve, Interaktionsindex |

Datensatz: **25 selbst aufgenommene und annotierte Sessions**, ein Annotator.

---

## Die fünf Schichten

```
5. Ist der Unterschied Zufall?        <- Statistik            (später)
4. Wer hat wie viel beigetragen?      <- Shapley              Schritt 4
3. Wie gut war eine Vorhersage?       <- F1 mit Toleranz      Schritt 1
2. Wie entsteht eine Vorhersage?      <- Random Forest        Schritt 2+3
1. Was liefern die Modalitäten?       <- fertig (segmentation/)
```

Schicht 1 ist implementiert: `video.py`, `audio.py`, `events.py` liefern je ein
`BoundaryEvidence(times_s, score, boundaries_s)`. Die Module importieren einander
nicht — diese Unabhängigkeit ist die Voraussetzung dafür, dass die Ablation
überhaupt gültig ist.

Schichten 2–4 sind die vier Bauschritte unten. Schicht 5 folgt danach.

---

## Schritt 1 — F1 mit Toleranz

**Ziel:** Aus einer Liste vorhergesagter Grenzen und der Ground Truth eine Zahl
machen.

Eine Vorhersage zählt als Treffer, wenn sie innerhalb der Toleranz `tau` an einer
GT-Grenze liegt:

```
Ground Truth:   10,0s        25,0s        40,0s
Vorhersage:     10,3s        24,8s                    55,0s
                 TP           TP           FN         FP

Precision = TP/(TP+FP) = 2/3 = 0,67
Recall    = TP/(TP+FN) = 2/3 = 0,67
F1                            = 0,67
```

**Das Matching muss 1:1 sein.** Ohne diese Bedingung könnte eine einzelne
Vorhersage mehrere GT-Grenzen bedienen und den Recall künstlich aufblähen.
Verfahren: optimales bipartites Matching (Hungarian) über die Paare mit
`|t_pred - t_gt| <= tau`.

**Toleranz:** `tau = 1,0 s` als Primärwert, zusätzlich ein Sweep über
0,25 s bis 3,0 s als Robustheitsnachweis.

**Verifikation des Messinstruments** (Unit-Tests, bevor irgendein Ergebnis
entsteht):

| Prüfung | Erwartung |
|---|---|
| Vorhersage identisch zur GT | F1 = 1,0 |
| Leere Vorhersage | F1 = 0,0 |
| Eine Vorhersage, zwei GT innerhalb tau | genau 1 TP, 1 FN |
| Vorhersage um tau + epsilon verschoben | TP wird zu FN |

**Nullmodell:** Ein Zufallsgenerator, der Grenzen mit derselben Rate wie die GT
setzt, liefert das Zufallsniveau. Ohne diesen Referenzwert ist ein F1 von 0,65
nicht interpretierbar.

**Ergebnis:** `metrics.py`

---

## Schritt 2 — Kandidaten und Merkmale

**Ziel:** Für eine Modalitätskombination S die Zeitpunkte bestimmen, über die
überhaupt entschieden wird, und sie beschreiben.

**Kandidaten:** die Vereinigung der Vorschläge aller Modalitäten in S.

> Nur Modalitäten aus S. Zöge die Kombination `{audio}` Kandidaten aus dem Video,
> würde Video-Information gemessen und Audio zugeschrieben — genau der Confound,
> den die Modultrennung verhindern soll.

Nebeneffekt: die leere Menge erzeugt keine Kandidaten, also keine Vorhersage,
also `v({}) = 0`. Das macht die Shapley-Zerlegung sauber.

**Merkmale je Kandidat:** die Score-Werte der beteiligten Modalitäten an diesem
Zeitpunkt.

| Kandidat | video_score | events_score | Label |
|---|---|---|---|
| 10,3s | 0,82 | 0,71 | Grenze |
| 24,8s | 0,76 | 0,68 | Grenze |
| 55,0s | 0,54 | 0,12 | keine Grenze |

**Label:** aus der Ground Truth — liegt eine echte Grenze innerhalb `tau`?

**Warum kandidatenbasiert und nicht pro Zeitraster?** Bei 50 Hz und rund
10 Grenzen je Session läge das Klassenverhältnis bei etwa 1:10.000. Kandidaten
reduzieren das auf ein trainierbares Maß.

**Ergebnis:** `dataset.py`, `fusion.py` (Kandidatenteil)

---

## Schritt 3 — Random Forest je Kombination

**Ziel:** Pro Kombination ein Modell, das je Kandidat entscheidet: echte Grenze
oder nicht.

Sieben Modelle (die leere Menge braucht keines). Jedes sieht ausschliesslich die
Merkmalsspalten seiner eigenen Modalitäten.

**Leave-One-Session-Out:** Das Modell, das Session *i* vorhersagt, darf Session
*i* nie im Training gesehen haben. 25 Durchläufe je Kombination, jeweils 24
Sessions zum Trainieren, 1 zum Testen. Auch die Entscheidungsschwelle wird nur
auf den Trainingsfolds bestimmt — sonst leakt die Testsession über die Schwelle
ein.

**Gegenprobe ohne Training:** dieselbe Auswertung mit einer einfachen Regel
(Maximum der beteiligten Scores, Schwelle 0,5). Ergibt sie in Schritt 4 dieselbe
Rangfolge, hängt das Ergebnis nicht an der Wahl des Modells.

**Ergebnis:** eine Tabelle mit einer Zeile je (Session x Kombination x tau):
`session_id, subset, tau, tp, fp, fn, precision, recall, f1`

Daraus lässt sich jede weitere Auswertung ableiten. Export als CSV.

> **Kosten:** Die teuren VLM/LLM-Aufrufe entstehen einmal je Session und
> Modalität, nicht je Kombination. Alle 8 Kombinationen verwenden dieselben
> zwischengespeicherten Evidenzkurven. Das 2^3-Design kostet 3 Extraktionen pro
> Session, nicht 8.

**Ergebnis:** `fusion.py`, `experiment.py`

---

## Schritt 4 — Shapley

**Ziel:** Die Gesamtleistung auf die drei Modalitäten aufteilen.

Bei drei Modalitäten gibt es 3! = 6 Reihenfolgen. Der Shapley-Wert einer
Modalität ist der mittlere Zuwachs, den sie beisteuert, gemittelt über alle
Reihenfolgen ihres Hinzukommens.

Beispiel für Video, bei angenommenen Werten:

| Reihenfolge | Video kommt | Zuwachs |
|---|---|---|
| V,A,E | zuerst | 0,65 - 0,00 = 0,65 |
| V,E,A | zuerst | 0,65 - 0,00 = 0,65 |
| A,V,E | nach Audio | 0,73 - 0,30 = 0,43 |
| E,V,A | nach Events | 0,75 - 0,50 = 0,25 |
| A,E,V | zuletzt | 0,80 - 0,55 = 0,25 |
| E,A,V | zuletzt | 0,80 - 0,55 = 0,25 |

`phi_video = Mittelwert = 0,41`

**Exakt, nicht approximiert.** Bei drei Spielern werden alle 8 Koalitionen
vollständig ausgewertet — es gibt keinen Monte-Carlo-Fehler zu rechtfertigen,
anders als in weiten Teilen der Shapley-Literatur.

**Summenprobe:** `phi_video + phi_audio + phi_events = v({alle drei})`. Das ist
eine mathematische Identität (Effizienzeigenschaft). Geht sie nicht auf, liegt
ein Implementierungsfehler vor — das Ergebnis prüft sich selbst.

**Theoretische Begründung:** Der Shapley-Wert ist die eindeutige Zuordnung, die
die Axiome Effizienz, Symmetrie, Dummy und Additivität erfüllt (Shapley 1953).
Die Wahl ist damit nicht Geschmack, sondern unter benannten Anforderungen
alternativlos.

**Für TF3 zusätzlich:**

- **Sättigungskurve:** mittlerer F1 je Anzahl beteiligter Modalitäten
  (k = 0,1,2,3) und der Zuwachs von Stufe zu Stufe.
- **Interaktionsindex** (Grabisch & Roubens 1999): `I(i,j) > 0` bedeutet
  Synergie, `I(i,j) < 0` bedeutet Redundanz. Beantwortet die Redundanzfrage
  direkt und unabhängig vom Signifikanztest.

**Ergebnis:** `analysis.py`

---

## Nach Schritt 4: Schicht 5

Nach den vier Schritten liegt ein vollständiges Ergebnis vor. Was dann folgt,
beantwortet ausschliesslich die Frage: *Ist der gemessene Unterschied echt oder
Zufall der 25 gewählten Sessions?*

Dafür werden die F1-Werte **je Session** benötigt (nicht nur die Mittelwerte):

| Session | {V,E} | {V,A,E} | Differenz |
|---|---|---|---|
| 1 | 0,71 | 0,79 | +0,08 |
| 2 | 0,80 | 0,80 | 0,00 |
| 3 | 0,62 | 0,74 | +0,12 |

Sind die Differenzen überwiegend gleichgerichtet, liegt ein echter Effekt vor.

| Baustein | Zweck |
|---|---|
| Wilcoxon-Vorzeichen-Rang-Test | Sind die Differenzen überwiegend gleichgerichtet? |
| BCa-Bootstrap | Konfidenzintervall statt Punktschätzer |
| Holm-Korrektur | Schutz vor Zufallstreffern bei vielen Vergleichen |
| Sensitivitäts-Poweranalyse (MDE) | Welche Effektgrösse kann n = 25 überhaupt zeigen? |
| TOST | Nachweis, dass ein Effekt *klein* ist (für TF3) |
| Intrarater-Reliabilität | Messlatte und empirische Herleitung der Relevanzschwelle |

Details werden vor Beginn von Schicht 5 festgelegt und committet, damit
nachweisbar ist, dass der Analyseplan vor den Ergebnissen feststand.

---

## Modulaufbau

```
evaluation/
  metrics.py       Matching, Precision/Recall/F1, Nullmodell      Schritt 1
  dataset.py       Sessions laden, Evidenz cachen                 Schritt 2
  fusion.py        Kandidaten, Merkmale, Random Forest + Regel     Schritt 2+3
  experiment.py    LOSO über 25 Sessions x 8 Kombinationen         Schritt 3
  analysis.py      Shapley, Interaktionsindex, Sättigungskurve     Schritt 4
  statistics.py    Signifikanz, Konfidenzintervalle, Power         Schicht 5
  reliability.py   Intrarater-Auswertung                           Schicht 5
```

Die Bibliothek muss **headless lauffähig** sein: ein Aufruf, der die
Ergebnistabelle schreibt. Die GUI orchestriert und zeigt an, sie rechnet nicht.
Der Lauf dauert Stunden und darf nicht an einem offenen Fenster hängen; ausserdem
muss das Experiment mit einem Befehl reproduzierbar sein.

Abbildungen für die Arbeit werden aus der exportierten CSV erzeugt, nicht aus
Qt-Fenstern.

---

## Offene Punkte

- **Entwicklungsset abgrenzen:** `_ACTIVITY_QUIET` und `_MIN_DWELL_S` wurden auf
  session_07 gesweept. Diese Session (und alles weitere, woran getunt wurde) aus
  dem Evaluationskorpus ausschliessen und im Methodikteil benennen.
- **Annotationsleitfaden rekonstruieren:** Die Regeln A (nutzerausgelöst),
  B (neuer persistenter Zustand), C (Ziel statt Mittel) existieren derzeit nur in
  den Modell-Prompts. Für den Anhang der Arbeit wird ein eigenständiges Dokument
  benötigt, das belegt, dass die Regeln vor der Annotation feststanden.
- **Blind annotieren:** Während des Annotierens dürfen keine Modellvorhersagen
  sichtbar sein, sonst bestätigt die Ground Truth nur noch das Modell.

---

## Literatur

- Shapley, L. S. (1953): *A Value for n-Person Games.*
- Grabisch, M.; Roubens, M. (1999): *An axiomatic approach to the concept of
  interaction among players in cooperative games.*
- Lakens, D. (2022): *Sample Size Justification.*
- Lakens, D. (2017): *Equivalence Tests.*
- Holm, S. (1979): *A simple sequentially rejective multiple test procedure.*
- Efron, B.; Tibshirani, R. (1993): *An Introduction to the Bootstrap.*
