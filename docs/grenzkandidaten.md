# Grenzkandidaten je Modalität

Dieses Dokument beschreibt, wie jede Modalität aus **einer** Aufzeichnung
Kandidaten für Handlungsgrenzen ermittelt. Der Code liegt in
`src/docupilot/segmentation/`.

## Definition einer Handlungsgrenze

Eine **Handlungsgrenze** ist der Zeitpunkt, an dem eine benutzerausgelöste
Aktion *abgeschlossen* ist und das System in einen **neuen, dauerhaften Zustand**
übergeht.

- Ein Menü zu öffnen ist keine Grenze — es ist ein transienter Zwischenschritt.
- Den Menüpunkt zu wählen, der den Zustand ändert, **ist** eine Grenze.

Grenzen sind in den Rohdaten nicht explizit vorhanden und müssen algorithmisch
bestimmt werden. In der Ground Truth (`ground_truth.json`) sind sie als
Zeitpunkte in Millisekunden annotiert; die Auswertung rechnet in Sekunden.

## Gemeinsamer Vertrag: `BoundaryEvidence`

Jede Modalität liefert dasselbe Ergebnisobjekt (`evidence.py`):

| Feld | Bedeutung |
|---|---|
| `times_s` | Zeitstempel je Abtastpunkt |
| `score` | **abgestufte** Evidenz in `[0, 1]` je Abtastpunkt (für Anzeige & Fusion) |
| `boundaries_s` | Grenzen, auf die sich die Modalität festlegt (`score ≥ 0.5`) |

- **Zeitraster:** Audio und Events nutzen ein 50-Hz-Raster (`GRID_HZ`); Video
  nutzt seine eigenen Framezeiten (aus dem MP4 via `ffprobe`).
- **Zeichen-Primitive:** `apply_gaussian` (symmetrischer Peak) und `apply_window`
  (raised-cosine-Fenster, Peak asymmetrisch platzierbar). Bei Überlappung gewinnt
  jeweils der höhere Wert.
- **Unabhängigkeit:** Jede Modalität liest **ausschließlich ihre eigene Quelle**.
  Das ist die Voraussetzung der Shapley-Analyse — ein einziger Blick in eine
  fremde Modalität würde den gemessenen Beitrag verfälschen.
- **Einstieg:** `pipeline.segment()` führt alle Modalitäten über eine Session aus
  und meldet jedes Ergebnis per Callback.

Wichtig: Die Modalitäten liefern **abgestufte Evidenz**, keine harten
Entscheidungen. Die eigentliche Schwellwertbildung passiert erst nachgelagert in
der Fusion (siehe [modalitaetsbeitraege.md](modalitaetsbeitraege.md)) — für alle
Modalitäten identisch.

---

## Video (`video.py`) — liest nur den Bildstrom

1. **Aktivitätssignal:** Pro Frame wird ein 8×8-Raster perzeptueller Hashes
   (pHash) berechnet; die Frame-Aktivität ist die größte Kachel-Distanz zum
   Vorframe.
2. **Dwells (Ruhezustände):** Zusammenhängende Läufe mit Aktivität unter
   `ACTIVITY_QUIET` (0.08), die mindestens `_MIN_DWELL_S` (0.5 s) dauern. Ein
   Dwell ist ein *eingerasteter* Bildschirmzustand; die Bursts dazwischen sind
   Übergänge.
3. **Kandidat & Urteil:** Je Dwell wird ein settled-Frame (0.2 s hineingesampelt)
   gegen einen **Anchor** verglichen, der den zuletzt etablierten Zustand hält.
   Ein VLM (Vision-Modell) beurteilt das `BEFORE|AFTER`-Kompositbild — mit einer
   Set-of-Mark-Box auf der geänderten Region — und liefert `p_boundary`.
   Pixel-identische Zustände werden ohne Modellaufruf übersprungen.
4. **Ausgabe:** `p_boundary` wird per `apply_gaussian` am Dwell-Beginn eingetragen.
   Bei `p_boundary ≥ 0.5` wird eine Grenze gesetzt **und** der Anchor rückt vor.

> Recall entsteht in der pHash-Stufe (jeder Dwell wird geprüft), Präzision im VLM.
> Ein `_MAX_CALLS`-Deckel (400) schützt vor flackernden Aufnahmen.

## Events (`events.py`) — liest nur `events.json`

1. **Bursts:** Alle Eingabeereignisse werden chronologisch gruppiert; eine Pause
   ≥ `_BURST_PAUSE_S` (2.0 s, nach Wengelin 2006) trennt zwei Bursts.
2. **Kandidat & Score:** Jeder Burst erzeugt genau einen Kandidaten auf seinem
   **letzten** Ereignis. Der Score ist die *Ruhe danach*:
   `min(Pause_bis_zum_nächsten_Burst / _REST_FULL_S, 1)` mit `_REST_FULL_S` = 8 s.
3. **Ausgabe:** `apply_gaussian` am Kandidaten; Grenze bei `score ≥ 0.5`.

> Idee: „Bursts und die Ruhe nach ihnen“ — eine abgeschlossene Aktion wird von
> einer Denkpause gefolgt.

## Audio (`audio.py`) — liest nur die Tonspur

1. **Transkript → Sätze:** Whisper (`small`) liefert Text + Wortzeiten; spaCy
   segmentiert in Sätze `(t_s, text)`.
2. **Kandidat & Urteil:** Ein LLM beurteilt jeden Satz → `p_boundary`. Jeder Satz
   öffnet ein **Ausführungsfenster** bis zum nächsten Satz (das letzte bis zum
   Median-Ansagenabstand bzw. Aufnahmeende). `apply_window` legt einen
   raised-cosine-Bump ins Fenster, Peak bei `_COMPLETION_POSITION` (0.75).
3. **Ausgabe:** Grenze am **Peak** des Fensters (nicht am Satzanfang), bei
   `p_boundary ≥ 0.5`.

> Audio kennt nur das **Intervall**, nie den exakten Zeitpunkt: Die Ansage kündigt
> Schritte in Reihenfolge an; Schritt *i* ist zwischen Ansage *i* und *i+1*
> abgeschlossen.

---

## Überblick

| Modalität | Quelle | Kandidat entsteht aus | Score | Grenze bei |
|---|---|---|---|---|
| **Video** | MP4-Bildstrom | Dwell (Ruhezustand) | VLM-Urteil `p_boundary` | `p ≥ 0.5`, Anchor rückt vor |
| **Events** | `events.json` | Eingabe-Burst | Ruhe nach dem Burst | `score ≥ 0.5` |
| **Audio** | Tonspur | Narrierter Satz | LLM-Urteil `p_boundary` | `p ≥ 0.5`, am Fenster-Peak |

Alle drei liefern denselben `BoundaryEvidence`-Vertrag und bleiben strikt
voneinander unabhängig.
