# Abgleich: Grundlagen 2.4 gegen die Implementierung

Grundlage ist die Vorschau von Abschnitt 2.4 (Unterabschnitte 2.4.1–2.4.5) und
der Code in `src/docupilot/segmentation/` und `src/docupilot/evaluation/`.

**Zuschnitt:** 2.4 behandelt die Modalitäten Video und Audio; die Ereignisse
werden in 2.5 gesondert behandelt. Dieser Abgleich prüft daher nur, ob 2.4 die
Video- und Audio-Mechanismen trägt. Abschnitt 2.5 liegt mir nicht vor — was der
Ereignisstrom an Grundlagen bräuchte, steht am Ende als offener Punkt.

Geprüft wurde eine Frage: Kann die Methodik in Kapitel 3 jeden Mechanismus der
Implementierung auf einen in 2.4 eingeführten Begriff zurückführen, ohne ihn dort
erst einzuführen?

## Befund in Kürze

| Mechanismus im Code | Fundstelle in 2.4 | Status |
|---|---|---|
| Fensterreihe, Unähnlichkeit benachbarter Fenster, Schwellenwert | 2.4.1 | trägt |
| Über-Segmentierung als Kernproblem | 2.4.1 | benannt, aber ohne Gegenmaßnahme |
| Transkript als symbolische Zwischenrepräsentation (Whisper) | 2.4.3 | trägt |
| Vortrainiertes Modell als Urteilsinstanz, Verzerrungen | 2.4.3 | trägt im Grundsatz |
| Zweistufige Architektur Kandidat → semantische Entscheidung | 2.4.5 | trägt |
| **Unähnlichkeitsmaß für Bildfolgen (perzeptueller Hash, Kachelraster)** | — | **fehlt** |
| **Kandidaten aus Stabilität statt aus Änderung (Dwells)** | — | **fehlt** |
| **Vergleich gegen einen Bezugszustand statt gegen das Nachbarfenster (Anchor)** | — | **fehlt** |
| **Unterdrückung redundanter Kandidaten (Mindestdauer, Non-Maximum-Suppression)** | — | **fehlt** |
| **Zeitlicher Versatz zwischen Ansage und Ausführung (Intervall statt Zeitpunkt)** | — | **fehlt** |
| **Mechanik der Modellbefragung (Positionsverzerrung, Set-of-Mark, Schema, Konfidenz)** | 2.4.3 nur im Grundsatz | **unvollständig** |
| **Abgestufte Evidenz + lokale Maxima statt geschwellter Grenzen** | — | **fehlt** |
| Zusammenführung mehrerer Modalitäten (späte Fusion) | — | außerhalb von 2.4 zu prüfen |

Alle sieben Lücken betreffen die Kandidatenstufe der beiden Signalmodalitäten und
liegen damit im Zuschnitt von 2.4. Vier davon (perzeptueller Hash, Dwells,
Anchor, Mindestdauer) tragen zusammen die gesamte Video-Kandidatenstufe: In der
jetzigen Fassung könnte Kapitel 3 zu `video.py` auf keinen einzigen in 2.4
eingeführten Begriff zurückgreifen, außer auf den der Unähnlichkeit selbst.

Bemerkenswert ist die vierte Zeile: 2.4.1 benennt die Über-Segmentierung
ausdrücklich als Kernproblem der framebasierten Segmentierung, nennt aber keine
Gegenmaßnahme. Die Implementierung hat davon drei — Mindestdauer eines Dwells,
nachrückender Anchor, Non-Maximum-Suppression — und keine ist dort verankert.

---

## Textvorschläge

Alle Vorschläge sind im Duktus des bestehenden Kapitels gehalten. Fußnoten sind
als `\footnote{Vgl. ... , S. ??.}` notiert — bitte an dein Zitat-Makro anpassen.
Zur Belastbarkeit der Quellen siehe den letzten Abschnitt.

### 1. In 2.4.1: das Unähnlichkeitsmaß

Anzuhängen an den Absatz über die zweite Bauform („Die zweite vergleicht zwei
benachbarte Fenster …"), da das Kapitel die Unähnlichkeit bisher nur voraussetzt:

> Womit diese Unähnlichkeit gemessen wird, ist eine eigene Entscheidung. Für
> Bildfolgen sind perzeptuelle Hashverfahren verbreitet, die ein Einzelbild über
> die niederfrequenten Koeffizienten seiner diskreten Kosinustransformation auf
> eine kurze Bitfolge abbilden; die Hamming-Distanz zweier solcher Bitfolgen ist
> ein Maß für die wahrnehmbare Verschiedenheit der Bilder und bleibt gegenüber
> Kompressionsartefakten und geringfügigem Rauschen stabil.\footnote{Vgl. Zauner
> 2010, S. ??.} Wird der Hash nicht über das ganze Bild, sondern über die Kacheln
> eines Rasters gebildet, bleibt zusätzlich erhalten, an welcher Stelle sich
> etwas geändert hat — für Bildschirmoberflächen wesentlich, in denen eine
> bedeutsame Änderung häufig nur einen kleinen Ausschnitt betrifft, ein
> unbedeutender Vorgang wie ein sich öffnendes Menü dagegen einen großen.

### 2. In 2.4.1: Kandidaten aus Stabilität

Als eigener Absatz nach den beiden Bauformen:

> Eine dritte Lesart derselben Fensterreihe kehrt die Blickrichtung um: Statt die
> Änderung zu suchen, sucht sie die Ruhe. Zusammenhängende Fensterfolgen, deren
> Unähnlichkeit unter einem Schwellenwert bleibt und die eine Mindestdauer
> überschreiten, bilden die eingerasteten Zustände; die Übergänge sind das, was
> zwischen ihnen übrig bleibt. Für Bildschirmaufzeichnungen ist diese Form
> naheliegend, weil die Oberfläche zwischen zwei Bedienschritten unverändert
> stehen bleibt und ein solcher Stillstand zuverlässiger zu messen ist als der
> Übergang, der ihn beendet.\footnote{Vgl. Bao u. a. 2017, S. ??.} Die
> Mindestdauer wirkt dabei als erste Gegenmaßnahme gegen die Über-Segmentierung:
> Sie verwirft Ruhephasen, die zu kurz sind, um einen abgeschlossenen Zustand
> darzustellen.

### 3. In 2.4.1: Bezugszustand statt Nachbarfenster

Als Absatz vor dem Übergang zu 2.4.2 — er bereitet dessen Argument vor:

> Beide Bauformen vergleichen benachbarte Fenster, und ihre Entscheidungen sind
> damit ausschließlich lokal. Eine Änderung, die sich über mehrere Fenster
> erstreckt, zerfällt in ebenso viele Kandidaten. Die Shot Boundary Detection
> begegnet dem bei allmählichen Übergängen, indem nicht gegen das unmittelbare
> Vorgängerbild verglichen wird, sondern gegen ein festgehaltenes Bezugsbild, das
> erst mit einer akzeptierten Grenze nachrückt.\footnote{Vgl. Smeaton, Over und
> Doherty 2010, S. ??.} Der Bezugszustand macht die Folge der Entscheidungen
> voneinander abhängig: Was gegenüber dem zuletzt etablierten Zustand keine
> Änderung darstellt, erzeugt keinen Kandidaten, auch wenn es sich von seinem
> unmittelbaren Vorgänger unterscheidet. Das ist eine schwächere, lokal
> berechenbare Form dessen, was die Change-Point-Detection global leistet.

### 4. In 2.4.1: Unterdrückung redundanter Kandidaten

Als Absatz am Ende von 2.4.1, direkt nachdem die Über-Segmentierung als
Kernproblem benannt wurde — das Kapitel nennt bislang das Problem, aber keine
Antwort darauf:

> Wo Kandidaten unabhängig voneinander entstehen, beschreiben mehrere von ihnen
> dieselbe Änderung. Die übliche Gegenmaßnahme ist die Unterdrückung
> nicht-maximaler Antworten: Innerhalb einer Umgebung fester Breite überlebt nur
> der stärkste Kandidat, die schwächeren werden verworfen.\footnote{Vgl. Neubeck
> und Van Gool 2006, S. ??.} Sie ist nicht kosmetisch, sondern folgt aus der
> Bewertung: Ordnet diese Vorhersage und Referenz eins zu eins zu, so wird von
> einer Häufung um dieselbe wahre Grenze genau eine als Treffer gezählt und jede
> weitere als Fehlalarm. Eine unterlassene Unterdrückung senkt damit die
> Präzision, ohne den Recall zu erhöhen.

### 5. In 2.4.3 (semantische Verfahren): Ansage und Ausführung fallen auseinander

Anzuhängen an den Absatz über den Repräsentationswechsel (Spracherkennung →
Transkript → Segmentierung auf dem Text):

> Der Repräsentationswechsel verschiebt allerdings die Bezugsgröße: Das
> Transkript liegt auf der Zeitachse der Äußerung, die Handlungsgrenze auf der
> der Ausführung, und beide fallen nicht zusammen. In narrierten Anleitungsvideos
> ist die Ansage der gezeigten Handlung systematisch vor- oder nachgelagert, und
> der Versatz ist weder konstant noch klein; Verfahren, die Narration und
> Handlung aufeinander beziehen, behandeln ihn deshalb als eigenes
> Problem.\footnote{Vgl. Miech u. a. 2020, S. ??.}\footnote{Vgl. Alayrac u. a.
> 2016, S. ??.} Für die Segmentierung folgt daraus, dass eine Ansage keinen
> Zeitpunkt bestimmt, sondern ein Intervall aufspannt, innerhalb dessen die
> angekündigte Handlung abgeschlossen wird; geschlossen wird dieses Intervall
> durch die Ansage der nächsten Handlung. Eine auf dem Transkript gewonnene
> Grenze ist damit von vornherein von geringerer zeitlicher Auflösung als eine
> aus dem Bildstrom gewonnene — ein Umstand, der bei der Wahl des
> Toleranzbereichs aus Abschnitt ?? zu berücksichtigen ist.

### 6. In 2.4.3: die Mechanik der Modellbefragung

Der dritte Weg ist im Kapitel in zwei Sätzen abgehandelt. Die Implementierung
hängt an drei konkreten Eigenschaften der Befragung, die dort eingeführt sein
müssen:

> Wird ein vortrainiertes Modell als Urteilsinstanz eingesetzt, tritt die Form
> der Befragung selbst als methodische Größe hinzu. Die Reihenfolge, in der
> Alternativen vorgelegt werden, beeinflusst das Urteil messbar;\footnote{Vgl.
> Zheng u. a. 2023, S. ??.} bei einem Vorher-Nachher-Vergleich schlägt das
> unmittelbar durch und spricht dafür, beide Zustände in einer einzigen,
> räumlich festgelegten Darstellung zu übergeben statt als zwei getrennte
> Eingaben. Für Bild-Sprach-Modelle erhöht zudem eine explizite Markierung der zu
> beurteilenden Bildregion die Verlässlichkeit der Bezugnahme; das Verfahren ist
> als Set-of-Mark-Prompting beschrieben.\footnote{Vgl. Yang u. a. 2023, S. ??.}
> Schließlich lässt sich die Ausgabe während der Dekodierung auf ein vorgegebenes
> Schema einschränken, sodass ein maschinell weiterverarbeitbares Ergebnis nicht
> von der Formulierungsdisziplin des Modells abhängt.\footnote{Vgl. Willard und
> Louf 2023, S. ??.}
>
> Von Interesse ist dabei nicht allein die vergebene Kategorie, sondern auch die
> Sicherheit, mit der das Modell sie vergibt. Sprachmodelle können eine Konfidenz
> in Worten ausdrücken, die mit ihrer tatsächlichen Trefferquote zusammenhängt,
> wenn auch mit systematischer Überschätzung.\footnote{Vgl. Lin, Hilton und Evans
> 2022, S. ??.} Kategorie und Konfidenz zusammen ergeben eine Größe in [0,1] und
> damit die abgestufte Evidenz, die der folgende Abschnitt verlangt.

### 7. In 2.4.5: abgestufte Evidenz als Schnittstelle der beiden Stufen

Anzuhängen an den Schlussabsatz, vor dem Verweis auf die Klassifikation:

> Damit die semantische Stufe entscheiden kann, darf die vorgelagerte ihr die
> Entscheidung nicht abnehmen. Reicht die Kandidatenstufe bereits geschwellte
> Grenzen weiter, so kann die nachgelagerte Stufe daraus nur noch entfernen und
> nie ergänzen; ihr erreichbarer Recall wäre durch den Schwellenwert der
> vorgelagerten Stufe nach oben begrenzt. Die Kandidatenstufe gibt deshalb eine
> abgestufte Kurve über der Zeit aus, und die Kandidaten sind deren lokale
> Maxima, nicht ihre Schwellenüberschreitungen. Dieselbe Trennung findet sich in
> der Generic Event Boundary Detection, deren Modelle eine Grenzwahrscheinlichkeit
> je Zeitpunkt ausgeben, die erst nachgelagert in Zeitpunkte überführt
> wird.\footnote{Vgl. Shou u. a. 2021, S. ??.}

---

## Quellen

Die im Vorschautext bereits zitierten Werke sind wiederverwendet, wo sie tragen
(Smeaton u. a. 2010, Shou u. a. 2021, Zheng u. a. 2023). Neu hinzu kommen:

| Quelle | wofür | Belastbarkeit |
|---|---|---|
| Zauner 2010 | DCT-basierter perzeptueller Hash, Hamming-Distanz | ● Standardreferenz für pHash |
| Neubeck und Van Gool 2006 | Non-Maximum-Suppression | ● |
| Yang u. a. 2023 | Set-of-Mark-Prompting | ● |
| Willard und Louf 2023 | schema-beschränkte Dekodierung | ● |
| Lin, Hilton und Evans 2022 | verbalisierte Konfidenz und ihre Kalibrierung | ● ggf. zusätzlich Xiong u. a. 2024 |
| Miech u. a. 2020 | Versatz zwischen Narration und gezeigter Handlung | ● |
| Alayrac u. a. 2016 | dito, unüberwachte Ausrichtung in Anleitungsvideos | ● |
| Bao u. a. 2017 | stabile Frames in Bildschirmaufzeichnungen | ◐ Aussage am Paper prüfen |
| Smeaton, Over und Doherty 2010 | Bezugsbild bei allmählichen Übergängen | ◐ bereits zitiert, aber **diese** Aussage im Text verifizieren |

● = Werk und Zuordnung sind belastbar, Seitenzahl ergänzen.
◐ = Werk existiert, aber prüfe am Volltext, ob es genau die Aussage stützt, an
die es hier gehängt ist. Bei Smeaton u. a. geht es um die Frage, ob dort
tatsächlich der nachrückende Bezugsrahmen beschrieben wird und nicht nur
Twin-Comparison-Verfahren allgemein; falls nicht, ist Zhang, Kankanhalli und
Smoliar 1993 die wahrscheinlichere Primärquelle.

## Offen — außerhalb von 2.4 zu prüfen

Diese Mechanismen des Codes liegen außerhalb des Zuschnitts von 2.4. Ich konnte
die betreffenden Abschnitte nicht einsehen; die Liste ist als Prüfliste gedacht,
nicht als Mängelliste.

1. **Ereignisstrom (Abschnitt 2.5).** `events.py` gruppiert Eingabeereignisse zu
   Bursts, getrennt an Pausen ab 2 s, und bewertet jeden Burst mit der *Länge der
   Ruhe danach* (gesättigt bei 8 s). Zu prüfen ist, ob 2.5 zwei Dinge trägt: die
   Pausenschwelle als Segmentierungskriterium auf einer Ereignisfolge — die
   Schwelle ist aus der Schreibprozessforschung importiert und im Code mit
   Wengelin 2006 belegt, in der Web-Nutzungsanalyse findet sich dieselbe
   Konstruktion als Sitzungsrekonstruktion über eine Inaktivitätsschwelle
   (Spiliopoulou u. a. 2003) — und die Deutung, dass nicht das Ereignis die
   Grenze trägt, sondern die Ruhe danach, die als abgestuftes Maß dient. Wenn du
   mir 2.5 gibst, mache ich denselben Abgleich dort.
2. **Zusammenführung der Modalitäten.** Die Implementierung ist dreistufig:
   Kandidat → semantisches Urteil je Modalität → gelernte Fusion über die
   Modalitäten. 2.4.5 beschreibt die ersten beiden Stufen für Video und Audio;
   die dritte kann dort nicht stehen, weil sie die Ereignisse einschließt. Zu
   prüfen ist, ob sie nach 2.5 eingeführt wird — insbesondere die Unterscheidung
   von früher und später Fusion (Baltrušaitis, Ahuja und Morency 2019) und die
   Feststellung, dass nur die späte Form die Unabhängigkeit der Modalitäten
   erhält, die die Ablation voraussetzt.
3. **Ziel-/Mittel-Unterscheidung.** Beide Prompts (`video_scoring._SYSTEM`,
   `audio_scoring._SYSTEM`) trennen das erreichte Ziel vom Weg dorthin — ein
   geöffnetes Menü ist keine Grenze, der ausgewählte Menüpunkt schon. Das ist die
   partonomische Ereignisstruktur und sollte in der Definition der
   Handlungsgrenze verankert sein (Zacks und Tversky 2001), nicht erst im Prompt.
4. **Random Forest als Entscheider**, `class_weight="balanced"` gegen die starke
   Klassenungleichheit (Breiman 2001), sowie **Leave-one-session-out** als
   Validierungsschema.
5. **Shapley-Werte und Interaktionsindex** (Shapley 1953; Grabisch und Roubens
   1999) — laut `docs/modalitaetsbeitraege.md` vorhanden, vermutlich in Kapitel
   2.6 oder 3.
6. **Eins-zu-eins-Zuordnung von Vorhersage und Referenz** (Kuhn-Munkres) und der
   Toleranzbereich τ — im Text als „Abschnitt ??" referenziert, also vermutlich
   bereits vorhanden.

---

*Randnotiz, nicht Teil des Abgleichs:* Die untrainierte Gegenprobe
(`RuleFuser`) existiert im Code nicht; `docs/modalitaetsbeitraege.md` ist auf
den aktuellen Stand gebracht (OOB-kalibrierte Schwelle, zwei Kandidatenpool-
Designs, Kopplungskontrollen). Soll eine untrainierte Gegenprobe in die Arbeit,
muss sie neu implementiert werden.
