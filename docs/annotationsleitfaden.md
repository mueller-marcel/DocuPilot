# Annotationsleitfaden — Wann ist eine Grenze zu setzen?

Selbsttragende, app-übergreifende Definition zum Setzen von Segmentgrenzen in
den aufgezeichneten Videos. Der Annotator entscheidet allein aus dem Video,
ohne Skript und ohne die Aufgabe zu kennen.

## Kerndefinition

Eine **Grenze** markiert den **Abschluss einer Nutzerhandlung**: den Moment, in
dem eine vom Nutzer ausgelöste Änderung in einem **neuen, bleibenden Zustand zur
Ruhe kommt**.

Eine Handlung ist abgeschlossen, wenn **alle drei** Bedingungen erfüllt sind:

- **A · Vom Nutzer ausgelöst.** Kurz davor gab es eine Nutzeraktion (Klick,
  Taste, Ziehen). *Ausnahme:* verzögerte Ergebnisse (Build, Test, Rendern,
  langsamer Filter) — der Auslöser liegt früher, das Ergebnis erscheint autonom
  später; gilt trotzdem als ausgelöst.
- **B · Neuer bleibender Zustand.** Danach zeigt der Bildschirm einen Zustand,
  der bestehen bleibt — kein Overlay, das sich wieder schließt, keine bloße
  Markierung, kein Zwischenstand beim Tippen.
- **C · Ein Ziel, kein Mittel.** Die Änderung ist das, worauf der Nutzer
  hinarbeitete, nicht ein Zwischenschritt dorthin. Beobachtbar: nach der
  Änderung hält der Nutzer inne (Pause / nächste Ansage / wendet sich einem
  anderen Objekt zu). Arbeitet er ohne Pause am selben Objekt weiter, war es ein
  Mittel — die Grenze kommt erst am Ende dieser fortgesetzten Handlung.

## Wo genau setzen

Auf den Frame, in dem das Ergebnis **zuerst erscheint und sich nicht mehr
ändert** (Einschwing-Moment). Nicht auf die Ansage, nicht auf den auslösenden
Klick.

## Zusammenfassen / Trennen

- **Eine** Grenze, wenn mehrere Änderungen in einem ununterbrochenen Schwung
  ohne Pause passieren (= eine Handlung).
- **Getrennte** Grenzen, wenn eine Pause / ein Einschwingen dazwischen liegt.

## Keine Grenze

- geöffnetes Menü / Dropdown / Dialog / Tooltip (Overlay, schließt sich wieder)
- Tippen, Scrollen, Ziehen *in Bewegung* (noch nicht fertig)
- Markierung / Auswahl / Zwischenablage (Laufrahmen) ohne Inhaltsänderung
- Navigation, die nur den Zielort der *nächsten* Handlung ansteuert (zur Zelle
  scrollen, zum Blatt wechseln, in einen Ordner navigieren, um dort zu arbeiten)
  — **außer** der Wechsel selbst ist das Ziel und der Nutzer hält danach inne
- automatische Änderungen ohne Auslöser (Benachrichtigung, Autospeichern-Hinweis)

## Vorgehen (Schritt für Schritt)

1. Abspielen, bis sich etwas ändert, das **bestehen bleibt** (kein Menü, kein
   Zwischentippen).
2. **Hat der Nutzer es ausgelöst?** Klick/Taste kurz davor — oder früherer
   Auslöser bei verzögertem Ergebnis. Nein → keine Grenze.
3. **Ziel oder Mittel?** Hält er danach inne / sagt den nächsten Schritt an /
   wendet sich anderem zu → Ziel. Arbeitet er ohne Pause am selben Objekt weiter
   → Mittel, warte auf den echten Abschluss.
4. Grenze auf den Frame setzen, in dem das Ergebnis erscheint und stillsteht.
5. Mehrere bleibende Änderungen ohne Pause hintereinander → nur **eine** Grenze
   am Ende des Schwungs.

## Kalibrier-Beispiele (app-übergreifend)

| ✅ Grenze setzen | ❌ Keine Grenze |
|---|---|
| Filter/Sortierung greift, Zeilen ordnen sich neu und stehen still | Dropdown/Filterdialog ist offen |
| Absatz getippt, Tippen stoppt, Text bleibt | Zeichen erscheinen noch beim Tippen |
| Neue Folie / neues Blatt / neuer Ordner erscheint | Zeile/Zelle nur markiert (Laufrahmen) |
| Datei verschoben, erscheint am neuen Ort | Zum Blatt wechseln, um dort einzufügen (Mittel) |
| Leseansicht aktiviert, bleibt, Nutzer hält inne | Zur Zelle scrollen vor dem Bearbeiten |
| Build fertig, Ausgabe/Fehler erscheinen (verzögert) | Tooltip / Hover-Vorschau; Systembenachrichtigung |

---

**Anker:** Die Grenze wird am visuellen Einschwingen platziert; über Ziel-vs-Mittel
entscheidet die *Bedeutung* der Handlung, nicht der Peak eines Signals. Die
Pause/Ansage ist nur Verständnishilfe. So bleibt die Ground Truth
modalitätsneutral.

**Handbuch:** Derselbe Satz schneidet die Handbuch-Schritte —
*ein Schritt = eine Handlung, die in einem bleibenden Zustand abschließt.*
