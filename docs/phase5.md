# Phase 5: Laufzeit- und Speicheranalyse

## Ziel

Phase 2 und Phase 3 haben gezeigt, wie gut die komprimierten Repräsentationen
die Struktur des Float-Raums erhalten. Phase 5 stellt die praktische
Effizienzfrage:

**Wie viel Speicher und Laufzeit sparen 1-Bit, 2-Bit und 4-Bit gegenüber
Float32 im effizientesten praktisch vertretbaren NumPy-vektorisierten Suchpfad, und welcher
Qualitätsverlust entsteht dafür?**

Damit wird aus den bisherigen Qualitätsmessungen eine Kosten-Nutzen-Analyse.
Eine Kombination aus Dimension und Bittiefe ist nur dann praktisch interessant,
wenn sie nicht nur gute Qualität liefert, sondern auch im realistischen
Speicherlayout und im NumPy-vektorisierten Suchpfad messbar Speicher oder Rechenzeit spart.

Wichtig ist die methodische Trennung: Phase 2 und Phase 3 bewerten die
geometrische Qualität. Phase 5 bewertet Effizienz. Eine schnellere
Implementierung ist nicht automatisch qualitativ besser, und eine hohe
Qualitätsmetrik ist kein Beleg für gute Laufzeit.

---

## Versuchsaufbau

### Vier native Repräsentationen

Phase 5 misst jede Repräsentation einzeln:

- **Float32:** unkomprimierte Referenz mit nativer Cosine-/Dot-Product-Berechnung
- **4-Bit:** gepackte 4-Bit-Codes mit nativer Distanz- und Top-k-Berechnung
- **2-Bit:** gepackte 2-Bit-Codes mit nativer Distanz- und Top-k-Berechnung
- **1-Bit:** gepackte Binärcodes mit nativer Hamming- und Top-k-Berechnung

Für jede Repräsentation werden zwei Effizienzbereiche untersucht:

1. **Geschwindigkeit:** reine native Rechenzeit für Pairwise-Distanzen und Top-k-Suche
2. **Speicherbedarf:** tatsächlicher Speicherbedarf des fertigen Index einschließlich Metadaten

Beide Bereiche werden zuerst theoretisch berechnet und danach praktisch
gemessen. Dadurch wird sichtbar, was unter idealen Annahmen möglich sein sollte
und was die konkrete NumPy-Implementierung tatsächlich erreicht.

### Gleiche Versuchsmatrix wie in Phase 2 und Phase 3

Die Effizienzanalyse nutzt dieselbe Matrix wie die vorherigen Phasen:

- **Bittiefe:** Float32, 4-Bit, 2-Bit, 1-Bit
- **Dimensionen:** 64, 128, 256, 384, 768
- **Datasets:** SciFact, FiQA, TREC-COVID

Dadurch können Speicher, Laufzeit und Qualität direkt gemeinsam betrachtet
werden. Jede Kombination aus Bittiefe und Dimension hat bereits Qualitätswerte
aus Phase 2 und Phase 3. Phase 5 ergänzt dazu die Effizienzdaten.

### Kontrollierte Korpusgrößen

Für Laufzeitmessungen sollten feste Korpusgrößen verwendet werden:

- 1.000 Dokumente
- 5.000 Dokumente
- 10.000 Dokumente
- optional die volle Datasetgröße, wenn die Messung praktikabel ist

Das ist notwendig, weil Laufzeit und Peak-Speicher stark mit der Anzahl der
Dokumente skalieren. Ein einzelner Messpunkt auf 5.000 Dokumenten reicht nicht,
um Skalierung zu beurteilen.

### Nicht zur Kernmessung zählen

Nicht zur Kernlaufzeit zählen:

- Dataset-Download
- Dataset-Parsing
- Embedding-Modell-Encoding
- Plotting
- JSON-Schreiben
- erstmaliges Laden großer Modelle

Diese Schritte gehören zur Gesamtpipeline, aber nicht zur Effizienz der
komprimierten Repräsentation.

---

## Speicheranalyse

### Theoretischer Speicherbedarf

Der theoretische Speicherbedarf wird für Float32, 4-Bit, 2-Bit und 1-Bit aus
Dimension, Bittiefe und Korpusgröße berechnet:

```text
bits_per_vector = dim * bit_depth
bytes_per_vector = bits_per_vector / 8
total_bytes = n_vectors * bytes_per_vector
```

Die Float32-Baseline mit 768 Dimensionen benötigt:

```text
768 * 32 bit = 24.576 bit = 3.072 byte pro Vektor
```

Beispiele:

| Repräsentation | Bits pro Vektor | Bytes pro Vektor | Kompression vs. 768d Float32 |
|----------------|----------------:|-----------------:|------------------------------:|
| Float32, 768d | 24.576 | 3.072 | 1x |
| 4-Bit, 768d | 3.072 | 384 | 8x |
| 2-Bit, 768d | 1.536 | 192 | 16x |
| 1-Bit, 768d | 768 | 96 | 32x |
| 4-Bit, 384d | 1.536 | 192 | 16x |
| 2-Bit, 384d | 768 | 96 | 32x |
| 1-Bit, 384d | 384 | 48 | 64x |
| 1-Bit, 64d | 64 | 8 | 384x |

Diese Werte sind ideale Untergrenzen. Sie gelten nur, wenn die Codes tatsächlich
bitgepackt gespeichert werden.

### Praktischer Speicherbedarf

Nach der theoretischen Berechnung wird der tatsächliche Speicherbedarf der
NumPy-Implementierung gemessen.

Für jede Repräsentation werden getrennt berichtet:

1. **Theoretischer Code-Speicher:** Speicher bei idealer Bitpackung.
2. **NumPy-Layout-Speicher:** tatsächlich belegter Speicher des fertigen NumPy-Layout, inklusive Metadaten.
3. **Peak-Working-Memory:** maximaler Arbeitsspeicher während Pairwise- oder Top-k-Operationen.
4. **NumPy-Payload-Speicher:** Speicher der aktuellen Python/NumPy-Implementierung als Referenz, nicht als Hauptwert.

Der Hauptwert für die Speicheranalyse ist der NumPy-Layout-Speicher. Das aktuelle
`uint8`-Layout für 2-Bit und 4-Bit wird nur als Zwischenstand dokumentiert, weil
es nicht die effizienteste Speicherung darstellt.

---

## Laufzeitanalyse

### Theoretische Geschwindigkeitsanalyse

Vor der praktischen Messung wird für jede Repräsentation abgeschätzt, welche
Arbeit pro Distanzvergleich anfällt:

- Float32: Float-Multiplikationen und Additionen pro Dimension
- 4-Bit: Code-Loads, Entpackoperationen, Lookup- oder Integer-Operationen pro Dimension
- 2-Bit: Code-Loads, Entpackoperationen, Lookup- oder Integer-Operationen pro Dimension
- 1-Bit: XOR- und Popcount-Operationen pro gepacktem Maschinenwort

Diese theoretische Analyse liefert keine echte Laufzeit, erklärt aber, warum
bestimmte Repräsentationen schneller oder langsamer sein sollten.

### Praktische Pairwise-Distanzen

Pairwise-Distanzen sind der kontrollierte Microbenchmark. Gemessen wird für eine
feste Menge identischer Dokumentpaare:

- Float32: native Cosine-Distanz auf `float32`-Arrays
- 4-Bit: native Distanz direkt auf gepackten 4-Bit-Codes
- 2-Bit: native Distanz direkt auf gepackten 2-Bit-Codes
- 1-Bit: native Hamming-Distanz direkt auf gepackten Bits

Diese Messung beantwortet:

**Wie schnell kann jede Repräsentation Distanzen für bekannte Paare berechnen?**

### Praktische Top-k-Suche

Top-k-Suche ist der retrieval-nähere Effizienzbenchmark. Gemessen wird für
identische Query-Indizes und Korpusgrößen:

- Float32: NumPy-vektorisierte Top-k-Suche per Cosine-Similarity auf `float32`-Arrays
- 4-Bit: NumPy-vektorisierte Top-k-Suche direkt auf gepackten 4-Bit-Codes
- 2-Bit: NumPy-vektorisierte Top-k-Suche direkt auf gepackten 2-Bit-Codes
- 1-Bit: NumPy-vektorisierte Top-k-Suche per Hamming-Distanz direkt auf gepackten Bits

Diese Messung beantwortet:

**Wie schnell kann jede Repräsentation die nächsten Nachbarn finden?**

Die Top-k-Messung soll keine dequantisierten Vollmatrizen als Hauptpfad nutzen.
Temporäre Scores pro Query oder Batch sind erlaubt, aber der Index selbst bleibt
gepackt.

### Indexaufbau und Umwandlungszeiten

Getrennt von der Suchlaufzeit werden Vorverarbeitungsschritte gemessen:

- PCA-Transformation
- TurboQuant-Encoding
- Binarisierung
- 1-Bit-, 2-Bit- und 4-Bit-Packing
- optional Dequantisierung als Diagnosewert

Diese Werte werden getrennt von der Query-Latenz berichtet. Encoding ist
Indexaufbau, nicht Suche auf einem fertigen Index.

---

## Die Effizienzmetriken

### Speicher pro Vektor

```text
bytes_per_vector = total_index_bytes / n_vectors
```

### Kompressionsfaktor

```text
compression_ratio = baseline_float32_bytes / compressed_bytes
```

Berichtet werden:

- theoretischer Kompressionsfaktor bei idealer Bitpackung
- NumPy-Layout-Kompression mit echter Bitpackung
- NumPy-Payload-Kompression als Referenz

### Peak-Working-Memory

Peak-Working-Memory misst den maximalen Speicherbedarf während einer Operation.
Diese Metrik ist besonders wichtig für Top-k, weil temporäre Score- oder
Distanzstrukturen größer sein können als der eigentliche Index.

### Median-Laufzeit

```text
median_ms = median(measurement_runs)
```

Der Median ist robuster als eine Einzelmessung und weniger empfindlich gegen
kurze Systemstörungen.

### Durchsatz

Für Pairwise-Distanzen:

```text
pairs_per_second = n_pairs / runtime_seconds
```

Für Top-k-Suche:

```text
queries_per_second = n_queries / runtime_seconds
```

---

## Native Implementierung

### Warum ein nativer gepackter Kern nötig ist

Python/NumPy ist für Qualitätsmetriken akzeptabel, aber für die effizienteste
Laufzeitmessung nicht ausreichend. Python-Orchestrierung, NumPy-Allokationen und
Dequantisierung können die Messung überlagern.

Deshalb ist ein vektorisierter NumPy-Kern vorgesehen. Python bleibt die
Referenzimplementierung und steuert die Experimente. C übernimmt die
zeitkritischen Operationen auf den gepackten Codes.

Native Hauptfunktionen:

- Float32-Cosine als native Referenz
- Hamming-Distanzen direkt auf gepackten 1-Bit-Vektoren
- Distanzfunktionen direkt auf gepackten 2-Bit- und 4-Bit-Codes
- Top-k-Suche auf Float32-, 1-Bit-, 2-Bit- und 4-Bit-Indizes

Der native Hauptpfad darf 2-Bit- und 4-Bit-Codes nicht vollständig
dequantisieren. Falls Lookup-Tabellen oder vorab berechnete Skalierungen
verwendet werden, werden sie als Teil der nativen Methode dokumentiert und gegen
die Python-Referenz validiert.

Die C-Ergebnisse müssen gegen die Python-Referenz validiert werden:

- Hamming-Distanzen exakt gleich
- 2-Bit-/4-Bit-Distanzen numerisch gleich oder mit dokumentierter Approximation
- Float-Distanzen numerisch gleich innerhalb dokumentierter Toleranz
- Top-k-Ergebnisse gleich oder mit dokumentierten Tie-Breaking-Unterschieden

---

## Messprotokoll

Jede theoretische und praktische Messung sollte dokumentieren:

- Dataset
- Anzahl Dokumente
- Dimension
- Repräsentation (`float32`, `4bit`, `2bit`, `1bit`)
- Operation (`memory`, `pairwise_distance`, `top_k`, `index_build`)
- Messart (`theoretical`, `practical`)
- Implementierung (`numpy_vectorized`, optional `numpy_vectorized`, optional `dequantized_baseline`)
- Speicherlayout
- Anzahl Paare oder Queries
- k-Wert bei Top-k-Suche
- Warmup-Läufe
- Mess-Läufe
- Median
- Minimum
- Mittelwert
- Standardabweichung oder Interquartilsabstand
- CPU-Modell
- RAM
- Betriebssystem
- Python- und NumPy-Version
- Compiler und Compiler-Flags bei nativer Messung

Empfohlen:

- 3 Warmup-Läufe
- 10 Mess-Läufe
- Median als Hauptwert

Alle Varianten müssen mit denselben Inputs gemessen werden:

- gleicher Dokument-Subsample
- gleiche Paare
- gleiche Query-Indizes
- gleiche PCA-Projektion
- gleiche quantisierte Codes
- gleiches Speicherlayout pro gemessener Implementierung
- gleicher Seed

---

## Ergebnisdateien und Grafiken

### Ergebnisdateien

```text
results/
  phase5/
    memory.json
    runtime.json
    figures/
```

`memory.json` enthält für Float32, 4-Bit, 2-Bit und 1-Bit jeweils theoretischen
Speicher, NumPy-Layoutspeicher, NumPy-Payload-Speicher und Peak-Working-Memory.

`runtime.json` enthält für Float32, 4-Bit, 2-Bit und 1-Bit jeweils theoretische
Arbeitsabschätzungen sowie praktisch gemessene NumPy-Laufzeiten und
Durchsatzwerte für Pairwise-Distanzen und Top-k-Suche. Der Hauptwert ist
`numpy_vectorized`; Python/NumPy und dequantisierte Varianten sind Referenz- bzw.
Baseline-Werte.

### Speicher-Grafiken

> **Grafik `memory_theoretical_vs_numpy.pdf`:** Vergleich zwischen
> theoretischem und nativ gemessenem Speicherbedarf pro Repräsentation.

> **Grafik `memory_compression_by_dim.pdf`:** Kompressionsfaktor als Funktion der
> Dimension und Bittiefe.

### Laufzeit-Grafiken

> **Grafik `runtime_pairs_by_dim.pdf`:** Native Pairwise-Distanzlaufzeit nach
> Dimension und Bittiefe.

> **Grafik `runtime_knn_by_n.pdf`:** Native Top-k-Laufzeit als Funktion der
> Korpusgröße.

> **Grafik `throughput_by_representation.pdf`:** Durchsatz in Paaren oder
> Queries pro Sekunde.

### Pareto-Fronten

> **Grafik `quality_memory_pareto.pdf`:** Qualität gegen nativen
> Indexspeicherbedarf.

> **Grafik `quality_runtime_pareto.pdf`:** Qualität gegen NumPy-Laufzeit.

> **Grafik `quality_efficiency_pareto.pdf`:** Gemeinsame Betrachtung von
> Qualität, Speicher und Laufzeit.

---

## Methodische Risiken

### Theoretischer Speicher ist nicht realer Speicher

Theoretische Bitzahlen sind nur Untergrenzen. Entscheidend für praktische
Aussagen ist der NumPy-Layout-Speicher mit tatsächlicher Bitpackung.

Gegenmaßnahme: theoretische, native und NumPy-Kompressionsfaktoren getrennt
berichten.

### Theoretische Geschwindigkeit ist keine echte Laufzeit

Die theoretische Arbeitsabschätzung erklärt erwartete Unterschiede, ersetzt aber
keine Messung auf realer Hardware.

Gegenmaßnahme: theoretische Analyse immer mit praktischer nativer Messung
kombinieren.

### Dequantisierung ist kein gültiger Hauptpfad für Effizienz

Wenn TurboQuant vor jeder Distanzmessung voll dequantisiert wird, misst man
nicht die effiziente Suche auf komprimierten Codes. Eine solche Messung ist als
Baseline erlaubt, aber nicht als Hauptaussage zur Effizienz.

Gegenmaßnahme: Der Hauptpfad für 2-Bit und 4-Bit nutzt gepackte Codes und native
Distanzberechnung ohne vollständige Rekonstruktion der Vektoren.

### Python misst nicht reine Algorithmuszeit

Python/NumPy-Laufzeiten enthalten Speicherallokationen und Bibliotheksverhalten.

Gegenmaßnahme: Python/NumPy als Referenz ausweisen und einen validierten nativen
Kern separat messen.

### Temporäre Matrizen können den Speicher dominieren

Bei Top-k entstehen je nach Implementierung große Strukturen der Form
`batch_size * n_vectors`. Diese können größer sein als der eigentliche Index.

Gegenmaßnahme: Indexspeicher und Peak-Working-Memory getrennt messen und die
Batchgröße dokumentieren.

---

## Umsetzungsschritte

### Schritt 1: Theoretische Speicher- und Geschwindigkeitsanalyse

Aufgaben:

- theoretische Speicherwerte für Float32, 4-Bit, 2-Bit und 1-Bit berechnen
- theoretische Operationen pro Distanzvergleich abschätzen
- theoretische Skalierung für Pairwise-Distanzen und Top-k-Suche beschreiben
- erwartete Kompressionsfaktoren berechnen

Dieser Schritt benötigt keinen Compiler und liefert die Erwartungswerte für die
praktische Messung.

### Schritt 2: Native Speicherlayouts aufbauen

Aufgaben:

- Float32-Indexlayout festlegen
- 1-Bit-Packing als bestehende Baseline prüfen
- 4-Bit-Packing implementieren
- 2-Bit-Packing implementieren
- Roundtrip-Tests schreiben
- NumPy-Layoutgröße für Float32, 4-Bit, 2-Bit und 1-Bit messen

### Schritt 3: Praktische native Speicheranalyse

Aufgaben:

- NumPy-Layoutspeicher pro Repräsentation messen
- Metadaten wie `col_min` und `col_max` berücksichtigen
- Peak-Working-Memory für Distanz- und Top-k-Operationen messen
- `results/phase5/memory.json` mit praktischen Messwerten erzeugen

### Schritt 4: Native C-Kernfunktionen

Aufgaben:

- NumPy-Laufzeitpfad verwenden
- Distanzen auf Float32, 1-Bit, 2-Bit und 4-Bit implementieren
- Top-k-Suche auf Float32, 1-Bit, 2-Bit und 4-Bit implementieren
- Tests gegen Python-Referenz schreiben
- Runtime-Benchmark ueber `numpy_vectorized` ausfuehren

Voraussetzung ist ein C-Compiler.

### Schritt 5: Praktische native Geschwindigkeitsanalyse

Aufgaben:

- NumPy-vektorisierte Pairwise-Distanzen für Float32, 4-Bit, 2-Bit und 1-Bit messen
- NumPy-vektorisierte Top-k-Suche für Float32, 4-Bit, 2-Bit und 1-Bit messen
- Durchsatzwerte berechnen
- `results/phase5/runtime.json` erzeugen

### Schritt 6: Python/NumPy- und Dequantisierungsbaselines ergänzen

Die bestehenden Python/NumPy-Funktionen werden zusätzlich gemessen, aber nur als
Referenz und zur Einordnung.

Aufgaben:

- Pairwise-Distanzfunktionen der bestehenden Implementierung messen
- Top-k-Funktionen der bestehenden Implementierung messen
- Dequantisierungskosten separat messen
- Ergebnisse als `numpy_vectorized` und `dequantized_baseline` speichern

Diese Werte sind nicht die finale Effizienzaussage, sondern zeigen, wie stark
die naive oder bequeme Implementierung vom nativen Hauptpfad abweicht.

---

## Zusammenfassung: Was diese Phase beantwortet

| Frage | Metrik |
|-------|--------|
| Wie viel Speicher benötigt jede Repräsentation theoretisch? | Theoretische Bytes pro Vektor |
| Wie viel Speicher benötigt jede Repräsentation praktisch im NumPy-Layout? | NumPy-Layout-Speicher |
| Wie viel RAM wird während der Suche benötigt? | Peak-Working-Memory |
| Wie schnell sollten die Repräsentationen theoretisch sein? | Operationen pro Vergleich |
| Wie schnell sind native Distanzberechnungen praktisch? | Median-Laufzeit, Paare/s |
| Wie schnell ist native brute-force Top-k-Suche praktisch? | Median-Laufzeit, Queries/s |
| Welche Kombination ist optimal für ein Speicherbudget? | Quality-Memory-Pareto |
| Welche Kombination ist optimal für ein Laufzeitbudget? | Quality-Runtime-Pareto |
| Welche Kombination ist insgesamt nicht dominiert? | Gemeinsame Pareto-Front |
