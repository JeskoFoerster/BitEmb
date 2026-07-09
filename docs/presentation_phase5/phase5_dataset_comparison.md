# Phase 5: Dataset-Vergleich

## Warum dieser Vergleich

Die bisherigen Phase-5-Ergebnisse basierten auf SciFact mit `n = 1000`. Für einen sauberen Vergleich zwischen SciFact, FiQA und TREC-COVID muss man zwei Effekte trennen:

- **Dataset-Inhalt:** Welche Texte und Embeddings werden gemessen?
- **Korpusgröße:** Wie viele Vektoren werden verarbeitet?

Für Speicher und Laufzeit ist die Korpusgröße in Phase 5 deutlich wichtiger als der konkrete Dataset-Inhalt. Deshalb wurde zuerst ein kontrollierter Vergleich mit gleicher Dokumentanzahl durchgeführt.

## Durchgeführter Vergleich

Gemessen wurden alle drei Datensätze mit gleicher Stichprobengröße:

| Einstellung | Wert |
|---|---:|
| Datensätze | SciFact, FiQA, TREC-COVID |
| Dokumente pro Datensatz | 1.000 |
| Dimensionen | 64 und 768 |
| Pairwise-Stichprobe | 5.000 zufällige Dokumentpaare |
| Top-k | `k = 10`, brute-force über den gesamten jeweiligen Ausschnitt |
| Implementierung | Python/NumPy, ohne projektspezifischen C-Build |

Die Ergebnisse liegen unter `results/phase5/scaling_dim/`.

## Zentrale Beobachtungen

### Speicher

Der Speicherbedarf hängt bei gleicher Dokumentanzahl und gleicher Dimension praktisch nicht vom Dataset ab. SciFact, FiQA und TREC-COVID erzeugen bei `n = 1000` dieselben Speichergrößen, weil alle drei dieselbe Anzahl von Vektoren mit derselben Dimensionalität verwenden.

Damit misst dieser Teil vor allem den Effekt der Repräsentation:

| Repräsentation | 768 Dimensionen, Bytes pro Vektor | Verhältnis zu Float32 |
|---|---:|---:|
| Float32 | 3072,0 | 1,0x |
| 4-bit | 396,3 | 7,8x kleiner |
| 2-bit | 204,3 | 15,0x kleiner |
| 1-bit | 96,0 | 32,0x kleiner |

Der kleine Unterschied zwischen theoretischen 4-bit/2-bit-Werten und NumPy-Werten entsteht durch `col_min`/`col_max`-Metadaten für die Dequantisierung.

### Pairwise-Distanzen

Bei der Distanzmessung über zufällige Paare ist die Rangfolge über alle drei Datensätze hinweg weitgehend gleich:

| Rang | Variante | Interpretation |
|---:|---|---|
| 1 | 1-bit | schnellste Paar-Distanz durch XOR und Popcount |
| 2 | Float32 | sehr effiziente NumPy-Vektoroperationen |
| 3 | 4-bit / 2-bit | langsamer wegen Entpacken und Dequantisierung |

Das ist wichtig: Weniger Speicher bedeutet hier nicht automatisch weniger Laufzeit. Die gepackten 2-bit- und 4-bit-Daten müssen vor oder während der Distanzberechnung interpretiert werden. Dieser Zusatzaufwand kann den Vorteil der kleineren Datenmenge überdecken.

### Top-k-Suche

Bei brute-force Top-k ist die Rangfolge anders:

| Rang | Variante | Interpretation |
|---:|---|---|
| 1 | Float32 | Matrixmultiplikation ist in NumPy/BLAS sehr stark optimiert |
| 2 | 2-bit / 4-bit | Speicher kleiner, aber Dequantisierung kostet Zeit |
| 3 | 1-bit | in dieser Python/NumPy-Implementierung langsam bei vollständiger Top-k-Suche |

Der Unterschied zwischen Pairwise und Top-k ist methodisch entscheidend. Pairwise misst eine feste Anzahl einzelner Distanzberechnungen. Top-k durchsucht dagegen für jede Query den gesamten Korpus-Ausschnitt. Dadurch dominieren Speicherzugriff, Batch-Verarbeitung, Dequantisierung und `argpartition`.

## Lohnt sich der Vergleich aller drei Datensätze?

Ja, aber mit klarer Interpretation:

- **Für Speicher** liefert der Vergleich keine neuen qualitativen Erkenntnisse, solange `n` und `dim` gleich sind. Der Speicher ist dann deterministisch durch Repräsentation, Dimension und Vektoranzahl bestimmt.
- **Für Laufzeit** lohnt sich der Vergleich als Robustheitsprüfung. Die absolute Laufzeit schwankt leicht zwischen Datensätzen, die relative Rangfolge der Repräsentationen bleibt aber stabil.
- **Für Skalierung** ist ein Vergleich mit unterschiedlichen Korpusgrößen aussagekräftiger als ein Vergleich mit gleicher Größe. Genau dafür wurde zusätzlich `results/phase5/scaling_n/` erzeugt.

## Fazit

Der kontrollierte Dataset-Vergleich zeigt, dass Phase 5 primär Struktur- und Größenunterschiede der Repräsentationen misst, nicht inhaltliche Besonderheiten einzelner Datensätze. Für die Arbeit ist deshalb folgende Formulierung belastbar:

> Bei gleicher Korpusgröße verhalten sich SciFact, FiQA und TREC-COVID in Phase 5 qualitativ ähnlich. Die Effizienzunterschiede werden hauptsächlich durch Bit-Tiefe, Dimension und Anzahl der Vektoren bestimmt. Der konkrete Dataset-Inhalt beeinflusst die Laufzeit nur sekundär.
