# Phase 5: Skalierungsvergleich über alle Datensätze

## Messaufbau

Für den Skalierungsvergleich wurden alle drei BEIR-Subsets mit mehreren Korpusgrößen ausgewertet. Ziel war nicht die Qualitätsbewertung der Retrieval-Ergebnisse, sondern die praktische Effizienz der Repräsentationen bei wachsender Vektoranzahl.

| Einstellung | Wert |
|---|---:|
| Datensätze | SciFact, FiQA, TREC-COVID |
| Korpusgrößen | 250, 500, 1000 Dokumente |
| Dimensionen | 64 für alle Größen, 768 zusätzlich bei 1000 Dokumenten |
| Pairwise-Messung | 3000 zufällige Dokumentpaare |
| Top-k-Messung | brute-force Suche mit `k = 10` |
| Wiederholungen | 1 Warmup, 3 Messläufe |
| Implementierung | Python/NumPy, ohne projektspezifischen C-Build |

Die Ergebnisse liegen unter `results/phase5/scaling_n/`. Die zugehörigen Plots befinden sich in `results/phase5/scaling_n/figures/`.

## Speicherverhalten

Der Speicherbedarf skaliert linear mit der Anzahl der Vektoren. Bei 64 Dimensionen ergibt sich pro Vektor folgendes NumPy-Layout:

| Repräsentation | Bytes pro Vektor bei 250 | Bytes pro Vektor bei 500 | Bytes pro Vektor bei 1000 |
|---|---:|---:|---:|
| Float32 | 256,000 | 256,000 | 256,000 |
| 4-bit | 36,096 | 34,048 | 33,024 |
| 2-bit | 20,096 | 18,048 | 17,024 |
| 1-bit | 8,000 | 8,000 | 8,000 |

Die Bytes pro Vektor sinken bei 2-bit und 4-bit leicht, weil die festen `col_min`/`col_max`-Metadaten auf mehr Vektoren verteilt werden. Bei Float32 und 1-bit gibt es in dieser Messung keinen zusätzlichen Metadatenblock, daher bleibt der Wert konstant.

Bei 768 Dimensionen und 1000 Dokumenten zeigt sich der erwartete Vollraum-Vergleich:

| Repräsentation | Bytes pro Vektor | Verhältnis zu Float32 |
|---|---:|---:|
| Float32 | 3072,000 | 1,0x |
| 4-bit | 396,288 | 7,8x kleiner |
| 2-bit | 204,288 | 15,0x kleiner |
| 1-bit | 96,000 | 32,0x kleiner |

## Laufzeit: Pairwise-Distanzen

Die Pairwise-Messung verwendet immer 3000 zufällige Paare. Deshalb wächst diese Messung nicht stark mit der Korpusgröße, sondern vor allem mit der Dimension und dem Aufwand pro Distanz.

Bei 64 Dimensionen ist die qualitative Rangfolge über alle Datensätze ähnlich:

| Repräsentation | Beobachtung |
|---|---|
| 1-bit | meist schnellste oder sehr schnelle Paar-Distanz durch XOR und Popcount |
| Float32 | sehr schnell durch vektorisierte NumPy-Operationen |
| 2-bit / 4-bit | langsamer, weil Entpacken und Dequantisierung zusätzlichen Aufwand erzeugen |

Bei 768 Dimensionen wird der Unterschied deutlicher. Für 1000 Dokumente lagen die Medianzeiten der Pairwise-Messung ungefähr in diesen Bereichen:

| Repräsentation | Medianbereich über Datensätze |
|---|---:|
| 1-bit | 1,4 bis 2,3 ms |
| Float32 | 6,1 bis 11,0 ms |
| 4-bit | 20,6 bis 34,0 ms |
| 2-bit | 24,6 bis 36,4 ms |

Interpretation: Für einzelne Distanzberechnungen ist 1-bit besonders attraktiv. Die 2-bit- und 4-bit-Varianten sparen Speicher, sind in der aktuellen NumPy-Implementierung aber nicht automatisch schneller als Float32.

## Laufzeit: Top-k-Suche

Die Top-k-Messung durchsucht den gesamten Korpus-Ausschnitt. Dadurch wächst die Laufzeit sichtbar mit der Korpusgröße.

Bei 64 Dimensionen zeigt sich über alle drei Datensätze hinweg derselbe Trend:

| Korpusgröße | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 250 | ca. 1,5 bis 1,9 ms | ca. 1,4 bis 2,8 ms | ca. 1,6 bis 2,7 ms | ca. 3,9 bis 8,1 ms |
| 500 | ca. 3,9 bis 4,6 ms | ca. 4,8 bis 8,0 ms | ca. 4,7 bis 7,5 ms | ca. 15,5 bis 17,6 ms |
| 1000 | ca. 14,8 bis 19,9 ms | ca. 26,0 bis 34,8 ms | ca. 25,4 bis 39,7 ms | ca. 70,6 bis 71,9 ms |

Bei 768 Dimensionen und 1000 Dokumenten verschärft sich dieser Effekt:

| Repräsentation | Medianbereich über Datensätze |
|---|---:|
| Float32 | 16,8 bis 23,1 ms |
| 2-bit | 46,1 bis 57,6 ms |
| 4-bit | 47,2 bis 62,6 ms |
| 1-bit | 452,8 bis 535,0 ms |

Interpretation: Für vollständige brute-force Top-k-Suche profitiert Float32 stark von optimierter Matrixmultiplikation. Die komprimierten Varianten reduzieren zwar den Speicherbedarf, zahlen in dieser Implementierung aber Laufzeitkosten für Entpacken, Dequantisierung oder bitweise Distanzberechnung.

## Vergleich der Datensätze

Der Vergleich über SciFact, FiQA und TREC-COVID zeigt keine grundsätzlich andere Rangfolge der Repräsentationen. Unterschiede zwischen Datensätzen sind vorhanden, aber kleiner als die Unterschiede zwischen den Repräsentationen und Korpusgrößen.

Für Phase 5 bedeutet das:

- Die Korpusgröße ist der wichtigere Skalierungsfaktor als der konkrete Dataset-Inhalt.
- Speicherwerte sind bei gleicher Größe und Dimension praktisch identisch.
- Laufzeitwerte schwanken zwischen Datensätzen, die qualitative Interpretation bleibt aber stabil.
- Der Vergleich aller drei Datensätze lohnt sich vor allem als Robustheitsprüfung und zur Darstellung des Skalierungsverhaltens.

## Fazit

Die Skalierungsmessung stützt die bisherige Interpretation von Phase 5: Kompression reduziert den Speicherbedarf zuverlässig und stark, führt in der aktuellen Python/NumPy-Implementierung aber nicht automatisch zu schnellerer Suche. Besonders bei brute-force Top-k bleibt Float32 wegen hochoptimierter NumPy/BLAS-Pfade die schnellste Referenz. 1-bit ist für einzelne Paar-Distanzen sehr effizient, aber für vollständige Top-k-Suche in dieser Implementierung deutlich langsamer.
