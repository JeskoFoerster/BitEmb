# Phase 5: Ergebniszusammenfassung

## Ziel der Phase

Phase 5 bewertet die praktische Effizienz der untersuchten Embedding-Repräsentationen. Im Unterschied zu Phase 2 und Phase 3 steht nicht die geometrische oder retrieval-bezogene Qualität im Vordergrund, sondern die Frage:

> Wie stark reduzieren 1-bit, 2-bit und 4-bit den Speicherbedarf, und wie verhalten sich diese Repräsentationen in NumPy-basierten Laufzeitmessungen?

Die aktuelle Phase-5-Auswertung nutzt alle drei betrachteten BEIR-Subsets: SciFact, FiQA und TREC-COVID. Für den Skalierungsvergleich wurden kontrollierte Korpusgrößen von `250`, `500` und `1000` Dokumenten gemessen. Dadurch lässt sich der Einfluss der Korpusgröße sauberer untersuchen als bei einem reinen Vergleich der vollständigen Datensätze.

## Messaufbau

| Einstellung | Aktueller Stand |
|---|---:|
| Datensätze | SciFact, FiQA, TREC-COVID |
| Korpusgrößen | 250, 500, 1000 Dokumente |
| Dimensionen | 64 für alle Größen, 768 zusätzlich bei 1000 Dokumenten |
| Pairwise-Messung | 3000 zufällige Dokumentpaare |
| Top-k-Messung | brute-force Suche mit `k = 10` |
| Wiederholungen | 1 Warmup, 3 Messläufe |
| Implementierung | Python/NumPy, ohne projektspezifischen C-Build |

Die Messung arbeitet bewusst mit kontrollierten Ausschnitten. Es wurden also nicht die vollständigen Korpora von SciFact, FiQA und TREC-COVID für alle Messpunkte verwendet. Der Vorteil ist, dass die Datensätze direkt vergleichbar sind und die Skalierung von `n = 250` bis `n = 1000` sichtbar wird.

## Gemessene Varianten

| Repräsentation | Bedeutung | Speicherlayout |
|---|---|---|
| `float32` | unkomprimierte Referenz | zusammenhängende NumPy-Float32-Matrix |
| `4bit` | TurboQuant mit 4-bit Codes | gepackte 4-bit Codes plus `col_min`/`col_max` |
| `2bit` | TurboQuant mit 2-bit Codes | gepackte 2-bit Codes plus `col_min`/`col_max` |
| `1bit` | binäre Vorzeichen-Codes | `np.packbits`-kompatible Bitvektoren |

Wichtig: Die aktuelle Implementierung ist einheitlich Python/NumPy. Es gibt keinen projektspezifischen C-Build mehr. Die rechenintensiven Operationen laufen über vektorisierte NumPy-Operationen.

## Ergebnisdateien und Graphen

| Datei | Inhalt | Aussage |
|---|---|---|
| `results/phase5/scaling_n/memory.json` | Speicherwerte pro Datensatz, Korpusgröße, Dimension und Repräsentation | Wie stark wird das Speicherlayout komprimiert? |
| `results/phase5/scaling_n/runtime.json` | Median-Laufzeiten und Durchsatz | Wie schnell sind Pairwise- und Top-k-Operationen? |
| `results/phase5/scaling_n/figures/memory_theoretical_vs_numpy.pdf` | theoretischer vs. praktischer Speicher | Wie nah liegt NumPy am idealen Speicherbedarf? |
| `results/phase5/scaling_n/figures/memory_compression_by_dim.pdf` | Kompression nach Dimension und Repräsentation | Wie wirken Dimensionsreduktion und Quantisierung zusammen? |
| `results/phase5/scaling_n/figures/runtime_pairs_by_dim.pdf` | Pairwise-Distanzlaufzeit | Wie schnell sind einzelne Distanzberechnungen? |
| `results/phase5/scaling_n/figures/runtime_knn_by_dim.pdf` | brute-force Top-k-Laufzeit | Wie schnell ist die Suche über den jeweiligen Korpus-Ausschnitt? |

Ergänzende Auswertungen liegen in `docs/presentation_phase5/phase5_dataset_comparison.md` und `docs/presentation_phase5/phase5_scaling_comparison.md`.

## Speicherergebnisse

Die Speicherkompression ist der eindeutigste Effekt in Phase 5. Der Speicherbedarf skaliert linear mit der Anzahl der Vektoren. Bei gleicher Dimension und gleicher Korpusgröße sind die Speicherwerte für SciFact, FiQA und TREC-COVID praktisch identisch, weil sie nur von Vektoranzahl, Dimension und Repräsentation abhängen.

Bei 64 Dimensionen ergibt sich im aktuellen NumPy-Layout folgender Speicherbedarf pro Vektor:

| Repräsentation | 250 Dokumente | 500 Dokumente | 1000 Dokumente |
|---|---:|---:|---:|
| Float32 | 256,000 Byte | 256,000 Byte | 256,000 Byte |
| 4-bit | 36,096 Byte | 34,048 Byte | 33,024 Byte |
| 2-bit | 20,096 Byte | 18,048 Byte | 17,024 Byte |
| 1-bit | 8,000 Byte | 8,000 Byte | 8,000 Byte |

Bei 2-bit und 4-bit sinkt der praktische Wert pro Vektor leicht mit wachsendem `n`, weil die festen `col_min`/`col_max`-Metadaten auf mehr Vektoren verteilt werden.

Bei 768 Dimensionen und 1000 Dokumenten zeigt sich der Vollraum-Vergleich:

| Repräsentation | Bytes pro Vektor | Kompression gegenüber Float32 |
|---|---:|---:|
| Float32 | 3072,000 | 1,0x |
| 4-bit | 396,288 | 7,8x kleiner |
| 2-bit | 204,288 | 15,0x kleiner |
| 1-bit | 96,000 | 32,0x kleiner |

## Pairwise-Distanzen

Pairwise-Distanzen messen eine feste Anzahl bekannter Dokumentpaare. Im aktuellen Skalierungslauf sind das 3000 zufällige Paare. Deshalb skaliert diese Messung nicht stark mit der Korpusgröße, sondern vor allem mit Dimension und Rechenaufwand pro Distanz.

Die qualitative Rangfolge ist über die Datensätze hinweg stabil:

| Repräsentation | Beobachtung |
|---|---|
| `1bit` | meist schnellste Paar-Distanz durch XOR und Popcount |
| `float32` | sehr schnell durch vektorisierte NumPy-Operationen |
| `2bit` / `4bit` | langsamer, weil Entpacken und Dequantisierung zusätzlichen Aufwand erzeugen |

Bei 768 Dimensionen und 1000 Dokumenten lagen die Medianzeiten der Pairwise-Messung über die drei Datensätze ungefähr in diesen Bereichen:

| Repräsentation | Medianbereich |
|---|---:|
| `1bit` | 1,4 bis 2,3 ms |
| `float32` | 6,1 bis 11,0 ms |
| `4bit` | 20,6 bis 34,0 ms |
| `2bit` | 24,6 bis 36,4 ms |

Interpretation: Für einzelne Distanzvergleiche ist `1bit` sehr effizient. `2bit` und `4bit` sparen Speicher, sind in dieser NumPy-Implementierung aber nicht automatisch schneller als Float32.

## Top-k-Suche

Top-k ist die retrieval-nähere Messung. Für jeden Vektor wird brute-force nach den `k = 10` nächsten Nachbarn im jeweiligen Korpus-Ausschnitt gesucht. Dadurch wächst die Laufzeit sichtbar mit der Korpusgröße.

Bei 64 Dimensionen zeigen alle drei Datensätze denselben Trend:

| Korpusgröße | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 250 | ca. 1,5 bis 1,9 ms | ca. 1,4 bis 2,8 ms | ca. 1,6 bis 2,7 ms | ca. 3,9 bis 8,1 ms |
| 500 | ca. 3,9 bis 4,6 ms | ca. 4,8 bis 8,0 ms | ca. 4,7 bis 7,5 ms | ca. 15,5 bis 17,6 ms |
| 1000 | ca. 14,8 bis 19,9 ms | ca. 26,0 bis 34,8 ms | ca. 25,4 bis 39,7 ms | ca. 70,6 bis 71,9 ms |

Bei 768 Dimensionen und 1000 Dokumenten verschärft sich der Unterschied:

| Repräsentation | Medianbereich |
|---|---:|
| `float32` | 16,8 bis 23,1 ms |
| `2bit` | 46,1 bis 57,6 ms |
| `4bit` | 47,2 bis 62,6 ms |
| `1bit` | 452,8 bis 535,0 ms |

Interpretation: Float32 ist bei brute-force Top-k in der aktuellen NumPy-Implementierung am schnellsten, weil dichte Matrixmultiplikation und `argpartition` sehr gut optimiert sind. Die komprimierten Varianten reduzieren den Speicherbedarf deutlich, zahlen aber Laufzeitkosten durch Entpacken, Dequantisierung oder bitweise Distanzberechnung.

## Vergleich der Datensätze

Der Vergleich aller drei Datensätze lohnt sich vor allem als Robustheitsprüfung. Bei gleicher Korpusgröße und gleicher Dimension sind die Speicherwerte identisch oder nahezu identisch. Die Laufzeiten schwanken zwischen SciFact, FiQA und TREC-COVID, die qualitative Rangfolge der Repräsentationen bleibt aber stabil.

Damit ist die wichtigste methodische Aussage:

> In Phase 5 werden die Effizienzunterschiede hauptsächlich durch Bit-Tiefe, Dimension und Anzahl der Vektoren bestimmt. Der konkrete Dataset-Inhalt beeinflusst die Laufzeit nur sekundär.

## Wie die Graphen gelesen werden sollten

### `memory_theoretical_vs_numpy.pdf`

Dieser Graph zeigt pro Repräsentation den theoretischen Speicher und den praktisch gemessenen NumPy-Speicher.

- Niedriger ist besser.
- Float32 ist die unkomprimierte Referenz.
- 1-bit sollte am niedrigsten liegen.
- Kleine Abweichungen bei 2-bit und 4-bit entstehen durch Metadaten.

### `memory_compression_by_dim.pdf`

Dieser Graph zeigt die Kompressionsrate gegenüber `768d float32`.

- Höher ist besser.
- Kleinere Dimensionen erhöhen die Kompression zusätzlich.
- 1-bit liegt oben, Float32 unten.
- Der Graph kombiniert den Effekt von PCA und Quantisierung.

### `runtime_pairs_by_dim.pdf`

Dieser Graph zeigt die Laufzeit für die festen Pairwise-Distanzmessungen.

- Niedriger ist besser.
- 1-bit liegt typischerweise unten.
- 2-bit und 4-bit liegen höher, weil Entpacken und Skalieren teuer ist.
- Die Messung hängt eher an der Dimension und der Pair-Anzahl als an der gesamten Korpusgröße.

### `runtime_knn_by_dim.pdf`

Dieser Graph zeigt die brute-force Top-k-Suche.

- Niedriger ist besser.
- Float32 liegt in dieser Implementierung meist am besten.
- 1-bit wird besonders bei 768 Dimensionen langsam.
- Der Graph zeigt Implementierungskosten, nicht nur theoretische Bitkosten.

## Zentrale Aussagen

1. Quantisierung spart deutlich Speicher.
2. Die größte Speicherersparnis erreicht `1bit`, gefolgt von `2bit` und `4bit`.
3. Dimensionsreduktion und Quantisierung wirken gemeinsam: kleine Dimension plus niedrige Bittiefe erzeugt die stärkste Kompression.
4. Speicherersparnis bedeutet nicht automatisch Laufzeitgewinn.
5. Für Pairwise-Distanzen ist `1bit` am schnellsten oder sehr nah an der Spitze.
6. Für brute-force Top-k ist Float32 in NumPy am schnellsten.
7. 2-bit und 4-bit sind in dieser Implementierung primär Speicheroptimierungen, nicht automatisch Laufzeitoptimierungen.
8. Der Vergleich über SciFact, FiQA und TREC-COVID bestätigt die Stabilität dieser Trends.

## Fazit für die Präsentation

Phase 5 zeigt einen klaren Trade-off:

> Quantisierung ist sehr effektiv für Speicherreduktion, aber nicht automatisch für Laufzeitreduktion.

Die stärkste praktische Speicherkompression entsteht bei `1bit`. Für einzelne paarweise Distanzvergleiche ist `1bit` ebenfalls besonders effizient. Für die brute-force Top-k-Suche ist dagegen Float32 in NumPy am schnellsten, weil NumPy/BLAS für dichte Float-Matrixoperationen stark optimiert ist.

Damit gewinnt keine Repräsentation in allen Metriken. Die geeignete Variante hängt vom Ziel ab:

| Ziel | Geeignete Variante |
|---|---|
| maximal Speicher sparen | `1bit`, möglichst kleine Dimension |
| guter Speicher-Kompromiss | `2bit` oder `4bit` |
| schnelle Pairwise-Distanzen | `1bit` |
| schnelle brute-force Top-k-Suche in NumPy | `float32` |
| ausgewogene Bewertung | gemeinsam mit Phase 2 und Phase 3 betrachten |

## Verbindung zu Phase 2 und Phase 3

Phase 5 allein entscheidet nicht, welche Repräsentation insgesamt die beste ist. Sie muss mit den Qualitätsmetriken aus Phase 2 und Phase 3 kombiniert werden.

- Phase 2 beantwortet: Wie stark werden Distanzen verzerrt?
- Phase 3 beantwortet: Wie gut bleiben Nachbarschaften erhalten?
- Phase 5 beantwortet: Was kosten die Repräsentationen in Speicher und Laufzeit?

Eine sinnvolle Endbewertung entsteht erst aus allen drei Perspektiven: Qualität, Speicher und Laufzeit.
