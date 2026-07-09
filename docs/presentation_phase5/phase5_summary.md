# Phase 5: Ergebniszusammenfassung

## Ziel der Phase

Phase 5 untersucht nicht mehr die geometrische Qualität der Quantisierung, sondern ihre praktische Effizienz. Die zentrale Frage lautet:

> Wie viel Speicher sparen 1-bit, 2-bit und 4-bit gegenüber Float32, und wie verhalten sich diese Repräsentationen in NumPy-basierten Laufzeitmessungen?

Die Messungen wurden für SciFact mit `n = 1000` Dokumenten durchgeführt. Für jede Dimension wurden 10.000 zufällige Dokumentpaare für Pairwise-Distanzen und eine brute-force Top-k-Suche mit `k = 10` gemessen.

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
| `results/phase5/memory.json` | Speicherwerte pro Dimension und Repräsentation | Wie stark wird der Index komprimiert? |
| `results/phase5/runtime.json` | Median-Laufzeiten und Durchsatz | Wie schnell sind Distanz- und Top-k-Operationen? |
| `results/phase5/figures/memory_theoretical_vs_numpy.pdf` | theoretischer vs. praktischer Speicher | Wie nah liegt NumPy am idealen Speicherbedarf? |
| `results/phase5/figures/memory_compression_by_dim.pdf` | Kompression über Dimensionen | Wie wirken Dimensionsreduktion und Quantisierung zusammen? |
| `results/phase5/figures/runtime_pairs_by_dim.pdf` | Pairwise-Distanzlaufzeit | Wie schnell sind einzelne Distanzberechnungen? |
| `results/phase5/figures/runtime_knn_by_dim.pdf` | brute-force Top-k-Laufzeit | Wie schnell ist die Suche über den Korpus? |

## Speicherergebnisse

Die Speicherkompression ist der stärkste und eindeutigste Effekt in Phase 5. Je kleiner die Dimension und je niedriger die Bittiefe, desto kleiner wird der Index.

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 12.0x | 93.0x | 180.5x | 384.0x |
| 128 | 6.0x | 46.5x | 90.2x | 192.0x |
| 256 | 3.0x | 23.3x | 45.1x | 96.0x |
| 384 | 2.0x | 15.5x | 30.1x | 64.0x |
| 768 | 1.0x | 7.8x | 15.0x | 32.0x |

Die Werte sind Kompressionsraten gegenüber der Baseline `768d float32`.

### Interpretation

- `1bit` erreicht die höchste Speicherreduktion: bei 768 Dimensionen 32x, bei 64 Dimensionen 384x.
- `2bit` liegt zwischen 1-bit und 4-bit und erreicht bei 768 Dimensionen etwa 15x praktische Kompression.
- `4bit` spart immer noch deutlich Speicher, bleibt aber näher an Float32 als 1-bit und 2-bit.
- Float32 profitiert nur von PCA-Dimensionsreduktion: `64d float32` ist 12x kleiner als `768d float32`.
- Bei 2-bit und 4-bit liegt der praktische Speicher leicht über dem theoretischen Minimum, weil `col_min` und `col_max` als Metadaten gespeichert werden.

## Pairwise-Distanzen

Pairwise-Distanzen messen, wie schnell eine feste Menge bekannter Dokumentpaare verglichen werden kann. Gemessen wurde jeweils der Median für 10.000 Paare.

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 2.21 ms | 5.89 ms | 7.07 ms | 0.62 ms |
| 128 | 3.86 ms | 15.92 ms | 15.65 ms | 0.93 ms |
| 256 | 6.87 ms | 24.04 ms | 22.43 ms | 1.63 ms |
| 384 | 9.82 ms | 34.77 ms | 33.32 ms | 2.26 ms |
| 768 | 20.26 ms | 68.78 ms | 66.98 ms | 4.02 ms |

### Interpretation

- `1bit` ist bei Pairwise-Distanzen klar am schnellsten.
- Float32 ist trotz größerem Speicher schneller als 2-bit und 4-bit, weil NumPy Float32-Operationen sehr effizient ausführt.
- 2-bit und 4-bit sparen Speicher, müssen aber Codes entpacken, skalieren und Distanzen rekonstruieren. Dieser Zusatzaufwand dominiert die Laufzeit.
- Alle Varianten werden mit steigender Dimension langsamer. Das ist erwartbar, weil mehr Koordinaten verarbeitet werden müssen.

## Top-k-Suche

Top-k misst die retrieval-nähere Operation: Für jeden Vektor werden die nächsten Nachbarn im Korpus gesucht. Gemessen wurde brute-force Top-k mit `k = 10`.

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 22.14 ms | 46.35 ms | 52.56 ms | 74.47 ms |
| 128 | 20.86 ms | 32.81 ms | 40.30 ms | 105.86 ms |
| 256 | 17.43 ms | 36.66 ms | 38.51 ms | 186.77 ms |
| 384 | 17.07 ms | 42.72 ms | 42.92 ms | 256.80 ms |
| 768 | 15.81 ms | 47.70 ms | 49.00 ms | 462.23 ms |

### Interpretation

- Float32 ist in der aktuellen NumPy-Implementierung bei Top-k am schnellsten.
- Das wirkt zunächst kontraintuitiv, ist aber plausibel: Float32-Top-k nutzt effiziente Matrixmultiplikation und `argpartition`.
- 1-bit ist bei einzelnen Pairwise-Distanzen sehr schnell, aber bei brute-force Top-k langsam. Die batched Hamming-Berechnung erzeugt große Zwischenarrays und profitiert weniger stark von optimierten BLAS-Routinen.
- 2-bit und 4-bit sparen Speicher, sind aber in der Suche langsamer als Float32, weil Dequantisierung bzw. rekonstruktionsnahe Distanzberechnung Aufwand erzeugt.

## Zentrale Aussagen

1. Quantisierung spart sehr deutlich Speicher.
2. Die größte Speicherersparnis erreicht `1bit`, gefolgt von `2bit` und `4bit`.
3. Dimensionsreduktion und Quantisierung wirken multiplikativ: kleine Dimension plus niedrige Bittiefe erzeugt die stärkste Kompression.
4. Speicherersparnis bedeutet nicht automatisch Laufzeitgewinn.
5. Für Pairwise-Distanzen ist `1bit` am schnellsten.
6. Für brute-force Top-k ist Float32 in NumPy am schnellsten, weil die Operation sehr gut optimiert ist.
7. 2-bit und 4-bit sind vor allem Speicheroptimierungen, nicht automatisch Laufzeitoptimierungen.

## Wie die Graphen gelesen werden sollten

### `memory_theoretical_vs_numpy.pdf`

Dieser Graph zeigt pro Repräsentation den theoretischen Speicher und den praktisch gemessenen NumPy-Speicher.

So liest man ihn:

- Je niedriger der Balken, desto kleiner der Speicherbedarf.
- Float32 ist die Referenz.
- 1-bit sollte am niedrigsten liegen.
- Kleine Abweichungen bei 2-bit und 4-bit entstehen durch Metadaten.

### `memory_compression_by_dim.pdf`

Dieser Graph zeigt die Kompressionsrate gegenüber `768d float32`.

So liest man ihn:

- Höher ist besser.
- Kurven steigen bei kleinerer Dimension.
- 1-bit liegt oben, Float32 unten.
- Der Graph zeigt gleichzeitig den Effekt von PCA und Quantisierung.

### `runtime_pairs_by_dim.pdf`

Dieser Graph zeigt die Laufzeit für 10.000 paarweise Distanzberechnungen.

So liest man ihn:

- Niedriger ist besser.
- 1-bit sollte deutlich unten liegen.
- 2-bit und 4-bit liegen höher, weil Entpacken und Skalieren teuer ist.
- Laufzeit steigt mit der Dimension.

### `runtime_knn_by_dim.pdf`

Dieser Graph zeigt die brute-force Top-k-Suche.

So liest man ihn:

- Niedriger ist besser.
- Float32 liegt in dieser Implementierung am besten.
- 1-bit wird bei größeren Dimensionen besonders langsam.
- Der Graph zeigt Implementierungskosten, nicht nur theoretische Bitkosten.

## Fazit für die Präsentation

Phase 5 zeigt einen klaren Trade-off:

> Quantisierung ist sehr effektiv für Speicherreduktion, aber nicht automatisch für Laufzeitreduktion.

Die stärkste praktische Speicherkompression entsteht bei `1bit` und niedrigen PCA-Dimensionen. Für einzelne paarweise Distanzvergleiche ist `1bit` auch die schnellste Variante. Für die brute-force Top-k-Suche ist dagegen Float32 in NumPy am schnellsten, weil NumPy/BLAS für dichte Float-Matrixoperationen stark optimiert ist.

Damit ist die wichtigste Aussage nicht, dass eine Repräsentation in allen Metriken gewinnt. Stattdessen zeigt Phase 5, dass die beste Wahl vom Ziel abhängt:

| Ziel | Geeignete Variante |
|---|---|
| maximal Speicher sparen | `1bit`, möglichst kleine PCA-Dimension |
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
