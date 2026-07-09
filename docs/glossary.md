# Glossar der Begriffe

Dieses Glossar erklärt die wichtigsten Begriffe, die in den Grafiken und
Ergebnisdateien des Projekts vorkommen.

## Grundbegriffe

**Embedding**  
Zahlenvektor, der einen Text repräsentiert. Ähnliche Texte sollen Ähnliche
Vektoren haben.

**Float32**  
Unkomprimierte Referenzdarstellung. Jede Dimension wird als 32-bit
Fließkommazahl gespeichert.

**Quantisierung**  
Kompression von Embeddings durch weniger Bits pro Dimension. Spart Speicher,
verliert aber Information.

**Bit-Tiefe**  
Anzahl Bits pro Dimension. Im Projekt: 4-bit, 2-bit und 1-bit.

**Binary / 1-bit**  
Jede Dimension wird nur als 0 oder 1 gespeichert, meist anhand des Vorzeichens.
Das ist sehr stark komprimiert, aber informationsarm.

**TurboQuant**  
Quantisierungsverfahren für 2-bit und 4-bit. Die Embeddings werden zuerst
rotiert und danach gleichmäßig quantisiert.

**PCA**  
Principal Component Analysis. Verfahren zur Dimensionsreduktion, das die
wichtigsten Varianzrichtungen beibehält.

**PCA-Dimensionen**  
Zieldimensionen nach PCA. Im Projekt: 64, 128, 256, 384 und 768. 768 bedeutet
keine Reduktion.

**Kompressionsfaktor**  
Gibt an, wie viel kleiner ein quantisierter Vektor gegenüber Float32 768d ist.
Float32 768d entspricht `768 * 32 = 24576` Bits.

**Bits per vector**  
Speicherbedarf eines Vektors in Bits: `Dimensionen * Bit-Tiefe`.

## Phase 1: Float-Raum

**Norm**  
Länge eines Vektors. Bei normalisierten Embeddings liegt sie etwa bei 1.

**L2-normalisiert**  
Alle Vektoren werden auf Länge 1 skaliert. Dann ist vor allem der Winkel
zwischen Vektoren relevant.

**Unit sphere / Einheitssphäre**  
Geometrische Beschreibung für Vektoren mit Norm 1.

**Coefficient of variation / CV**  
Relative Streuung: Standardabweichung geteilt durch Mittelwert.

**Mean / Mittelwert**  
Durchschnittswert einer Dimension.

**Standard deviation / Std**  
Streuung der Werte einer Dimension.

**Skewness / Schiefe**  
Misst, ob eine Verteilung asymmetrisch ist. Hohe Schiefe kann einfache
Schwellwert-Quantisierung verschlechtern.

**Kurtosis**  
Misst, wie stark Ausreißer oder schwere Verteilungsenden auftreten. Hohe
Kurtosis kann Quantisierung erschweren.

**Intrinsic dimensionality / intrinsische Dimensionalität**  
Schätzung, wie viele Dimensionen die Datenstruktur wirklich benötigt.

**TwoNN**  
Verfahren zur Schätzung intrinsischer Dimensionalität anhand der Abstände zum
ersten und zweiten nächsten Nachbarn.

**PCA 95% variance**  
Anzahl PCA-Komponenten, die 95 Prozent der Varianz erklären.

**Cumulative explained variance**  
Kumulierte erklärte Varianz durch die ersten PCA-Komponenten.

**Variance spectrum**  
Zeigt, wie viel Varianz jede einzelne PCA-Komponente erklärt.

## Phase 2: Distanzverzerrung

**Pairwise distance**  
Abstand zwischen zwei Dokument-Embeddings.

**Cosine similarity**  
Ähnlichkeit über den Winkel zwischen zwei Vektoren. Bei L2-normalisierten
Vektoren entspricht sie dem Skalarprodukt.

**Cosine distance**  
Distanzmaß aus Cosine Similarity: `1 - cosine similarity`.

**Hamming distance**  
Distanz zwischen binären Vektoren. Zählt, an wie vielen Bitpositionen sich
zwei Vektoren unterscheiden.

**Distance distortion / Distanzverzerrung**  
Veränderung der Abstände durch Quantisierung.

**Pearson r**  
Lineare Korrelation zwischen Float-Distanzen und quantisierten Distanzen.
Höher ist besser.

**Spearman rho**  
Rangkorrelation zwischen Float-Distanzen und quantisierten Distanzen. Höher ist
besser und für Ranking besonders relevant.

**MAE**  
Mean Absolute Error. Durchschnittlicher absoluter Fehler. Niedriger ist besser.

**RMSE**  
Root Mean Squared Error. Bestraft große Fehler stärker als MAE. Niedriger ist
besser.

**Normalized distance**  
Auf den Bereich 0 bis 1 skalierte Distanzwerte.

**Scatter plot**  
Punktdiagramm mit Float-Distanz auf der x-Achse und quantisierter Distanz auf
der y-Achse. Punkte nahe der Diagonale bedeuten geringe Verzerrung.

**Error histogram**  
Histogramm der Distanzfehler. Eine schmale Verteilung nahe 0 ist gut.

**Heatmap**  
Matrixgrafik. Im Projekt meist: Bit-Tiefe gegen PCA-Dimension, Farbe zeigt die
Metrik.

**Pareto plot**  
Zeigt Qualität gegen Speicherbedarf. Gute Punkte liegen möglichst links oben:
wenig Speicher, hohe Qualität.

## Phase 3: Nachbarschaftserhaltung

**Nearest neighbor / nächster Nachbar**  
Das Dokument, dessen Embedding einem anderen Dokument am ähnlichsten ist.

**k-NN**  
Die `k` nächsten Nachbarn, zum Beispiel Top-10, Top-50 oder Top-100.

**Neighborhood overlap**  
Anteil gemeinsamer Nachbarn zwischen Float-Raum und quantisiertem Raum.
Höher ist besser.

**Random baseline**  
Erwarteter Overlap bei zufälligen Nachbarn: `k / N`.

**Trustworthiness**  
Metrik dafür, ob im quantisierten Raum falsche Nachbarn auftauchen. Höher ist
besser.

**False neighbor**  
Dokument, das im quantisierten Raum unter den Top-k liegt, im Float-Raum aber
nicht.

**Rank displacement**  
Gibt an, wie weit ein falscher Nachbar im Float-Raum eigentlich entfernt war.
Größere Verschiebungen werden stärker bestraft.


## Phase 5: Effizienz

**Laufzeitanalyse / Runtime analysis**  
Messung, wie lange eine Operation praktisch benötigt. In Phase 5 betrifft das
vor allem Pairwise-Distanzen und brute-force Top-k-Suche.

**Speicheranalyse / Memory analysis**  
Messung, wie viel Speicher eine Repräsentation benötigt. Unterschieden wird
zwischen theoretischem Speicherbedarf und praktisch gemessenem NumPy-Layout.

**NumPy-vectorized / NumPy-vektorisiert**  
Implementierung, bei der Python die Experimente steuert, die eigentliche Arbeit
aber über vektorisierte NumPy-Operationen ausgeführt wird. Dadurch werden
Python-Schleifen in den rechenintensiven Teilen vermieden.

**NumPy-Layout**  
Konkrete Speicherform einer Repräsentation in NumPy, zum Beispiel eine
zusammenhängende `float32`-Matrix oder gepackte Codes plus Metadaten.

**Theoretischer Speicherbedarf**  
Idealer Speicherbedarf, berechnet aus `Dimensionen * Bit-Tiefe`. Dieser Wert ist
eine Untergrenze und enthält keine Metadaten oder Implementierungsdetails.

**Praktischer Speicherbedarf**  
Tatsächlich gemessener Speicherbedarf des verwendeten Layouts. Dieser kann über
dem theoretischen Wert liegen, etwa durch `col_min` und `col_max` bei TurboQuant.

**Bytes per vector**  
Speicherbedarf pro Vektor in Bytes. In Phase 5 wird dieser Wert genutzt, um
Repräsentationen direkt miteinander zu vergleichen.

**Metadaten**  
Zusatzinformationen, die neben den eigentlichen Codes gespeichert werden. Bei
TurboQuant sind das vor allem `col_min` und `col_max`, um quantisierte Werte
wieder auf den Wertebereich der jeweiligen Dimension abzubilden.

**Packed codes / gepackte Codes**  
Speicherform, bei der mehrere niedrig-bitige Codes in einem Byte abgelegt
werden. Dadurch können 1-bit, 2-bit oder 4-bit Repräsentationen tatsächlich
weniger Speicher als `uint8`-Codes verbrauchen.

**Bitpacking**  
Technik zum Verdichten mehrerer kleiner Codes in Bytes. Beispiel: vier 2-bit
Codes passen in ein Byte, zwei 4-bit Codes passen in ein Byte.

**Pairwise runtime**  
Laufzeit für eine feste Menge von Dokumentpaaren. Diese Messung ist ein
Microbenchmark für Distanzberechnungen.

**Top-k runtime**  
Laufzeit für die Suche der `k` nächsten Nachbarn. Diese Messung ist näher an
der eigentlichen Retrieval-Aufgabe als Pairwise-Distanzen.

**Brute-force Top-k**  
Suchverfahren, bei dem jede Query mit allen Kandidaten im Korpus verglichen
wird. Es ist einfach und kontrollierbar, skaliert aber teuer mit der Anzahl der
Dokumente.

**Median-Laufzeit**  
Median über mehrere Messläufe. Der Median ist robuster gegen einzelne Ausreißer
als ein einzelner Lauf oder nur der Mittelwert.

**Warmup-Lauf**  
Messlauf, der vor den eigentlichen Messungen ausgeführt wird. Er reduziert
Effekte durch einmalige Initialisierung, Caches oder erste Speicherallokationen.

**Durchsatz / Throughput**  
Anzahl verarbeiteter Einheiten pro Sekunde. Bei Pairwise-Distanzen sind das
Paare pro Sekunde, bei Top-k-Suche Queries pro Sekunde.

**BLAS**  
Optimierte Bibliotheken für lineare Algebra, die von NumPy intern genutzt
werden können. Sie erklären, warum Float32-Matrixoperationen trotz höherem
Speicherbedarf sehr schnell sein können.

**Argpartition**  
NumPy-Funktion, die effizient die kleinsten oder größten `k` Werte findet,
ohne die gesamte Distanzliste vollständig zu sortieren. In Phase 5 wird sie für
Top-k-Suche verwendet.

**Dequantisierung**  
Rekonstruktion näherungsweiser Float-Werte aus quantisierten Codes. Sie kann
hilfreich für Distanzberechnungen sein, verursacht aber zusätzlichen
Rechenaufwand und kann Laufzeitvorteile der Quantisierung reduzieren.

**Speicher-Laufzeit-Trade-off**  
Abwägung zwischen geringem Speicherbedarf und schneller Laufzeit. Phase 5 zeigt,
dass starke Kompression nicht automatisch schnellere Suche bedeutet.

**Quality-Efficiency-Trade-off**  
Gemeinsame Betrachtung von Qualität, Speicher und Laufzeit. Eine
Repräsentation ist nur dann praktisch attraktiv, wenn sie genügend Qualität
bei akzeptablen Effizienzkosten erhält.

## Datasets

**SciFact**  
Wissenschaftliche Claims und Evidenzdokumente. Eher kurze wissenschaftliche
Texte.

**FiQA**  
Finanzbezogene Fragen und Dokumente.

**TREC-COVID**  
Biomedizinische COVID-Dokumente, oft länger und fachlich.

