# Glossar der Begriffe

Dieses Glossar erklärt die wichtigsten Begriffe, die in den Grafiken und
Ergebnisdateien des Projekts vorkommen.

## Grundbegriffe

**Embedding**  
Zahlenvektor, der einen Text repräsentiert. Ähnliche Texte sollen ähnliche
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
Zieldimensionen nach PCA. Im Projekt: 64, 128, 256, 384, 512, 768 und 1024. 1024 bedeutet
keine Reduktion.

**Kompressionsfaktor**  
Gibt an, wie viel kleiner ein quantisierter Vektor gegenüber Float32 1024d ist.
Float32 1024d entspricht `1024 * 32 = 32768` Bits.

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


## Phase 4: Retrieval-Evaluation

**Retrieval**  
Suchprozess, bei dem eine Query gegen alle Dokumente im Korpus verglichen wird.
Das Ergebnis ist ein Ranking der Dokumente nach Ähnlichkeit oder Distanz.

**Query**  
Suchanfrage aus dem BEIR-Testset. In Phase 4 wird jede Query mit dem
BGE-Query-Prefix eingebettet und anschließend gegen den Korpus gesucht.

**Korpus / Corpus**  
Menge aller Dokumente, in denen gesucht wird. Für SciFact umfasst der aktuelle
Phase-4-Run 5.183 Dokumente.

**BEIR qrels**  
Relevanzurteile des BEIR-Datensatzes. Sie geben an, welche Dokumente für eine
Query relevant sind und mit welchem Relevanzgrad sie bewertet wurden.

**Ranking**  
Sortierte Ergebnisliste für eine Query. Gute Rankings platzieren relevante
Dokumente möglichst weit oben.

**Exakte Suche / Brute-force Retrieval**  
Suchverfahren, bei dem jede Query mit jedem Dokument im Korpus verglichen wird.
Phase 4 nutzt diese exakte Suche bewusst, damit keine Fehler durch
approximative Indizes wie HNSW in die Messung eingehen.

**Approximative Suche / ANN**  
Approximate Nearest Neighbor Search. Schnelle Suchverfahren, die nicht immer das
exakte Ranking finden. In Phase 4 werden sie bewusst nicht verwendet, weil sie
den Quantisierungseffekt überlagern würden.

**Cosine-Similarity-Ranking**  
Ranking nach Cosine Similarity. Im Float32-Raum ist dies die Referenzmethode.
Da die Embeddings L2-normalisiert sind, entspricht Cosine Similarity dem
Skalarprodukt.

**Asymmetrische TurboQuant-Distanz**  
Distanzberechnung, bei der die Query exakt rotiert bleibt, während der Korpus
quantisiert und dequantisiert vorliegt. Diese Variante wird für TurboQuant 2-bit
und 4-bit in Phase 4 genutzt.

**Hamming-Ranking**  
Ranking nach Hamming-Distanz zwischen binären Vektoren. Niedrigere Distanz
bedeutet höhere Ähnlichkeit.

**Top-k**  
Die ersten `k` Dokumente eines Rankings. In Phase 4 sind vor allem Top-10 und
Top-100 relevant.

**NDCG@10**  
Normalized Discounted Cumulative Gain bei den ersten 10 Treffern. Die Metrik
berücksichtigt sowohl Relevanzgrade als auch Rankingpositionen. Höher ist
besser. In Phase 4 ist NDCG@10 die primäre Qualitätsmetrik.

**DCG**  
Discounted Cumulative Gain. Bewertet relevante Treffer höher, wenn sie weiter
oben im Ranking stehen.

**IDCG**  
Ideal Discounted Cumulative Gain. Bestmöglicher DCG-Wert für eine Query. NDCG
normalisiert den tatsächlichen DCG durch diesen Idealwert.

**Recall@10**  
Anteil der relevanten Dokumente, die in den ersten 10 Treffern enthalten sind.
Höher ist besser.

**Recall@100**  
Anteil der relevanten Dokumente, die in den ersten 100 Treffern enthalten sind.
Diese Metrik ist besonders wichtig für zweistufige Retrieval-Systeme, bei denen
die erste Stufe Kandidaten sammelt und eine zweite Stufe genauer rerankt.

**MRR / Mean Reciprocal Rank**  
Mittelwert des Reciprocal Rank über alle Queries. MRR misst, wie früh das erste
relevante Dokument im Ranking erscheint. Höher ist besser.

**Reciprocal Rank**  
Kehrwert des Rangs des ersten relevanten Dokuments. Wenn das erste relevante
Dokument auf Rang 1 steht, ist der Wert 1. Bei Rang 5 ist der Wert 0,2.

**Per-Query-Metriken**  
Metrikwerte, die nicht nur als Durchschnitt über alle Queries gespeichert
werden, sondern einzeln pro Query. Sie sind die Grundlage für die statistischen
Tests in Phase 4.

**Wilcoxon-Signed-Rank-Test**  
Nichtparametrischer Test für gepaarte Stichproben. In Phase 4 vergleicht er die
per-Query-Metriken zweier Repräsentationen. Er wird verwendet, weil
Retrieval-Metriken typischerweise nicht normalverteilt sind.

**p-Wert**  
Wahrscheinlichkeit, unter der Nullhypothese einen mindestens so starken
Unterschied zu beobachten. Kleine p-Werte sprechen gegen die Nullhypothese.

**Signifikanzniveau / Alpha**  
Schwelle für statistische Signifikanz. In Phase 4 gilt zunächst `alpha = 0,05`.

**Bonferroni-Korrektur**  
Korrektur für multiples Testen. Da pro Metrik sechs paarweise Vergleiche
zwischen vier Repräsentationen durchgeführt werden, nutzt Phase 4
`alpha_korr = 0,05 / 6 = 0,0083`.

**Signifikanter Unterschied**  
Ein Unterschied gilt in Phase 4 als signifikant, wenn der p-Wert kleiner als die
Bonferroni-korrigierte Schwelle ist. Signifikant bedeutet nicht automatisch
praktisch groß, sondern statistisch belastbar über Queries hinweg.

**Kandidatengenerator**  
Erste Stufe eines zweistufigen Retrieval-Systems. Sie soll möglichst viele
relevante Dokumente in eine größere Kandidatenmenge holen. Recall@100 ist dafür
wichtiger als NDCG@10.

**Zweistufiges Retrieval-System**  
Retrieval-Architektur mit grober erster Suche und anschließendem Reranking. Eine
komprimierte Repräsentation kann für die erste Stufe geeignet sein, auch wenn
sie für das finale Top-10-Ranking etwas Qualität verliert.
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

