# Glossar der Begriffe

Dieses Glossar erklaert die wichtigsten Begriffe, die in den Grafiken und
Ergebnisdateien des Projekts vorkommen.

## Grundbegriffe

**Embedding**  
Zahlenvektor, der einen Text repraesentiert. Aehnliche Texte sollen aehnliche
Vektoren haben.

**Float32**  
Unkomprimierte Referenzdarstellung. Jede Dimension wird als 32-bit
Fliesskommazahl gespeichert.

**Quantisierung**  
Kompression von Embeddings durch weniger Bits pro Dimension. Spart Speicher,
verliert aber Information.

**Bit-Tiefe**  
Anzahl Bits pro Dimension. Im Projekt: 4-bit, 2-bit und 1-bit.

**Binary / 1-bit**  
Jede Dimension wird nur als 0 oder 1 gespeichert, meist anhand des Vorzeichens.
Das ist sehr stark komprimiert, aber informationsarm.

**TurboQuant**  
Quantisierungsverfahren fuer 2-bit und 4-bit. Die Embeddings werden zuerst
rotiert und danach gleichmaessig quantisiert.

**PCA**  
Principal Component Analysis. Verfahren zur Dimensionsreduktion, das die
wichtigsten Varianzrichtungen beibehaelt.

**PCA-Dimensionen**  
Zieldimensionen nach PCA. Im Projekt: 64, 128, 256, 384 und 768. 768 bedeutet
keine Reduktion.

**Kompressionsfaktor**  
Gibt an, wie viel kleiner ein quantisierter Vektor gegenueber Float32 768d ist.
Float32 768d entspricht `768 * 32 = 24576` Bits.

**Bits per vector**  
Speicherbedarf eines Vektors in Bits: `Dimensionen * Bit-Tiefe`.

## Phase 1: Float-Raum

**Norm**  
Laenge eines Vektors. Bei normalisierten Embeddings liegt sie etwa bei 1.

**L2-normalisiert**  
Alle Vektoren werden auf Laenge 1 skaliert. Dann ist vor allem der Winkel
zwischen Vektoren relevant.

**Unit sphere / Einheitssphaere**  
Geometrische Beschreibung fuer Vektoren mit Norm 1.

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
Misst, wie stark Ausreisser oder schwere Verteilungsenden auftreten. Hohe
Kurtosis kann Quantisierung erschweren.

**Intrinsic dimensionality / intrinsische Dimensionalitaet**  
Schaetzung, wie viele Dimensionen die Datenstruktur wirklich benoetigt.

**TwoNN**  
Verfahren zur Schaetzung intrinsischer Dimensionalitaet anhand der Abstaende zum
ersten und zweiten naechsten Nachbarn.

**PCA 95% variance**  
Anzahl PCA-Komponenten, die 95 Prozent der Varianz erklaeren.

**Cumulative explained variance**  
Kumulierte erklaerte Varianz durch die ersten PCA-Komponenten.

**Variance spectrum**  
Zeigt, wie viel Varianz jede einzelne PCA-Komponente erklaert.

## Phase 2: Distanzverzerrung

**Pairwise distance**  
Abstand zwischen zwei Dokument-Embeddings.

**Cosine similarity**  
Aehnlichkeit ueber den Winkel zwischen zwei Vektoren. Bei L2-normalisierten
Vektoren entspricht sie dem Skalarprodukt.

**Cosine distance**  
Distanzmass aus Cosine Similarity: `1 - cosine similarity`.

**Hamming distance**  
Distanz zwischen binaeren Vektoren. Zaehlt, an wie vielen Bitpositionen sich
zwei Vektoren unterscheiden.

**Distance distortion / Distanzverzerrung**  
Veraenderung der Abstaende durch Quantisierung.

**Pearson r**  
Lineare Korrelation zwischen Float-Distanzen und quantisierten Distanzen.
Hoeher ist besser.

**Spearman rho**  
Rangkorrelation zwischen Float-Distanzen und quantisierten Distanzen. Hoeher ist
besser und fuer Ranking besonders relevant.

**MAE**  
Mean Absolute Error. Durchschnittlicher absoluter Fehler. Niedriger ist besser.

**RMSE**  
Root Mean Squared Error. Bestraft grosse Fehler staerker als MAE. Niedriger ist
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
Zeigt Qualitaet gegen Speicherbedarf. Gute Punkte liegen moeglichst links oben:
wenig Speicher, hohe Qualitaet.

## Phase 3: Nachbarschaftserhaltung

**Nearest neighbor / naechster Nachbar**  
Das Dokument, dessen Embedding einem anderen Dokument am aehnlichsten ist.

**k-NN**  
Die `k` naechsten Nachbarn, zum Beispiel Top-10, Top-50 oder Top-100.

**Neighborhood overlap**  
Anteil gemeinsamer Nachbarn zwischen Float-Raum und quantisiertem Raum.
Hoeher ist besser.

**Random baseline**  
Erwarteter Overlap bei zufaelligen Nachbarn: `k / N`.

**Trustworthiness**  
Metrik dafuer, ob im quantisierten Raum falsche Nachbarn auftauchen. Hoeher ist
besser.

**False neighbor**  
Dokument, das im quantisierten Raum unter den Top-k liegt, im Float-Raum aber
nicht.

**Rank displacement**  
Gibt an, wie weit ein falscher Nachbar im Float-Raum eigentlich entfernt war.
Groessere Verschiebungen werden staerker bestraft.

## Datasets

**SciFact**  
Wissenschaftliche Claims und Evidenzdokumente. Eher kurze wissenschaftliche
Texte.

**FiQA**  
Finanzbezogene Fragen und Dokumente.

**TREC-COVID**  
Biomedizinische COVID-Dokumente, oft laenger und fachlich.

