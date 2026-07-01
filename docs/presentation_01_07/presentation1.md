# Empfohlene Grafiken für die Projektvorstellung

Diese Datei nennt sechs aussagekräftige Grafiken, mit denen der aktuelle
Projektstand kurz und verständlich vorgestellt werden kann. Die Auswahl folgt
einer klaren Erzählung: Redundanz im Float-Raum, Distanzverzerrung durch
Quantisierung und Auswirkungen auf lokale Nachbarschaften.

## 1. PCA: Kumulierte erklärte Varianz

**Datei:** [`pca_cumulative_variance.pdf`](pca_cumulative_variance.pdf)

**Was zeigt die Grafik?**  
Die Grafik zeigt, wie viel Varianz durch die ersten PCA-Komponenten erhalten
bleibt.

**Kernaussage:**  
Der Embedding-Raum ist redundant. Für 95 Prozent erklärte Varianz werden je
nach Dataset nur etwa 390 bis 417 der 768 Dimensionen benötigt.

**Warum ist sie wichtig?**  
Sie begründet, warum Dimensionsreduktion überhaupt untersucht wird. Wenn viele
Dimensionen redundant sind, kann man Speicher sparen, ohne sofort die gesamte
Struktur zu verlieren.

## 2. Distanz-Pareto: Speicher gegen Strukturtreue

**Datei:** [`distortion_pareto.pdf`](distortion_pareto.pdf)

**Was zeigt die Grafik?**  
Der Plot vergleicht Speicherbedarf mit Distanzqualität. Gute Punkte liegen
möglichst links oben: wenig Speicher, hohe Qualität.

**Kernaussage:**  
4-bit Quantisierung erhält globale Distanzen sehr gut. 2-bit ist noch
brauchbar, 1-bit verliert deutlich mehr Struktur, besonders bei reduzierter
Dimension.

**Warum ist sie wichtig?**  
Sie zeigt den zentralen Trade-off des Projekts: Wie viel Kompression ist
möglich, bevor die geometrische Struktur sichtbar leidet?

## 3. Spearman-Heatmap für SciFact

**Datei:** [`distortion_heatmap_spearman_rho_scifact.pdf`](distortion_heatmap_spearman_rho_scifact.pdf)

**Was zeigt die Grafik?**  
Die Heatmap zeigt die Spearman-Rangkorrelation für Kombinationen aus Bit-Tiefe
und PCA-Dimension.

**Kernaussage:**  
Nicht nur die Bit-Tiefe und nicht nur die Dimension zählen, sondern ihre
Kombination. 4-bit bleibt auch bei kleinerer Dimension sehr stabil, während
1-bit bei reduzierten Dimensionen stark abfällt.

**Warum ist sie wichtig?**  
Spearman ist für Retrieval relevant, weil Rankings wichtiger sind als exakte
Distanzwerte.

## 4. Scatterplot: Float-Distanz gegen quantisierte Distanz

**Datei:** [`distance_scatter_scifact.pdf`](distance_scatter_scifact.pdf)

**Was zeigt die Grafik?**  
Die x-Achse zeigt die Float-Distanz, die y-Achse die quantisierte Distanz.
Punkte nahe der Diagonale bedeuten geringe Verzerrung.

**Kernaussage:**  
4-bit bleibt nahe an der Float-Referenz. 2-bit streut stärker. 1-bit erzeugt
die sichtbar größte Verzerrung.

**Warum ist sie wichtig?**  
Sie macht die abstrakten Korrelationswerte visuell intuitiv. Man sieht direkt,
wie die Quantisierung die Geometrie verschiebt.

## 5. Trustworthiness@10 Pareto

**Datei:** [`neighborhood_pareto_trustworthiness_k10.pdf`](neighborhood_pareto_trustworthiness_k10.pdf)

**Was zeigt die Grafik?**  
Der Plot zeigt, wie vertrauenswürdig die Top-10-Nachbarschaften nach
Quantisierung bleiben, relativ zum Speicherbedarf.

**Kernaussage:**  
4-bit bleibt fast perfekt vertrauenswürdig. 2-bit bleibt oft gut, aber mit mehr
Verlust. 1-bit wird bei starker Dimensionsreduktion deutlich riskanter.

**Warum ist sie wichtig?**  
Trustworthiness@10 ist retrieval-näher als globale Distanzkorrelation, weil
Suchsysteme besonders von den obersten Treffern abhängen.

## 6. Neighborhood Overlap@10 für SciFact

**Datei:** [`neighborhood_overlap_k10_scifact.pdf`](neighborhood_overlap_k10_scifact.pdf)

**Was zeigt die Grafik?**  
Die Heatmap zeigt, wie viele Top-10-Nachbarn aus dem Float-Raum auch im
quantisierten Raum erhalten bleiben.

**Kernaussage:**  
4-bit erhält die Top-10-Nachbarschaften sehr gut. Bei SciFact bleiben bei
4-bit und 768 Dimensionen etwa 95 Prozent der Top-10-Nachbarn erhalten. 2-bit
liegt niedriger, 1-bit deutlich darunter.

**Warum ist sie wichtig?**  
Overlap misst nicht nur, ob die neuen Nachbarn plausibel sind, sondern ob es
tatsächlich dieselben Nachbarn bleiben.

## Gesamtfazit

Der aktuelle Projektstand zeigt: Der Float-Embedding-Raum enthält Redundanz,
Quantisierung kann diese Redundanz nutzen, aber die Qualitätsverluste hängen
stark von Bit-Tiefe und Dimension ab. 4-bit ist bisher die robusteste
Kompressionsvariante. 2-bit ist ein möglicher Kompromiss. 1-bit bietet sehr
starke Kompression, verliert aber deutlich mehr Distanz- und
Nachbarschaftsstruktur.
