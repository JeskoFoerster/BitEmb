# Empfohlene Grafiken fuer die Projektvorstellung

Diese Datei nennt sechs aussagekraeftige Grafiken, mit denen der aktuelle
Projektstand kurz und verstaendlich vorgestellt werden kann. Die Auswahl folgt
einer klaren Erzaehlung: Redundanz im Float-Raum, Distanzverzerrung durch
Quantisierung und Auswirkungen auf lokale Nachbarschaften.

## 1. PCA: Kumulierte erklaerte Varianz

**Datei:** `results/phase1/figures/pca_cumulative_variance.pdf`

**Was zeigt die Grafik?**  
Die Grafik zeigt, wie viel Varianz durch die ersten PCA-Komponenten erhalten
bleibt.

**Kernaussage:**  
Der Embedding-Raum ist redundant. Fuer 95 Prozent erklaerte Varianz werden je
nach Dataset nur etwa 390 bis 417 der 768 Dimensionen benoetigt.

**Warum ist sie wichtig?**  
Sie begruendet, warum Dimensionsreduktion ueberhaupt untersucht wird. Wenn viele
Dimensionen redundant sind, kann man Speicher sparen, ohne sofort die gesamte
Struktur zu verlieren.

## 2. Distanz-Pareto: Speicher gegen Strukturtreue

**Datei:** `results/phase2/figures/distortion_pareto.pdf`

**Was zeigt die Grafik?**  
Der Plot vergleicht Speicherbedarf mit Distanzqualitaet. Gute Punkte liegen
moeglichst links oben: wenig Speicher, hohe Qualitaet.

**Kernaussage:**  
4-bit Quantisierung erhaelt globale Distanzen sehr gut. 2-bit ist noch
brauchbar, 1-bit verliert deutlich mehr Struktur, besonders bei reduzierter
Dimension.

**Warum ist sie wichtig?**  
Sie zeigt den zentralen Trade-off des Projekts: Wie viel Kompression ist
moeglich, bevor die geometrische Struktur sichtbar leidet?

## 3. Spearman-Heatmap fuer SciFact

**Datei:** `results/phase2/figures/distortion_heatmap_spearman_rho_scifact.pdf`

**Was zeigt die Grafik?**  
Die Heatmap zeigt die Spearman-Rangkorrelation fuer Kombinationen aus Bit-Tiefe
und PCA-Dimension.

**Kernaussage:**  
Nicht nur die Bit-Tiefe und nicht nur die Dimension zaehlen, sondern ihre
Kombination. 4-bit bleibt auch bei kleinerer Dimension sehr stabil, waehrend
1-bit bei reduzierten Dimensionen stark abfaellt.

**Warum ist sie wichtig?**  
Spearman ist fuer Retrieval relevant, weil Rankings wichtiger sind als exakte
Distanzwerte.

## 4. Scatterplot: Float-Distanz gegen quantisierte Distanz

**Datei:** `results/phase2/figures/distance_scatter_scifact.pdf`

**Was zeigt die Grafik?**  
Die x-Achse zeigt die Float-Distanz, die y-Achse die quantisierte Distanz.
Punkte nahe der Diagonale bedeuten geringe Verzerrung.

**Kernaussage:**  
4-bit bleibt nahe an der Float-Referenz. 2-bit streut staerker. 1-bit erzeugt
die sichtbar groesste Verzerrung.

**Warum ist sie wichtig?**  
Sie macht die abstrakten Korrelationswerte visuell intuitiv. Man sieht direkt,
wie die Quantisierung die Geometrie verschiebt.

## 5. Trustworthiness@10 Pareto

**Datei:** `results/phase3/figures/neighborhood_pareto_trustworthiness_k10.pdf`

**Was zeigt die Grafik?**  
Der Plot zeigt, wie vertrauenswuerdig die Top-10-Nachbarschaften nach
Quantisierung bleiben, relativ zum Speicherbedarf.

**Kernaussage:**  
4-bit bleibt fast perfekt vertrauenswuerdig. 2-bit bleibt oft gut, aber mit mehr
Verlust. 1-bit wird bei starker Dimensionsreduktion deutlich riskanter.

**Warum ist sie wichtig?**  
Trustworthiness@10 ist retrieval-naeher als globale Distanzkorrelation, weil
Suchsysteme besonders von den obersten Treffern abhaengen.

## 6. Neighborhood Overlap@10 fuer SciFact

**Datei:** `results/phase3/figures/neighborhood_overlap_k10_scifact.pdf`

**Was zeigt die Grafik?**  
Die Heatmap zeigt, wie viele Top-10-Nachbarn aus dem Float-Raum auch im
quantisierten Raum erhalten bleiben.

**Kernaussage:**  
4-bit erhaelt die Top-10-Nachbarschaften sehr gut. Bei SciFact bleiben bei
4-bit und 768 Dimensionen etwa 95 Prozent der Top-10-Nachbarn erhalten. 2-bit
liegt niedriger, 1-bit deutlich darunter.

**Warum ist sie wichtig?**  
Overlap misst nicht nur, ob die neuen Nachbarn plausibel sind, sondern ob es
tatsaechlich dieselben Nachbarn bleiben.

## Gesamtfazit

Der aktuelle Projektstand zeigt: Der Float-Embedding-Raum enthaelt Redundanz,
Quantisierung kann diese Redundanz nutzen, aber die Qualitaetsverluste haengen
stark von Bit-Tiefe und Dimension ab. 4-bit ist bisher die robusteste
Kompressionsvariante. 2-bit ist ein moeglicher Kompromiss. 1-bit bietet sehr
starke Kompression, verliert aber deutlich mehr Distanz- und
Nachbarschaftsstruktur.

