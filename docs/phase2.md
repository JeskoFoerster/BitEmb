# Phase 2: Distanz- und Fehleranalyse

## Ziel

Phase 1 hat gezeigt, wie der Float-Raum statistisch aussieht. Phase 2 stellt die erste direkte Frage an die Quantisierung:

**Bleiben die Abstände zwischen Dokumenten nach der Kompression erhalten?**

Wenn zwei Dokumente im Float-Raum sehr ähnlich sind, sollten sie auch im komprimierten Raum nah beieinander liegen. Und wenn zwei Dokumente unähnlich sind, sollten sie auch nach der Kompression weit voneinander entfernt sein. Wie gut das funktioniert, messen die Metriken in dieser Phase.

Das ist eine notwendige Vorstufe zu Phase 3 (Nachbarschaftserhaltung) und Phase 4 (Retrieval-Qualität): Wenn schon die grundlegende Abstandsstruktur zerstört wird, können weder Nachbarschaften noch Suchergebnisse stimmen.

---

## Versuchsaufbau

### Zufällige Paare statt aller Paare

Den Abstand zwischen allen möglichen Dokumentpaaren zu berechnen wäre bei 57.000 Dokumenten (FiQA) rechnerisch unmöglich: das wären über 1,6 Milliarden Paare. Stattdessen werden **10.000 zufällig gezogene Paare** verwendet, gleichmäßig über den gesamten Korpus verteilt.

Die Stichprobengröße von 10.000 ist so gewählt, dass statistische Aussagen zuverlässig sind. Das 95%-Konfidenzintervall für eine Korrelation von r = 0,9 ist schmaler als ±0,005, Unterschiede zwischen Verfahren sind also klar messbar.

### Distanzmaße

Für jedes Paar werden die Abstände in allen Repräsentationsräumen berechnet:

- **Float-Raum:** Cosine-Distanz = 1 − Cosine-Similarity. Theoretisch zwischen 0 (identisch) und 2 (exakt entgegengesetzt). Für Sprachmodell-Embeddings liegen die Werte empirisch meist unter 1, weil entgegengesetzte Vektoren in der Praxis selten vorkommen.
- **TurboQuant 4-Bit / 2-Bit:** L2²-Distanz zwischen dequantisierten Vektoren. Die Vektoren werden für die Analyse rekonstruiert, um einen fairen Vergleich auf einer gemeinsamen Skala zu ermöglichen. In einem echten Produktivsystem würde man direkt auf den komprimierten Codes operieren.
- **Binär (1-Bit):** Hamming-Distanz, zählt in wie vielen der 768 Bits zwei Vektoren sich unterscheiden.

Da diese Distanzmaße unterschiedliche Skalen haben, werden alle auf [0, 1] normiert, bevor sie verglichen werden. So sind MAE und RMSE direkt interpretierbar.

---

## Die vier Metriken

Für jede Kombination aus Bittiefe und Dimensionszahl werden vier Kennzahlen berechnet. Sie messen zwei verschiedene Dinge: **relative Struktur** (Pearson, Spearman) und **absolute Verzerrung** (MAE, RMSE).

### Pearson r – Lineare Korrelation

Pearson r misst, ob Float-Distanz und komprimierte Distanz in einem linearen Zusammenhang stehen. Ein Wert nahe 1 bedeutet: wenn der Float-Abstand groß ist, ist auch der komprimierte Abstand groß, und zwar proportional.

Stell dir eine Punktwolke vor, bei der die x-Achse die Float-Distanz zeigt und die y-Achse die komprimierte Distanz. Pearson r = 1 wäre eine perfekte Gerade. Pearson r = 0,5 wäre eine breite Wolke ohne klare Struktur.

> **Grafik `distance_scatter_<dataset>.pdf`:** Genau diese Punktwolke, jeder Punkt ist ein Dokumentpaar. Je enger die Wolke um die Diagonale liegt, desto höher Pearson r. Für jedes der drei Datasets eine eigene Grafik.

Ein hohes Pearson r bedeutet, dass die Quantisierung die Abstände lediglich linear transformiert hat und die geometrische Struktur intakt ist. Das ist eine notwendige Bedingung für gute Suche.

### Spearman ρ – Rangkorrelation

Spearman ρ misst nicht ob die Abstände proportional erhalten bleiben, sondern nur ob die **Rangordnung** stimmt. Also: Ist Paar A wirklich näher als Paar B, wenn es im Float-Raum näher war?

Für Retrieval ist das oft die relevantere Frage. Suchmaschinen ranken Ergebnisse, sie zeigen das ähnlichste Dokument an erster Stelle, das zweitähnlichste an zweiter Stelle usw. Dafür zählt nur die Reihenfolge, nicht der genaue Abstandswert. Ein Verfahren mit hohem Spearman ρ aber niedrigem Pearson r verzerrt die Absolutwerte, sortiert aber trotzdem richtig. Für Ranking ist das ausreichend, für Schwellenwertentscheidungen nicht.

| Pearson r | Spearman ρ | Bedeutung |
|-----------|------------|-----------|
| Hoch | Hoch | Abstände linear erhalten, strukturtreu |
| Niedrig | Hoch | Reihenfolge stimmt, aber Werte verzerrt. Für Ranking gut, für Schwellenwerte schlecht |
| Niedrig | Niedrig | Geometrische Struktur beschädigt, schlechte Suche zu erwarten |

### MAE – Mittlerer absoluter Fehler

MAE misst die durchschnittliche absolute Abweichung zwischen Float-Distanz und komprimierter Distanz, nach Normierung auf [0, 1].

Beispiel: MAE = 0.05 bedeutet, dass die komprimierte Distanz im Schnitt um 5 Prozentpunkte vom Float-Wert abweicht. Das klingt klein, kann aber bei ähnlichen Dokumenten (Float-Distanz ≈ 0.05) bedeuten, dass die relative Abweichung 100% beträgt.

MAE ist robust gegenüber Ausreißern, ein einzelnes sehr schlecht komprimiertes Paar reißt den Wert nicht stark nach oben.

> **Grafik `error_histogram_<dataset>.pdf`:** Histogramm der Fehler |d_float − d_quant| für alle 10.000 Paare, getrennt nach Verfahren. Zeigt die Fehlerverteilung: Sind die Fehler gleichmäßig verteilt oder gibt es einen langen Schwanz mit wenigen sehr großen Fehlern?

### RMSE – Wurzel des mittleren quadratischen Fehlers

RMSE ähnelt MAE, gewichtet aber große Fehler stärker, weil die Abweichungen vor der Mittelung quadriert werden. Wenn MAE und RMSE stark voneinander abweichen, gibt es einzelne Paare mit sehr hohem Fehler. Ein Verhältnis RMSE / MAE > 1.5 deutet auf eine schiefe Fehlerverteilung mit einzelnen großen Ausreißern hin.

> **Grafik `distortion_heatmap_<metrik>_<dataset>.pdf`:** Vier Heatmaps pro Dataset, eine für Pearson r, Spearman ρ, MAE und RMSE. Zeilen = Bittiefe (4-Bit, 2-Bit, 1-Bit), Spalten = Dimensionszahl (64, 128, 256, 384, 768). Damit sieht man auf einen Blick, wie sich jede Metrik über die gesamte Versuchsmatrix verhält.

---

## Die zwei Perspektiven zusammen

Warum braucht man alle vier Metriken? Sie beantworten verschiedene Fragen.

**Pearson + Spearman** sagen: Ist die relative Struktur erhalten? Wird Näheres als näher erkannt?

**MAE + RMSE** sagen: Wie weit liegen die absoluten Werte daneben? Auch wenn die Reihenfolge stimmt, kann das Abstandsniveau verschoben sein.

Ein Verfahren kann hohe Korrelation aber hohen MAE haben: Es sortiert Paare perfekt nach Abstand, aber alle Abstände sind um +0.3 verschoben. Für ein System das mit einer festen Ähnlichkeitsschwelle arbeitet ("zeige nur Dokumente mit Ähnlichkeit > 0.8") wäre das fatal, weil alle Ergebnisse herausgefiltert würden.

---

## Versuchsmatrix und Pareto-Front

Alle vier Metriken werden für jede Kombination aus Bittiefe und Dimensionszahl berechnet:

- **Bittiefe:** 4-Bit (TurboQuant), 2-Bit (TurboQuant), 1-Bit (binär)
- **Dimensionen:** 64, 128, 256, 384, 768

Das ergibt 15 Kombinationen pro Dataset. Der Speicherbedarf einer Kombination beträgt `Dimensionen × Bittiefe` Bit pro Vektor, von 768 × 4 = 3072 Bit (4-Bit, volle Dimension) bis 64 × 1 = 64 Bit (binär, minimale Dimension).

> **Grafik `distortion_pareto.pdf`:** Pareto-Front über alle Datasets und Kombinationen. X-Achse = Speicherbedarf in Bit pro Vektor (links stark komprimiert, rechts wenig komprimiert), Y-Achse = Spearman ρ als Proxy für Retrieval-Qualität. Kombinationen auf der Pareto-Front sind nicht dominiert, es gibt also keine andere Kombination mit gleichem oder kleinerem Speicherbedarf und besserer Qualität.

---

## Zusammenfassung: Was diese Phase beantwortet

| Frage | Metrik |
|-------|--------|
| Sind die Abstände proportional erhalten? | Pearson r |
| Stimmt die Rangordnung der Abstände? | Spearman ρ |
| Wie groß ist der typische Fehler? | MAE |
| Gibt es einzelne sehr schlecht komprimierte Paare? | RMSE vs. MAE |
| Welche Kombination ist optimal für ein gegebenes Speicherbudget? | Pareto-Front |