# Phase 4: Graphen, Lesart und offene Punkte

## Verfuegbare Graphen

Die Phase-4-Plots liegen unter `results/phase4/figures/`.

| Datei | Inhalt | Lesart |
|---|---|---|
| `retrieval_ndcg_at_10_by_dim.pdf` | NDCG@10 über PCA-Dimensionen | Primäre Qualitätskurve |
| `retrieval_recall_at_10_by_dim.pdf` | Recall@10 über PCA-Dimensionen | Sichtbare Top-10-Trefferquote |
| `retrieval_recall_at_100_by_dim.pdf` | Recall@100 über PCA-Dimensionen | Kandidatengenerator-Qualität |
| `retrieval_mrr_by_dim.pdf` | MRR über PCA-Dimensionen | Position des ersten relevanten Dokuments |
| `retrieval_pareto_ndcg_at_10.pdf` | NDCG@10 gegen Bits pro Vektor | Qualität pro Speicherbudget |
| `retrieval_pareto_recall_at_100.pdf` | Recall@100 gegen Bits pro Vektor | Kandidatenqualität pro Speicherbudget |
| `retrieval_heatmap_ndcg_at_10_scifact.pdf` | Heatmap für NDCG@10 | schneller Gesamtüberblick |
| `retrieval_heatmap_recall_at_100_scifact.pdf` | Heatmap für Recall@100 | zweistufige Retrieval-Perspektive |

## Wie die By-Dim-Graphen gelesen werden sollten

Die By-Dim-Graphen zeigen die Qualität als Funktion der PCA-Dimension.

Erwartetes Muster im aktuellen SciFact-Run:

- Float32 steigt von 64d bis 256d deutlich und saturiert danach.
- 4-bit folgt Float32 fast deckungsgleich.
- 2-bit steigt stark mit der Dimension und wird bei 768d deutlich stabiler.
- 1-bit verbessert sich bis 384d, faellt bei 768d in NDCG@10 aber wieder leicht.

Die wichtigste visuelle Aussage:

> Die Kurve von 4-bit liegt fast auf der Float32-Kurve. 2-bit ist
> dimensionsabhaengig. 1-bit bleibt sichtbar darunter.

## Wie die Pareto-Graphen gelesen werden sollten

Pareto-Plots verbinden Qualität mit Speicherbudget.

X-Achse:

- Bits pro Vektor
- logarithmisch
- niedriger ist speichereffizienter

Y-Achse:

- Retrieval-Metrik
- höher ist besser

Interpretation für NDCG@10:

- 4-bit bei 256d/384d/768d bietet eine starke Qualität bei deutlich weniger Bits als Float32.
- 2-bit bei 384d/768d kann ein sinnvoller Kompromiss sein, wenn etwas Qualitätsverlust akzeptiert wird.
- 1-bit ist nur attraktiv, wenn Speicher die dominierende Nebenbedingung ist und Qualitätsverlust toleriert wird.

Interpretation für Recall@100:

- 2-bit wirkt besser als bei NDCG@10.
- Das spricht dafür, 2-bit eher als Kandidatengenerator für ein zweistufiges System zu betrachten.
- Für finales Ranking bleibt NDCG@10 die wichtigere Warnmetrik.

## Verbindung zu Phase 2 und Phase 3

Phase 4 bestaetigt die Richtung der geometrischen Analysen:

| Beobachtung aus Phase 2/3 | Phase-4-Entsprechung |
|---|---|
| 4-bit erhält Distanzen und Nachbarschaften gut | 4-bit erhält NDCG@10 und Recall fast vollständig |
| 2-bit ist schlechter, aber nicht zerstoert | 2-bit verliert besonders bei kleinen Dimensionen, stabilisiert sich bei 768d |
| 1-bit verändert Nachbarschaften stark | 1-bit verliert signifikant in Retrieval-Metriken |

Damit ist Phase 4 die Brücke von geometrischer Struktur zu praktischer
Suchqualität.

## Empfohlene Folienstruktur

### Folie 1: Setup

- SciFact, 5.183 Dokumente, 300 Queries
- exakte brute-force Suche
- vier Repräsentationen
- Metriken: NDCG@10, Recall@10, Recall@100, MRR
- Wilcoxon mit Bonferroni-Korrektur

### Folie 2: NDCG@10 Hauptbefund

- Tabelle oder `retrieval_ndcg_at_10_by_dim.pdf`
- 4-bit fast identisch zu Float32
- 2-bit dimensionssensitiv
- 1-bit deutlich schlechter

### Folie 3: Recall@100 für zweistufige Systeme

- `retrieval_recall_at_100_by_dim.pdf`
- 2-bit bei 768d nahe an Float32
- Qualitätsverlust bei Kandidatenmenge kleiner als im Top-10-Ranking

### Folie 4: Signifikanz

- Float32 vs. 4-bit: kein signifikanter NDCG@10-Unterschied
- Float32 vs. 2-bit: bei 768d nicht mehr signifikant für NDCG@10
- Float32 vs. 1-bit: immer signifikant schlechter

### Folie 5: Trade-off

- `retrieval_pareto_ndcg_at_10.pdf`
- 4-bit als beste moderate Kompression
- 2-bit als möglicher Kompromiss bei hoher Dimension
- 1-bit nur bei extremem Speicherzwang

## Offene Punkte

Der aktuelle Ergebnisstand umfasst nur SciFact. Für eine robuste Abschlussaussage
müssen FiQA und TREC-COVID noch mit demselben Protokoll gerechnet werden.

Erwartete Fragestellungen für die naechsten Runs:

- Bleibt 4-bit auch auf FiQA und TREC-COVID retrieval-neutral?
- Ist 2-bit auf längeren oder schwierigeren Korpora stärker betroffen?
- Verändert mehrstufige Relevanz in TREC-COVID die NDCG@10-Lesart?
- Ist Recall@100 über Datasets hinweg stabiler als NDCG@10?

## Vorsicht bei der Interpretation

Die Phase-4-Suche ist exakt und brute-force. Dadurch ist der Vergleich methodisch
sauber, aber nicht als Produktivlatenz zu lesen. Für reale größere Systeme
wären approximative Indizes relevant. Die Interaktion zwischen Quantisierung
und HNSW/ANN ist hier bewusst ausgeklammert.

