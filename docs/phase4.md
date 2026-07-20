# Phase 4: Retrieval-Evaluation

## Ziel

Die Phasen 2 und 3 messen, ob geometrische Struktur erhalten bleibt: globale
Distanzen und lokale Nachbarschaften. Phase 4 prüft die praxisnähere Frage:

**Führen diese geometrischen Beobachtungen zu gleicher oder schlechterer
Suchqualität auf echten BEIR-Queries?**

Dafür wird nicht mehr Dokument gegen Dokument verglichen, sondern jede Query
gegen den gesamten Korpus gerankt. Bewertet wird gegen die BEIR-qrels des
jeweiligen Testsets.

---

## Retrieval-Protokoll

Für jede Query wird eine exakte Brute-Force-Suche in allen zehn
Repräsentationsräumen ausgeführt:

- **Float32:** Ranking nach Cosine-Similarity. Da die Embeddings normalisiert
  sind, entspricht das dem Skalarprodukt.
- **Float16:** Ranking nach Cosine-Similarity auf float16-gecasteten Vektoren.
- **Naive 8-Bit:** Ranking nach asymmetrischer L2-Distanz mit naiver
  (unrotierter) 8-Bit-Quantisierung.
- **TurboQuant 8-Bit:** Ranking nach asymmetrischer L2-Distanz mit rotierter
  8-Bit-Quantisierung.
- **Naive 4-Bit:** Ranking nach asymmetrischer L2-Distanz mit naiver
  4-Bit-Quantisierung.
- **TurboQuant 4-Bit:** Ranking nach asymmetrischer L2-Distanz mit rotierter
  4-Bit-Quantisierung.
- **Naive 2-Bit:** Ranking nach asymmetrischer L2-Distanz mit naiver
  2-Bit-Quantisierung.
- **TurboQuant 2-Bit:** Ranking nach asymmetrischer L2-Distanz mit rotierter
  2-Bit-Quantisierung.
- **Naive 1-Bit:** Ranking nach Hamming-Distanz auf gepackten Sign-Bits
  (unrotiert).
- **TurboQuant 1-Bit:** Ranking nach Hamming-Distanz auf gepackten Sign-Bits
  (rotiert).

Approximative Indizes wie HNSW werden bewusst nicht verwendet. Das Experiment
isoliert den Quantisierungseffekt. Eine approximative Suche würde eigene
Rankingfehler einführen und damit den Effekt der Kompression überlagern.

Die Konsequenz ist wichtig: Die gemessene Retrieval-Qualität ist methodisch
sauber vergleichbar, aber die Laufzeiten sind nicht direkt als
Produktivlatenzen zu interpretieren. Bei großen Korpora wären approximative
Indizes in realen Systemen praktisch unvermeidbar.

---

## Versuchsmatrix

Phase 4 nutzt dieselbe Matrix wie die vorherigen Phasen:

- **Repräsentationen:** Float32, Float16, Naive 8-Bit, TurboQuant 8-Bit,
  Naive 4-Bit, TurboQuant 4-Bit, Naive 2-Bit, TurboQuant 2-Bit, Naive 1-Bit,
  TurboQuant 1-Bit
- **Dimensionen:** 64, 128, 256, 384, 512, 768, 1024
- **Datasets:** SciFact, FiQA, TREC-COVID

PCA wird nur auf dem Korpus gefittet und danach auf Korpus und Queries
angewandt. Relevanzlabels fließen nicht in die Reduktion ein.

---

## Metriken

### NDCG@10

NDCG@10 ist die primäre Retrieval-Metrik. Sie berücksichtigt sowohl die
Relevanzgrade als auch die Position im Ranking. Das ist besonders relevant für
TREC-COVID, wo qrels mehrstufige Relevanzen enthalten können.

Ein hoher NDCG@10 bedeutet: Relevante und stark relevante Dokumente erscheinen
weit oben in den Top-10.

### Recall@10 und Recall@100

Recall@k misst, welcher Anteil aller relevanten Dokumente einer Query in den
Top-k gefunden wird.

- **Recall@10** beschreibt die direkt sichtbare Ergebnisqualität.
- **Recall@100** ist besonders relevant für zweistufige Retrieval-Systeme:
  Die erste Stufe erzeugt eine Kandidatenmenge, die später rerankt werden kann.

### MRR

MRR misst die Position des ersten relevanten Dokuments:

```
RR(q) = 1 / rank_first_relevant(q)
MRR = mean_q RR(q)
```

Die Implementierung berechnet den Reciprocal Rank exakt über den ganzen Korpus,
nicht nur innerhalb der Top-100.

---

## Statistische Signifikanz

Retrieval-Metriken schwanken stark zwischen Queries. Deshalb wird nicht nur der
Mittelwert verglichen. Für jede Metrik und jede PCA-Dimension wird ein
paarweiser Wilcoxon-Signed-Rank-Test auf den per-Query-Metriken ausgeführt.

Verglichen werden alle 45 Paare der zehn Repräsentationen (C(10,2) = 45).

Wilcoxon wird statt eines gepaarten t-Tests verwendet, weil Retrieval-Metriken
typischerweise nicht normalverteilt sind. Das Signifikanzniveau ist:

```
alpha = 0.05
alpha_korr = 0.05 / 45 = 0.001111...
```

Ein Vergleich wird nur als signifikant markiert, wenn `p_value < alpha_korr`.

---

## Output

Das Script schreibt zwei JSON-Dateien:

- `results/phase4/retrieval_metrics.json`
  - Mittelwerte je Dataset, Dimension und Repräsentation
  - Wilcoxon-Tests mit Bonferroni-korrigierter Schwelle
- `results/phase4/retrieval_per_query.json`
  - per-Query-Metrikarrays je Dataset, Dimension und Repräsentation
  - Grundlage für die Signifikanztests

Zusätzlich werden PDF-Grafiken unter `results/phase4/figures/` erzeugt:

- `retrieval_ndcg_at_10_by_dim.pdf`
- `retrieval_recall_at_10_by_dim.pdf`
- `retrieval_recall_at_100_by_dim.pdf`
- `retrieval_mrr_by_dim.pdf`
- `retrieval_pareto_<metric>.pdf`
- `retrieval_heatmap_<metric>_<dataset>.pdf`

---

## Ausführung

```bash
python scripts/phase4_retrieval.py --dataset scifact
python scripts/phase4_retrieval.py --all
python scripts/phase4_retrieval.py --all --max-docs 10000
```

`--max-docs` ist für schnelle Vorläufe gedacht. Dabei wird der Korpus
deterministisch subsampled und qrels werden auf die verbleibenden Dokumente
remappt. Queries ohne verbleibende relevante Dokumente werden entfernt. Für die
finale Auswertung sollte ohne `--max-docs` gerechnet werden.

---

## Interpretation

Phase 4 beantwortet, ob Kompression die tatsächliche Suchqualität messbar
verschlechtert. Die wichtigste Lesart ist:

| Beobachtung | Interpretation |
|-------------|----------------|
| Hohe Phase-3-Werte und stabile NDCG@10 | Geometrische Erhaltung schlägt sich in Retrieval-Qualität nieder |
| Gute Distanzmetriken, aber schlechter NDCG@10 | Globale Struktur reicht nicht aus; Top-Ranking wird gestört |
| Stabiler Recall@100, aber schlechter NDCG@10 | Kandidaten bleiben erhalten, aber Reihenfolge leidet |
| Signifikanter NDCG@10-Abfall gegen Float32 | Kompression verursacht query-robusten Qualitätsverlust |

Damit verbindet Phase 4 die geometrische Analyse mit der eigentlichen
Retrieval-Aufgabe.
