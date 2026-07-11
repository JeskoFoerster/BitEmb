# Phase 4: Ergebniszusammenfassung

## Ziel der Phase

Phase 4 prüft, ob die geometrischen Effekte aus Phase 2 und Phase 3 in echter
Retrieval-Qualität sichtbar werden. Statt Dokumente untereinander zu
vergleichen, wird jede BEIR-Query gegen den gesamten Korpus gesucht.

Die zentrale Frage lautet:

> Wie stark verändert Quantisierung die tatsächliche Suchqualität gemessen an
> NDCG@10, Recall@10, Recall@100 und MRR?

## Aktueller Ergebnisstand

Der aktuelle Run liegt für **SciFact** vor.

| Einstellung | Wert |
|---|---:|
| Dataset | SciFact |
| Korpusgröße | 5.183 Dokumente |
| Queries | 300 |
| Dimensionen | 64, 128, 256, 384, 768 |
| Repräsentationen | Float32, 4-bit, 2-bit, 1-bit |
| Suche | exakt, brute-force |
| Signifikanztest | Wilcoxon Signed-Rank |
| Korrektur | Bonferroni, alpha_korr = 0,0083 |

Die Ergebnisdateien liegen unter:

| Datei | Inhalt |
|---|---|
| `results/phase4/retrieval_metrics.json` | Mittelwerte und Signifikanztests |
| `results/phase4/retrieval_per_query.json` | per-Query-Metriken für Wilcoxon |
| `results/phase4/figures/` | Heatmaps, Pareto-Plots und Dimensionsplots |

## Wichtigste Ergebnisse

### 4-bit ist praktisch verlustfrei

TurboQuant 4-bit liegt auf SciFact in allen Dimensionen sehr nah an Float32.
Bei 768 Dimensionen ist 4-bit sogar minimal höher als Float32:

| Repräsentation | NDCG@10 | Recall@10 | Recall@100 | MRR |
|---|---:|---:|---:|---:|
| Float32, 768d | 0,7287 | 0,8566 | 0,9517 | 0,7001 |
| 4-bit, 768d | 0,7291 | 0,8592 | 0,9517 | 0,7006 |

Das ist keine belastbare Aussage, dass 4-bit besser ist als Float32. Der
Wilcoxon-Test zeigt keinen signifikanten Unterschied. Methodisch ist die
sinnvolle Lesart:

> 4-bit erhält die Retrieval-Qualität auf SciFact nahezu vollständig.

### 2-bit verliert bei kleinen Dimensionen, stabilisiert sich aber im Vollraum

2-bit zeigt bei 64 Dimensionen einen deutlichen NDCG@10-Abfall:

| Dimension | Float32 | 2-bit | Delta |
|---:|---:|---:|---:|
| 64 | 0,6211 | 0,4673 | -0,1538 |
| 128 | 0,6856 | 0,6304 | -0,0552 |
| 256 | 0,7242 | 0,6692 | -0,0550 |
| 384 | 0,7254 | 0,7015 | -0,0240 |
| 768 | 0,7287 | 0,7097 | -0,0190 |

Bei 768 Dimensionen ist der Unterschied in NDCG@10 nicht mehr
Bonferroni-signifikant. Recall@100 bleibt ebenfalls nahe an Float32.

Interpretation:

> 2-bit ist bei ausreichender Dimension ein brauchbarer Kompromiss, aber bei
> starker Dimensionsreduktion sichtbar riskant.

### 1-bit ist deutlich schwächer

Die binäre 1-bit-Repräsentation verliert durchgängig Qualität. Der beste
NDCG@10-Wert liegt bei 384 Dimensionen:

| Dimension | 1-bit NDCG@10 |
|---:|---:|
| 64 | 0,3432 |
| 128 | 0,5069 |
| 256 | 0,6040 |
| 384 | 0,6214 |
| 768 | 0,6065 |

1-bit ist gegen Float32 in NDCG@10 bei allen Dimensionen signifikant schlechter.
Auch Recall@100 bleibt deutlich niedriger, besonders bei 64 und 768 Dimensionen.

Interpretation:

> 1-bit spart stark Speicher, ist für direkte Single-stage-Retrieval-Qualität
> auf SciFact aber klar unterlegen.

## NDCG@10 über alle Dimensionen

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 0,6211 | 0,6148 | 0,4673 | 0,3432 |
| 128 | 0,6856 | 0,6842 | 0,6304 | 0,5069 |
| 256 | 0,7242 | 0,7213 | 0,6692 | 0,6040 |
| 384 | 0,7254 | 0,7163 | 0,7015 | 0,6214 |
| 768 | 0,7287 | 0,7291 | 0,7097 | 0,6065 |

## Recall@100 als Kandidatenqualität

Recall@100 ist für zweistufige Systeme relevant. Hier ist der Verlust kleiner
als bei NDCG@10, besonders für 4-bit und 2-bit.

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 0,9183 | 0,9133 | 0,8730 | 0,7478 |
| 128 | 0,9383 | 0,9383 | 0,8967 | 0,8259 |
| 256 | 0,9517 | 0,9517 | 0,9383 | 0,8666 |
| 384 | 0,9517 | 0,9517 | 0,9350 | 0,8609 |
| 768 | 0,9517 | 0,9517 | 0,9417 | 0,8202 |

Wichtig für die Interpretation:

> 2-bit kann als Kandidatengenerator deutlich besser aussehen als
> finales Top-10-Ranking. Recall@100 bleibt im Vollraum nahe an Float32,
> obwohl NDCG@10 etwas sinkt.

## Zentrale Aussagen für die Präsentation

1. Phase 4 bestätigt die geometrischen Befunde aus Phase 2 und Phase 3.
2. 4-bit ist auf SciFact nahezu retrieval-neutral.
3. 2-bit ist dimensionssensitiv: bei 64d schlecht, bei 768d deutlich näher an Float32.
4. 1-bit verliert deutlich und signifikant gegen Float32.
5. Recall@100 faellt weniger stark als NDCG@10, besonders bei 2-bit.
6. Für zweistufige Systeme ist 2-bit daher plausibler als für finales Single-stage-Ranking.
7. Die bisherigen Ergebnisse gelten aktuell nur für SciFact; FiQA und TREC-COVID müssen noch gerechnet werden.

## Kurzes Fazit

Der wichtigste Befund ist die Trennung zwischen aggressiver und moderater
Kompression:

> 4-bit erhält die Suchqualität auf SciFact fast vollständig. 2-bit ist bei
> hoher Dimension ein möglicher Kompromiss, verliert aber bei starker
> Dimensionsreduktion. 1-bit ist für direkte Retrieval-Qualität zu aggressiv.

