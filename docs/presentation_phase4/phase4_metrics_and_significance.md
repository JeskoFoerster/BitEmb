# Phase 4: Metriken und Signifikanz

## Metriktabellen

### NDCG@10

NDCG@10 ist die primäre Metrik, weil sie Rankingposition und Relevanzgrad
kombiniert.

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 0,6211 | 0,6148 | 0,4673 | 0,3432 |
| 128 | 0,6856 | 0,6842 | 0,6304 | 0,5069 |
| 256 | 0,7242 | 0,7213 | 0,6692 | 0,6040 |
| 384 | 0,7254 | 0,7163 | 0,7015 | 0,6214 |
| 768 | 0,7287 | 0,7291 | 0,7097 | 0,6065 |

### Recall@10

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 0,7526 | 0,7536 | 0,6746 | 0,4697 |
| 128 | 0,8186 | 0,8052 | 0,7412 | 0,6298 |
| 256 | 0,8432 | 0,8432 | 0,8002 | 0,7161 |
| 384 | 0,8492 | 0,8416 | 0,8202 | 0,7247 |
| 768 | 0,8566 | 0,8592 | 0,8436 | 0,7069 |

### Recall@100

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 0,9183 | 0,9133 | 0,8730 | 0,7478 |
| 128 | 0,9383 | 0,9383 | 0,8967 | 0,8259 |
| 256 | 0,9517 | 0,9517 | 0,9383 | 0,8666 |
| 384 | 0,9517 | 0,9517 | 0,9350 | 0,8609 |
| 768 | 0,9517 | 0,9517 | 0,9417 | 0,8202 |

### MRR

| Dimension | Float32 | 4-bit | 2-bit | 1-bit |
|---:|---:|---:|---:|---:|
| 64 | 0,5938 | 0,5849 | 0,4179 | 0,3217 |
| 128 | 0,6586 | 0,6616 | 0,6112 | 0,4852 |
| 256 | 0,6996 | 0,6965 | 0,6397 | 0,5827 |
| 384 | 0,6982 | 0,6895 | 0,6794 | 0,6039 |
| 768 | 0,7001 | 0,7006 | 0,6802 | 0,5878 |

## Signifikanzsetup

Für jede Metrik und jede Dimension wurden sechs paarweise Vergleiche gerechnet:

- Float32 vs. 4-bit
- Float32 vs. 2-bit
- Float32 vs. 1-bit
- 4-bit vs. 2-bit
- 4-bit vs. 1-bit
- 2-bit vs. 1-bit

Test:

| Einstellung | Wert |
|---|---:|
| Test | Wilcoxon Signed-Rank |
| Basis | per-Query-Metriken |
| alpha | 0,05 |
| Bonferroni-Korrektur | 6 Vergleiche pro Metrik |
| alpha_korr | 0,0083 |

## Signifikante Vergleiche pro Metrik

Die Tabelle zeigt, wie viele der sechs paarweisen Vergleiche je Dimension nach
Bonferroni-Korrektur signifikant waren.

| Dimension | NDCG@10 | Recall@10 | Recall@100 | MRR |
|---:|---:|---:|---:|---:|
| 64 | 5/6 | 5/6 | 5/6 | 5/6 |
| 128 | 5/6 | 5/6 | 5/6 | 5/6 |
| 256 | 5/6 | 3/6 | 3/6 | 5/6 |
| 384 | 4/6 | 3/6 | 3/6 | 6/6 |
| 768 | 3/6 | 3/6 | 3/6 | 4/6 |

Interpretation:

- Kleine Dimensionen erzeugen klarere Unterschiede zwischen den Verfahren.
- Bei hoher Dimension konvergieren Float32, 4-bit und teilweise 2-bit.
- 1-bit bleibt auch bei hoher Dimension klar getrennt.

## Float32 gegen komprimierte Varianten: NDCG@10

| Dimension | Float32 vs. 4-bit | Float32 vs. 2-bit | Float32 vs. 1-bit |
|---:|---|---|---|
| 64 | nicht signifikant (p = 0,15) | signifikant (p = 7,6e-17) | signifikant (p = 4,5e-24) |
| 128 | nicht signifikant (p = 0,66) | signifikant (p = 1,7e-4) | signifikant (p = 4,5e-16) |
| 256 | nicht signifikant (p = 0,51) | signifikant (p = 5,1e-6) | signifikant (p = 2,8e-11) |
| 384 | nicht signifikant (p = 0,019) | signifikant (p = 0,0029) | signifikant (p = 2,6e-8) |
| 768 | nicht signifikant (p = 0,87) | nicht signifikant (p = 0,015) | signifikant (p = 3,1e-11) |

Wichtig: Bei 384d ist p = 0,019 für Float32 vs. 4-bit zwar unter 0,05, aber
nicht unter der Bonferroni-Schwelle von 0,0083. Es wird daher nicht als
signifikant gewertet.

## Float32 gegen komprimierte Varianten: Recall@100

| Dimension | Float32 vs. 4-bit | Float32 vs. 2-bit | Float32 vs. 1-bit |
|---:|---|---|---|
| 64 | nicht signifikant (p = 0,18) | signifikant (p = 8,2e-4) | signifikant (p = 2,6e-11) |
| 128 | nicht signifikant (p = 1,00) | signifikant (p = 4,7e-4) | signifikant (p = 2,2e-8) |
| 256 | nicht signifikant (p = 1,00) | nicht signifikant (p = 0,046) | signifikant (p = 5,4e-7) |
| 384 | nicht signifikant (p = 1,00) | nicht signifikant (p = 0,025) | signifikant (p = 1,9e-7) |
| 768 | nicht signifikant (p = 1,00) | nicht signifikant (p = 0,083) | signifikant (p = 5,6e-10) |

Recall@100 zeigt damit die stärkste Stabilität für 4-bit und eine deutlich
entschärfte Bewertung von 2-bit bei größeren Dimensionen.

## Präsentationsaussage

Für die Signifikanzfolie eignet sich diese Kernaussage:

> 4-bit unterscheidet sich auf SciFact in keiner Dimension signifikant von
> Float32 bei NDCG@10 oder Recall@100. 2-bit ist bei niedriger Dimension
> signifikant schlechter, im Vollraum aber nicht mehr. 1-bit bleibt in allen
> Dimensionen signifikant schlechter als Float32.

