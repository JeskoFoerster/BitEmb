# Phase 1: Charakterisierung des Float-Raums

## Ziel

Bevor wir komprimieren, müssen wir den Ausgangszustand verstehen. Phase 1 beantwortet:
- Wie sieht der unkomprimierte Vektorraum statistisch aus?
- Welche Eigenschaften begünstigen oder erschweren Quantisierung?
- Wie viel Redundanz steckt in den 768 Dimensionen?

Die Ergebnisse erlauben es, **vorherzusagen**, wo Quantisierung gut funktionieren wird und wo nicht – bevor wir überhaupt komprimieren.

## Die drei Messungen

### 1. Normverteilung

**Was wird gemessen:** Die Länge (L2-Norm) jedes Vektors.

**Warum:** Unser Modell normalisiert alle Vektoren auf Länge 1 (Einheitskugel). Binarisierung behält nur das Vorzeichen jeder Dimension – wenn alle Vektoren gleich lang sind, geht durch das Weglassen der Länge keine Information verloren.

**Interpretation:**
- **CV (Coefficient of Variation) ≈ 0**: Alle Vektoren haben gleiche Länge → gut für Binarisierung
- **CV > 0.01**: Unterschiedliche Längen → Binarisierung verliert Magnitudeninformation

> *Warum 0.01?* CV = Standardabweichung / Mittelwert. Bei normalisierten Vektoren (Norm = 1.0) bedeutet CV = 0.01, dass die Normen nur um ±0.01 schwanken. Das ist die Grenze, ab der Cosine-Similarity und Dot-Product nicht mehr äquivalent sind – und Binarisierung setzt diese Äquivalenz voraus.

### 2. Dimensionsstatistiken (Skewness & Kurtosis)

**Was wird gemessen:** Für jede der 768 Dimensionen wird die Verteilung der Werte über alle Dokumente analysiert.

**Warum:** Quantisierung teilt den Wertebereich in gleichmäßige Stufen ein. Das funktioniert am besten, wenn die Werte symmetrisch und gleichmäßig verteilt sind.

#### Skewness (Schiefe)

Misst, ob die Verteilung symmetrisch ist oder nach einer Seite "kippt".

```
Symmetrisch (gut):     Schief (schlecht):
    ╱╲                      ╱╲
   ╱  ╲                    ╱  ╲___
  ╱    ╲                  ╱
 ╱      ╲                ╱
```

**Interpretation:**
- **|Skewness| < 1**: Annähernd symmetrisch → Binarisierung bei Schwelle 0 funktioniert gut
- **|Skewness| > 1**: Stark schief → der Nullpunkt ist keine gute Trennlinie, viele Dokumente landen auf der gleichen Seite

> *Warum 1?* Binarisierung setzt den Schwellenwert bei 0: positive Werte → 1, negative → 0. Bei einer symmetrischen Verteilung fallen ~50% der Dokumente auf jede Seite – maximaler Informationsgehalt pro Bit. Ab |Skew| > 1 verschiebt sich dieses Verhältnis deutlich (z.B. 70/30), und das eine Bit unterscheidet kaum noch zwischen Dokumenten. In der Statistik gilt |Skew| > 1 als Grenze für "deutlich asymmetrisch".

#### Kurtosis (Wölbung)

Misst, ob es viele Ausreißer gibt (schwere Ränder). Verwendet wird die "Excess-Kurtosis" – die Normalverteilung hat den Wert 0.

**Interpretation:**
- **Kurtosis ≈ 0**: Normalverteilungsähnlich → Quantisierungsstufen werden gleichmäßig genutzt
- **Kurtosis > 3**: Deutlich schwere Ränder mit vielen Extremwerten → bei wenigen Stufen (2-Bit, 4-Bit) werden Ausreißer schlecht erfasst

> *Warum 3?* Excess-Kurtosis misst den Unterschied zur Normalverteilung (= 0). Bei Kurtosis = 3 hat die Verteilung so schwere Ränder wie eine Laplace-Verteilung – dort liegen ~5% der Werte in den äußeren 1% des Wertebereichs. Bei nur 4 Quantisierungsstufen (2-Bit) werden diese Ausreißer alle in denselben Bucket gequetscht und sind danach ununterscheidbar.

### 3. Intrinsische Dimensionalität

**Was wird gemessen:** Wie viele Dimensionen tragen wirklich Information bei?

Obwohl der Raum 768 Dimensionen hat, ist die tatsächliche Informationsmenge oft viel geringer – viele Dimensionen sind redundant oder korreliert.

**Zwei Methoden:**
- **PCA (95% Varianz):** Wie viele Hauptkomponenten braucht man, um 95% der Gesamtvariation zu erklären?
- **TwoNN:** Lokale Schätzung der Mannigfaltigkeitsdimension (wie viele "echte" Freiheitsgrade hat der Raum?)

**Interpretation:**
- **PCA_95 = 200 bei 768 Dimensionen**: Nur ~200 Dimensionen tragen wesentliche Information → PCA-Reduktion auf 256d sollte kaum Qualität kosten
- **Hohe Redundanz (PCA_95 << 768)**: Dimensionsreduktion ist vielversprechend
- **TwoNN << PCA_95**: Der Raum hat nichtlineare Struktur, die PCA nicht vollständig erfasst

## Grafiken

### Cumulative PCA Variance (`pca_cumulative_variance.pdf`)

**Was zeigt sie:** X-Achse = Anzahl Hauptkomponenten, Y-Achse = kumulativ erklärte Varianz (0–100%).

**Wie lesen:**
- Je steiler die Kurve am Anfang ansteigt, desto konzentrierter ist die Information in wenigen Dimensionen
- Die horizontale Linie bei 95% zeigt, ab wie vielen Komponenten "genug" Information erhalten bleibt
- Vertikale Linien markieren die PCA-Reduktionsziele (64, 128, 256, 384)

**Beispiel:** Wenn die Kurve bei 128 Komponenten schon 90% erreicht, dann enthält ein 128d-Vektor fast die gleiche Information wie der 768d-Originalvektor.

### Variance Spectrum (`pca_variance_spectrum.pdf`)

**Was zeigt sie:** Wie viel Varianz jede einzelne Hauptkomponente erklärt (log-Skala, erste 100 Komponenten).

**Wie lesen:**
- Steiler Abfall = wenige dominante Dimensionen → gut komprimierbar
- Flacher Abfall = Information gleichmäßig verteilt → Dimensionsreduktion kostet mehr
- Die gestrichelte Linie "uniform (1/768)" zeigt den Idealfall nach TurboQuant-Rotation: alle Dimensionen gleich wichtig → optimale Nutzung der Quantisierungsstufen

**Warum relevant:** TurboQuant rotiert den Raum VOR der Quantisierung, damit die Varianz gleichmäßiger verteilt ist. Diese Grafik zeigt, wie ungleichmäßig der Originalraum ist und damit wie viel die Rotation bringt.

### Dimension Distribution (`dimension_distribution.pdf`)

**Was zeigt sie:** Zwei Histogramme – links |Skewness|, rechts Kurtosis – über alle 768 Dimensionen.

**Wie lesen:**
- **Links (Skewness):** Je mehr Masse rechts von der gestrichelten Linie (|Skew| > 1), desto mehr Dimensionen sind problematisch für naive Binarisierung
- **Rechts (Kurtosis):** Je mehr Masse rechts von der gestrichelten Linie (Kurtosis > 3), desto mehr Dimensionen haben Ausreißer-Probleme bei grober Quantisierung

**Vergleich zwischen Datasets:** Wenn ein Dataset (z.B. FiQA) mehr schiefe Dimensionen hat als ein anderes (z.B. SciFact), erwarten wir dort schlechtere Binarisierungsergebnisse in den folgenden Phasen.

### Summary Table (`phase1_summary.tex`)

LaTeX-Tabelle mit allen Kennzahlen pro Dataset. Dient der schnellen Übersicht und dem Einbau in die Arbeit.
