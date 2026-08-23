# Phase 6: Praktische Anwendungsfall-Analyse & AWS-Kostenrechnung

## Ziel

Aufbauend auf den Effizienz- und Qualitätsergebnissen aus Phase 4 und Phase 5 bewertet Phase 6 die **praktische Eignung** und die **betriebswirtschaftlichen Einsparungen (AWS-Hosting-Kosten)** für drei reale Einsatzszenarien:

1. **Szenario 1: Minimaler Speicherverbrauch (Edge / Mobile / On-Device)**
2. **Szenario 2: Der beste Kompromiss (Kostenoptimierte SaaS / Cloud Search)**
3. **Szenario 3: Enterprise-Optimierung (High Precision / Low Loss)**

Dafür wird der Indexbedarf für drei typische Korpusgrößen hochgerechnet:
- $N = 100.000$ Vektoren (Lokale Fach-Suchmaschine / Edge-Gerät)
- $N = 1.000.000$ Vektoren (1 Million) (Mittelständischer SaaS-Vektorindex)
- $N = 10.000.000$ Vektoren (10 Millionen) (Enterprise-Suche / Cloud-Infrastruktur)

Zusätzlich wird für das Handy-Szenario eine feinere Skalierung unterhalb von 10 Millionen Vektoren betrachtet, weil dort Speicherbudgets im zweistelligen bis niedrigen dreistelligen MiB-Bereich praktisch relevant sind.

---

## 1. Übersicht der Anwendungsfälle

| Anwendungsfall | Empfohlene Konfiguration | NDCG@10 | Relative Qualität | Speicher / Vektor | Kompressionsfaktor | RAM bei 1 Mio. Vektoren | RAM bei 10 Mio. Vektoren |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Float32 Baseline** | `float32 1024d` | 0.7463 | 100.0% | 4096.0 B | 1.0x | 3.81 GiB | 38.15 GiB |
| **Szenario 1: Edge / Mobile** | `tq_1bit 768d` | 0.7007 | **93.9%** | 96.0 B | **42.7x** | **91.6 MiB** | 0.89 GiB |
| **Szenario 2: Business Sweet Spot** | `tq_2bit 1024d` | 0.7318 | **98.1%** | 272.4 B | **15.0x** | **259.8 MiB** | 2.54 GiB |
| **Szenario 3: Enterprise Precision** | `tq_4bit 1024d` | 0.7422 | **99.5%** | 528.4 B | **7.8x** | 503.9 MiB | **4.92 GiB** |

---

## 2. Detailanalyse der drei Szenarien

### Szenario 1: Minimaler Speicherverbrauch (Edge / Mobile / WASM)
- **Ziel:** Vektorsuche soll ohne Cloud-Abhängigkeit lokal auf Laptops, Smartphones oder IoT-Geräten im Arbeitsspeicher (oder WASM/Browser-Cache) laufen.
- **Empfehlung:** `tq_1bit` bei 768d (oder 384d).
- **Leistung:** Erreicht **93.9%** der unkomprimierten Float32-Baseline (NDCG@10 = `0.7007` vs. `0.7463`).
- **Speicher:** Benötigt nur **96 Bytes pro Vektor** (Kompressionsfaktor **42.7x**).
- **Praxisnutzen:** Ein Index aus 100.000 Vektoren belegt lediglich **9.16 MiB** RAM (1 Mio. Vektoren nur **91.6 MiB**). Damit passt der Suchindex vollständig in den mobilen Arbeitsspeicher; für L3-Cache-Größen ist der 100k-Index plausibler als der 1M-Index.

#### Skalierung unter 10M Vektoren im Handy-Szenario

Für Smartphones ist ein 10M-Index eher eine Obergrenze als ein typischer lokaler Bestand. Interessanter sind kleinere On-Device-Korpora wie persönliche Notizen, lokale Produktkataloge, App-Hilfen oder ausgewählte PDF-Sammlungen. Bei `tq_1bit 768d` skaliert der reine Vektorspeicher linear mit **96 Bytes pro Vektor**:

| Vektoren | `tq_1bit 768d` RAM | Float32-Baseline RAM | Einordnung für Handy / On-Device |
| :---: | :---: | :---: | :--- |
| 10.000 | **0.92 MiB** | 39.1 MiB | Sehr kleiner lokaler Suchindex, praktisch unkritisch. |
| 50.000 | **4.58 MiB** | 195.3 MiB | Realistisch für App-Hilfen, FAQs oder kuratierte Dokumentsets. |
| 100.000 | **9.16 MiB** | 390.6 MiB | Guter Zielbereich für lokale Fachsuche mit vielen Chunks. |
| 250.000 | **22.9 MiB** | 976.6 MiB | Noch klar mobil nutzbar; Metadaten werden wichtiger als der Vektoranteil. |
| 500.000 | **45.8 MiB** | 1.91 GiB | Für leistungsfähigere Geräte plausibel, Float32 wäre bereits unpraktisch. |
| 1.000.000 | **91.6 MiB** | 3.81 GiB | Oberer mobiler Arbeitsbereich, abhängig von ANN-Index und Metadaten. |
| 5.000.000 | **457.8 MiB** | 19.07 GiB | Eher Tablet/Laptop oder Spezialfall; für typische Smartphones schon budgetrelevant. |

> **Interpretation:** Unterhalb von 1M Embeddings ist das Handy-Szenario mit TurboQuant nicht durch den Vektorspeicher limitiert, sondern eher durch Zusatzstrukturen: ANN-Index, Text-/Metadaten, App-Runtime und verfügbare Speicherlimits des Betriebssystems. Der Bereich **100k bis 500k Embeddings** wirkt deshalb als besonders realistischer On-Device-Korridor.

#### Dimensionsvergleich im Handy-Szenario

Zusätzlich lohnt sich eine Betrachtung der Dimensionszahl. Bei `tq_1bit` wächst der Speicher linear mit der Dimension: 64d benötigt 8 Bytes pro Vektor, 768d benötigt 96 Bytes und 1024d benötigt 128 Bytes. Die Qualitätswerte sind empirische `NDCG@10`-Ergebnisse auf `scifact`, relativ zur Float32-1024d-Baseline:

| Dimension | Speicher / Vektor | RAM bei 100k | RAM bei 500k | RAM bei 1M | Relative Qualität |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 64d | 8 B | 0.76 MiB | 3.8 MiB | 7.6 MiB | 44.7% |
| 128d | 16 B | 1.53 MiB | 7.6 MiB | 15.3 MiB | 72.6% |
| 256d | 32 B | 3.05 MiB | 15.3 MiB | 30.5 MiB | 84.7% |
| 384d | 48 B | 4.58 MiB | 22.9 MiB | 45.8 MiB | 90.5% |
| 512d | 64 B | 6.10 MiB | 30.5 MiB | 61.0 MiB | 90.4% |
| 768d | 96 B | 9.16 MiB | 45.8 MiB | 91.6 MiB | **93.9%** |
| 1024d | 128 B | 12.21 MiB | 61.0 MiB | 122.1 MiB | 90.3% |

> **Interpretation:** Für sehr kleine Speicherbudgets sind 384d oder 512d attraktiv, weil sie bei 1M Embeddings unter 64 MiB bleiben und noch etwa 90% der Float32-Qualität erreichen. Die beste gemessene Mobile-Konfiguration bleibt in diesen Ergebnissen jedoch **768d**, weil sie trotz höherem Speicherbedarf den besten Qualitätswert liefert. 1024d bringt bei `tq_1bit` auf `scifact` keinen zusätzlichen Qualitätsgewinn gegenüber 768d.

> Die zugehörige gestapelte PDF ist nach **Dimensionen** aufgebaut: Für jede Dimension von 64d bis 1024d zeigt ein eigener Teilplot die Speicherskalierung über **10k, 50k, 100k, 250k, 500k, 1M und 5M Embeddings**.


### Szenario 2: Business Sweet Spot (Kostenoptimierte SaaS)
- **Ziel:** Ein kostenempfindliches SaaS-Unternehmen möchte Server-RAM-Kosten massiv senken, ohne dass Kunden eine Verschlechterung der Suchergebnisse wahrnehmen.
- **Empfehlung:** `tq_2bit` bei 1024d.
- **Leistung:** Erreicht **98.1%** der Float32-Qualität (NDCG@10 = `0.7318` $\rightarrow$ **nur 1.9% Qualitätsverlust!**).
- **Speicher:** Benötigt **272.4 Bytes pro Vektor** (Kompressionsfaktor **15.0x**).
- **Praxisnutzen:** Ein Index aus 1 Million Dokumenten schrumpft von **3.81 GiB** auf **259.8 MiB**. Pro Million Vektoren werden **3.56 GiB High-Speed RAM** eingespart.

### Szenario 3: Enterprise-Optimierung (High Precision / Low Loss)
- **Ziel:** Große Enterprise-Systeme (Legal, Medizintechnik, Finanzen), bei denen kein nennenswerter Qualitätsverlust akzeptabel ist ($\le 1\%$ Abweichung), die jedoch bei vielen Millionen Vektoren Serverkosten und Latenzen optimieren wollen.
- **Empfehlung:** `tq_4bit` bei 1024d.
- **Leistung:** Hält **99.5%** der Float32-Baseline-Qualität (NDCG@10 = `0.7422` vs. `0.7463` $\rightarrow$ **nur 0.5% Abweichung!**).
- **Speicher:** Benötigt **528.4 Bytes pro Vektor** (Kompressionsfaktor **7.8x**).
- **Praxisnutzen:** Bei 10 Millionen Dokumenten sinkt der Speicherbedarf von **38.15 GiB** auf **4.92 GiB**. Über **33 GiB teurer Server-RAM** werden frei.

---

## 3. Umrechnung: Wie viele PDF-Seiten entsprechen 10 Millionen Vektoren? (Detaillierte Analyse & Bounds)

> **Hinweis zur Einordnung:** Dieser Abschnitt ist eine grobe Veranschaulichung zur Größenordnung und beruht auf frei gewählten, nicht empirisch belegten Annahmen zu Chunk-Größe, Überlappung und Textdichte pro Seite. Die genannten Wörter- und Chunk-pro-Seite-Werte sind Erfahrungswerte ohne Quelle und dienen nur der Intuition; sie sind nicht Teil der reproduzierbaren Kernauswertung (Skript und JSON) und fließen nicht in den wissenschaftlichen Bericht ein. Die Spannbreite der Ergebnisse (Faktor 16 zwischen unterer und oberer Grenze) spiegelt genau diese Annahmenunsicherheit wider.

Die Umrechnung von Vektoren in PDF-Seiten hängt maßgeblich vom **Dokumententyp** und der verwendeten **Chunking-Strategie** (Passagengröße) ab:
- **Standard RAG-Chunk-Größe:** 200 bis 400 Wörter (ca. 1.000 bis 2.000 Zeichen pro Vektor / Chunk).
- **Chunk-Überlappung (Overlap):** Typischerweise 10%–15% (~30–50 Wörter Überlappung).

### Differenzierte Betrachtung nach Dokumententyp:

1. **Untere Grenze (Sehr dichte Texte / Patente / Gesetzestexte / Wiss. Paper):**
   - Dichter Fließtext (einzeilig, wenig Absätze, 600–900 Wörter/Seite).
   - Dichte: ca. **2.5 bis 4.0 Chunks pro PDF-Seite** (1 Vektor $\approx$ 0.25–0.40 PDF-Seiten).
   - **10 Mio. Vektoren = ca. 2.500.000 bis 4.000.000 (2.5 bis 4 Millionen) PDF-Seiten**.

2. **Mittlere Grenze (Standard Business-Dokumente / Fachberichte / Handbücher / E-Books):**
   - Mischung aus Text, Überschriften, Tabellen und Abständen (250–450 Wörter/Seite).
   - Dichte: ca. **1.0 bis 2.0 Chunks pro PDF-Seite** (1 Vektor $\approx$ 0.5–1.0 PDF-Seiten).
   - **10 Mio. Vektoren = ca. 5.000.000 bis 10.000.000 (5 bis 10 Millionen) PDF-Seiten**.

3. **Obere Grenze (Layout-intensive PDFs / Präsentationsfolien / Datenblätter / Formulare):**
   - Dokumente mit viel Freiraum, Diagrammen, Folien (PowerPoint in PDF) oder Aufzählungen (50–200 Wörter/Seite).
   - Dichte: ca. **0.25 bis 0.5 Chunks pro PDF-Seite** (1 Vektor erfasst 2.0 bis 4.0 PDF-Seiten).
   - **10 Mio. Vektoren = ca. 20.000.000 bis 40.000.000 (20 bis 40 Millionen) PDF-Seiten**.

---

### Zusammenfassende Bounds für 10 Millionen Vektoren:

- **Gesamter Seiten-Bereich:** **2,5 Millionen bis 40 Millionen PDF-Seiten** (je nach Layout & Dichte).
- **In vollständigen PDF-Dokumenten (bei 20 Seiten/PDF):** **ca. 125.000 bis 2.000.000 (125k bis 2 Millionen) vollständige PDF-Dokumente**.
- **In Fachbüchern (bei 300 Seiten/Buch):** **ca. 8.300 bis 133.000 Fachbücher / Monografien**.

---

## 4. AWS-Hosting-Kostenanalyse (Cloud-Server-Vergleich)

Für den Serverbetrieb von In-Memory-Vektorindizes im Cloud-Hosting (AWS EC2 mit Memory-Optimized `r6i`- oder General-Purpose `t4g`/`m6g`-Instanzen) ergeben sich für **10 Millionen Vektoren (~2.5 bis 40 Mio. PDF-Seiten)** folgende monatliche Hosting-Kosten (On-Demand-Listenpreise, Linux, eu-central-1, abgerufen am 23.08.2026, 730 h/Monat):

| Szenario | Benötigte AWS-Instanz | RAM-Kapazität | Monatl. Kosten (USD) | Ersparnis (monatl.) | Ersparnis (jährlich) | Prozentuale Ersparnis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Float32 Baseline** | `r6i.2xlarge` | 64 GiB | $368.00 | - | - | Baseline (0.0%) |
| **Szenario 3: Enterprise** | `m6g.large` | 8 GiB | **$49.00** | **+$319.00 / Mo.** | **+$3,828.00 / Jahr** | **86.7% günstiger** |
| **Szenario 2: Business** | `t4g.medium` | 4 GiB | **$24.50** | **+$343.50 / Mo.** | **+$4,122.00 / Jahr** | **93.3% günstiger** |

> **Hinweis:** *Szenario 1 (Edge / Mobile)* wird hier nicht aufgeführt, da der Vektorindex lokal auf Endgeräten (Smartphone/Laptop) betrieben wird und somit **0 USD Cloud-Hosting-Kosten** verursacht.

> **Fußnote zur AWS-Kostenrechnung:**  
> Grundlage sind AWS EC2 On-Demand-Listenpreise (Linux) in der Region eu-central-1 (Frankfurt), abgerufen am 23.08.2026 von https://aws.amazon.com/ec2/pricing/on-demand/. Die Umrechnung auf Monatskosten erfolgt mit der üblichen Näherung von 730 Stunden pro Monat (Monatskosten = Stundenpreis x 730). Unkomprimiertes Float32 (1024d) benötigt bei 10M Vektoren mindestens eine `r6i.2xlarge`-Instanz ($368.00 USD/Monat). Durch `tq_2bit` sinkt der RAM-Bedarf auf 2.54 GiB, was auf einer `t4g.medium`-Instanz ($24.50 USD/Monat) betrieben werden kann. Das entspricht einer **Betriebskostenersparnis von 93.3%** bei vernachlässigbarem Qualitätsverlust (1.9%). Die Ersparnis speist sich aus zwei Quellen: dem geringeren Speicherbedarf und dem dadurch möglichen Wechsel von einer speicheroptimierten `r6i`-Instanz zu einer günstigeren Allzweck-/Graviton-Instanz (`t4g`/`m6g`); ein Teil der prozentualen Differenz geht also auf die andere Preisstruktur der Instanzfamilie zurück. Die Zuordnung der Float32-Baseline zur speicherstarken `r6i.2xlarge` ist eine bewusst konservative Annahme und markiert das obere Ende der darstellbaren Ersparnis. Die Werte sind gerundete Größenordnungen und ein Best-Case-Szenario, kein Beschaffungsangebot oder garantierter Mindestwert: Listenpreise ändern sich über die Zeit und enthalten weder Speicher-, Netzwerk- noch Reserved-/Spot-Rabatte.

---

## 5. Generierte Abbildungen (Phase 6)

- `results/phase6/figures/phase6_scenario_memory_comparison.pdf` (Vergleich des Speicherbedarfs bei 10M Vektoren)
- `results/phase6/figures/phase6_tradeoff_scenarios_highlighted.pdf` (3-Zonen Trade-off-Diagramm mit Highlighted Scenarios)
- `results/phase6/figures/phase6_aws_cost_savings.pdf` (Monatliche AWS-Hostingkosten im Vergleich)
- `results/phase6/figures/phase6_mobile_sub10m_scaling.pdf` (gestapelte TurboQuant-Varianten `tq_1bit` bis `tq_8bit`, jeweils mit gruppierten Balken nach Dimension und Vektoranzahl)
- `results/phase6/scenarios_summary.json` (Vollständige strukturierte Ergebnis-Matrix)
