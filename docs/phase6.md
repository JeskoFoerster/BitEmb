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

---

## 1. Übersicht der Anwendungsfälle

| Anwendungsfall | Empfohlene Konfiguration | NDCG@10 | Relative Qualität | Speicher / Vektor | Kompressionsfaktor | RAM bei 1 Mio. Vektoren | RAM bei 10 Mio. Vektoren |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Float32 Baseline** | `float32 1024d` | 0.7463 | 100.0% | 4096.0 B | 1.0x | 4.10 GB | 41.00 GB |
| **Szenario 1: Edge / Mobile** | `tq_1bit 768d` | 0.7007 | **93.9%** | 96.0 B | **42.7x** | **96 MB** | 0.96 GB |
| **Szenario 2: Business Sweet Spot** | `tq_2bit 1024d` | 0.7318 | **98.1%** | 272.4 B | **15.0x** | **272 MB** | 2.72 GB |
| **Szenario 3: Enterprise Precision** | `tq_4bit 1024d` | 0.7422 | **99.5%** | 528.4 B | **7.8x** | 528 MB | **5.28 GB** |

---

## 2. Detailanalyse der drei Szenarien

### Szenario 1: Minimaler Speicherverbrauch (Edge / Mobile / WASM)
- **Ziel:** Vektorsuche soll ohne Cloud-Abhängigkeit lokal auf Laptops, Smartphones oder IoT-Geräten im Arbeitsspeicher (oder WASM/Browser-Cache) laufen.
- **Empfehlung:** `tq_1bit` bei 768d (oder 384d).
- **Leistung:** Erreicht **93.9%** der unkomprimierten Float32-Baseline (NDCG@10 = `0.7007` vs. `0.7463`).
- **Speicher:** Benötigt nur **96 Bytes pro Vektor** (Kompressionsfaktor **42.7x**).
- **Praxisnutzen:** Ein Index aus 100.000 Vektoren belegt lediglich **9.6 MB** RAM (1 Mio. Vektoren nur **96 MB**). Damit passt der Suchindex vollständig in den L3-Cache oder den mobilen Arbeitsspeicher.

### Szenario 2: Business Sweet Spot (Kostenoptimierte SaaS)
- **Ziel:** Ein kostenempfindliches SaaS-Unternehmen möchte Server-RAM-Kosten massiv senken, ohne dass Kunden eine Verschlechterung der Suchergebnisse wahrnehmen.
- **Empfehlung:** `tq_2bit` bei 1024d.
- **Leistung:** Erreicht **98.1%** der Float32-Qualität (NDCG@10 = `0.7318` $\rightarrow$ **nur 1.9% Qualitätsverlust!**).
- **Speicher:** Benötigt **272.4 Bytes pro Vektor** (Kompressionsfaktor **15.0x**).
- **Praxisnutzen:** Ein Index aus 1 Million Dokumenten schrumpft von **4.1 GB** auf **272 MB**. Pro Million Vektoren werden **3.8 GB High-Speed RAM** eingespart.

### Szenario 3: Enterprise-Optimierung (High Precision / Low Loss)
- **Ziel:** Große Enterprise-Systeme (Legal, Medizintechnik, Finanzen), bei denen kein nennenswerter Qualitätsverlust akzeptabel ist ($\le 1\%$ Abweichung), die jedoch bei vielen Millionen Vektoren Serverkosten und Latenzen optimieren wollen.
- **Empfehlung:** `tq_4bit` bei 1024d.
- **Leistung:** Hält **99.5%** der Float32-Baseline-Qualität (NDCG@10 = `0.7422` vs. `0.7463` $\rightarrow$ **nur 0.5% Abweichung!**).
- **Speicher:** Benötigt **528.4 Bytes pro Vektor** (Kompressionsfaktor **7.8x**).
- **Praxisnutzen:** Bei 10 Millionen Dokumenten sinkt der Speicherbedarf von **41.0 GB** auf **5.28 GB**. Über **35 GB teurer Server-RAM** werden frei.

---

## 3. Umrechnung: Wie viele PDF-Seiten entsprechen 10 Millionen Vektoren? (Detaillierte Analyse & Bounds)

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

Für den Serverbetrieb von In-Memory-Vektorindizes im Cloud-Hosting (AWS EC2 mit Memory-Optimized `r6i`- oder General-Purpose `t4g`/`m6g`-Instanzen) ergeben sich für **10 Millionen Vektoren (~2.5 bis 40 Mio. PDF-Seiten)** folgende monatliche Hosting-Kosten (On-Demand pricing, eu-central-1):

| Szenario | Benötigte AWS-Instanz | RAM-Kapazität | Monatl. Kosten (USD) | Ersparnis (monatl.) | Ersparnis (jährlich) | Prozentuale Ersparnis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Float32 Baseline** | `r6i.2xlarge` | 64 GB | $368.00 | - | - | Baseline (0.0%) |
| **Szenario 3: Enterprise** | `m6g.large` | 8 GB | **$49.00** | **+$319.00 / Mo.** | **+$3,828.00 / Jahr** | **86.7% günstiger** |
| **Szenario 2: Business** | `t4g.medium` | 4 GB | **$24.50** | **+$343.50 / Mo.** | **+$4,122.00 / Jahr** | **93.3% günstiger** |

> **Hinweis:** *Szenario 1 (Edge / Mobile)* wird hier nicht aufgeführt, da der Vektorindex lokal auf Endgeräten (Smartphone/Laptop) betrieben wird und somit **0 USD Cloud-Hosting-Kosten** verursacht.

> **Fußnote zur AWS-Kostenrechnung:**  
> Die Berechnung basiert auf aktuellen AWS EC2 On-Demand-Tarifen (eu-central-1 / us-east-1). Unkomprimiertes Float32 (1024d) benötigt bei 10M Vektoren mindestens eine `r6i.2xlarge`-Instanz ($368.00 USD/Monat). Durch `tq_2bit` sinkt der RAM-Bedarf auf 2.72 GB, was auf einer `t4g.medium`-Instanz ($24.50 USD/Monat) betrieben werden kann. Das entspricht einer **Betriebskostenersparnis von 93.3%** bei vernachlässigbarem Qualitätsverlust (1.9%).

---

## 5. Generierte Abbildungen (Phase 6)

- `results/phase6/figures/phase6_scenario_memory_comparison.pdf` (Vergleich des Speicherbedarfs bei 10M Vektoren)
- `results/phase6/figures/phase6_tradeoff_scenarios_highlighted.pdf` (3-Zonen Trade-off-Diagramm mit Highlighted Scenarios)
- `results/phase6/figures/phase6_aws_cost_savings.pdf` (Monatliche AWS-Hostingkosten im Vergleich)
- `results/phase6/scenarios_summary.json` (Vollständige strukturierte Ergebnis-Matrix)
