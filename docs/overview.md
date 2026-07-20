# BitEmb – Projektübersicht

## Worum geht es?

Moderne Suchsysteme (z.B. semantische Suche, RAG) wandeln Texte in Zahlenvektoren ("Embeddings") um. Diese Vektoren sind normalerweise 1024-dimensional mit je 32 Bit pro Zahl – das braucht viel Speicher und macht die Suche langsam.

**Quantisierung** komprimiert diese Vektoren, indem sie die Genauigkeit reduziert: statt 32 Bit pro Dimension nur noch 4, 2 oder sogar 1 Bit. Das spart massiv Speicher (bis Faktor 384×), aber es geht Information verloren.

**Die zentrale Frage:** Was genau geht bei der Kompression verloren, und ab wann werden die Suchergebnisse dadurch schlechter?

## Aufbau des Experiments

Das Experiment läuft in sechs Phasen, die aufeinander aufbauen:

| Phase | Was wird gemessen? | Warum? |
|-------|-------------------|--------|
| **1. Charakterisierung** | Statistische Eigenschaften des Float-Raums | Verstehen, *warum* Quantisierung funktioniert oder scheitert |
| **2. Distanzanalyse** | Werden Abstände zwischen Dokumenten erhalten? | Globale Strukturerhaltung prüfen |
| **3. Nachbarschaft** | Bleiben die nächsten Nachbarn gleich? | Lokale Strukturerhaltung (retrieval-relevant) |
| **4. Retrieval** | Tatsächliche Suchqualität (NDCG, Recall) | Praxisrelevante Auswirkung messen |
| **5. Effizienz** | Speicher & Geschwindigkeit | Kosten-Nutzen-Abwägung |
| **6. Anwendung** | MCP-Server im Agenten-Szenario | Praxistauglichkeit validieren |

## Versuchsmatrix

Jede Phase wird über eine 2D-Matrix evaluiert:

- **Bittiefe**: Float32 (Referenz), 4-Bit, 2-Bit, 1-Bit (binär)
- **Dimensionen**: 1024, 768, 384, 256, 128, 64 (via PCA-Reduktion)

Das ergibt Kompressionsraten von 1× (Float, 1024d) bis 384× (1-Bit, 64d).

## Datasets

Drei BEIR-Datasets mit maximaler Variation:

| Dataset | Domäne | Dokumentlänge | Annotation |
|---------|--------|--------------|------------|
| SciFact | Wissenschaft | Kurz | Binär |
| FiQA | Finanz | Mittel | Binär |
| TREC-COVID | Biomedizin | Lang | Mehrstufig |

Jedes Dataset wird **einzeln** analysiert, um domänenspezifische Effekte sichtbar zu machen.

## Modell

`BAAI/bge-large-en-v1.5` – bewusst ein Modell, das **nicht** auf Quantisierung optimiert wurde. Damit messen wir den reinen Kompressionsverlust, nicht den Effekt eines speziellen Trainings.
