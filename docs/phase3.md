# Phase 3: Nachbarschaftserhaltung

## Ziel

Phase 2 hat gezeigt, ob die globale Abstandsstruktur nach der Quantisierung erhalten bleibt. Phase 3 stellt eine schärfere Frage:

**Bleiben die nächsten Nachbarn eines Dokuments nach der Kompression dieselben?**

Das ist der entscheidende Unterschied. Für Retrieval interessiert nicht, ob zwei weit entfernte Dokumente auch nach der Kompression weit entfernt sind. Was zählt, ist die lokale Nachbarschaft: Wenn ein Nutzer nach einem Dokument sucht, bekommt er die Top-10 oder Top-100 Ergebnisse angezeigt. Diese sollten dieselben sein wie im Float-Raum.

Ein Verfahren kann in Phase 2 gut abschneiden (globale Distanzstruktur erhalten) und trotzdem in Phase 3 schwächeln, wenn die lokale Nachbarschaft für einzelne Dokumente stark verändert wird. Das wäre in der Praxis spürbar als verschlechterte Suchqualität.

---

## Versuchsaufbau

### Exakte k-NN statt Näherungsverfahren

Für jedes Dokument werden die exakten k nächsten Nachbarn per Brute-Force berechnet, sowohl im Float-Raum als auch im komprimierten Raum. Approximative Indizes (z.B. HNSW) werden bewusst nicht verwendet, weil sie eigene Fehler einführen würden, die den Quantisierungsfehler überlagern. Das Experiment soll ausschließlich den Effekt der Kompression messen.

### Distanzfunktionen je Repräsentation

Die k-NN werden jeweils mit der natürlichen Distanzfunktion des Repräsentationsraums berechnet:

- **Float-Raum:** Cosine-Similarity (da die Vektoren normalisiert sind, äquivalent zum Skalarprodukt)
- **TurboQuant 4-Bit / 2-Bit:** L2-Distanz auf dequantisierten Vektoren
- **Binär (1-Bit):** Hamming-Distanz

### Wahl von k

Die Nachbarschaftsgröße k wird für k ∈ {5, 10, 20} ausgewertet.

- **k = 5** und **k = 10** entsprechen dem engsten Retrieval-Szenario.
- **k = 20** ist etwas robuster und fängt eine breitere Nachbarschaft ab.
- Der Vergleich der verschiedenen k-Werte zeigt, ob Quantisierung die engste Nachbarschaft stärker beschädigt als die weitere.

---

## Die zwei Metriken

### Neighborhood Overlap

Für jedes Dokument im Korpus werden die k nächsten Nachbarn im Float-Raum und im komprimierten Raum bestimmt. Der Overlap ist der Anteil der Nachbarn, die in beiden Mengen vorkommen:

```
Overlap(i) = |N_k_float(i) ∩ N_k_quant(i)| / k
```

Ein Overlap von 1.0 bedeutet: die k nächsten Nachbarn sind nach der Kompression exakt dieselben. Ein Overlap von 0.0 bedeutet: kein einziger Nachbar überschneidet sich, die Nachbarschaft wurde vollständig zerstört.

**Zufallsbaseline:** Bei zufälligem Ranking wäre der erwartete Overlap k/N, also für k = 10 und N = 5000 genau 0.002. Jeder Overlap deutlich über dieser Baseline zeigt echte Strukturerhaltung.

Neighborhood Overlap ist ein **symmetrisches Maß**: Es behandelt fehlende echte Nachbarn und eingeführte falsche Nachbarn gleich. Es zählt nur, wie viel die beiden Mengen überlappen, ohne zu unterscheiden, welche Art von Fehler vorliegt.

> **Grafiken `neighborhood_overlap_k5_<dataset>.pdf`, `neighborhood_overlap_k10_<dataset>.pdf`, `neighborhood_overlap_k20_<dataset>.pdf`:** Overlap pro Verfahren und Dimensionszahl für einen festen k-Wert, je Dataset. Zeigt wie stark Overlap mit steigender Dimensionszahl zunimmt und wie stark der Abfall von 4-Bit zu 1-Bit ist.

> **Grafiken `neighborhood_overlap_by_dim_k5.pdf`, `neighborhood_overlap_by_dim_k20.pdf`:** Aggregiert über alle Datasets, Overlap als Funktion der Dimensionszahl. Erlaubt direkten Vergleich ob ein Dataset systematisch besser oder schlechter abschneidet.

### Trustworthiness

Neighborhood Overlap sagt, wie viel übereinstimmt. Trustworthiness stellt eine gezieltere Frage: Werden im komprimierten Raum **falsche Nachbarn eingeführt**, also Dokumente die im Float-Raum weit entfernt sind, aber im komprimierten Raum nah erscheinen?

Das ist der für Retrieval schädlichere Fehlertyp. Ein fehlendes relevantes Ergebnis fällt dem Nutzer oft nicht auf, aber ein irrelevantes Ergebnis auf Position 1 oder 2 ist direkt sichtbar.

Formal (Venna & Kaski, 2006):

```
T(k) = 1 - (2 / (N·k·(2N - 3k - 1))) · Σ_i Σ_{j ∈ U_k(i)} (r(i,j) - k)
```

Dabei ist U_k(i) die Menge der Dokumente, die im komprimierten Raum unter den k Nachbarn von i sind, aber im Float-Raum nicht. r(i,j) ist der Rang von j bezüglich i im Float-Raum.

Intuitiv: Für jeden falschen Nachbarn wird eine Strafe addiert, die proportional dazu ist, wie weit er im Float-Raum eigentlich entfernt ist. Ein falscher Nachbar, der im Float-Raum auf Rang 11 liegt (knapp außerhalb der k = 10 Nachbarschaft), wird wenig bestraft. Ein falscher Nachbar auf Rang 5000 (das am weitesten entfernte Dokument) wird stark bestraft.

Der Wertebereich ist [0, 1], wobei T = 1 bedeutet, dass keine falschen Nachbarn eingeführt werden.

**Warum nicht Continuity?** Continuity misst den komplementären Fehler: echte Nachbarn die in der komprimierten Nachbarschaft fehlen. Das ist ebenfalls ein Fehler, aber für Retrieval weniger kritisch als falsche Nachbarn. Solange genug relevante Dokumente in den Top-k bleiben, ist das Fehlen einzelner verkraftbar. Continuity wird daher nicht separat ausgewertet.

> **Grafiken `neighborhood_trustworthiness_k5_<dataset>.pdf`, `neighborhood_trustworthiness_k10_<dataset>.pdf`, `neighborhood_trustworthiness_k20_<dataset>.pdf`:** Trustworthiness pro Verfahren und Dimensionszahl für einen festen k-Wert, je Dataset.

> **Grafiken `neighborhood_trustworthiness_by_dim_k5.pdf`, `neighborhood_trustworthiness_by_dim_k20.pdf`:** Aggregiert über alle Datasets, analog zu den Overlap-Grafiken.

---

## Overlap vs. Trustworthiness: Was sagen beide zusammen?

Beide Metriken zusammen erlauben eine differenzierte Aussage:

| Overlap | Trustworthiness | Interpretation |
|---------|-----------------|----------------|
| Hoch | Hoch | Nachbarschaft gut erhalten, kaum falsche Nachbarn |
| Niedrig | Hoch | Viele echte Nachbarn fehlen, aber keine falschen eingeführt |
| Hoch | Niedrig | Viele falsche Nachbarn, obwohl Überschneidung gut ist (unwahrscheinlich) |
| Niedrig | Niedrig | Nachbarschaft stark verändert und falsche Nachbarn eingeführt |

Ein Verfahren mit niedrigem Overlap aber hoher Trustworthiness ist für Retrieval besser geeignet als eines mit vergleichbarem Overlap aber niedriger Trustworthiness, weil es zwar Nachbarn "verliert", aber keine irrelevanten Ergebnisse einführt.

---

## Versuchsmatrix und Pareto-Fronten

Beide Metriken werden für die volle 2D-Matrix ausgewertet:

- **Bittiefe:** 4-Bit (TurboQuant), 2-Bit (TurboQuant), 1-Bit (binär)
- **Dimensionen:** 64, 128, 256, 384, 512, 768, 1024
- **k-Werte:** 5, 10, 20

> **Grafiken `neighborhood_pareto_overlap_k5.pdf`, `neighborhood_pareto_overlap_k20.pdf`:** Pareto-Front mit Speicherbedarf (Bit pro Vektor) auf der x-Achse und Overlap auf der y-Achse. Zeigt welche Kombination den besten Overlap für ein gegebenes Speicherbudget liefert.

> **Grafiken `neighborhood_pareto_trustworthiness_k5.pdf`, `neighborhood_pareto_trustworthiness_k20.pdf`:** Pareto-Front analog für Trustworthiness. Relevant um zu prüfen, ob die optimale Kombination für Overlap auch für Trustworthiness optimal ist, oder ob es einen Trade-off zwischen den beiden gibt.

> **Grafiken `overlap_by_k.pdf`, `trustworthiness_by_k.pdf`:** Beide Metriken als Funktion von k, aggregiert über Datasets. Zeigt ob der Qualitätsverlust mit wachsendem k zunimmt oder abnimmt.

---

## Zusammenfassung: Was diese Phase beantwortet

| Frage | Metrik |
|-------|--------|
| Wie viele der k nächsten Nachbarn bleiben erhalten? | Neighborhood Overlap |
| Werden falsche Nachbarn eingeführt, und wie weit lagen diese im Float-Raum? | Trustworthiness |
| Ist k = 5 oder k = 20 stärker betroffen? | Overlap/Trustworthiness bei verschiedenen k |
| Welche Kombination ist optimal für ein gegebenes Speicherbudget? | Pareto-Fronten |