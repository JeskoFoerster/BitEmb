# Phase 1: Charakterisierung des Float-Raums

## Ziel

Bevor wir komprimieren, müssen wir den Ausgangszustand verstehen. Phase 1 beantwortet:
- Wie sieht der unkomprimierte Vektorraum statistisch aus?
- Welche Eigenschaften begünstigen oder erschweren Quantisierung?
- Wie viel Redundanz steckt in den 768 Dimensionen?

Die Ergebnisse sind nicht Selbstzweck, sondern liefern konkrete **Vorhersagen** darüber, wo Quantisierung gut funktionieren wird und wo nicht. Diese Vorhersagen werden in den späteren Phasen empirisch überprüft.

---

## Die drei Messungen

### 1. Normverteilung

**Was wird gemessen:** Die Länge (L2-Norm) jedes Vektors.

Ein Embedding-Vektor ist eine Liste von 768 Zahlen, z.B. `[0.3, -0.1, 0.8, ...]`. Die "Länge" dieses Vektors ist der euklidische Abstand vom Nullpunkt – wie lang der Pfeil wäre, wenn man ihn im Raum einzeichnet.

**Warum könnten Vektoren unterschiedliche Längen haben?** Ohne Normalisierung hängt die Länge davon ab, wie stark das Modell auf einen Text "reagiert" – ein langer, inhaltsreicher Text erzeugt möglicherweise stärkere Aktivierungen als ein kurzer. Das wäre für die Ähnlichkeitssuche problematisch, weil dann ein langer Text automatisch "ähnlicher" zu allem wirkt als ein kurzer, unabhängig vom Inhalt.

Deshalb normalisiert `bge-large-en-v1.5` alle Vektoren auf die Einheitskugel (Länge = 1). Damit hängen Ähnlichkeiten nur noch vom *Winkel* zwischen Vektoren ab, nicht von ihrer Länge.

**Warum ist das für Binarisierung relevant?** Binarisierung behält nur das *Vorzeichen* jeder Dimension (positiv → 1, negativ → 0) und wirft die tatsächlichen Zahlenwerte – und damit auch die Länge – vollständig weg. Wenn alle Vektoren gleich lang sind, ist die Länge konstant und trägt keine Unterscheidungskraft zwischen Dokumenten – ihr Verlust ist also irrelevant. Hätten Vektoren dagegen sehr unterschiedliche Längen, würde Binarisierung echte Unterschiede zwischen Dokumenten unsichtbar machen.

**Kennzahl:** CV (Coefficient of Variation) = Standardabweichung / Mittelwert. Misst, wie stark die Längen streuen, relativ zu ihrer durchschnittlichen Größe.

| CV | Bedeutung |
|----|-----------|
| ≈ 0 | Alle Vektoren gleich lang → Länge ist konstant, kein Informationsverlust durch Binarisierung |
| > 0.01 | Nennenswerte Längenunterschiede → Länge trägt Unterscheidungskraft, die Binarisierung verwirft |

> **Warum 0.01?** Wenn alle Normen nahezu identisch sind, ist die Länge praktisch keine Variable mehr – sie trägt nichts zur Unterscheidung von Dokumenten bei. Ab CV = 0.01 beginnen die Längenunterschiede groß genug zu werden, dass Cosine-Similarity und Dot-Product spürbar auseinanderlaufen. Binarisierung approximiert Cosine-Similarity über Hamming-Distanz – diese Approximation setzt voraus, dass die Normen vernachlässigbar variieren.

---

### 2. Dimensionsstatistiken (Skewness & Kurtosis)

**Was wird gemessen:** Jeder Vektor hat 768 Dimensionen. Für jede dieser Dimensionen schaut man sich an, welche Werte die verschiedenen Dokumente dort annehmen – also: "Was ist der Wertebereich von Dimension 42 über alle 5.000 Dokumente hinweg?"

Das ergibt für jede Dimension eine Verteilung von Zahlen. Diese Verteilungen werden auf zwei Eigenschaften untersucht: Schiefe und Wölbung.

**Warum ist die Form dieser Verteilungen wichtig?** Naive Quantisierung teilt den Wertebereich einer Dimension in gleichmäßige Stufen ein (uniform quantization). Bei 2-Bit gibt es 4 Stufen, bei 4-Bit gibt es 16. Diese Stufen sind gleichmäßig über den Wertebereich verteilt. Das funktioniert gut, wenn die Werte ebenfalls gleichmäßig verteilt sind – dann nutzt jede Stufe einen ähnlichen Anteil der Dokumente. Ist die Verteilung ungleichmäßig, fallen viele Dokumente in dieselbe Stufe und werden danach ununterscheidbar. TurboQuant geht einen Schritt weiter und wendet vor der Quantisierung eine zufällige orthogonale Rotation an (Hadamard-Transformation + zufällige Vorzeichenflips). Diese Rotation verteilt Ausreißer und ungleichmäßige Varianz gleichmäßiger über alle Dimensionen – ohne die Daten dafür zu kennen. Auch dort gilt: je günstiger die Ausgangsverteilung, desto weniger Arbeit muss die Rotation leisten.

#### Skewness (Schiefe)

Schiefe beschreibt, ob eine Verteilung symmetrisch ist oder nach einer Seite "zieht".

Stell dir vor, du schaust dir die Werte einer Dimension über alle Dokumente an und zeichnest ein Histogramm. Eine symmetrische Verteilung sieht wie eine Glocke aus – die meisten Werte liegen in der Mitte, nach links und rechts fällt die Häufigkeit gleichmäßig ab. Eine schiefe Verteilung hat dagegen einen langen Schwanz auf einer Seite: die meisten Werte sind klein, aber ein paar wenige sind sehr groß (oder umgekehrt).

```
Symmetrisch (günstig):          Schief (ungünstig, Masse auf einer Seite):
      ╱╲                                ╱╲
     ╱  ╲                             ╱╲  ╲
    ╱    ╲                           ╱  ╲  ╲___
───╱──────╲───                  ───╱────╲──────────
  -1   0   1                      -1   0   1   2
          ↑                               ↑
     Schwelle 0                      Schwelle 0
    ~50% | ~50%                      ~80% | ~20%
                                    (→ 0) (→ 1)
```

**Warum ist das schlecht für naive Binarisierung?** Die Binarisierung setzt den Schwellenwert bei 0: alles über 0 wird zu 1, alles darunter zu 0. Das ergibt nur dann sinnvolle Bits, wenn die Verteilung symmetrisch um 0 liegt – dann landen ~50% der Dokumente auf jeder Seite, das Bit trennt die Dokumente in zwei gleich große Gruppen und trägt maximale Information.

Ist die Verteilung schief – d.h. die meisten Werte liegen auf einer Seite der 0 – dann werden z.B. 80% der Dokumente zu 0 und nur 20% zu 1 (oder umgekehrt, je nach Richtung der Schiefe). Das Bit sagt damit kaum noch etwas aus: fast alle Dokumente sind "gleich", egal was der genaue Wert war.

| \|Skewness\| | Bedeutung |
|-------------|-----------|
| < 1 | Annähernd symmetrisch → Binarisierung bei Schwelle 0 funktioniert gut |
| > 1 | Stark schief → der Nullpunkt ist keine gute Trennlinie |

> **Warum 1?** Ab |Skew| > 1 verschiebt sich das 50/50-Verhältnis deutlich (auf ca. 70/30 oder mehr). In der Statistik gilt |Skew| > 1 als Grenze für "deutlich asymmetrisch".

#### Kurtosis (Wölbung)

Wölbung beschreibt, ob eine Verteilung viele Ausreißer hat – also ob es oft Werte gibt, die weit vom Durchschnitt entfernt liegen.

Eine Verteilung mit niedriger Kurtosis ist "flach": die Werte streuen gleichmäßig und es gibt kaum extreme Ausreißer. Eine Verteilung mit hoher Kurtosis hat eine sehr spitze Mitte und gleichzeitig lange, schwere Ränder: die meisten Werte liegen sehr dicht beieinander, aber es gibt regelmäßig einzelne extreme Ausreißer weit weg vom Zentrum.

```
Niedrige Kurtosis (günstig):   Hohe Kurtosis (ungünstig):
                                          │
    ████████████                          █
   ██████████████                        ███
  ████████████████                   ▄▄▄█████▄▄▄
 ██████████████████       ▄▄▄▄▄▄▄▄▄▄▄           ▄▄▄▄▄▄▄▄▄▄▄
```

**Warum sind Ausreißer für Quantisierung ein Problem?** Bei 2-Bit gibt es nur 4 Stufen, gleichmäßig über den gesamten Wertebereich verteilt. Der Wertebereich wird von den Extremwerten bestimmt. Wenn also ein paar Ausreißer den Bereich auf z.B. [-5, +5] ausdehnen, aber 95% der Dokumente zwischen [-0.5, +0.5] liegen, dann quetschen sich fast alle Dokumente in die mittleren zwei Stufen – und sind danach kaum noch unterscheidbar. Die Ausreißer "verschwenden" die Auflösung, ohne selbst viele Dokumente zu sein.

| Kurtosis | Bedeutung |
|----------|-----------|
| ≈ 0 | Normalverteilungsähnlich → Stufen werden gleichmäßig genutzt |
| > 3 | Viele Ausreißer → bei wenigen Stufen (2-Bit) verlieren die meisten Dokumente ihre Unterscheidbarkeit |

> **Warum 3?** Die Normalverteilung hat Excess-Kurtosis = 0. Ab einem Wert von ~3 sind die Ränder so schwer, dass bei grober Quantisierung ein spürbarer Anteil der Werte in den äußersten Bucket fällt und dort ununterscheidbar wird. Die Grenze ist nicht scharf – sie markiert den Bereich, ab dem Ausreißer bei 4 Stufen (2-Bit) praktisch relevant werden.

> **Grafik `dimension_distribution.pdf`:** Zwei Histogramme nebeneinander – links |Skewness|, rechts Kurtosis – über alle 768 Dimensionen, alle drei Datasets überlagert. Jeder Balken steht für eine Gruppe von Dimensionen mit ähnlichem Wert. Die gestrichelten Linien markieren die Warnschwellen (|Skew| = 1, Kurtosis = 3). Je mehr Masse rechts der Schwelle liegt, desto mehr Dimensionen sind problematisch für die jeweilige Quantisierungsmethode.

---

### 3. Intrinsische Dimensionalität

**Was wird gemessen:** Wie viele Dimensionen tragen wirklich unabhängige Information?

768 Dimensionen klingt nach viel – aber viele davon könnten redundant sein. Stell dir vor, Dimension 42 misst irgendwie "wie medizinisch der Text ist", und Dimension 137 misst "wie viele Fachbegriffe aus der Medizin vorkommen". Diese beiden Dimensionen werden für medizinische Texte immer gleichzeitig hoch sein – sie sind korreliert. Wenn man die eine kennt, weiß man schon viel über die andere. Solche Korrelationen bedeuten, dass der Raum weniger echte Freiheitsgrade hat als seine 768 Dimensionen suggerieren.

Das ist wichtig für die **Dimensionsreduktion via PCA**: PCA findet die Richtungen im Raum, die die meiste Variation erklären, und wirft die unwichtigsten weg. Je mehr Redundanz im Original steckt, desto verlustloser kann man reduzieren. Steckt dagegen in jeder Dimension echte, unabhängige Information, kostet jede weggeworfene Dimension wirklich etwas.

**Zwei komplementäre Methoden:**

- **PCA (95% Varianz):** Wie viele Hauptkomponenten braucht man, um 95% der Gesamtvariation zu erklären? Gibt an, wie viel lineare Redundanz es gibt. Wenn 200 Komponenten bereits 95% erklären, dann "steckt" die Information praktisch in 200 Richtungen, nicht 768.

- **TwoNN:** Ein anderer Ansatz: Für jedes Dokument schaut man sich an, wie weit der erste und der zweite Nachbar entfernt sind. Aus dem Verhältnis dieser Abstände lässt sich schätzen, wie viele Dimensionen der Raum *lokal* wirklich hat – also wie viele Richtungen es gibt, in die man sich von einem Punkt aus "sinnvoll" bewegen kann. Dieser Schätzer erfasst auch nichtlineare Strukturen, die PCA nicht sieht.

Wenn PCA sagt "150 Dimensionen reichen", TwoNN aber "der Raum hat lokal nur 15 echte Freiheitsgrade" – dann hat der Raum nichtlineare Struktur. Die 150 PCA-Komponenten beschreiben diese Struktur zwar linear korrekt, aber eine PCA-Reduktion auf z.B. 64d schneidet dann echte nichtlineare Zusammenhänge ab, die in den verworfenen Komponenten stecken. Der tatsächliche Qualitätsverlust bei Dimensionsreduktion ist in diesem Fall größer, als die erklärte Varianz allein vermuten lässt.

**Interpretation:**
- **PCA_95 << 768:** Viel lineare Redundanz → PCA-Reduktion auf z.B. 128d oder 256d verlustarm möglich
- **PCA_95 nahe 768:** Information gleichmäßig verteilt → jede weggeworfene Dimension kostet etwas
- **TwoNN << PCA_95:** Der Raum hat nichtlineare Struktur – PCA unterschätzt den tatsächlichen Verlust bei Dimensionsreduktion

> **Grafik `pca_cumulative_variance.pdf`:** X-Achse = Anzahl Hauptkomponenten, Y-Achse = wie viel Prozent der Gesamtvarianz diese Komponenten zusammen erklären (0–100%). Man liest ab: "Mit 256 Komponenten erkläre ich X% der Varianz." Die horizontale Linie bei 95% zeigt, wie früh oder spät dieser Schwellenwert erreicht wird. Vertikale Linien markieren die Reduktionsziele 64, 128, 256, 384. Ein steiler Anstieg am Anfang wäre günstig (wenige Dimensionen erklären fast alles); ein flacher Anstieg bedeutet, dass die Information gleichmäßig über viele Dimensionen verteilt ist und Reduktion teuer wird.

> **Grafik `pca_variance_spectrum.pdf`:** Zeigt für jede der ersten 100 Hauptkomponenten einzeln, wie viel Varianz sie jeweils erklärt (log-Skala). Eine dominante erste Komponente mit steilem Abfall danach wäre gut komprimierbar. Ein gleichmäßig flacher Verlauf bedeutet: alle Komponenten sind gleich wichtig, keine kann ohne Verlust weggeworfen werden. Die gestrichelte Linie "uniform (1/768)" ist der Zielzustand nach TurboQuant-Rotation: TurboQuant dreht den Raum so, dass alle Dimensionen gleich viel Varianz tragen – damit werden die Quantisierungsstufen gleichmäßig ausgenutzt. Je ungleichmäßiger das Spektrum vor der Rotation, desto stärker hilft TurboQuant.