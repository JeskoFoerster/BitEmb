# Anleitung: Evaluation auf Kaggle GPU ausführen

Da die Berechnung der Embeddings und Nachbarschafts-Distanzen auf CPU-Hardware sehr zeitaufwendig ist, kann die gesamte Pipeline vollautomatisch auf einer kostenlosen, leistungsstarken Cloud-GPU von Kaggle (Tesla T4) ausgeführt werden. 

Diese Anleitung beschreibt Schritt für Schritt, wie Kaggle eingerichtet und die Pipeline gestartet wird.

---

## Schritt 1: Kaggle Account & API-Token einrichten

1. Registriere dich auf [Kaggle.com](https://www.kaggle.com/).
2. Gehe in deine Einstellungen (Klick auf dein Profilbild oben rechts $\to$ **Settings**).
3. Scrolle nach unten zum Bereich **API**.
4. Klicke auf **"Create New Token"**. Dies lädt eine Datei namens `kaggle.json` herunter, die deine Zugangsdaten enthält.
5. Kopiere diese Datei in das entsprechende Verzeichnis auf deinem lokalen Rechner:
   * **Linux/macOS:** `~/.kaggle/kaggle.json`
   * **Windows:** `C:\Users\<DeinBenutzername>\.kaggle\kaggle.json`
6. Ändere die Lese-/Schreibrechte der Datei (unter Linux/macOS zwingend erforderlich):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```
7. **Telefonnummer verifizieren (Zwingend erforderlich für GPU-Nutzung):**
   * Kaggle verlangt für die Zuteilung von kostenlosen GPU-Ressourcen eine Bestätigung deines Kontos.
   * Gehe in deinen Einstellungen (**Settings**) zum Bereich **Phone verification**.
   * Trage deine Mobilfunknummer ein und bestätige den empfangenen SMS-Code.
   * **Wichtig:** Ohne diese Verifizierung kann der GPU-Accelerator im Browser nicht auf "GPU T4" umgestellt werden und CLI-Runs brechen mit Berechtigungsfehlern ab!

---

## Schritt 2: Kaggle CLI installieren

Installiere das offizielle Kaggle-Kommandozeilenwerkzeug in deiner lokalen Python-Umgebung:
```bash
pip install kaggle
```
Verifiziere die Installation, indem du eine Liste deiner bestehenden Notebooks abfragst:
```bash
kaggle kernels list
```

---

## Schritt 3: Notebook-Optionen im Webinterface festlegen (Einmaliges Setup)

Kaggle-Notebooks benötigen für diese Pipeline Internetzugang (zum Herunterladen des SciFact-Datensatzes und des Hugging-Face-Modells) und GPU-Beschleunigung. Diese Optionen müssen **einmalig im Browser** als Standard festgelegt werden:

1. **Ersten Push ausführen:** Starte den ersten Deploy-Versuch über das Hilfsskript im Repository (dies erstellt das Notebook auf Kaggle):
   ```bash
   ./scripts/run_on_kaggle.sh
   ```
2. **Notebook im Browser öffnen:** Gehe auf `https://www.kaggle.com/code/<dein-username>/bitemb-evaluation-pipeline`.
3. **In den Editor wechseln:** Klicke oben rechts auf den blauen Button **"Edit"** (oder "Copy & Edit").
4. **Einstellungen anpassen:** Öffne das Einstellungs-Panel auf der rechten Seite des Editors (Zahnrad-Symbol oder `<` einklappen):
   * **Accelerator:** Wähle **"GPU T4 x2"** (oder "GPU T4") anstelle von "GPU P100".
   * **Internet on:** Aktiviere den Schalter für den Internetzugang.
5. **Als Standard speichern:** Klicke oben rechts auf den blauen Button **"Save Version"**, wähle **"Quick Save"** und klicke auf **"Save"**. 
   * *Hinweis:* Erst durch dieses Speichern merkt sich Kaggle die T4 GPU und den Internetzugang als Standard für alle zukünftigen CLI-Pushes!
6. **Laufende Session stoppen:** Schließe die interaktive Session oder stoppe sie links unten unter **"Active Events"** (rotes X/Mülleimer-Symbol), um dein wöchentliches GPU-Kontingent zu schonen.

---

## Schritt 4: Pipeline ausführen & Ergebnisse herunterladen

Nach dem einmaligen Setup in Schritt 3 kannst du die Pipeline jederzeit vollständig über die Konsole steuern:

1. **Pipeline starten:**
   ```bash
   ./scripts/run_on_kaggle.sh
   ```
2. **Status überwachen:**
   Frage den aktuellen Status des Hintergrund-Jobs ab (kann `RUNNING`, `COMPLETE` oder `ERROR` sein):
   ```bash
   kaggle kernels status <dein-username>/bitemb-evaluation-pipeline
   ```
3. **Ergebnisse herunterladen:**
   Sobald der Status `COMPLETE` anzeigt, kannst du die berechneten JSON-Ergebnisse und die neu generierten Diagramme direkt in dein lokales Repository herunterladen:
   ```bash
   kaggle kernels output <dein-username>/bitemb-evaluation-pipeline -p output/
   ```
4. **Dateien kopieren:**
   Kopiere die heruntergeladenen Daten an die richtigen Stellen in deinem lokalen Projektordner:
   ```bash
   cp -r output/cache .
   cp -r output/results .
   rm -rf output/
   ```

---

## Konfiguration in `scripts/run_on_kaggle.sh`

Solltest du deinen Kaggle-Benutzernamen ändern oder das CLI an einem anderen Ort installiert haben, kannst du dies einfach am Anfang der Datei `scripts/run_on_kaggle.sh` anpassen:
```bash
# Kaggle Configuration
KAGGLE_USER="<dein-username>"
KAGGLE_SLUG="bitemb-evaluation-pipeline"
KAGGLE_CLI="~/.local/bin/kaggle"
```
