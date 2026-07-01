# Native Build Setup für Phase 5

## Ziel

Phase 5 nutzt einen optionalen nativen C-Kern, um Speicher- und Laufzeitmessungen
für Float32, 4-Bit, 2-Bit und 1-Bit möglichst praktisch zu messen. Dafür muss
unter Windows ein C/C++-Compiler installiert sein.

Eine VS-Code-Extension allein reicht nicht aus. Sie kann beim Bearbeiten von
C-Code helfen, installiert aber normalerweise nicht die benötigte Buildchain.

---

## Benötigte Komponenten

### Erforderlich

- Microsoft C++ Build Tools
- MSVC C/C++ Compiler
- Windows SDK

### Optional

- VS Code Extension: `C/C++` von Microsoft
- VS Code Extension: `CMake Tools`

Die optionalen Extensions sind nur Komfortwerkzeuge. Entscheidend für den Build
ist der Compiler aus den Microsoft C++ Build Tools.

---

## Installation unter Windows

1. Offizielle Microsoft-Seite öffnen:

   <https://visualstudio.microsoft.com/visual-cpp-build-tools/>

2. **Build Tools** herunterladen und starten.

3. Im Installer den Workload auswählen:

   - **Desktop development with C++**

4. Darauf achten, dass diese Komponenten ausgewählt sind:

   - **MSVC v14.x C++ build tools**
   - **Windows 10 SDK** oder **Windows 11 SDK**
   - optional: **C++ CMake tools for Windows**

5. Installation abschließen.

6. Danach ein neues PowerShell-Fenster öffnen.

---

## Build im Projekt prüfen

Im Projektordner ausführen:

```powershell
.\.venv\Scripts\python.exe -m bitemb.native.build_native
```

Wenn der Build erfolgreich ist, wurde das native CFFI-Modul erzeugt und kann von
Phase 5 verwendet werden.

---

## Phase 5 mit synthetischen Daten testen

Zum schnellen Test ohne Dataset-Download:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --synthetic --max-docs 1000 --dims 64 128 256
```

Die Ergebnisse werden gespeichert unter:

```text
results/phase5/memory.json
results/phase5/runtime.json
```

`memory.json` enthält theoretische und praktisch gemessene Speicherwerte.
`runtime.json` enthält theoretische Arbeitsabschätzungen und native Laufzeiten.

---

## Phase 5 mit echtem Dataset ausführen

Beispiel für SciFact:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --dataset scifact --max-docs 5000
```

Für alle Datasets:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --all --max-docs 5000
```

Für größere Messungen kann `--max-docs` erhöht oder weggelassen werden. Top-k ist
brute-force und kann bei großen Korpora sehr langsam werden.

---

## Prüfen, ob der native Backend genutzt wird

Wenn der native Backend fehlt, meldet das Skript:

```text
Native backend is not built; runtime.json contains unavailable native runtime records.
Build it with: python -m bitemb.native.build_native
```

Dann wurde zwar die theoretische Analyse und Speicheranalyse erzeugt, aber keine
echte native Laufzeitmessung.

Wenn der Build erfolgreich war, sollten in `runtime.json` Einträge mit folgendem
Feld stehen:

```json
"implementation": "native_packed"
```

und

```json
"status": "ok"
```

---

## Git und erzeugte Dateien

Der Native-Build und Phase-5-Laeufe erzeugen lokale Artefakte:

```text
Release/
bitemb/native/_bitemb_native.c
bitemb/native/_bitemb_native.*.pyd
results/phase5/
```

Diese Dateien sind Build- oder Ergebnisartefakte und werden nicht committed. Sie
sind in `.gitignore` eingetragen. Committen solltest du nur Quellcode,
Dokumentation und Tests.

## Häufige Fehler

### Microsoft Visual C++ 14.0 or greater is required

Bedeutung: Die MSVC Build Tools fehlen oder sind nicht korrekt installiert.

Lösung:

- Microsoft C++ Build Tools installieren
- Workload **Desktop development with C++** auswählen
- neues PowerShell-Fenster öffnen
- Build erneut ausführen

### cl.exe wird nicht gefunden

Bedeutung: Der Compiler ist nicht im aktuellen Environment verfügbar.

Lösung:

- neues PowerShell-Fenster öffnen
- alternativ über **Developer PowerShell for VS** starten
- erneut ausführen:

```powershell
.\.venv\Scripts\python.exe -m bitemb.native.build_native
```

### VS Code Extension ist installiert, Build geht trotzdem nicht

Das ist erwartbar. Die Extension liefert IntelliSense und Editorfunktionen, aber
nicht zwingend den Compiler. Die Microsoft C++ Build Tools müssen separat
installiert sein.

---

## Empfohlener Ablauf

1. Microsoft C++ Build Tools installieren.
2. Neues PowerShell-Fenster öffnen.
3. Native Erweiterung bauen:

   ```powershell
   .\.venv\Scripts\python.exe -m bitemb.native.build_native
   ```

4. Schnellen Smoke-Test ausführen:

   ```powershell
   .\.venv\Scripts\python.exe scripts\phase5_efficiency.py --synthetic --max-docs 1000 --dims 64
   ```

5. Tests laufen lassen:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests\test_efficiency.py
   ```

6. Danach echte Phase-5-Messungen starten.
