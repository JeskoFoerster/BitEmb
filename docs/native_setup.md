# Phase 5 Setup

Phase 5 verwendet keine projektspezifische C-Erweiterung mehr. Die Laufzeitmessungen laufen über vektorisierte NumPy-Operationen, deren innere Schleifen in NumPy/BLAS ausgeführt werden. Dadurch ist kein separater C-Compiler, kein CFFI-Build und keine Installation der Microsoft C++ Build Tools notwendig.

## Ausführen

Schneller Test ohne Dataset-Download:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --synthetic --max-docs 1000 --dims 64
```

Mit Dataset:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --dataset scifact --max-docs 5000
```

Vergleich nach Korpusgröße:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --all --max-docs-list 250 500 1000 --dims 64 768 --output results\phase5\scaling_n
```

Vergleich nach Dimension:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --all --max-docs 1000 --dims 64 128 256 384 768 --output results\phase5\scaling_dim
```

Standardmäßig werden die Ergebnisse unter `results/phase5/` gespeichert. Vergleichsläufe sollten unter `results/phase5/scaling_n/` oder `results/phase5/scaling_dim/` abgelegt werden. `runtime_metrics.json` enthält Einträge mit `"implementation": "numpy_vectorized"`.
