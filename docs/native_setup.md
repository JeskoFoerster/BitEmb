# Phase 5 Setup

Phase 5 verwendet keine projektspezifische C-Erweiterung mehr. Die Laufzeitmessungen laufen ueber vektorisierte NumPy-Operationen, deren innere Schleifen in NumPy/BLAS ausgefuehrt werden. Dadurch ist kein separater C-Compiler, kein CFFI-Build und keine Microsoft C++ Build Tools Installation notwendig.

## Ausfuehren

Schneller Test ohne Dataset-Download:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --synthetic --max-docs 1000 --dims 64
```

Mit Dataset:

```powershell
.\.venv\Scripts\python.exe scripts\phase5_efficiency.py --dataset scifact --max-docs 5000
```

Die Ergebnisse werden unter `results/phase5/` gespeichert. `runtime.json` enthaelt Eintraege mit `"implementation": "numpy_vectorized"`.
