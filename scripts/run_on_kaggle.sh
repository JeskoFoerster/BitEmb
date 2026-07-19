#!/bin/bash
# Helper script to run the BitEmb evaluation pipeline on a free Kaggle GPU.

set -e

# Kaggle Configuration
KAGGLE_USER="jeskofoerster"
KAGGLE_SLUG="bitemb-evaluation-pipeline"
KAGGLE_CLI="/home/jesko/.local/bin/kaggle"

REPO_DIR=$(pwd)
JOB_DIR="$REPO_DIR/kaggle_job"

echo "=== Preparing Kaggle Job Directory ==="
rm -rf "$JOB_DIR"
mkdir -p "$JOB_DIR"

echo "=== Creating Project ZIP Archive ==="
# Zip project files excluding pycache and pdfs/assets
zip -r "$JOB_DIR/project.zip" bitemb/ scripts/ requirements.txt pyproject.toml Makefile \
    -x "**/__pycache__/*" -x "*.pdf" -x "*.png" -x "*.jpg" -x "*.zip" > /dev/null

# Convert zip to base64
BASE64_ZIP=$(base64 -w 0 "$JOB_DIR/project.zip")
rm -f "$JOB_DIR/project.zip"

# Create the main script that Kaggle will execute
echo "=== Creating run_pipeline.py with embedded source code ==="
cat << EOF > "$JOB_DIR/run_pipeline.py"
import base64
import io
import zipfile
import subprocess
import os
import shutil
from pathlib import Path

# Let PyTorch use the Kaggle GPU (T4 is highly recommended)
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

ZIP_DATA = b"$BASE64_ZIP"

print("=== PIPELINE RUNNING ON KAGGLE GPU ===")
print("Extracting project source code...")
z = zipfile.ZipFile(io.BytesIO(base64.b64decode(ZIP_DATA)))
z.extractall("/kaggle/working")

# Change working directory to /kaggle/working/
os.chdir("/kaggle/working")

# Create symlinks or output directories in current working dir
os.makedirs("results", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# Install requirements
print("\n--- Installing PyTorch & dependencies ---")
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

# Run Phase 2
print("\n--- Running Phase 2 (Distance Distortion) ---")
subprocess.run(["python3", "scripts/phase2_distance_analysis.py", "--dataset", "scifact"], check=True, env=dict(os.environ, PYTHONPATH="."))

# Run Phase 3
print("\n--- Running Phase 3 (Neighborhood Preservation) ---")
subprocess.run(["python3", "scripts/phase3_neighborhood.py", "--dataset", "scifact"], check=True, env=dict(os.environ, PYTHONPATH="."))

# Run Phase 4
print("\n--- Running Phase 4 (Exact Retrieval) ---")
subprocess.run(["python3", "scripts/phase4_retrieval.py", "--dataset", "scifact"], check=True, env=dict(os.environ, PYTHONPATH="."))

# Run Phase 5
print("\n--- Running Phase 5 (NumPy Efficiency) ---")
subprocess.run(["python3", "scripts/phase5_efficiency.py", "--dataset", "scifact", "--max-docs", "1000"], check=True, env=dict(os.environ, PYTHONPATH="."))

# Run Trade-off calculations and plots
print("\n--- Calculating trade-off metrics & generating plots ---")
subprocess.run(["python3", "scripts/calculate_tradeoff_metrics.py"], check=True)

print("\n=== Pipeline completed successfully! ===")
EOF

# Initialize kaggle metadata
echo "=== Generating Kaggle Metadata ==="
cd "$JOB_DIR"
cat << EOF > kernel-metadata.json
{
  "id": "$KAGGLE_USER/$KAGGLE_SLUG",
  "title": "BitEmb Evaluation Pipeline",
  "code_file": "run_pipeline.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
EOF

echo "=== Pushing Job to Kaggle ==="
"$KAGGLE_CLI" kernels push -p "$JOB_DIR"

echo ""
echo "========================================================================="
echo " JOB SUBMITTED SUCCESSFULLY!"
echo "========================================================================="

