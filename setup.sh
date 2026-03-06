#!/bin/bash
# L5PC DESCARTES — Vast.ai Setup Script
# Run: bash setup.sh
set -euo pipefail

echo "=== L5PC DESCARTES Setup ==="

# 1. System deps
echo "--- Installing system dependencies ---"
apt-get update && apt-get install -y libx11-dev libreadline-dev default-jre wget unzip

# 2. Python deps
echo "--- Installing Python packages ---"
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download Bahl model (ModelDB 146026)
echo "--- Downloading Bahl model (ModelDB 146026) ---"
BAHL_DIR="data/models/bahl"
if [ ! -d "$BAHL_DIR" ]; then
    mkdir -p "$BAHL_DIR"
    wget -q "https://senselab.med.yale.edu/ModelDB/eavBinDown?o=146026&a=23&mime=application/zip" \
         -O /tmp/bahl_model.zip || {
        echo "ModelDB download failed. Trying OpenSourceBrain..."
        git clone https://github.com/OpenSourceBrain/BahlEtAl2012_ReducedL5PyrCell.git "$BAHL_DIR"
    }
    if [ -f /tmp/bahl_model.zip ]; then
        unzip -o /tmp/bahl_model.zip -d "$BAHL_DIR"
        rm /tmp/bahl_model.zip
    fi
    echo "Bahl model downloaded to $BAHL_DIR"
else
    echo "Bahl model already exists, skipping"
fi

# 4. Download Hay model (ModelDB 139653)
echo "--- Downloading Hay model (ModelDB 139653) ---"
HAY_DIR="data/models/hay"
if [ ! -d "$HAY_DIR" ]; then
    mkdir -p "$HAY_DIR"
    wget -q "https://senselab.med.yale.edu/ModelDB/eavBinDown?o=139653&a=23&mime=application/zip" \
         -O /tmp/hay_model.zip || {
        echo "ModelDB download failed. Trying OpenSourceBrain..."
        git clone https://github.com/OpenSourceBrain/L5bPyrCellHayEtAl2011.git "$HAY_DIR"
    }
    if [ -f /tmp/hay_model.zip ]; then
        unzip -o /tmp/hay_model.zip -d "$HAY_DIR"
        rm /tmp/hay_model.zip
    fi
    echo "Hay model downloaded to $HAY_DIR"
else
    echo "Hay model already exists, skipping"
fi

# 5. Compile NEURON mod files for Bahl
echo "--- Compiling Bahl mod files ---"
BAHL_MECH=$(find "$BAHL_DIR" -name "*.mod" -printf "%h\n" | sort -u | head -1)
if [ -n "$BAHL_MECH" ]; then
    cd "$BAHL_MECH"
    nrnivmodl .
    cd -
    echo "Bahl mechanisms compiled"
else
    echo "WARNING: No .mod files found for Bahl model"
fi

# 6. Compile NEURON mod files for Hay
echo "--- Compiling Hay mod files ---"
HAY_MECH=$(find "$HAY_DIR" -name "*.mod" -printf "%h\n" | sort -u | head -1)
if [ -n "$HAY_MECH" ]; then
    cd "$HAY_MECH"
    nrnivmodl .
    cd -
    echo "Hay mechanisms compiled"
else
    echo "WARNING: No .mod files found for Hay model"
fi

# 7. Download Beniaguev pre-trained TCN (Phase 2)
echo "--- Downloading Beniaguev pre-trained TCN ---"
TCN_PATH="data/models/beniaguev_tcn"
if [ ! -d "$TCN_PATH" ]; then
    mkdir -p "$TCN_PATH"
    echo "NOTE: Download pre-trained TCN manually from Kaggle:"
    echo "  https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-tcn"
    echo "  Place NMDA_TCN__DxWxT_7x128x153.h5 in $TCN_PATH/"
else
    echo "TCN directory exists, skipping"
fi

echo ""
echo "=== Setup complete ==="
echo "Run Phase 1: python scripts/run_phase1.py"
echo "Run Phase 2: python scripts/run_phase2.py"
echo "Run Phase 3: python scripts/run_phase3.py"
