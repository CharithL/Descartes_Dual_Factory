#!/bin/bash
# Download BBP ion channel mechanism .mod files and compile with nrnivmodl.
# These are the Hay et al. 2011 / Bahl et al. 2012 mechanisms used by the L5PC model.
#
# Usage: bash scripts/download_mechanisms.sh
#
# Sources (tried in order):
#   1. BlueBrain/BluePyOpt GitHub repo (fast, reliable)
#   2. ModelDB 139653 direct download (Hay 2011, slower)

set -euo pipefail
cd "$(dirname "$0")/.."

MOD_DIR="mechanisms"
mkdir -p "$MOD_DIR"

echo "=== Downloading BBP ion channel mechanisms ==="

# List of mechanism files needed
MECHANISMS=(
    "Ca_HVA.mod"
    "Ca_LVAst.mod"
    "CaDynamics_E2.mod"
    "Ih.mod"
    "Im.mod"
    "K_Pst.mod"
    "K_Tst.mod"
    "Nap_Et2.mod"
    "NaTa_t.mod"
    "SK_E2.mod"
    "SKv3_1.mod"
)

GITHUB_BASE="https://raw.githubusercontent.com/BlueBrain/BluePyOpt/master/examples/l5pc/mechanisms"

DOWNLOADED=0
for MOD_FILE in "${MECHANISMS[@]}"; do
    if [ -f "$MOD_DIR/$MOD_FILE" ]; then
        echo "  [exists] $MOD_FILE"
        DOWNLOADED=$((DOWNLOADED + 1))
        continue
    fi

    echo -n "  Downloading $MOD_FILE ... "
    if wget -q --timeout=15 --tries=2 -O "$MOD_DIR/$MOD_FILE" "$GITHUB_BASE/$MOD_FILE" 2>/dev/null; then
        echo "OK"
        DOWNLOADED=$((DOWNLOADED + 1))
    elif curl -sf --connect-timeout 15 --retry 2 -o "$MOD_DIR/$MOD_FILE" "$GITHUB_BASE/$MOD_FILE" 2>/dev/null; then
        echo "OK (curl)"
        DOWNLOADED=$((DOWNLOADED + 1))
    else
        echo "FAILED"
        rm -f "$MOD_DIR/$MOD_FILE"
    fi
done

echo ""
echo "Downloaded $DOWNLOADED/${#MECHANISMS[@]} mechanism files."

if [ "$DOWNLOADED" -lt "${#MECHANISMS[@]}" ]; then
    echo "WARNING: Some mechanisms failed to download."
    echo "Trying ModelDB 139653 bulk download as fallback..."

    TMPFILE=$(mktemp /tmp/hay_model_XXXX.zip)
    if wget -q --timeout=60 --tries=2 -O "$TMPFILE" \
        "https://senselab.med.yale.edu/modeldb/eavBinDown?o=139653&a=23&mime=application/zip" 2>/dev/null; then
        echo "  Extracting mod files from ModelDB archive..."
        unzip -o -j "$TMPFILE" "*.mod" -d "$MOD_DIR" 2>/dev/null || true
    fi
    rm -f "$TMPFILE"
fi

echo ""
echo "=== Compiling mechanisms with nrnivmodl ==="
cd "$MOD_DIR"
nrnivmodl .
cd ..

echo ""
echo "=== Done ==="
echo "Compiled mechanisms are in: $MOD_DIR/x86_64/"
echo ""
echo "Now run: python scripts/run_phase1.py"
