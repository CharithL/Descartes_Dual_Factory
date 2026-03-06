#!/bin/bash
# L5PC DESCARTES — Run all 3 phases sequentially.
# Usage: bash scripts/run_all.sh
#
# For individual phases:
#   python scripts/run_phase1.py
#   python scripts/run_phase2.py
#   python scripts/run_phase3.py

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=================================================="
echo "L5PC DESCARTES — Full Pipeline"
echo "=================================================="

echo ""
echo "--- Phase 1: Bahl Reduced Model ---"
python scripts/run_phase1.py "$@"

echo ""
echo "--- Phase 2: Hay Full Model ---"
python scripts/run_phase2.py "$@"

echo ""
echo "--- Phase 3: Circuit Integration ---"
python scripts/run_phase3.py "$@"

echo ""
echo "=================================================="
echo "ALL PHASES COMPLETE"
echo "=================================================="
