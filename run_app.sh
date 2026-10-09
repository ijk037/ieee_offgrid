#!/usr/bin/env bash
# =============================================================================
# CivicPulse AI — Local Application Launcher
# Person B Integration: Launches the full end-to-end backend & UXPilot portal
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "  Starting CivicPulse AI — Urban Development & Civic Intelligence Portal"
echo "========================================================================"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

PORT="${1:-8000}"

echo "[INFO] Launching CivicPulse AI backend and interactive dashboard on port $PORT..."
echo "[INFO] Dashboard URL: http://localhost:$PORT"
echo "[INFO] API Data URL:  http://localhost:$PORT/api/data"
echo ""

# Execute app.py
python3 app.py "$PORT"
