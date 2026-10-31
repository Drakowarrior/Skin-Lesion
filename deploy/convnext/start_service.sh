#!/bin/bash
# Startup script for isolated ConvNeXt-Small deployment service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=========================================================="
echo "Starting Isolated ConvNeXt-Small Service (Port 8002)"
echo "=========================================================="

# Check virtual environment
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

# Export configuration
export API_PORT="${API_PORT:-8002}"
export API_HOST="${API_HOST:-0.0.0.0}"

echo "Using Python runtime: $(${PYTHON_EXEC} --version)"
echo "Target port: ${API_PORT}"

# Execute uvicorn
exec ${PYTHON_EXEC} -m uvicorn backend.convnext_app:app --host "${API_HOST}" --port "${API_PORT}"
