#!/bin/bash
#
# Run the real-time trading dashboard
#
# Usage:
#   ./scripts/run_dashboard.sh        # Run on default port 8080
#   ./scripts/run_dashboard.sh 3000   # Run on custom port
#

cd "$(dirname "$0")/.."

PORT=${1:-8080}

echo "=========================================="
echo "  AI Trading Dashboard"
echo "=========================================="
echo ""
echo "Starting dashboard on http://localhost:${PORT}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Check if Flask is installed
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flask dependencies..."
    pip install flask flask-cors
fi

# Run the dashboard
DASHBOARD_PORT=$PORT python dashboard/app.py
