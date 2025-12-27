#!/bin/bash
#
# Alpaca Paper Trading Quick Start Script
#
# Usage:
#   ./scripts/start_alpaca_paper.sh              # Run single session
#   ./scripts/start_alpaca_paper.sh --scheduled  # Run in scheduled mode
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=========================================="
echo "  Alpaca Paper Trading System"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    echo "   Please copy .env.example to .env and configure your API keys."
    echo ""
    echo "   cp .env.example .env"
    echo "   # Then edit .env with your Alpaca API keys"
    exit 1
fi

# Check for Alpaca credentials
if ! grep -q "ALPACA_API_KEY=" .env || grep -q 'ALPACA_API_KEY=""' .env; then
    echo "❌ Alpaca API key not configured!"
    echo "   Please set ALPACA_API_KEY in .env"
    echo ""
    echo "   Get your keys at: https://alpaca.markets/"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q alpaca-py 2>/dev/null || true

# Start MCP services in background
echo ""
echo "🚀 Starting MCP services..."
python agent_tools/start_alpaca_services.py &
MCP_PID=$!

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 5

# Check if services started
if ! kill -0 $MCP_PID 2>/dev/null; then
    echo "❌ MCP services failed to start"
    exit 1
fi

echo "✅ MCP services running (PID: $MCP_PID)"
echo ""

# Trap to clean up on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $MCP_PID 2>/dev/null || true
    wait $MCP_PID 2>/dev/null || true
    echo "✅ Cleanup complete"
}
trap cleanup EXIT

# Run the trading agent
echo "🤖 Starting trading agent..."
echo ""

if [ "$1" == "--scheduled" ]; then
    python main_alpaca.py --scheduled
else
    python main_alpaca.py "$@"
fi
