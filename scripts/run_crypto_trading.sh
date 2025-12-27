#!/bin/bash
#
# 24/7 Crypto Trading Script
#
# This script runs the crypto trading agent in the background with automatic restart.
# Uses screen or tmux for persistent sessions.
#
# Usage:
#   ./scripts/run_crypto_trading.sh start     # Start trading in background
#   ./scripts/run_crypto_trading.sh stop      # Stop trading
#   ./scripts/run_crypto_trading.sh status    # Check status
#   ./scripts/run_crypto_trading.sh logs      # View live logs
#   ./scripts/run_crypto_trading.sh compare   # Run multi-model comparison
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SESSION_NAME="ai-trader-crypto"
LOG_FILE="$PROJECT_DIR/logs/crypto_trading.log"
PID_FILE="$PROJECT_DIR/logs/crypto_trading.pid"
CONFIG_FILE="${2:-$PROJECT_DIR/configs/alpaca_crypto_config.json}"

# Conda environment
CONDA_ENV="ai-trader"
PYTHON_PATH="$HOME/opt/anaconda3/envs/$CONDA_ENV/bin/python"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

check_dependencies() {
    # Check if screen or tmux is available
    if command -v screen &> /dev/null; then
        TERMINAL_MUX="screen"
    elif command -v tmux &> /dev/null; then
        TERMINAL_MUX="tmux"
    else
        error "Neither screen nor tmux is installed. Please install one:"
        echo "  brew install screen"
        echo "  or"
        echo "  brew install tmux"
        exit 1
    fi
}

start_mcp_services() {
    log "Starting MCP services..."
    cd "$PROJECT_DIR"

    # Start MCP services in background
    nohup "$PYTHON_PATH" agent_tools/start_alpaca_services.py > "$PROJECT_DIR/logs/mcp_services.log" 2>&1 &
    MCP_PID=$!
    echo $MCP_PID > "$PROJECT_DIR/logs/mcp_services.pid"

    # Wait for services to start
    sleep 5
    log "MCP services started (PID: $MCP_PID)"
}

stop_mcp_services() {
    if [ -f "$PROJECT_DIR/logs/mcp_services.pid" ]; then
        MCP_PID=$(cat "$PROJECT_DIR/logs/mcp_services.pid")
        if kill -0 $MCP_PID 2>/dev/null; then
            log "Stopping MCP services (PID: $MCP_PID)..."
            kill $MCP_PID 2>/dev/null || true
        fi
        rm -f "$PROJECT_DIR/logs/mcp_services.pid"
    fi
}

start_trading() {
    local mode="${1:-scheduled}"

    check_dependencies

    # Check if already running
    if [ "$TERMINAL_MUX" = "screen" ]; then
        if screen -list | grep -q "$SESSION_NAME"; then
            warn "Trading session already running. Use 'stop' first."
            exit 1
        fi
    else
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            warn "Trading session already running. Use 'stop' first."
            exit 1
        fi
    fi

    log "Starting 24/7 crypto trading..."
    log "Config: $CONFIG_FILE"
    log "Mode: $mode"

    # Start MCP services first
    start_mcp_services

    cd "$PROJECT_DIR"

    # Build the command
    if [ "$mode" = "compare" ]; then
        CMD="$PYTHON_PATH main_alpaca.py $CONFIG_FILE --compare"
    else
        CMD="$PYTHON_PATH main_alpaca.py $CONFIG_FILE --scheduled"
    fi

    # Start in terminal multiplexer
    if [ "$TERMINAL_MUX" = "screen" ]; then
        screen -dmS "$SESSION_NAME" bash -c "
            cd $PROJECT_DIR
            while true; do
                echo '=== Starting trading session at $(date) ===' >> $LOG_FILE
                $CMD 2>&1 | tee -a $LOG_FILE
                echo '=== Session ended at $(date). Restarting in 60 seconds... ===' >> $LOG_FILE
                sleep 60
            done
        "
        log "Trading started in screen session: $SESSION_NAME"
        echo "  View logs:    ./scripts/run_crypto_trading.sh logs"
        echo "  Attach:       screen -r $SESSION_NAME"
        echo "  Detach:       Ctrl+A, then D"
    else
        tmux new-session -d -s "$SESSION_NAME" "
            cd $PROJECT_DIR
            while true; do
                echo '=== Starting trading session at $(date) ===' >> $LOG_FILE
                $CMD 2>&1 | tee -a $LOG_FILE
                echo '=== Session ended at $(date). Restarting in 60 seconds... ===' >> $LOG_FILE
                sleep 60
            done
        "
        log "Trading started in tmux session: $SESSION_NAME"
        echo "  View logs:    ./scripts/run_crypto_trading.sh logs"
        echo "  Attach:       tmux attach -t $SESSION_NAME"
        echo "  Detach:       Ctrl+B, then D"
    fi
}

stop_trading() {
    check_dependencies

    log "Stopping crypto trading..."

    if [ "$TERMINAL_MUX" = "screen" ]; then
        if screen -list | grep -q "$SESSION_NAME"; then
            screen -S "$SESSION_NAME" -X quit
            log "Trading session stopped"
        else
            warn "No active trading session found"
        fi
    else
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux kill-session -t "$SESSION_NAME"
            log "Trading session stopped"
        else
            warn "No active trading session found"
        fi
    fi

    # Stop MCP services
    stop_mcp_services
}

show_status() {
    check_dependencies

    echo ""
    echo "=== AI-Trader Crypto Status ==="
    echo ""

    # Check trading session
    if [ "$TERMINAL_MUX" = "screen" ]; then
        if screen -list | grep -q "$SESSION_NAME"; then
            echo -e "Trading Session: ${GREEN}RUNNING${NC} (screen: $SESSION_NAME)"
        else
            echo -e "Trading Session: ${RED}STOPPED${NC}"
        fi
    else
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo -e "Trading Session: ${GREEN}RUNNING${NC} (tmux: $SESSION_NAME)"
        else
            echo -e "Trading Session: ${RED}STOPPED${NC}"
        fi
    fi

    # Check MCP services
    if [ -f "$PROJECT_DIR/logs/mcp_services.pid" ]; then
        MCP_PID=$(cat "$PROJECT_DIR/logs/mcp_services.pid")
        if kill -0 $MCP_PID 2>/dev/null; then
            echo -e "MCP Services:    ${GREEN}RUNNING${NC} (PID: $MCP_PID)"
        else
            echo -e "MCP Services:    ${RED}STOPPED${NC}"
        fi
    else
        echo -e "MCP Services:    ${RED}STOPPED${NC}"
    fi

    # Show recent log entries
    echo ""
    echo "=== Recent Activity ==="
    if [ -f "$LOG_FILE" ]; then
        tail -10 "$LOG_FILE"
    else
        echo "No logs yet"
    fi
    echo ""
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        error "No log file found at $LOG_FILE"
        exit 1
    fi
}

# Main command handler
case "${1:-help}" in
    start)
        start_trading "scheduled"
        ;;
    compare)
        start_trading "compare"
        ;;
    stop)
        stop_trading
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    restart)
        stop_trading
        sleep 2
        start_trading "scheduled"
        ;;
    *)
        echo "AI-Trader Crypto Trading Script"
        echo ""
        echo "Usage: $0 <command> [config_file]"
        echo ""
        echo "Commands:"
        echo "  start     Start 24/7 scheduled trading in background"
        echo "  compare   Run multi-model comparison once"
        echo "  stop      Stop trading session"
        echo "  status    Show current status"
        echo "  logs      View live logs (Ctrl+C to exit)"
        echo "  restart   Restart trading session"
        echo ""
        echo "Examples:"
        echo "  $0 start                                    # Use default config"
        echo "  $0 start configs/alpaca_crypto_config.json  # Use custom config"
        echo "  $0 compare                                  # Run model comparison"
        echo ""
        ;;
esac
