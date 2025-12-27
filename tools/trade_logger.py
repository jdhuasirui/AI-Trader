"""
Trade Logger - Detailed trade history and summary logging

Provides comprehensive logging for:
1. Trade execution history (trades.jsonl)
2. Daily summaries (daily_summary.json)
3. Performance metrics over time
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TradeLogger:
    """Logger for detailed trade history and performance tracking"""

    def __init__(self, log_path: str, signature: str):
        """
        Initialize trade logger

        Args:
            log_path: Base path for logs (e.g., ./data/agent_data_crypto)
            signature: Agent signature (e.g., gpt-5.2-crypto)
        """
        self.log_path = log_path
        self.signature = signature
        self.base_dir = Path(log_path) / signature
        self.trades_file = self.base_dir / "trades.jsonl"
        self.summary_file = self.base_dir / "summary.json"

        # Ensure directories exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def log_trade(self, trade: Dict[str, Any]) -> None:
        """
        Log a single trade execution

        Args:
            trade: Trade details including:
                - symbol: Trading symbol (e.g., BTC/USD)
                - side: 'buy' or 'sell'
                - qty: Quantity traded
                - price: Execution price
                - value: Total value (qty * price)
                - timestamp: Execution time
                - order_id: Order ID from broker
                - status: Order status
                - model: Model that made the decision
        """
        trade_record = {
            "timestamp": trade.get("timestamp", datetime.now().isoformat()),
            "symbol": trade.get("symbol"),
            "side": trade.get("side"),
            "qty": trade.get("qty"),
            "price": trade.get("price"),
            "value": trade.get("value"),
            "order_id": trade.get("order_id"),
            "status": trade.get("status", "executed"),
            "model": self.signature,
        }

        # Append to trades file
        with open(self.trades_file, "a") as f:
            f.write(json.dumps(trade_record) + "\n")

    def log_session_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log a trading session summary

        Args:
            summary: Session summary including:
                - date: Session date
                - start_equity: Starting portfolio value
                - end_equity: Ending portfolio value
                - pnl: Profit/loss
                - pnl_pct: P&L percentage
                - trades_count: Number of trades
                - positions: Current positions
        """
        summary_record = {
            "date": summary.get("date", datetime.now().strftime("%Y-%m-%d")),
            "timestamp": datetime.now().isoformat(),
            "start_equity": summary.get("start_equity"),
            "end_equity": summary.get("end_equity"),
            "pnl": summary.get("pnl", 0),
            "pnl_pct": summary.get("pnl_pct", 0),
            "trades_count": summary.get("trades_count", 0),
            "positions": summary.get("positions", []),
            "cash": summary.get("cash"),
            "model": self.signature,
        }

        # Load existing summaries or create new list
        summaries = []
        if self.summary_file.exists():
            try:
                with open(self.summary_file, "r") as f:
                    summaries = json.load(f)
            except (json.JSONDecodeError, IOError):
                summaries = []

        # Append new summary
        summaries.append(summary_record)

        # Save updated summaries
        with open(self.summary_file, "w") as f:
            json.dump(summaries, f, indent=2)

    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trade history

        Args:
            limit: Maximum number of trades to return

        Returns:
            List of trade records
        """
        trades = []
        if self.trades_file.exists():
            with open(self.trades_file, "r") as f:
                for line in f:
                    try:
                        trades.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue

        return trades[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary

        Returns:
            Performance metrics including:
                - total_pnl: Total profit/loss
                - total_trades: Total number of trades
                - win_rate: Percentage of profitable trades
                - best_trade: Best single trade
                - worst_trade: Worst single trade
        """
        trades = self.get_trade_history(limit=10000)

        if not trades:
            return {
                "total_pnl": 0,
                "total_trades": 0,
                "win_rate": 0,
                "best_trade": None,
                "worst_trade": None,
            }

        # Calculate metrics
        total_trades = len(trades)
        buy_trades = [t for t in trades if t.get("side") == "buy"]
        sell_trades = [t for t in trades if t.get("side") == "sell"]

        return {
            "total_trades": total_trades,
            "buy_trades": len(buy_trades),
            "sell_trades": len(sell_trades),
            "symbols_traded": list(set(t.get("symbol") for t in trades if t.get("symbol"))),
            "first_trade": trades[0].get("timestamp") if trades else None,
            "last_trade": trades[-1].get("timestamp") if trades else None,
        }

    def export_to_csv(self, output_path: Optional[str] = None) -> str:
        """
        Export trade history to CSV format

        Args:
            output_path: Optional output path. If None, uses default.

        Returns:
            Path to the exported CSV file
        """
        import csv

        output_path = output_path or str(self.base_dir / "trades.csv")
        trades = self.get_trade_history(limit=10000)

        if not trades:
            return output_path

        fieldnames = ["timestamp", "symbol", "side", "qty", "price", "value", "order_id", "status", "model"]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in trades:
                writer.writerow({k: trade.get(k, "") for k in fieldnames})

        return output_path


def get_trade_logger(log_path: str, signature: str) -> TradeLogger:
    """
    Get or create a trade logger instance

    Args:
        log_path: Base path for logs
        signature: Agent signature

    Returns:
        TradeLogger instance
    """
    return TradeLogger(log_path, signature)
