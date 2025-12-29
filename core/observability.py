"""
Observability Layer for AI-Trader

This module provides:
- Real-time metrics collection (PnL, drawdown, Sharpe, win rate)
- Alert system (Slack/email for circuit breaker triggers)
- Audit logging (Signal → TargetPortfolio → OrderIntent → ExecutionReport)
"""

import json
import logging
import os
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional
import math

from .data_structures import (
    Signal,
    TargetPortfolio,
    OrderIntent,
    ExecutionReport,
    Portfolio,
    RiskViolation,
    TradeRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)

    # PnL metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    mtd_pnl: float = 0.0
    ytd_pnl: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    drawdown_duration_days: int = 0

    # Risk-adjusted returns (rolling 30d)
    sharpe_ratio_30d: float = 0.0
    sortino_ratio_30d: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # Position metrics
    avg_position_size: float = 0.0
    avg_holding_period_hours: float = 0.0

    # Model performance (by model)
    model_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and calculates real-time performance metrics.

    Maintains rolling windows of returns for efficient metric calculation.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.04,  # 4% annual risk-free rate
        window_size: int = 30,  # 30-day rolling window
    ):
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size

        # Rolling windows
        self.daily_returns: Deque[float] = deque(maxlen=window_size)
        self.portfolio_values: Deque[float] = deque(maxlen=window_size * 2)

        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.trade_pnls: List[float] = []

        # Peak tracking for drawdown
        self.peak_value: float = initial_capital
        self.drawdown_start: Optional[datetime] = None

        # Current metrics
        self.metrics = PerformanceMetrics()

        # Model-specific tracking
        self._model_trades: Dict[str, List[TradeRecord]] = {}

    def update(self, portfolio: Portfolio) -> PerformanceMetrics:
        """
        Update metrics with current portfolio state.

        Should be called at end of each trading day or after each trade.
        """
        now = datetime.now()
        current_value = portfolio.total_value

        # Update portfolio value history
        self.portfolio_values.append(current_value)

        # Calculate daily return
        if len(self.portfolio_values) >= 2:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                daily_return = (current_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)

        # Update PnL metrics
        self.metrics.total_pnl = current_value - self.initial_capital
        self.metrics.total_pnl_pct = (
            self.metrics.total_pnl / self.initial_capital if self.initial_capital > 0 else 0
        )
        self.metrics.daily_pnl = portfolio.daily_pnl
        self.metrics.daily_pnl_pct = portfolio.daily_pnl_pct

        # Update drawdown
        if current_value > self.peak_value:
            self.peak_value = current_value
            self.drawdown_start = None
        else:
            if self.peak_value > 0:
                self.metrics.current_drawdown = (self.peak_value - current_value) / self.peak_value
                if self.metrics.current_drawdown > self.metrics.max_drawdown:
                    self.metrics.max_drawdown = self.metrics.current_drawdown

            if self.drawdown_start is None and self.metrics.current_drawdown > 0:
                self.drawdown_start = now
            elif self.drawdown_start:
                self.metrics.drawdown_duration_days = (now - self.drawdown_start).days

        # Calculate risk-adjusted returns
        if len(self.daily_returns) >= 5:  # Need at least 5 days
            self._calculate_risk_adjusted_metrics()

        # Update trade statistics
        self._calculate_trade_statistics()

        self.metrics.timestamp = now
        return self.metrics

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade for statistics."""
        self.trades.append(trade)

        # Track by model
        model = trade.model_name or "unknown"
        if model not in self._model_trades:
            self._model_trades[model] = []
        self._model_trades[model].append(trade)

    def _calculate_risk_adjusted_metrics(self) -> None:
        """Calculate Sharpe, Sortino, and Calmar ratios."""
        returns = list(self.daily_returns)
        if not returns:
            return

        # Annualization factor (252 trading days)
        annual_factor = 252

        # Mean and std of returns
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_return = math.sqrt(variance) if variance > 0 else 0

        # Daily risk-free rate
        daily_rf = self.risk_free_rate / annual_factor

        # Sharpe Ratio
        if std_return > 0:
            daily_sharpe = (mean_return - daily_rf) / std_return
            self.metrics.sharpe_ratio_30d = daily_sharpe * math.sqrt(annual_factor)
        else:
            self.metrics.sharpe_ratio_30d = 0

        # Sortino Ratio (downside deviation)
        downside_returns = [r for r in returns if r < daily_rf]
        if downside_returns:
            downside_variance = sum((r - daily_rf) ** 2 for r in downside_returns) / len(downside_returns)
            downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0

            if downside_std > 0:
                daily_sortino = (mean_return - daily_rf) / downside_std
                self.metrics.sortino_ratio_30d = daily_sortino * math.sqrt(annual_factor)
            else:
                self.metrics.sortino_ratio_30d = 0
        else:
            self.metrics.sortino_ratio_30d = self.metrics.sharpe_ratio_30d * 1.5  # Approximation

        # Calmar Ratio (annual return / max drawdown)
        annual_return = mean_return * annual_factor
        if self.metrics.max_drawdown > 0:
            self.metrics.calmar_ratio = annual_return / self.metrics.max_drawdown
        else:
            self.metrics.calmar_ratio = 0

    def _calculate_trade_statistics(self) -> None:
        """Calculate win rate, profit factor, expectancy."""
        if not self.trade_pnls:
            return

        wins = [p for p in self.trade_pnls if p > 0]
        losses = [p for p in self.trade_pnls if p < 0]

        self.metrics.total_trades = len(self.trade_pnls)
        self.metrics.winning_trades = len(wins)
        self.metrics.losing_trades = len(losses)

        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

        if wins:
            self.metrics.avg_win = sum(wins) / len(wins)
        if losses:
            self.metrics.avg_loss = abs(sum(losses) / len(losses))

        # Profit factor = gross profit / gross loss
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        if gross_loss > 0:
            self.metrics.profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            self.metrics.profit_factor = float('inf')
        else:
            self.metrics.profit_factor = 0

        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        self.metrics.expectancy = (
            self.metrics.win_rate * self.metrics.avg_win -
            (1 - self.metrics.win_rate) * self.metrics.avg_loss
        )

    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Get performance metrics for a specific model."""
        trades = self._model_trades.get(model_name, [])
        if not trades:
            return {}

        # Calculate model-specific win rate, etc.
        # This is a simplified version
        return {
            "total_trades": len(trades),
            "trade_count": len(trades),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return asdict(self.metrics)


class AuditLogger:
    """
    Audit logging for complete trade lifecycle.

    Logs: Signal → TargetPortfolio → OrderIntent → ExecutionReport
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        retention_days: int = 7,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

        # Current day's log file
        self._current_date: Optional[str] = None
        self._log_file: Optional[Path] = None

        # In-memory buffer for recent entries
        self._buffer: Deque[Dict] = deque(maxlen=1000)

    def _get_log_file(self) -> Path:
        """Get the log file for today, creating if necessary."""
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self._log_file = self.log_dir / f"audit_{today}.jsonl"
            self._cleanup_old_logs()
        return self._log_file

    def _cleanup_old_logs(self) -> None:
        """Remove logs older than retention period."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        for log_file in self.log_dir.glob("audit_*.jsonl"):
            try:
                date_str = log_file.stem.replace("audit_", "")
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                if file_date < cutoff:
                    log_file.unlink()
                    logger.info(f"Removed old audit log: {log_file}")
            except (ValueError, OSError) as e:
                logger.warning(f"Error cleaning up log file {log_file}: {e}")

    def _write_entry(self, entry_type: str, data: Dict) -> None:
        """Write an audit log entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": entry_type,
            "data": data,
        }

        # Add to buffer
        self._buffer.append(entry)

        # Write to file
        log_file = self._get_log_file()
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError as e:
            logger.error(f"Failed to write audit log: {e}")

    def log_signal(self, signal: Signal) -> None:
        """Log a trading signal."""
        self._write_entry("signal", {
            "id": signal.id,
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "strength": signal.strength,
            "confidence": signal.confidence,
            "target_position_pct": signal.target_position_pct,
            "model_name": signal.model_name,
            "reasoning": signal.reasoning[:500] if signal.reasoning else "",  # Truncate
        })

    def log_target_portfolio(self, target: TargetPortfolio) -> None:
        """Log target portfolio allocation."""
        self._write_entry("target_portfolio", {
            "positions": target.positions,
            "total_exposure": target.total_exposure,
            "net_exposure": target.net_exposure,
            "regime": target.regime.value,
            "model_agreement": target.model_agreement,
            "confidence": target.confidence,
            "signal_count": len(target.source_signals),
        })

    def log_order_intent(self, order: OrderIntent) -> None:
        """Log an order intent (before execution)."""
        self._write_entry("order_intent", {
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "limit_price": order.limit_price,
            "source_signal_id": order.source_signal_id,
        })

    def log_execution_report(self, report: ExecutionReport) -> None:
        """Log an execution report."""
        self._write_entry("execution_report", {
            "order_id": report.order_id,
            "client_order_id": report.client_order_id,
            "symbol": report.symbol,
            "side": report.side,
            "status": report.status.value,
            "filled_qty": report.filled_qty,
            "avg_fill_price": report.avg_fill_price,
            "slippage": report.slippage,
            "commission": report.commission,
            "reject_reason": report.reject_reason,
        })

    def log_risk_violation(self, violation: RiskViolation) -> None:
        """Log a risk constraint violation."""
        self._write_entry("risk_violation", {
            "violation_type": violation.violation_type,
            "description": violation.description,
            "current_value": violation.current_value,
            "limit_value": violation.limit_value,
            "action_taken": violation.action_taken.value,
            "order_rejected": violation.order_rejected,
        })

    def get_recent_entries(self, count: int = 100, entry_type: Optional[str] = None) -> List[Dict]:
        """Get recent audit entries from buffer."""
        entries = list(self._buffer)
        if entry_type:
            entries = [e for e in entries if e["type"] == entry_type]
        return entries[-count:]


class AlertManager:
    """
    Alert system for critical events.

    Supports multiple channels: console, file, Slack, email (webhook-based).
    """

    def __init__(
        self,
        slack_webhook_url: Optional[str] = None,
        email_webhook_url: Optional[str] = None,
        alert_log_file: str = "logs/alerts.jsonl",
    ):
        self.slack_webhook_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.email_webhook_url = email_webhook_url or os.environ.get("EMAIL_WEBHOOK_URL")
        self.alert_log_file = Path(alert_log_file)
        self.alert_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Alert handlers
        self._handlers: List[Callable[[str, str, Dict], None]] = []

        # Rate limiting
        self._last_alert: Dict[str, datetime] = {}
        self._min_interval = timedelta(minutes=5)

    def add_handler(self, handler: Callable[[str, str, Dict], None]) -> None:
        """Add a custom alert handler."""
        self._handlers.append(handler)

    def alert(
        self,
        level: str,  # "INFO", "WARNING", "CRITICAL"
        message: str,
        context: Optional[Dict] = None,
        force: bool = False,
    ) -> None:
        """
        Send an alert.

        Args:
            level: Alert level (INFO, WARNING, CRITICAL)
            message: Alert message
            context: Additional context data
            force: Force send even if rate limited
        """
        context = context or {}

        # Rate limiting (except for forced alerts)
        alert_key = f"{level}:{message[:50]}"
        now = datetime.now()
        if not force and alert_key in self._last_alert:
            if now - self._last_alert[alert_key] < self._min_interval:
                return
        self._last_alert[alert_key] = now

        # Create alert record
        alert_record = {
            "timestamp": now.isoformat(),
            "level": level,
            "message": message,
            "context": context,
        }

        # Log to file
        self._log_to_file(alert_record)

        # Log to console
        self._log_to_console(level, message, context)

        # Send to Slack if configured
        if self.slack_webhook_url and level in ("WARNING", "CRITICAL"):
            self._send_slack(level, message, context)

        # Call custom handlers
        for handler in self._handlers:
            try:
                handler(level, message, context)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

    def _log_to_file(self, record: Dict) -> None:
        """Log alert to file."""
        try:
            with open(self.alert_log_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            logger.error(f"Failed to write alert log: {e}")

    def _log_to_console(self, level: str, message: str, context: Dict) -> None:
        """Log alert to console."""
        log_func = {
            "INFO": logger.info,
            "WARNING": logger.warning,
            "CRITICAL": logger.critical,
        }.get(level, logger.info)

        context_str = f" | Context: {context}" if context else ""
        log_func(f"[ALERT] {message}{context_str}")

    def _send_slack(self, level: str, message: str, context: Dict) -> None:
        """Send alert to Slack webhook."""
        if not self.slack_webhook_url:
            return

        try:
            import urllib.request

            emoji = {"INFO": ":information_source:", "WARNING": ":warning:", "CRITICAL": ":rotating_light:"}.get(level, ":bell:")

            payload = {
                "text": f"{emoji} *{level}*: {message}",
                "attachments": [
                    {
                        "color": {"INFO": "#36a64f", "WARNING": "#ffc107", "CRITICAL": "#dc3545"}.get(level, "#6c757d"),
                        "fields": [
                            {"title": k, "value": str(v), "short": True}
                            for k, v in list(context.items())[:10]
                        ],
                    }
                ] if context else [],
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.slack_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status != 200:
                    logger.warning(f"Slack webhook returned status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    # Convenience methods for common alerts

    def circuit_breaker_triggered(self, action: str, daily_pnl_pct: float) -> None:
        """Alert for circuit breaker trigger."""
        self.alert(
            "CRITICAL",
            f"Circuit breaker triggered: {action}",
            {
                "daily_pnl_pct": f"{daily_pnl_pct * 100:.2f}%",
                "action": action,
            },
            force=True,
        )

    def kill_switch_activated(self, reason: str) -> None:
        """Alert for kill switch activation."""
        self.alert(
            "CRITICAL",
            f"KILL SWITCH ACTIVATED: {reason}",
            {"reason": reason},
            force=True,
        )

    def order_rejected(self, symbol: str, reason: str) -> None:
        """Alert for order rejection."""
        self.alert(
            "WARNING",
            f"Order rejected: {symbol}",
            {"symbol": symbol, "reason": reason},
        )

    def large_position_alert(self, symbol: str, position_pct: float) -> None:
        """Alert for large position concentration."""
        self.alert(
            "WARNING",
            f"Large position: {symbol} at {position_pct*100:.1f}%",
            {"symbol": symbol, "position_pct": f"{position_pct*100:.1f}%"},
        )

    def model_underperformance(self, model_name: str, sharpe: float) -> None:
        """Alert for model underperformance."""
        self.alert(
            "WARNING",
            f"Model underperforming: {model_name}",
            {"model": model_name, "sharpe_30d": f"{sharpe:.2f}"},
        )
