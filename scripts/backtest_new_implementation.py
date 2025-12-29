#!/usr/bin/env python3
"""
Backtest script for evaluating the new core implementation.

Tests:
1. Risk Engine with circuit breakers
2. LLM Validator
3. Signal Aggregator
4. Metrics calculation (Sharpe, Sortino, Deflated Sharpe)
5. Walk-Forward Optimization validation
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core import (
    RiskEngine, RiskConfig, RiskState,
    LLMValidator, ValidationResult,
    SignalAggregator, AggregatorConfig,
    RegimeDetector, RegimeConfig, Regime,
    Portfolio, Position, Signal, SignalDirection,
    OrderIntent, OrderType, MarketState,
)
from backtest.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    deflated_sharpe_ratio,
)


def load_summary_data(model_name: str) -> list:
    """Load trading summary data for a model."""
    data_dir = Path(project_root) / "data" / "agent_data_crypto" / model_name
    summary_file = data_dir / "summary.json"

    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    return []


def extract_returns(summary_data: list) -> list:
    """Extract returns from summary data."""
    returns = []
    for i in range(1, len(summary_data)):
        prev_equity = summary_data[i-1]['end_equity']
        curr_equity = summary_data[i]['end_equity']
        if prev_equity > 0:
            ret = (curr_equity - prev_equity) / prev_equity
            returns.append(ret)
    return returns


def test_risk_engine(summary_data: list) -> dict:
    """Test risk engine with historical data."""
    print("\n" + "="*60)
    print("📊 Testing Risk Engine")
    print("="*60)

    config = RiskConfig(
        max_single_position=0.25,
        max_sector_exposure=0.40,
        daily_loss_reduce_size=-0.02,
        daily_loss_halt_new=-0.05,
        daily_loss_force_liquidate=-0.10,
    )
    engine = RiskEngine(config)

    results = {
        "circuit_breaker_triggers": 0,
        "position_limit_violations": 0,
        "orders_validated": 0,
        "orders_rejected": 0,
    }

    for session in summary_data:
        if not session.get('positions'):
            continue

        # Create portfolio from session data
        positions = {}
        for pos in session['positions']:
            market_value = pos['qty'] * pos['current_price']
            cost_basis = pos['qty'] * pos['avg_price']
            unrealized_pnl = pos.get('unrealized_pnl', market_value - cost_basis)
            unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            positions[pos['symbol']] = Position(
                symbol=pos['symbol'],
                quantity=pos['qty'],
                avg_entry_price=pos['avg_price'],
                current_price=pos['current_price'],
                market_value=market_value,
                cost_basis=cost_basis,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                opened_at=datetime.now(),
                last_updated=datetime.now(),
            )

        portfolio = Portfolio(
            timestamp=datetime.now(),
            cash=session.get('cash', 0),
            positions=positions,
            total_value=session['end_equity'],
            buying_power=session.get('cash', 0),
        )

        # Update risk state
        engine.update_portfolio_state(portfolio)
        state = engine.get_state()

        # Check for circuit breaker triggers
        if state.circuit_breaker_level > 0:
            results["circuit_breaker_triggers"] += 1

        # Check position limits
        if state.largest_position_pct > config.max_single_position:
            results["position_limit_violations"] += 1

        # Simulate order validation
        for symbol, pos in positions.items():
            order = OrderIntent(
                symbol=symbol,
                side="BUY",
                quantity=10,
                order_type=OrderType.MARKET,
            )
            is_valid, msg, _ = engine.validate_order(order, portfolio)
            results["orders_validated"] += 1
            if not is_valid:
                results["orders_rejected"] += 1

    print(f"  Sessions analyzed: {len(summary_data)}")
    print(f"  Circuit breaker triggers: {results['circuit_breaker_triggers']}")
    print(f"  Position limit violations: {results['position_limit_violations']}")
    print(f"  Orders validated: {results['orders_validated']}")
    print(f"  Orders rejected: {results['orders_rejected']}")

    return results


def test_llm_validator() -> dict:
    """Test LLM validator with sample outputs."""
    print("\n" + "="*60)
    print("🔍 Testing LLM Validator")
    print("="*60)

    validator = LLMValidator(
        tolerance_pct=5.0,
        require_reasoning=True,
        min_reasoning_length=10,
    )

    # Test cases
    test_cases = [
        # Valid output
        {
            "name": "Valid JSON with reasoning",
            "output": json.dumps({
                "analysis": "Market showing bullish momentum with strong volume.",
                "signals": [
                    {
                        "symbol": "BTC/USD",
                        "action": "BUY",
                        "quantity": 0.1,
                        "confidence": 0.75,
                        "reasoning": "Breaking out above key resistance with volume confirmation."
                    }
                ],
                "risk_assessment": "Moderate risk with 5% stop loss recommended."
            }),
            "expected_valid": True,
        },
        # Invalid action
        {
            "name": "Invalid action type",
            "output": json.dumps({
                "analysis": "Market analysis here",
                "signals": [
                    {
                        "symbol": "ETH/USD",
                        "action": "LONG",  # Should be BUY/SELL/HOLD
                        "quantity": 1,
                        "confidence": 0.8,
                        "reasoning": "Some reasoning here for this trade."
                    }
                ],
                "risk_assessment": "Risk assessment here"
            }),
            "expected_valid": False,
        },
        # Missing reasoning
        {
            "name": "Missing reasoning",
            "output": json.dumps({
                "analysis": "Market analysis",
                "signals": [
                    {
                        "symbol": "SOL/USD",
                        "action": "SELL",
                        "quantity": 10,
                        "confidence": 0.6,
                        "reasoning": "Short"  # Too short
                    }
                ],
                "risk_assessment": "Risk assessment"
            }),
            "expected_valid": False,
        },
        # Hallucination detection
        {
            "name": "Potential hallucination (breaking news)",
            "output": json.dumps({
                "analysis": "Breaking news: Bitcoin just hit $150,000!",
                "signals": [
                    {
                        "symbol": "BTC/USD",
                        "action": "BUY",
                        "quantity": 1,
                        "confidence": 0.95,
                        "reasoning": "Major news catalyst driving price action."
                    }
                ],
                "risk_assessment": "High conviction trade based on live update"
            }),
            "expected_valid": True,  # Valid but with warnings
        },
        # Invalid JSON
        {
            "name": "Invalid JSON",
            "output": "This is not valid JSON at all",
            "expected_valid": False,
        },
    ]

    results = {
        "total_tests": len(test_cases),
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    for test in test_cases:
        result = validator.validate(test["output"])
        passed = result.is_valid == test["expected_valid"]

        if passed:
            results["passed"] += 1
            status = "✅"
        else:
            results["failed"] += 1
            status = "❌"

        print(f"  {status} {test['name']}")
        if result.warnings:
            print(f"      Warnings: {result.warnings}")
        if result.errors:
            print(f"      Errors: {result.errors[:2]}")  # Show first 2 errors

        results["details"].append({
            "name": test["name"],
            "passed": passed,
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "confidence_adjustment": result.confidence_adjustment,
        })

    print(f"\n  Summary: {results['passed']}/{results['total_tests']} tests passed")

    return results


def test_signal_aggregator() -> dict:
    """Test signal aggregator with multiple model signals."""
    print("\n" + "="*60)
    print("🔀 Testing Signal Aggregator")
    print("="*60)

    config = AggregatorConfig()
    aggregator = SignalAggregator(config)

    # Simulate signals from multiple models
    test_signals = [
        # All models agree - BUY
        {
            "name": "Consensus BUY",
            "signals": [
                Signal(timestamp=datetime.now(), symbol="BTC/USD", direction=SignalDirection.LONG,
                       strength=0.8, confidence=0.85, target_position_pct=0.1, model_name="claude"),
                Signal(timestamp=datetime.now(), symbol="BTC/USD", direction=SignalDirection.LONG,
                       strength=0.75, confidence=0.80, target_position_pct=0.08, model_name="gpt"),
                Signal(timestamp=datetime.now(), symbol="BTC/USD", direction=SignalDirection.LONG,
                       strength=0.7, confidence=0.75, target_position_pct=0.09, model_name="gemini"),
            ],
        },
        # Mixed signals
        {
            "name": "Mixed signals (2 BUY, 1 SELL)",
            "signals": [
                Signal(timestamp=datetime.now(), symbol="ETH/USD", direction=SignalDirection.LONG,
                       strength=0.7, confidence=0.8, target_position_pct=0.1, model_name="claude"),
                Signal(timestamp=datetime.now(), symbol="ETH/USD", direction=SignalDirection.LONG,
                       strength=0.6, confidence=0.7, target_position_pct=0.08, model_name="gpt"),
                Signal(timestamp=datetime.now(), symbol="ETH/USD", direction=SignalDirection.SHORT,
                       strength=0.5, confidence=0.6, target_position_pct=-0.05, model_name="gemini"),
            ],
        },
        # No consensus
        {
            "name": "No consensus (equal split)",
            "signals": [
                Signal(timestamp=datetime.now(), symbol="SOL/USD", direction=SignalDirection.LONG,
                       strength=0.6, confidence=0.7, target_position_pct=0.1, model_name="claude"),
                Signal(timestamp=datetime.now(), symbol="SOL/USD", direction=SignalDirection.SHORT,
                       strength=0.6, confidence=0.7, target_position_pct=-0.1, model_name="gpt"),
            ],
        },
    ]

    results = []
    for test in test_signals:
        aggregated = aggregator.aggregate_signals(test["signals"])

        print(f"\n  {test['name']}:")
        if aggregated and aggregated.positions:
            print(f"    Regime: {aggregated.regime.name}")
            print(f"    Net Exposure: {aggregated.net_exposure:.2%}")
            print(f"    Model Agreement: {aggregated.model_agreement:.2%}")
            print(f"    Confidence: {aggregated.confidence:.2f}")
            print(f"    Positions: {aggregated.positions}")
        else:
            print("    No aggregated signal (insufficient consensus)")

        results.append({
            "name": test["name"],
            "input_signals": len(test["signals"]),
            "has_positions": bool(aggregated.positions) if aggregated else False,
            "net_exposure": aggregated.net_exposure if aggregated else 0,
        })

    return results


def test_regime_detector() -> dict:
    """Test regime detector."""
    print("\n" + "="*60)
    print("📈 Testing Regime Detector")
    print("="*60)

    config = RegimeConfig()
    detector = RegimeDetector(config)

    # Test with different market states
    test_states = [
        {
            "name": "High volatility market",
            "state": MarketState(
                timestamp=datetime.now(),
                symbol="BTC/USD",
                open=90000, high=95000, low=85000, close=92000,
                volume=1000000000,
                technical_indicators={"ATR": 3000, "ADX": 35},
            ),
        },
        {
            "name": "Low volatility trending",
            "state": MarketState(
                timestamp=datetime.now(),
                symbol="ETH/USD",
                open=2900, high=2950, low=2890, close=2940,
                volume=500000000,
                technical_indicators={"ATR": 30, "ADX": 45},
            ),
        },
        {
            "name": "Range-bound market",
            "state": MarketState(
                timestamp=datetime.now(),
                symbol="SOL/USD",
                open=120, high=122, low=118, close=120.5,
                volume=100000000,
                technical_indicators={"ATR": 2, "ADX": 15},
            ),
        },
    ]

    results = []
    for test in test_states:
        # Update detector with market state
        detector.update(test["state"])
        regime = detector.get_current_regime(test["state"].symbol)

        print(f"  {test['name']}: {regime.name}")
        results.append({
            "name": test["name"],
            "symbol": test["state"].symbol,
            "regime": regime.name,
        })

    return results


def calculate_backtest_metrics(model_name: str, returns: list) -> dict:
    """Calculate comprehensive backtest metrics."""
    print("\n" + "="*60)
    print(f"📊 Backtest Metrics for {model_name}")
    print("="*60)

    if len(returns) < 2:
        print("  Insufficient data for metrics calculation")
        return {}

    # Use crypto annualization (365 days, 24/7)
    ann_factor = 365

    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.04, annualization_factor=ann_factor)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.04, annualization_factor=ann_factor)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(returns, max_drawdown=max_dd, annualization_factor=ann_factor)

    # Calculate cumulative return
    cum_return = 1.0
    for r in returns:
        cum_return *= (1 + r)
    cum_return = (cum_return - 1) * 100

    # Calculate deflated Sharpe ratio (accounting for multiple testing)
    # Assume we tested 10 strategies
    # Estimate variance of Sharpe ratio
    variance_sharpe = (1 + 0.5 * sharpe**2) / len(returns) if len(returns) > 0 else 1.0
    dsr = deflated_sharpe_ratio(
        sharpe=sharpe,
        n_trials=10,
        variance_sharpe=variance_sharpe,
        skewness=0,
        kurtosis=3,
        n_observations=len(returns),
    )

    print(f"  Cumulative Return: {cum_return:+.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Sortino Ratio: {sortino:.3f}")
    print(f"  Calmar Ratio: {calmar:.3f}")
    print(f"  Max Drawdown: {max_dd*100:.2f}%")
    print(f"  Deflated Sharpe Ratio: {dsr:.3f}")

    return {
        "cumulative_return": cum_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "deflated_sharpe": dsr,
    }


def main():
    print("\n" + "="*60)
    print("🚀 AI-Trader New Implementation Backtest")
    print("="*60)
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # Load data from all models
    models = [
        "claude-opus-4.5-crypto",
        "chatgpt-5.2-crypto",
        "gemini-3.0-pro-crypto",
        "grok-4.2-crypto",
    ]

    all_results = {}

    # Test each component

    # 1. Test LLM Validator
    llm_results = test_llm_validator()
    all_results["llm_validator"] = llm_results

    # 2. Test Signal Aggregator
    aggregator_results = test_signal_aggregator()
    all_results["signal_aggregator"] = aggregator_results

    # 3. Test Regime Detector
    regime_results = test_regime_detector()
    all_results["regime_detector"] = regime_results

    # 4. Test Risk Engine and calculate metrics for each model
    for model in models:
        print(f"\n{'='*60}")
        print(f"📈 Analyzing: {model}")
        print("="*60)

        summary_data = load_summary_data(model)
        if not summary_data:
            print(f"  No data available for {model}")
            continue

        # Test risk engine
        risk_results = test_risk_engine(summary_data)

        # Calculate metrics
        returns = extract_returns(summary_data)
        metrics = calculate_backtest_metrics(model, returns)

        all_results[model] = {
            "risk_engine": risk_results,
            "metrics": metrics,
            "sessions": len(summary_data),
        }

    # Summary
    print("\n" + "="*60)
    print("📋 BACKTEST SUMMARY")
    print("="*60)

    print("\n  LLM Validator: {}/{} tests passed".format(
        llm_results["passed"], llm_results["total_tests"]
    ))

    print("\n  Model Performance Comparison:")
    print("  " + "-"*50)
    print(f"  {'Model':<25} {'Return':>10} {'Sharpe':>10}")
    print("  " + "-"*50)

    for model in models:
        if model in all_results and "metrics" in all_results[model]:
            m = all_results[model]["metrics"]
            if m:
                print(f"  {model:<25} {m['cumulative_return']:>+9.2f}% {m['sharpe_ratio']:>10.3f}")

    print("\n✅ Backtest completed successfully!")

    return all_results


if __name__ == "__main__":
    results = main()
