#!/usr/bin/env python3
"""
Alpaca Paper Trading Main Entry Point

This script runs the AI trading agent with Alpaca paper trading.

Usage:
    # Single run (default config)
    python main_alpaca.py

    # With custom config
    python main_alpaca.py configs/alpaca_paper_config.json

    # Scheduled mode (runs at intervals)
    python main_alpaca.py --scheduled

    # With specific model
    python main_alpaca.py --model gpt-4o

Prerequisites:
    1. Set up Alpaca API credentials in .env:
       ALPACA_API_KEY=your_key
       ALPACA_SECRET_KEY=your_secret
       ALPACA_PAPER=true

    2. Start MCP services:
       python agent_tools/start_alpaca_services.py

    3. Run this script
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from agent.alpaca_agent.alpaca_agent import AlpacaAgent
from agent_tools.alpaca_client import AlpacaClientError, get_alpaca_client
from tools.general_tools import write_config_value


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        return json.load(f)


def verify_alpaca_connection() -> bool:
    """Verify Alpaca API connection"""
    try:
        client = get_alpaca_client()
        account = client.get_account()

        print("=" * 50)
        print("  Alpaca Paper Trading Connection")
        print("=" * 50)
        print(f"  Account Status: {account['status']}")
        print(f"  Buying Power:   ${account['buying_power']:,.2f}")
        print(f"  Cash:           ${account['cash']:,.2f}")
        print(f"  Portfolio:      ${account['portfolio_value']:,.2f}")
        print(f"  Equity:         ${account['equity']:,.2f}")

        # Check market status
        clock = client.get_clock()
        if clock["is_open"]:
            print(f"  Market:         OPEN (closes {clock['next_close']})")
        else:
            print(f"  Market:         CLOSED (opens {clock['next_open']})")
        print("=" * 50)

        return True

    except AlpacaClientError as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        print("\nPlease verify your credentials in .env:")
        print("  ALPACA_API_KEY=your_paper_api_key")
        print("  ALPACA_SECRET_KEY=your_paper_secret_key")
        print("  ALPACA_PAPER=true")
        return False


async def run_single_session(config: dict, model_name: str = None):
    """Run a single trading session"""

    # Find enabled model
    models = config.get("models", [])
    enabled_models = [m for m in models if m.get("enabled", True)]

    if model_name:
        # Find specific model
        model_config = next((m for m in models if m["name"] == model_name), None)
        if not model_config:
            print(f"❌ Model '{model_name}' not found in config")
            print(f"   Available models: {[m['name'] for m in models]}")
            return
    elif enabled_models:
        model_config = enabled_models[0]
    else:
        print("❌ No enabled models in configuration")
        return

    print(f"\n🤖 Using model: {model_config['name']} ({model_config['basemodel']})")

    # Create agent
    agent_config = config.get("agent_config", {})
    log_config = config.get("log_config", {})

    # Get API key - either directly or from environment variable
    api_key = model_config.get("openai_api_key")
    if not api_key and model_config.get("openai_api_key_env"):
        api_key = os.getenv(model_config["openai_api_key_env"])

    # Get symbols - support both stock_symbols and crypto_symbols
    symbols = config.get("stock_symbols") or config.get("crypto_symbols")

    agent = AlpacaAgent(
        signature=model_config["signature"],
        basemodel=model_config["basemodel"],
        stock_symbols=symbols,
        log_path=log_config.get("log_path", "./data/agent_data_alpaca"),
        max_steps=agent_config.get("max_steps", 15),
        max_retries=agent_config.get("max_retries", 3),
        base_delay=agent_config.get("base_delay", 1.0),
        verbose=agent_config.get("verbose", False),
        check_market_hours=agent_config.get("check_market_hours", True),
        openai_base_url=model_config.get("openai_base_url"),
        openai_api_key=api_key,
    )

    # Set global config values
    write_config_value("SIGNATURE", model_config["signature"])
    write_config_value("LOG_PATH", log_config.get("log_path", "./data/agent_data_alpaca"))
    write_config_value("MARKET", "us")

    # Initialize and run
    try:
        await agent.initialize()
        agent.register_agent()
        await agent.run_trading_session()

    except Exception as e:
        print(f"❌ Error during trading session: {e}")
        raise


async def run_multi_model_comparison(config: dict, virtual_mode: bool = True):
    """Run multiple models side by side for comparison

    Each model runs independently with virtual cash allocation.
    This allows comparing different AI models' trading decisions without interference.

    Args:
        config: Configuration dict
        virtual_mode: If True, each model gets virtual cash allocation (default: True)
    """
    models = config.get("models", [])
    enabled_models = [m for m in models if m.get("enabled", True)]

    if len(enabled_models) < 2:
        print("❌ Need at least 2 enabled models for comparison")
        print(f"   Currently enabled: {[m['name'] for m in enabled_models]}")
        return

    # Get total account value and calculate allocation per model
    client = get_alpaca_client()
    account = client.get_account()
    total_equity = account['equity']
    allocation_per_model = total_equity / len(enabled_models)

    print("\n" + "=" * 60)
    print("  MULTI-MODEL COMPARISON MODE")
    print("=" * 60)
    print(f"  Total Account Equity: ${total_equity:,.2f}")
    print(f"  Models: {[m['name'] for m in enabled_models]}")
    print(f"  Allocation per model: ${allocation_per_model:,.2f}")
    if virtual_mode:
        print(f"  Mode: VIRTUAL (recommendations only, no real trades)")
    else:
        print(f"  Mode: LIVE (real trades will be executed)")
    print("=" * 60 + "\n")

    agent_config = config.get("agent_config", {})
    log_config = config.get("log_config", {})
    results = {}

    for i, model_config in enumerate(enabled_models):
        print("\n" + "=" * 60)
        print(f"  [{i+1}/{len(enabled_models)}] Running: {model_config['name']}")
        print(f"  💰 Virtual Allocation: ${allocation_per_model:,.2f}")
        print("=" * 60)

        try:
            # Get API key - either directly or from environment variable
            api_key = model_config.get("openai_api_key")
            if not api_key and model_config.get("openai_api_key_env"):
                api_key = os.getenv(model_config["openai_api_key_env"])

            # Get symbols - support both stock_symbols and crypto_symbols
            symbols = config.get("stock_symbols") or config.get("crypto_symbols")

            agent = AlpacaAgent(
                signature=model_config["signature"],
                basemodel=model_config["basemodel"],
                stock_symbols=symbols,
                log_path=log_config.get("log_path", "./data/agent_data_alpaca"),
                max_steps=agent_config.get("max_steps", 15),
                max_retries=agent_config.get("max_retries", 3),
                base_delay=agent_config.get("base_delay", 1.0),
                verbose=agent_config.get("verbose", False),
                check_market_hours=agent_config.get("check_market_hours", True),
                allocated_cash=allocation_per_model,
                virtual_mode=virtual_mode,
                openai_base_url=model_config.get("openai_base_url"),
                openai_api_key=api_key,
            )

            # Set global config for this model
            write_config_value("SIGNATURE", model_config["signature"])
            write_config_value("LOG_PATH", log_config.get("log_path", "./data/agent_data_alpaca"))
            write_config_value("MARKET", "us")
            write_config_value("ALLOCATED_CASH", allocation_per_model)
            write_config_value("VIRTUAL_MODE", virtual_mode)

            await agent.initialize()
            agent.register_agent()

            # Load virtual portfolio if exists
            if virtual_mode:
                agent.load_virtual_portfolio()

            await agent.run_trading_session()

            # Save virtual portfolio
            if virtual_mode:
                agent.save_virtual_portfolio()
                summary = agent.get_virtual_portfolio_summary()
                results[model_config['name']] = {
                    "status": "success",
                    "signature": model_config["signature"],
                    "portfolio_value": summary['portfolio_value'],
                    "pnl": summary['pnl'],
                    "cash": summary['cash'],
                    "positions": len(summary['positions'])
                }
            else:
                results[model_config['name']] = {
                    "status": "success",
                    "signature": model_config["signature"]
                }

        except Exception as e:
            print(f"❌ Error running {model_config['name']}: {e}")
            results[model_config['name']] = {
                "status": "error",
                "error": str(e)
            }

    # Print comparison summary
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    log_path = log_config.get("log_path", "./data/agent_data_alpaca")

    for model_name, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"\n  {status_icon} {model_name}")

        if result["status"] == "success":
            print(f"     📁 Logs: {log_path}/{result['signature']}/")
            if virtual_mode and 'portfolio_value' in result:
                pnl = result['pnl']
                pnl_pct = (pnl / allocation_per_model) * 100 if allocation_per_model > 0 else 0
                pnl_icon = "📈" if pnl >= 0 else "📉"
                print(f"     💰 Portfolio: ${result['portfolio_value']:,.2f}")
                print(f"     {pnl_icon} P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                print(f"     📊 Positions: {result['positions']}")
        else:
            print(f"     ❌ Error: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)

    # Rank models by P&L if in virtual mode
    if virtual_mode:
        successful = {k: v for k, v in results.items() if v['status'] == 'success' and 'pnl' in v}
        if successful:
            ranked = sorted(successful.items(), key=lambda x: x[1]['pnl'], reverse=True)
            print("\n  🏆 RANKING BY P&L:")
            for rank, (name, data) in enumerate(ranked, 1):
                medal = ["🥇", "🥈", "🥉"][rank-1] if rank <= 3 else f"#{rank}"
                pnl = data['pnl']
                print(f"     {medal} {name}: ${pnl:+,.2f}")
        print("=" * 60)

    print("\n💡 Compare detailed results by reviewing each model's log files")


async def run_scheduled(config: dict, model_name: str = None):
    """Run trading in scheduled mode"""

    schedule_config = config.get("trading_schedule", {})
    interval = schedule_config.get("run_interval_minutes", 60)

    # Find model
    models = config.get("models", [])
    enabled_models = [m for m in models if m.get("enabled", True)]

    if model_name:
        model_config = next((m for m in models if m["name"] == model_name), None)
        if not model_config:
            print(f"❌ Model '{model_name}' not found")
            return
    elif enabled_models:
        model_config = enabled_models[0]
    else:
        print("❌ No enabled models")
        return

    print(f"\n🤖 Using model: {model_config['name']}")
    print(f"⏰ Running in scheduled mode (interval: {interval} minutes)")

    # Create agent
    agent_config = config.get("agent_config", {})
    log_config = config.get("log_config", {})

    # Get API key - either directly or from environment variable
    api_key = model_config.get("openai_api_key")
    if not api_key and model_config.get("openai_api_key_env"):
        api_key = os.getenv(model_config["openai_api_key_env"])

    # Get symbols - support both stock_symbols and crypto_symbols
    symbols = config.get("stock_symbols") or config.get("crypto_symbols")

    agent = AlpacaAgent(
        signature=model_config["signature"],
        basemodel=model_config["basemodel"],
        stock_symbols=symbols,
        log_path=log_config.get("log_path", "./data/agent_data_alpaca"),
        max_steps=agent_config.get("max_steps", 15),
        max_retries=agent_config.get("max_retries", 3),
        base_delay=agent_config.get("base_delay", 1.0),
        verbose=agent_config.get("verbose", False),
        check_market_hours=agent_config.get("check_market_hours", True),
        openai_base_url=model_config.get("openai_base_url"),
        openai_api_key=api_key,
    )

    # Set global config
    write_config_value("SIGNATURE", model_config["signature"])
    write_config_value("LOG_PATH", log_config.get("log_path", "./data/agent_data_alpaca"))
    write_config_value("MARKET", "us")

    # Initialize
    await agent.initialize()
    agent.register_agent()

    # Run scheduled
    await agent.run_scheduled(interval_minutes=interval)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Alpaca Paper Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_alpaca.py                          # Single run with default config
  python main_alpaca.py --model gpt-4o-mini     # Use specific model
  python main_alpaca.py --scheduled             # Run in scheduled mode
  python main_alpaca.py --compare               # Compare multiple models side by side
  python main_alpaca.py custom_config.json      # Use custom config file
        """
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="configs/alpaca_paper_config.json",
        help="Path to configuration file (default: configs/alpaca_paper_config.json)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Specify which model to use"
    )
    parser.add_argument(
        "--scheduled", "-s",
        action="store_true",
        help="Run in scheduled mode (at regular intervals)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Run all enabled models side by side for comparison"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify Alpaca connection, don't run trading"
    )

    args = parser.parse_args()

    # Verify Alpaca connection
    if not verify_alpaca_connection():
        sys.exit(1)

    if args.verify_only:
        print("\n✅ Alpaca connection verified successfully")
        sys.exit(0)

    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    print(f"\n📄 Loading config: {config_path}")
    config = load_config(config_path)

    # Run
    try:
        if args.compare:
            asyncio.run(run_multi_model_comparison(config))
        elif args.scheduled:
            asyncio.run(run_scheduled(config, args.model))
        else:
            asyncio.run(run_single_session(config, args.model))

    except KeyboardInterrupt:
        print("\n\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
