import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from pathlib import Path as _Path
from dotenv import load_dotenv

load_dotenv()

from prompts.agent_prompt import all_nasdaq_100_symbols
# Import tools and prompts
from tools.general_tools import get_config_value, write_config_value

# Import core infrastructure for multi-model signal aggregation
try:
    from core import (
        SignalAggregator, AggregatorConfig, ConfidenceCalibrator,
        Signal, SignalDirection, Regime,
        LLMValidator,
        # NEW: Trading Engine with integrated components
        TradingEngine, TradingEngineConfig, EngineMode,
        create_trading_engine,
        # NEW: Advanced components
        DriftDetector, DriftConfig,
        RiskEngine, RiskConfig,
        AdvancedCalibrator, CalibrationConfig,
    )
    SIGNAL_AGGREGATION_AVAILABLE = True
    TRADING_ENGINE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Signal aggregation modules not available: {e}")
    SIGNAL_AGGREGATION_AVAILABLE = False
    TRADING_ENGINE_AVAILABLE = False

# Agent class mapping table - for dynamic import and instantiation
AGENT_REGISTRY = {
    "BaseAgent": {
        "module": "agent.base_agent.base_agent",
        "class": "BaseAgent"
    },
    "BaseAgent_Hour": {
        "module": "agent.base_agent.base_agent_hour",
        "class": "BaseAgent_Hour"
    },
    "BaseAgentAStock": {
        "module": "agent.base_agent_astock.base_agent_astock",
        "class": "BaseAgentAStock"
    },
    "BaseAgentAStock_Hour": {
        "module": "agent.base_agent_astock.base_agent_astock_hour",
        "class": "BaseAgentAStock_Hour"
    },
    "BaseAgentCrypto": {
        "module": "agent.base_agent_crypto.base_agent_crypto",
        "class": "BaseAgentCrypto"
    },
    "AlpacaAgent": {
        "module": "agent.alpaca_agent.alpaca_agent",
        "class": "AlpacaAgent"
    }
}


def get_agent_class(agent_type):
    """
    Dynamically import and return the corresponding class based on agent type name

    Args:
        agent_type: Agent type name (e.g., "BaseAgent")

    Returns:
        Agent class

    Raises:
        ValueError: If agent type is not supported
        ImportError: If unable to import agent module
    """
    if agent_type not in AGENT_REGISTRY:
        supported_types = ", ".join(AGENT_REGISTRY.keys())
        raise ValueError(f"❌ Unsupported agent type: {agent_type}\n" f"   Supported types: {supported_types}")

    agent_info = AGENT_REGISTRY[agent_type]
    module_path = agent_info["module"]
    class_name = agent_info["class"]

    try:
        # Dynamic import module
        import importlib

        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        print(f"✅ Successfully loaded Agent class: {agent_type} (from {module_path})")
        return agent_class
    except ImportError as e:
        raise ImportError(f"❌ Unable to import agent module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"❌ Class {class_name} not found in module {module_path}: {e}")


def load_config(config_path=None):
    """
    Load configuration file from configs directory

    Args:
        config_path: Configuration file path, if None use default config

    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Default configuration file path
        config_path = Path(__file__).parent / "configs" / "default_config.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"❌ Configuration file does not exist: {config_path}")
        exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print(f"✅ Successfully loaded configuration file: {config_path}")
        return config
    except json.JSONDecodeError as e:
        print(f"❌ Configuration file JSON format error: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Failed to load configuration file: {e}")
        exit(1)


async def main(config_path=None):
    """Run trading experiment using BaseAgent class

    Args:
        config_path: Configuration file path, if None use default config
    """
    # Load configuration file
    config = load_config(config_path)

    # Get Agent type
    agent_type = config.get("agent_type", "BaseAgent")
    try:
        AgentClass = get_agent_class(agent_type)
    except (ValueError, ImportError, AttributeError) as e:
        print(str(e))
        exit(1)

    # Get market type from configuration
    market = config.get("market", "us")
    # Auto-detect market from agent_type (BaseAgentAStock always uses CN market)
    if agent_type == "BaseAgentAStock" or agent_type == "BaseAgentAStock_Hour":
        market = "cn"
    elif agent_type == "BaseAgentCrypto":
        market = "crypto"

    if market == "crypto":
        print(f"🌍 Market type: Cryptocurrency (24/7 trading)")
    elif market == "cn":
        print(f"🌍 Market type: A-shares (China)")
    else:
        print(f"🌍 Market type: US stocks")

    # Get date range from configuration file
    INIT_DATE = config["date_range"]["init_date"]
    END_DATE = config["date_range"]["end_date"]

    # Environment variables can override dates in configuration file
    if os.getenv("INIT_DATE"):
        INIT_DATE = os.getenv("INIT_DATE")
        print(f"⚠️  Using environment variable to override INIT_DATE: {INIT_DATE}")
    if os.getenv("END_DATE"):
        END_DATE = os.getenv("END_DATE")
        print(f"⚠️  Using environment variable to override END_DATE: {END_DATE}")

    # Validate date range
    # Support both YYYY-MM-DD and YYYY-MM-DD HH:MM:SS formats
    if ' ' in INIT_DATE:
        INIT_DATE_obj = datetime.strptime(INIT_DATE, "%Y-%m-%d %H:%M:%S")
    else:
        INIT_DATE_obj = datetime.strptime(INIT_DATE, "%Y-%m-%d")
    
    if ' ' in END_DATE:
        END_DATE_obj = datetime.strptime(END_DATE, "%Y-%m-%d %H:%M:%S")
    else:
        END_DATE_obj = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    if INIT_DATE_obj > END_DATE_obj:
        print("❌ INIT_DATE is greater than END_DATE")
        exit(1)

    # Get model list from configuration file (only select enabled models)
    enabled_models = [model for model in config["models"] if model.get("enabled", True)]

    # Get agent configuration
    agent_config = config.get("agent_config", {})
    log_config = config.get("log_config", {})
    max_steps = agent_config.get("max_steps", 10)
    max_retries = agent_config.get("max_retries", 3)
    base_delay = agent_config.get("base_delay", 0.5)
    initial_cash = agent_config.get("initial_cash", 10000.0)
    verbose = agent_config.get("verbose", False)

    # Display enabled model information
    model_names = [m.get("name", m.get("signature")) for m in enabled_models]

    print("🚀 Starting trading experiment")
    print(f"🤖 Agent type: {agent_type}")
    print(f"📅 Date range: {INIT_DATE} to {END_DATE}")
    print(f"🤖 Model list: {model_names}")
    print(
        f"⚙️  Agent config: max_steps={max_steps}, max_retries={max_retries}, base_delay={base_delay}, initial_cash={initial_cash}, verbose={verbose}"
    )

    for model_config in enabled_models:
        # Read basemodel and signature directly from configuration file
        model_name = model_config.get("name", "unknown")
        basemodel = model_config.get("basemodel")
        signature = model_config.get("signature")
        openai_base_url = model_config.get("openai_base_url",None)
        openai_api_key = model_config.get("openai_api_key",None)
        
        # Validate required fields
        if not basemodel:
            print(f"❌ Model {model_name} missing basemodel field")
            continue
        if not signature:
            print(f"❌ Model {model_name} missing signature field")
            continue

        print("=" * 60)
        print(f"🤖 Processing model: {model_name}")
        print(f"📝 Signature: {signature}")
        print(f"🔧 BaseModel: {basemodel}")
            
        # Initialize runtime configuration
        # Use the shared config file from RUNTIME_ENV_PATH in .env
        
        project_root = _Path(__file__).resolve().parent
        
        # Get log path configuration
        log_path = log_config.get("log_path", "./data/agent_data")
        
        # Check position file to determine if this is a fresh start
        position_file = project_root / log_path / signature / "position" / "position.jsonl"
        
        # If position file doesn't exist, reset config to start from INIT_DATE
        if not position_file.exists():
            # Clear the shared config file for fresh start
            from tools.general_tools import _resolve_runtime_env_path
            runtime_env_path = _resolve_runtime_env_path()
            if os.path.exists(runtime_env_path):
                os.remove(runtime_env_path)
                print(f"🔄 Position file not found, cleared config for fresh start from {INIT_DATE}")
        
        # Write config values to shared config file (from .env RUNTIME_ENV_PATH)
        write_config_value("SIGNATURE", signature)
        write_config_value("IF_TRADE", False)
        write_config_value("MARKET", market)
        write_config_value("LOG_PATH", log_path)
        
        print(f"✅ Runtime config initialized: SIGNATURE={signature}, MARKET={market}")

        # Select symbols based on agent type and market
        # Crypto agents don't use stock_symbols parameter
        if agent_type == "BaseAgentCrypto":
            stock_symbols = None  # Crypto agent uses its own crypto_symbols
        elif agent_type == "BaseAgentAStock" or agent_type == "BaseAgentAStock_Hour":
            stock_symbols = None  # Let BaseAgentAStock use its default SSE 50
        elif market == "cn":
            from prompts.agent_prompt import all_sse_50_symbols

            stock_symbols = all_sse_50_symbols
        else:
            stock_symbols = all_nasdaq_100_symbols

        try:
            # Dynamically create Agent instance
            # Crypto agents have different parameter requirements
            if agent_type == "BaseAgentCrypto":
                agent = AgentClass(
                    signature=signature,
                    basemodel=basemodel,
                    log_path=log_path,
                    max_steps=max_steps,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    initial_cash=initial_cash,
                    init_date=INIT_DATE,
                    openai_base_url=openai_base_url,
                    openai_api_key=openai_api_key
                )
            else:
                agent = AgentClass(
                    signature=signature,
                    basemodel=basemodel,
                    stock_symbols=stock_symbols,
                    log_path=log_path,
                    max_steps=max_steps,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    initial_cash=initial_cash,
                    init_date=INIT_DATE,
                    openai_base_url=openai_base_url,
                    openai_api_key=openai_api_key
                )

            print(f"✅ {agent_type} instance created successfully: {agent}")

            # Initialize MCP connection and AI model
            await agent.initialize()
            print("✅ Initialization successful")
            # Run all trading days in date range
            await agent.run_date_range(INIT_DATE, END_DATE)

            # Display final position summary
            summary = agent.get_position_summary()
            # Get currency symbol from agent's actual market (more accurate)
            if agent.market == "crypto":
                currency_symbol = "USDT"
            elif agent.market == "cn":
                currency_symbol = "¥"
            else:
                currency_symbol = "$"
            print(f"📊 Final position summary:")
            print(f"   - Latest date: {summary.get('latest_date')}")
            print(f"   - Total records: {summary.get('total_records')}")
            print(f"   - Cash balance: {currency_symbol}{summary.get('positions', {}).get('CASH', 0):,.2f}")

            # Show crypto positions if this is a crypto agent
            if agent.market == "crypto" and hasattr(agent, 'crypto_symbols'):
                crypto_positions = {k: v for k, v in summary.get('positions', {}).items() if k.endswith('-USDT') and v > 0}
                if crypto_positions:
                    print(f"   - Crypto positions:")
                    for symbol, amount in crypto_positions.items():
                        print(f"     • {symbol}: {amount}")

        except Exception as e:
            print(f"❌ Error processing model {model_name} ({signature}): {str(e)}")
            print(f"📋 Error details: {e}")
            # Can choose to continue processing next model, or exit
            # continue  # Continue processing next model
            exit()  # Or exit program

        print("=" * 60)
        print(f"✅ Model {model_name} ({signature}) processing completed")
        print("=" * 60)

    print("🎉 All models processing completed!")


async def main_multi_model_fusion(config_path=None):
    """
    Run multi-model signal fusion mode with integrated TradingEngine.

    This mode runs multiple LLM agents and aggregates their signals using
    the TradingEngine which integrates:
    - SignalAggregator for consensus-based trading decisions
    - DriftDetector for model health monitoring
    - ConfidenceCalibrator for LLM output calibration
    - RiskEngine for ATR-based position sizing
    - SlippageSimulator for execution cost estimation

    Args:
        config_path: Configuration file path
    """
    if not SIGNAL_AGGREGATION_AVAILABLE:
        print("❌ Signal aggregation modules not available. Install core dependencies.")
        exit(1)

    # Load configuration
    config = load_config(config_path)

    # Check if fusion mode is enabled
    fusion_config = config.get("fusion_config", {})
    if not fusion_config.get("enabled", False):
        print("⚠️ Fusion mode not enabled in config. Running standard mode.")
        await main(config_path)
        return

    print("🔀 Running in Multi-Model Signal Fusion Mode with TradingEngine")

    # Get enabled models
    enabled_models = [model for model in config["models"] if model.get("enabled", True)]
    if len(enabled_models) < 2:
        print("⚠️ Fusion mode requires at least 2 enabled models. Running standard mode.")
        await main(config_path)
        return

    # Get agent and market config
    agent_config = config.get("agent_config", {})
    log_config = config.get("log_config", {})
    market = config.get("market", "us")
    agent_type = config.get("agent_type", "BaseAgent")

    # Auto-detect market from agent_type
    if agent_type in ("BaseAgentAStock", "BaseAgentAStock_Hour"):
        market = "cn"
    elif agent_type == "BaseAgentCrypto":
        market = "crypto"

    # Initialize TradingEngine with integrated components
    if TRADING_ENGINE_AVAILABLE:
        # Create engine configuration
        engine_config = TradingEngineConfig(
            mode=EngineMode.PAPER,
            initial_capital=agent_config.get("initial_cash", 100000.0),
            drift_enabled=fusion_config.get("drift_detection", True),
            calibration_enabled=fusion_config.get("calibration", True),
            risk_enabled=fusion_config.get("risk_management", True),
            slippage_enabled=fusion_config.get("slippage_simulation", True),
            aggregation_enabled=True,
            log_dir=log_config.get("log_path", "./data/agent_data") + "/engine_logs",
        )

        # Configure aggregator
        engine_config.aggregator_config = AggregatorConfig(
            min_models_for_consensus=fusion_config.get("min_models", 2),
            consensus_threshold=fusion_config.get("consensus_threshold", 0.6),
            use_regime_weighting=fusion_config.get("regime_aware", True),
            default_regime=Regime.RANGING,
        )

        # Configure risk engine with ATR-based sizing
        engine_config.risk_config = RiskConfig(
            max_position_pct=0.20,  # 20% max per position
            max_portfolio_risk=0.02,  # 2% portfolio risk per trade
            atr_position_risk_pct=0.01,  # 1% risk per ATR
            atr_multiplier=2.0,
            kelly_fraction=0.25,  # 1/4 Kelly
            max_kelly_position=0.20,
            drawdown_reduce_25pct=-0.05,
            drawdown_reduce_50pct=-0.10,
            drawdown_halt_new=-0.15,
            drawdown_stop_all=-0.20,
        )

        # Configure drift detection
        engine_config.drift_config = DriftConfig(
            adwin_delta=0.002,
            page_hinkley_threshold=50.0,
            setar_threshold=2.0,
            min_samples=30,
            weight_decay_on_drift=0.5,
        )

        # Create and initialize engine
        trading_engine = TradingEngine(engine_config)
        trading_engine.initialize()
        print("✅ TradingEngine initialized with all components")
    else:
        # Fallback to legacy mode
        trading_engine = None
        aggregator_config = AggregatorConfig(
            min_models_for_consensus=fusion_config.get("min_models", 2),
            consensus_threshold=fusion_config.get("consensus_threshold", 0.6),
            use_regime_weighting=fusion_config.get("regime_aware", True),
            default_regime=Regime.RANGING,
        )
        signal_aggregator = SignalAggregator(aggregator_config)
        print("⚠️ TradingEngine not available, using legacy SignalAggregator")

    # Initialize LLM validator
    llm_validator = LLMValidator(
        tolerance_pct=5.0,
        require_reasoning=True,
        min_reasoning_length=20,
    )

    # Get date range
    INIT_DATE = config["date_range"]["init_date"]
    END_DATE = config["date_range"]["end_date"]

    print(f"📅 Date range: {INIT_DATE} to {END_DATE}")
    print(f"🤖 Models for fusion: {[m.get('name', m.get('signature')) for m in enabled_models]}")
    if trading_engine:
        print(f"⚙️ Engine config: drift={engine_config.drift_enabled}, "
              f"calibration={engine_config.calibration_enabled}, "
              f"risk={engine_config.risk_enabled}")

    # Initialize all agents
    agents = []
    AgentClass = get_agent_class(agent_type)

    for model_config in enabled_models:
        signature = model_config.get("signature")
        basemodel = model_config.get("basemodel")

        if not basemodel or not signature:
            continue

        try:
            agent = AgentClass(
                signature=signature,
                basemodel=basemodel,
                log_path=log_config.get("log_path", "./data/agent_data"),
                max_steps=agent_config.get("max_steps", 10),
                initial_cash=agent_config.get("initial_cash", 10000.0),
                init_date=INIT_DATE,
                openai_base_url=model_config.get("openai_base_url"),
                openai_api_key=model_config.get("openai_api_key"),
            )
            await agent.initialize()
            agents.append((signature, agent))

            # Register model with trading engine
            if trading_engine:
                trading_engine.register_model(signature)

            print(f"✅ Initialized agent: {signature}")
        except Exception as e:
            print(f"❌ Failed to initialize {signature}: {e}")

    if len(agents) < 2:
        print("❌ Need at least 2 agents for fusion mode")
        exit(1)

    print(f"\n🔀 Signal fusion ready with {len(agents)} agents")
    print("=" * 60)

    # Display engine and model stats
    print("\n📊 Multi-Model Fusion Summary:")
    print(f"   Agents: {[name for name, _ in agents]}")

    if trading_engine:
        engine_stats = trading_engine.get_engine_stats()
        print(f"   Engine mode: {engine_stats['mode']}")
        print(f"   Portfolio value: ${engine_stats['portfolio_value']:,.2f}")
        print(f"   Registered models: {engine_stats['model_count']}")

        for name, _ in agents:
            model_stats = trading_engine.get_model_stats(name)
            print(f"   - {name}: healthy={model_stats.get('is_healthy', True)}, "
                  f"weight={model_stats.get('weight_adjustment', 1.0):.2f}")
    else:
        print(f"   Aggregator: {aggregator_config}")

    print(f"   Validator stats: {llm_validator.get_validation_stats()}")

    print("\n🎉 Multi-model fusion with TradingEngine setup complete!")
    print("   Components integrated:")
    print("   - DriftDetector (ADWIN + Page-Hinkley) for model health monitoring")
    print("   - ConfidenceCalibrator (Platt/Isotonic) for LLM output calibration")
    print("   - RiskEngine with ATR-based position sizing and drawdown protection")
    print("   - SlippageSimulator for execution cost estimation")
    print("   - SignalAggregator for multi-model consensus")


if __name__ == "__main__":
    import sys

    # Support specifying configuration file through command line arguments
    # Usage: python main.py [config_path] [--fusion]
    # Example: python main.py configs/my_config.json
    # Example: python main.py configs/my_config.json --fusion

    config_path = None
    fusion_mode = False

    for arg in sys.argv[1:]:
        if arg == "--fusion":
            fusion_mode = True
        elif not arg.startswith("-"):
            config_path = arg

    if config_path:
        print(f"📄 Using specified configuration file: {config_path}")
    else:
        print(f"📄 Using default configuration file: configs/default_config.json")

    if fusion_mode:
        print("🔀 Running in multi-model fusion mode")
        asyncio.run(main_multi_model_fusion(config_path))
    else:
        asyncio.run(main(config_path))
