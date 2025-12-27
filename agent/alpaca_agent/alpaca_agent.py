"""
AlpacaAgent - Real-time trading agent using Alpaca Paper Trading API

This agent connects to Alpaca's paper trading environment for live market simulation.
Key differences from BaseAgent:
1. Uses real-time market data instead of historical data
2. Executes trades through Alpaca API instead of local file simulation
3. Runs in real-time with actual market hours
4. Gets positions from Alpaca account instead of local files
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.globals import set_debug, set_verbose
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from agent.base_agent.base_agent import BaseAgent, DeepSeekChatOpenAI
from agent_tools.alpaca_client import AlpacaClient, AlpacaClientError, get_alpaca_client
from prompts.agent_prompt_alpaca import STOP_SIGNAL, get_alpaca_system_prompt
from tools.general_tools import extract_conversation, extract_tool_messages, get_config_value, write_config_value
from tools.trade_logger import TradeLogger, get_trade_logger

# Best-effort import for console callback handler
try:
    from langchain.callbacks.stdout import StdOutCallbackHandler as _ConsoleHandler
except Exception:
    try:
        from langchain.callbacks import StdOutCallbackHandler as _ConsoleHandler
    except Exception:
        try:
            from langchain_core.callbacks.stdout import StdOutCallbackHandler as _ConsoleHandler
        except Exception:
            _ConsoleHandler = None

load_dotenv()


class AlpacaAgent(BaseAgent):
    """
    Real-time trading agent using Alpaca Paper Trading API.

    This agent extends BaseAgent with the following key differences:
    - Uses Alpaca MCP tools for trading and price data
    - Connects to real market data (with possible 15-min delay on free tier)
    - Executes orders through Alpaca's paper trading API
    - Respects actual market hours

    Usage:
        agent = AlpacaAgent(
            signature="gpt-4o-alpaca",
            basemodel="gpt-4o"
        )
        await agent.initialize()
        await agent.run_trading_session()  # Runs with current time
    """

    # Top traded US stocks for Alpaca trading
    DEFAULT_STOCK_SYMBOLS = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "NFLX",
        "COST", "PEP", "ADBE", "CSCO", "INTC", "QCOM", "TXN", "AMGN", "SBUX", "PYPL",
        "ASML", "ISRG", "AMAT", "LRCX", "KLAC", "MRVL", "PANW", "CRWD", "SNPS", "CDNS",
    ]

    def __init__(
        self,
        signature: str,
        basemodel: str,
        stock_symbols: Optional[List[str]] = None,
        mcp_config: Optional[Dict[str, Dict[str, Any]]] = None,
        log_path: Optional[str] = None,
        max_steps: int = 15,
        max_retries: int = 3,
        base_delay: float = 1.0,
        openai_base_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        verbose: bool = False,
        # Alpaca-specific options
        check_market_hours: bool = True,
        max_position_pct: float = 0.25,
        max_order_value: float = 5000.0,
        # Virtual allocation options
        allocated_cash: Optional[float] = None,
        virtual_mode: bool = False,
    ):
        """
        Initialize AlpacaAgent.

        Args:
            signature: Agent signature/name
            basemodel: Base model name (e.g., "gpt-4o", "claude-3-sonnet")
            stock_symbols: List of stock symbols to trade
            mcp_config: MCP tool configuration (uses Alpaca tools by default)
            log_path: Log path, defaults to ./data/agent_data_alpaca
            max_steps: Maximum reasoning steps per session
            max_retries: Maximum retry attempts
            base_delay: Base delay time for retries
            openai_base_url: OpenAI API base URL
            openai_api_key: OpenAI API key
            verbose: Enable verbose output
            check_market_hours: Whether to check market hours before trading
            max_position_pct: Maximum position as percentage of portfolio
            max_order_value: Maximum single order value
            allocated_cash: Virtual cash allocation for this agent (for multi-model comparison)
            virtual_mode: If True, track virtual portfolio without executing real trades
        """
        # Use Alpaca-specific defaults
        if log_path is None:
            log_path = "./data/agent_data_alpaca"

        if stock_symbols is None:
            stock_symbols = self.DEFAULT_STOCK_SYMBOLS

        # Initialize base class (skip initial_cash as Alpaca manages this)
        super().__init__(
            signature=signature,
            basemodel=basemodel,
            stock_symbols=stock_symbols,
            mcp_config=mcp_config,
            log_path=log_path,
            max_steps=max_steps,
            max_retries=max_retries,
            base_delay=base_delay,
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            initial_cash=allocated_cash or 0.0,
            init_date=datetime.now().strftime("%Y-%m-%d"),
            market="us",  # Alpaca only supports US markets
            verbose=verbose,
        )

        # Alpaca-specific settings
        self.check_market_hours = check_market_hours
        self.max_position_pct = max_position_pct
        self.max_order_value = max_order_value

        # Virtual allocation settings
        self.allocated_cash = allocated_cash
        self.virtual_mode = virtual_mode
        self.virtual_portfolio: Dict[str, float] = {}  # symbol -> quantity
        self.virtual_cash: float = allocated_cash or 0.0

        # Alpaca client (will be initialized on first use)
        self._alpaca_client: Optional[AlpacaClient] = None

        # Trade logger for detailed history
        self.trade_logger = get_trade_logger(log_path, signature)
        self.session_start_equity: float = 0.0

    def _get_default_mcp_config(self) -> Dict[str, Dict[str, Any]]:
        """Get Alpaca-specific MCP configuration"""
        # Support both localhost (local dev) and Docker service names
        math_host = os.getenv('MCP_MATH_HOST', 'localhost')
        search_host = os.getenv('MCP_SEARCH_HOST', 'localhost')
        price_host = os.getenv('MCP_PRICE_HOST', 'localhost')
        trade_host = os.getenv('MCP_TRADE_HOST', 'localhost')

        return {
            "math": {
                "transport": "streamable_http",
                "url": f"http://{math_host}:{os.getenv('MATH_HTTP_PORT', '8000')}/mcp",
            },
            "alpaca_price": {
                "transport": "streamable_http",
                "url": f"http://{price_host}:{os.getenv('ALPACA_PRICE_PORT', '8010')}/mcp",
            },
            "search": {
                "transport": "streamable_http",
                "url": f"http://{search_host}:{os.getenv('SEARCH_HTTP_PORT', '8001')}/mcp",
            },
            "alpaca_trade": {
                "transport": "streamable_http",
                "url": f"http://{trade_host}:{os.getenv('ALPACA_TRADE_PORT', '8011')}/mcp",
            },
        }

    @property
    def alpaca_client(self) -> AlpacaClient:
        """Get Alpaca client (lazy initialization)"""
        if self._alpaca_client is None:
            self._alpaca_client = get_alpaca_client()
        return self._alpaca_client

    def _get_virtual_portfolio_path(self) -> Path:
        """Get path to virtual portfolio file"""
        return Path(self.data_path) / "virtual_portfolio.json"

    def load_virtual_portfolio(self) -> None:
        """Load virtual portfolio from file"""
        portfolio_path = self._get_virtual_portfolio_path()
        if portfolio_path.exists():
            try:
                with open(portfolio_path, 'r') as f:
                    data = json.load(f)
                self.virtual_portfolio = data.get('positions', {})
                self.virtual_cash = data.get('cash', self.allocated_cash or 0.0)
                print(f"📂 Loaded virtual portfolio: ${self.virtual_cash:,.2f} cash, {len(self.virtual_portfolio)} positions")
            except Exception as e:
                print(f"⚠️ Failed to load virtual portfolio: {e}")
                self.virtual_portfolio = {}
                self.virtual_cash = self.allocated_cash or 0.0
        else:
            # Initialize new virtual portfolio
            self.virtual_portfolio = {}
            self.virtual_cash = self.allocated_cash or 0.0

    def save_virtual_portfolio(self) -> None:
        """Save virtual portfolio to file"""
        portfolio_path = self._get_virtual_portfolio_path()
        try:
            data = {
                'cash': self.virtual_cash,
                'positions': self.virtual_portfolio,
                'updated_at': datetime.now().isoformat(),
                'signature': self.signature
            }
            with open(portfolio_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save virtual portfolio: {e}")

    def get_virtual_portfolio_value(self) -> float:
        """Calculate total virtual portfolio value"""
        total = self.virtual_cash
        for symbol, qty in self.virtual_portfolio.items():
            if qty > 0:
                try:
                    quote = self.alpaca_client.get_quote(symbol)
                    price = quote.get('mid_price') or quote.get('bid_price', 0)
                    total += qty * price
                except Exception:
                    pass  # Skip if can't get price
        return total

    def get_virtual_portfolio_summary(self) -> Dict[str, Any]:
        """Get virtual portfolio summary for display"""
        portfolio_value = self.get_virtual_portfolio_value()
        positions = []
        for symbol, qty in self.virtual_portfolio.items():
            if qty > 0:
                try:
                    quote = self.alpaca_client.get_quote(symbol)
                    price = quote.get('mid_price') or quote.get('bid_price', 0)
                    positions.append({
                        'symbol': symbol,
                        'quantity': qty,
                        'price': price,
                        'value': qty * price
                    })
                except Exception:
                    positions.append({
                        'symbol': symbol,
                        'quantity': qty,
                        'price': 0,
                        'value': 0
                    })
        return {
            'cash': self.virtual_cash,
            'portfolio_value': portfolio_value,
            'positions': positions,
            'allocated_cash': self.allocated_cash,
            'pnl': portfolio_value - (self.allocated_cash or 0)
        }

    async def initialize(self) -> None:
        """Initialize agent with Alpaca connection verification"""
        print(f"🚀 Initializing Alpaca agent: {self.signature}")

        # Verify Alpaca connection first
        try:
            account = self.alpaca_client.get_account()
            print(f"✅ Connected to Alpaca Paper Trading")
            print(f"   Account Status: {account['status']}")
            print(f"   Buying Power: ${account['buying_power']:,.2f}")
            print(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
        except AlpacaClientError as e:
            raise RuntimeError(
                f"❌ Failed to connect to Alpaca: {e}\n"
                f"   Please verify ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )

        # Check market status
        clock = self.alpaca_client.get_clock()
        if clock["is_open"]:
            print(f"   Market Status: OPEN (closes at {clock['next_close']})")
        else:
            print(f"   Market Status: CLOSED (opens at {clock['next_open']})")

        # Initialize parent (MCP client, AI model)
        await super().initialize()

    async def run_trading_session(self, today_date: Optional[str] = None) -> None:
        """
        Run a trading session.

        For Alpaca, if no date is provided, uses current time.
        This is the main entry point for live trading.

        Args:
            today_date: Optional date string. If None, uses current time.
        """
        # Use current time if not specified
        if today_date is None:
            now = datetime.now()
            today_date = now.strftime("%Y-%m-%d %H:%M:%S")

        print(f"📈 Starting Alpaca trading session: {today_date}")

        # Check market hours if enabled
        if self.check_market_hours:
            clock = self.alpaca_client.get_clock()
            if not clock["is_open"]:
                print(f"⚠️  Market is closed. Next open: {clock['next_open']}")
                print("   Skipping trading session.")
                return

        # Get current account state for prompt
        account = self.alpaca_client.get_account()
        positions = self.alpaca_client.get_positions()

        # Record start equity for session tracking
        self.session_start_equity = account['equity']

        # Set up logging
        log_file = self._setup_logging(today_date.split()[0])  # Use date only for log path
        write_config_value("LOG_FILE", log_file)
        write_config_value("TODAY_DATE", today_date)
        write_config_value("SIGNATURE", self.signature)

        # Create agent with Alpaca-specific prompt
        system_prompt = get_alpaca_system_prompt(
            current_time=today_date,
            account=account,
            positions=positions,
            stock_symbols=self.stock_symbols,
        )

        self.agent = create_agent(
            self.model,
            tools=self.tools,
            system_prompt=system_prompt,
        )

        # Add verbose callbacks if enabled
        if self.verbose and _ConsoleHandler is not None:
            try:
                handler = _ConsoleHandler()
                self.agent = self.agent.with_config({
                    "callbacks": [handler],
                    "tags": [self.signature, today_date],
                    "run_name": f"{self.signature}-alpaca-session"
                })
            except Exception:
                pass

        # Initial user query
        user_query = [{
            "role": "user",
            "content": f"Please analyze current market conditions and manage the portfolio. Current time: {today_date}"
        }]
        message = user_query.copy()

        # Log initial message
        self._log_message(log_file, user_query)

        # Trading loop
        current_step = 0
        while current_step < self.max_steps:
            current_step += 1
            print(f"🔄 Step {current_step}/{self.max_steps}")

            try:
                # Call agent
                response = await self._ainvoke_with_retry(message)

                # Extract agent response
                agent_response = extract_conversation(response, "final")

                # Check stop signal
                if STOP_SIGNAL in agent_response:
                    print("✅ Received stop signal, trading session ended")
                    print(agent_response)
                    self._log_message(log_file, [{"role": "assistant", "content": agent_response}])
                    break

                # Extract tool messages
                tool_msgs = extract_tool_messages(response)
                # Handle content that can be either string or list
                def get_content_str(msg):
                    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    if isinstance(content, list):
                        return " ".join(str(c) for c in content)
                    return str(content) if content else ""
                tool_response = "\n".join([get_content_str(msg) for msg in tool_msgs])

                # Prepare new messages
                new_messages = [
                    {"role": "assistant", "content": agent_response},
                    {"role": "user", "content": f"Tool results: {tool_response}"},
                ]

                # Add new messages
                message.extend(new_messages)

                # Log messages
                self._log_message(log_file, new_messages[0])
                self._log_message(log_file, new_messages[1])

            except Exception as e:
                print(f"❌ Trading session error: {str(e)}")
                raise

        # Print final account state
        await self._print_session_summary()

    async def _print_session_summary(self) -> None:
        """Print summary after trading session and log to trade logger"""
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()

            end_equity = account['equity']
            pnl = end_equity - self.session_start_equity if self.session_start_equity > 0 else 0
            pnl_pct = (pnl / self.session_start_equity * 100) if self.session_start_equity > 0 else 0

            print("\n" + "=" * 50)
            print("📊 SESSION SUMMARY")
            print("=" * 50)
            print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"Cash: ${account['cash']:,.2f}")
            print(f"Buying Power: ${account['buying_power']:,.2f}")
            print(f"Daily P&L: ${account['equity'] - account['last_equity']:,.2f}")

            if positions:
                print(f"\nPositions ({len(positions)}):")
                for symbol, pos in positions.items():
                    print(f"  {symbol}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
                    print(f"         Current: ${pos['current_price']:.2f} | P&L: ${pos['unrealized_pl']:.2f}")
            else:
                print("\nNo open positions")

            print("=" * 50)

            # Log session summary
            self.trade_logger.log_session_summary({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "start_equity": self.session_start_equity,
                "end_equity": end_equity,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "cash": account['cash'],
                "positions": [
                    {
                        "symbol": symbol,
                        "qty": pos['qty'],
                        "avg_price": pos['avg_entry_price'],
                        "current_price": pos['current_price'],
                        "unrealized_pnl": pos['unrealized_pl']
                    }
                    for symbol, pos in positions.items()
                ] if positions else []
            })

        except Exception as e:
            print(f"⚠️  Could not print session summary: {e}")

    def register_agent(self) -> None:
        """
        Register agent for Alpaca trading.

        Unlike BaseAgent, this doesn't create initial positions since
        Alpaca manages the account state. It only creates the log directory.
        """
        # Create log directory structure
        position_dir = os.path.join(self.data_path, "position")
        log_dir = os.path.join(self.data_path, "log")

        for dir_path in [position_dir, log_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"📁 Created directory: {dir_path}")

        # Get initial account state from Alpaca
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()

            print(f"✅ Agent {self.signature} registered for Alpaca trading")
            print(f"📁 Log path: {self.data_path}")
            print(f"💰 Account Equity: ${account['equity']:,.2f}")
            print(f"💵 Buying Power: ${account['buying_power']:,.2f}")
            print(f"📊 Current Positions: {len(positions)}")

        except AlpacaClientError as e:
            print(f"⚠️  Could not get Alpaca account info: {e}")
            print(f"✅ Agent {self.signature} registered (Alpaca connection pending)")

    async def run_scheduled(self, interval_minutes: int = 60) -> None:
        """
        Run trading in scheduled mode.

        This method runs continuously, executing trading sessions
        at regular intervals. For crypto (check_market_hours=False),
        it runs 24/7. For stocks, it only runs during market hours.

        Args:
            interval_minutes: Minutes between trading sessions
        """
        print(f"⏰ Starting scheduled trading (interval: {interval_minutes} minutes)")
        if not self.check_market_hours:
            print(f"🌐 24/7 mode enabled (crypto trading)")

        while True:
            try:
                # For crypto trading (check_market_hours=False), always run
                # For stock trading, check if market is open
                should_run = True
                if self.check_market_hours:
                    clock = self.alpaca_client.get_clock()
                    if not clock["is_open"]:
                        print(f"💤 Market closed. Next open: {clock['next_open']}")
                        should_run = False

                if should_run:
                    print(f"\n🔔 Running trading session...")
                    await self.run_trading_session()

                # Wait for next interval
                print(f"⏳ Waiting {interval_minutes} minutes until next session...")
                await asyncio.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                print("\n🛑 Scheduled trading stopped by user")
                break
            except Exception as e:
                print(f"❌ Error in scheduled trading: {e}")
                print("   Waiting before retry...")
                await asyncio.sleep(60)

    def get_position_summary(self) -> Dict[str, Any]:
        """Get current position summary from Alpaca"""
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()

            return {
                "signature": self.signature,
                "source": "alpaca",
                "account": {
                    "equity": account["equity"],
                    "cash": account["cash"],
                    "buying_power": account["buying_power"],
                    "portfolio_value": account["portfolio_value"],
                },
                "positions": positions,
                "position_count": len(positions),
            }

        except AlpacaClientError as e:
            return {"error": f"Failed to get Alpaca positions: {e}"}

    def __str__(self) -> str:
        return (
            f"AlpacaAgent(signature='{self.signature}', basemodel='{self.basemodel}', "
            f"stocks={len(self.stock_symbols)})"
        )
