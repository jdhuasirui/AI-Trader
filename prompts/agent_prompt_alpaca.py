"""
Alpaca Trading Agent Prompts

System prompts specifically designed for real-time paper trading with Alpaca.
"""

from typing import Any, Dict, List, Optional

STOP_SIGNAL = "<FINISH_SIGNAL>"

# System prompt for Alpaca paper trading
ALPACA_SYSTEM_PROMPT = """
You are a professional stock trading assistant connected to Alpaca Paper Trading.

IMPORTANT: You are trading with REAL-TIME market data through Alpaca's API.
- Market hours: 9:30 AM - 4:00 PM Eastern Time (NYSE/NASDAQ)
- Orders are executed through Alpaca's paper trading API
- This is paper trading (simulated money), but uses real market prices
- Free tier data may have ~15 minute delay

## Your Current Account Status
- **Buying Power**: ${buying_power:,.2f}
- **Cash**: ${cash:,.2f}
- **Portfolio Value**: ${portfolio_value:,.2f}
- **Equity**: ${equity:,.2f}
- **Daily P&L**: ${daily_pnl:+,.2f}

## Your Current Positions
{positions_text}

## Watchlist Stocks
You can trade any of these stocks: {stock_list}

## Available Tools
1. **get_price(symbol)** - Get real-time quote for a stock
2. **get_prices(symbols)** - Get quotes for multiple stocks (comma-separated)
3. **get_price_history(symbol, days)** - Get historical daily prices
4. **get_market_status()** - Check if market is open
5. **buy(symbol, amount)** - Buy shares (market order)
6. **sell(symbol, amount)** - Sell shares (market order)
7. **get_account_info()** - Get current account details
8. **get_positions()** - Get all current positions
9. **search_news(query)** - Search for market news
10. **calculate(expression)** - Perform calculations

## Trading Guidelines
1. Always check current prices before trading
2. Consider your available buying power before buying
3. You can only sell shares you currently own
4. Market orders execute at the current market price
5. Be mindful of position sizing (don't put too much in one stock)
6. Consider market conditions and news before trading

## Your Task
Analyze the current market conditions and your portfolio. Make informed trading decisions to optimize returns while managing risk.

Think step by step:
1. Review your current positions and P&L
2. Check prices for stocks of interest
3. Research any relevant news
4. Decide on trades (if any)
5. Execute trades using buy/sell tools
6. Summarize your actions

Current time: {current_time}

When you have completed your analysis and any trades, output:
{STOP_SIGNAL}
"""

# Compact version for shorter context
ALPACA_SYSTEM_PROMPT_COMPACT = """
You are a trading assistant connected to Alpaca Paper Trading (real-time US market data).

## Account
- Buying Power: ${buying_power:,.2f}
- Cash: ${cash:,.2f}
- Portfolio: ${portfolio_value:,.2f}
- Daily P&L: ${daily_pnl:+,.2f}

## Positions
{positions_text}

## Watchlist
{stock_list}

## Tools
- get_price(symbol) / get_prices("SYM1,SYM2")
- buy(symbol, amount) / sell(symbol, amount)
- get_account_info() / get_positions()
- get_price_history(symbol, days)
- search_news(query)

## Task
Analyze market and portfolio. Make trading decisions to optimize returns.

Time: {current_time}

Output {STOP_SIGNAL} when done.
"""


def format_positions(positions: Dict[str, Dict[str, Any]]) -> str:
    """Format positions dictionary into readable text"""
    if not positions:
        return "No open positions"

    lines = []
    total_value = 0
    total_pnl = 0

    for symbol, pos in positions.items():
        qty = pos.get("qty", 0)
        avg_price = pos.get("avg_entry_price", 0)
        current_price = pos.get("current_price", 0)
        market_value = pos.get("market_value", 0)
        unrealized_pnl = pos.get("unrealized_pl", 0)
        pnl_pct = pos.get("unrealized_plpc", 0) * 100

        lines.append(
            f"- {symbol}: {qty} shares @ ${avg_price:.2f} | "
            f"Current: ${current_price:.2f} | "
            f"Value: ${market_value:,.2f} | "
            f"P&L: ${unrealized_pnl:+,.2f} ({pnl_pct:+.2f}%)"
        )
        total_value += market_value
        total_pnl += unrealized_pnl

    lines.append(f"\nTotal Position Value: ${total_value:,.2f}")
    lines.append(f"Total Unrealized P&L: ${total_pnl:+,.2f}")

    return "\n".join(lines)


def get_alpaca_system_prompt(
    current_time: str,
    account: Dict[str, Any],
    positions: Dict[str, Dict[str, Any]],
    stock_symbols: List[str],
    compact: bool = False,
) -> str:
    """
    Generate system prompt for Alpaca trading agent.

    Args:
        current_time: Current timestamp string
        account: Account info from Alpaca
        positions: Positions dict from Alpaca
        stock_symbols: List of tradeable stock symbols
        compact: Use compact prompt version

    Returns:
        Formatted system prompt string
    """
    # Format positions
    positions_text = format_positions(positions)

    # Format stock list
    stock_list = ", ".join(stock_symbols[:30])  # Limit to first 30 for brevity
    if len(stock_symbols) > 30:
        stock_list += f", ... ({len(stock_symbols)} total)"

    # Calculate daily P&L
    daily_pnl = account.get("equity", 0) - account.get("last_equity", 0)

    # Select template
    template = ALPACA_SYSTEM_PROMPT_COMPACT if compact else ALPACA_SYSTEM_PROMPT

    # Format prompt
    prompt = template.format(
        buying_power=account.get("buying_power", 0),
        cash=account.get("cash", 0),
        portfolio_value=account.get("portfolio_value", 0),
        equity=account.get("equity", 0),
        daily_pnl=daily_pnl,
        positions_text=positions_text,
        stock_list=stock_list,
        current_time=current_time,
        STOP_SIGNAL=STOP_SIGNAL,
    )

    return prompt


# For quick testing
if __name__ == "__main__":
    # Example usage
    test_account = {
        "buying_power": 45000.00,
        "cash": 45000.00,
        "portfolio_value": 55000.00,
        "equity": 55000.00,
        "last_equity": 54500.00,
    }

    test_positions = {
        "AAPL": {
            "qty": 25,
            "avg_entry_price": 180.50,
            "current_price": 185.25,
            "market_value": 4631.25,
            "unrealized_pl": 118.75,
            "unrealized_plpc": 0.0263,
        },
        "NVDA": {
            "qty": 10,
            "avg_entry_price": 450.00,
            "current_price": 485.00,
            "market_value": 4850.00,
            "unrealized_pl": 350.00,
            "unrealized_plpc": 0.0778,
        },
    }

    test_symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

    prompt = get_alpaca_system_prompt(
        current_time="2025-01-15 10:30:00",
        account=test_account,
        positions=test_positions,
        stock_symbols=test_symbols,
    )

    print(prompt)
