# Alpaca Enhanced Data Integration Spec

## Overview

Enhance the AI trader's decision-making capabilities by integrating additional data sources including Alpaca APIs, on-chain crypto metrics, and improved news analysis. Primary success metric: **Win rate increase**.

## Current State

### What We Have
- Real-time quotes (bid/ask prices) - 15-min delayed on free tier
- Historical OHLCV bars (1min to 1day)
- Account info and positions
- Market clock status
- Basic order execution

### What's Missing
- News data with relevance scoring
- Market snapshots (consolidated view)
- Trade-level data (institutional activity detection)
- Corporate actions (dividends, splits)
- On-chain crypto metrics (exchange flows, whale alerts)

---

## Design Decisions

### Data Delay Handling
**Decision:** Accept 15-min delay for free tier. Hourly trading cycle makes this acceptable.

### Service Architecture
**Decision:** Bundle services by data type:
- `alpaca-data-service`: price, snapshot, trades, corporate actions
- `alpaca-news-service`: news with clustering/summarization
- `onchain-service`: crypto on-chain metrics

### Error Handling Strategy
**Decision:**
- Retry with exponential backoff (3x attempts)
- On persistent failure: Return available data with **degraded mode flag**
- AI prompt includes hint: "Consider smaller positions when data is incomplete"

### Tool Access
**Decision:** All traders get equal access to all tools (fair model comparison)

---

## Feature Specifications

### 1. News Integration (Priority: HIGH)

#### Approach: Hybrid Relevance Scoring
Pre-compute relevance hints, let AI make final decision.

#### Relevance Context Package
Each news item includes:
| Field | Source | Description |
|-------|--------|-------------|
| sentiment_score | LLM/NLP | -1.0 to 1.0 sentiment |
| recency_weight | Computed | Decay function, newer = higher |
| source_count | Aggregated | # of sources reporting same story |
| peer_reaction | Computed | How sector peers moved since news |

#### Peer Reaction Calculation
- Use **top 5 historically correlated stocks** for peer group
- Check price movement in last hour since news publication
- Avoids look-ahead bias by only using past price data

#### News Deduplication
**Decision:** Cluster similar headlines and use **LLM summarization** (GPT-4o-mini) to create cluster summary.

Output format:
```json
{
  "cluster_id": "abc123",
  "headline_summary": "Apple announces new AI features...",
  "source_count": 5,
  "sources": ["Bloomberg", "Reuters", "CNBC", "WSJ", "TechCrunch"],
  "earliest_time": "2025-01-15T10:30:00Z",
  "sentiment_score": 0.65,
  "peer_reaction": {"MSFT": +0.3%, "GOOGL": +0.2%, "META": +0.5%}
}
```

#### New Tool
```python
@mcp.tool()
def get_stock_news(symbols: str, limit: int = 10) -> Dict[str, Any]:
    """
    Get recent news with relevance scoring for stock symbols.

    Args:
        symbols: Comma-separated symbols (e.g., "AAPL,TSLA")
        limit: Max news clusters (default 10, max 50)

    Returns:
        List of news clusters with:
        - Summarized headline (LLM-generated)
        - Source count and list
        - Sentiment score
        - Peer reaction metrics
        - Recency weight

    Usage Hint: Check news before making large trades to avoid
    entering during high-volatility news events.
    """
```

---

### 2. Market Snapshots (Priority: HIGH)

#### Description
Consolidated view of market state in single API call.

#### Caching Strategy
**Decision:** 5-minute cache
- Reduces API calls significantly
- Acceptable staleness for hourly trading cycle
- Cache invalidated on error or explicit refresh

#### New Tool
```python
@mcp.tool()
def get_market_snapshot(symbols: str) -> Dict[str, Any]:
    """
    Get complete market snapshot for symbols.

    Returns consolidated view including:
    - Latest trade and quote
    - Current minute bar
    - Today's daily bar with VWAP
    - Previous day's bar
    - Calculated metrics (change %, spread, momentum)

    Data may be cached for up to 5 minutes.
    Check 'data_age_seconds' field for freshness.

    Usage Hint: Use before trading to get complete picture
    in single call instead of multiple API requests.
    """
```

---

### 3. Trade Flow Analysis (Priority: MEDIUM)

#### Block Trade Detection
**Decision:** 0.05% of Average Daily Volume (ADV)
- Balanced sensitivity
- Catches significant institutional activity without noise
- ADV calculated from 20-day rolling average

#### Lit vs Dark Pool Metrics
**Decision:** Separate metrics for transparency
```json
{
  "total_volume": 1500000,
  "lit_volume": 1200000,
  "dark_pool_volume": 300000,
  "dark_pool_pct": 20.0,
  "large_blocks": [
    {"size": 50000, "price": 185.50, "venue": "NASDAQ", "type": "lit"},
    {"size": 75000, "price": 185.48, "venue": "DARK", "type": "dark"}
  ]
}
```

#### New Tools
```python
@mcp.tool()
def get_recent_trades(symbol: str, limit: int = 100) -> Dict[str, Any]:
    """Get recent trades for a symbol with venue classification."""

@mcp.tool()
def analyze_trade_flow(symbol: str, minutes: int = 5) -> Dict[str, Any]:
    """
    Analyze recent trade flow with institutional detection.

    Returns:
    - Total volume (lit vs dark pool breakdown)
    - Average trade size
    - Large block count (>0.05% of ADV)
    - Buy/sell pressure estimate
    - VWAP

    Usage Hint: Check before large orders to assess
    institutional activity and market depth.
    """
```

---

### 4. Corporate Actions (Priority: MEDIUM)

#### Ex-Date Risk Adjustment
**Decision:** Dynamic reduction by dividend yield

| Dividend Yield | Position Limit Reduction | Window |
|----------------|-------------------------|--------|
| < 1% | No change | - |
| 1-3% | 25% reduction | 1 day before/after |
| 3-5% | 50% reduction | 1 day before/after |
| > 5% | 75% reduction | 2 days before/after |

#### New Tool
```python
@mcp.tool()
def get_corporate_actions(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Get upcoming and recent corporate actions.

    Returns dividends, splits, and other actions with:
    - Risk adjustment recommendation
    - Position limit modifier
    - Days until ex-date

    Usage Hint: Check before opening new positions to
    avoid unexpected dividend-related price adjustments.
    """
```

---

### 5. On-Chain Crypto Metrics (Priority: MEDIUM)

#### Why On-Chain Instead of Crypto News
Alpaca's crypto news coverage is limited. On-chain data provides more actionable signals for crypto trading.

#### Metrics to Track
| Metric | Description | Signal |
|--------|-------------|--------|
| Exchange Net Flow | Inflows minus outflows to exchanges | High inflow = selling pressure |
| Whale Transactions | Transfers > $1M | Large movements = volatility coming |
| Active Addresses | Daily active wallets | Network health/adoption |
| MVRV Ratio | Market Value / Realized Value | >3 = overheated, <1 = undervalued |

#### Data Sources (Free Alternatives)
- **blockchain.com API**: BTC exchange flows (free, limited)
- **whale-alert.io API**: Large transaction alerts (free tier: 10 req/min)
- **Etherscan API**: ETH metrics (free tier: 5 req/sec)

#### Caching Strategy
**Decision:** Heavy caching (1 hour+) to work within free tier limits.
On-chain data changes slowly compared to price data.

#### New Tool
```python
@mcp.tool()
def get_onchain_metrics(symbol: str) -> Dict[str, Any]:
    """
    Get on-chain metrics for cryptocurrency.

    Currently supports: BTC, ETH

    Returns:
    - exchange_net_flow_24h: Net flow to exchanges (negative = bullish)
    - whale_transactions_24h: Count of >$1M transfers
    - active_addresses_24h: Network activity
    - mvrv_ratio: Market Value / Realized Value
    - data_freshness: How old the cached data is

    Note: Data cached for 1 hour to stay within free API limits.

    Usage Hint: High exchange inflows often precede selling.
    MVRV > 3 suggests overheated market conditions.
    """
```

---

### 6. WebSocket Streaming (Priority: LOW)

#### Why Low Priority
Current 60-minute trading intervals make real-time unnecessary.
LLM response latency (2-5s) negates sub-second data benefits.

#### Future Consideration
If trading frequency increases, implement event-driven architecture.

---

## Implementation Plan

### Phase 1: Core Data Enhancement (Week 1-2)

1. **News API Integration**
   - Create `tool_news_alpaca.py`
   - Implement LLM-based news clustering
   - Add peer reaction calculation
   - Integration test with live API

2. **Snapshot API**
   - Add `get_market_snapshot()` to price tool
   - Implement 5-minute caching layer
   - Add degraded mode handling

### Phase 2: Trade Analysis + Corporate Actions (Week 2-3)

3. **Trade Flow Analysis**
   - Create trade data tools
   - Implement ADV-based block detection
   - Add lit/dark pool classification

4. **Corporate Actions**
   - Integrate dividend/split data
   - Implement dynamic position limit adjustment
   - Add ex-date proximity warnings

### Phase 3: On-Chain Integration (Week 3-4)

5. **On-Chain Metrics**
   - Integrate free API sources
   - Implement aggressive caching
   - Add MVRV calculation from public data

---

## File Changes

### New Files
```
agent_tools/
├── tool_news_alpaca.py       # News with clustering/summarization
├── tool_trade_flow.py        # Trade analysis + block detection
├── tool_corporate_actions.py # Dividends, splits, risk adjustment
├── tool_onchain_metrics.py   # Crypto on-chain data
└── cache/
    ├── __init__.py
    └── cache_manager.py      # Unified caching with TTL
```

### Modified Files
```
agent_tools/
├── tool_get_price_alpaca.py  # Add snapshot endpoint + caching
├── alpaca_client.py          # Add news, trades, actions clients
└── start_alpaca_services.py  # Register new bundled services

prompts/
└── agent_prompt_alpaca.py    # Update tools list with usage hints

configs/
└── *.json                    # Add new service endpoints
```

---

## Updated System Prompt

```
## Available Tools

### Price Data
1. get_price(symbol) - Current quote
2. get_prices(symbols) - Multiple quotes
3. get_price_history(symbol, days) - Historical bars
4. get_market_snapshot(symbols) - Complete market state [NEW]
   Hint: Use before trading for complete picture in one call

### News & Research
5. search_news(query) - Web search (Jina)
6. get_stock_news(symbols) - News with relevance scoring [NEW]
   Hint: Check before large trades to avoid news volatility

### Trade Analysis
7. get_recent_trades(symbol) - Recent executions [NEW]
8. analyze_trade_flow(symbol) - Volume/institutional analysis [NEW]
   Hint: Check before large orders to assess market depth

### Corporate Actions
9. get_corporate_actions(symbol) - Dividends, splits [NEW]
   Hint: Check before new positions to avoid ex-date surprises

### On-Chain (Crypto)
10. get_onchain_metrics(symbol) - Exchange flows, whale alerts [NEW]
    Hint: High exchange inflows often precede selling pressure

### Trading
11. buy(symbol, amount)
12. sell(symbol, amount)
13. get_positions()
14. get_account_info()

## Data Quality Notes
- Stock quotes may be 15-min delayed (free tier)
- On-chain data cached for 1 hour
- Check 'degraded_mode' flag - if true, consider smaller positions
```

---

## Success Metrics

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| **Win Rate** | ~52% | **60%+** | PRIMARY |
| Data points per decision | ~5 | ~15 | Secondary |
| News awareness | None | Real-time (clustered) | Secondary |
| API calls per cycle | ~10 | ~6 (snapshots) | Efficiency |

---

## API Rate Limits

| Service | Free Tier | Paid Tier |
|---------|-----------|-----------|
| Alpaca Data | 200 req/min | 10,000 req/min |
| whale-alert.io | 10 req/min | 100 req/min |
| Etherscan | 5 req/sec | 10 req/sec |
| blockchain.com | 100 req/15min | N/A |

**Mitigation:** Aggressive caching + snapshot endpoints

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| API rate limits | Caching (5min-1hr), snapshot endpoints |
| LLM summarization cost | Use GPT-4o-mini (~$0.001 per cluster) |
| On-chain data delays | Accept 1hr cache, on-chain signals are slow-moving |
| Degraded mode confusion | Explicit prompt hint about conservative trading |
| Peer stock selection | Pre-compute correlations weekly, cache results |

---

## Testing Plan

1. **Unit Tests**
   - Mock API responses
   - Test caching TTL behavior
   - Test degraded mode flag propagation

2. **Integration Tests**
   - Live API calls (paper trading)
   - End-to-end tool execution
   - Rate limit compliance verification

3. **Performance Baseline**
   - Record current win rate before deployment
   - Compare to win rate after 30 days with enhanced data
   - No formal A/B testing (historical baseline comparison)

---

## Dependencies

```python
# Already installed
alpaca-py>=0.30.0

# New dependencies for on-chain
requests>=2.28.0  # Already installed
```

No new paid services required. All on-chain data from free APIs.

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| How to avoid look-ahead in news correlation? | Use sector peer reaction (past prices only) |
| What defines "large block"? | 0.05% of 20-day ADV |
| How to handle 15-min data delay? | Accept for hourly trading |
| Crypto news source? | Skip news, use on-chain metrics instead |
| A/B testing approach? | Skip, compare to historical baseline |

---

## References

- [Alpaca Market Data API](https://docs.alpaca.markets/docs/about-market-data-api)
- [Alpaca News API](https://docs.alpaca.markets/docs/news)
- [whale-alert.io API](https://docs.whale-alert.io/)
- [Etherscan API](https://docs.etherscan.io/)
- [blockchain.com API](https://www.blockchain.com/api)
