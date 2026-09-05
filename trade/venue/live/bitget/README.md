# Bitget live venue

`BitgetVenue` implements the same runner and dashboard interfaces as
`BinanceVenue`, using Bitget Classic Account REST v2 and private WebSocket v2.
It trades USDT perpetual futures and keeps the shared Binance market-data feed.
Unified Trading Accounts (UTA v3), spot and coin-margined futures are outside
this venue's scope.

## Configuration

Use the usual report hash, model path and broker settings. Select `Bitget` and
provide a credential directory, resolved relative to the live configuration file:

```json
{
  "venue": "Bitget",
  "Bitget": {
    "path": "../../LiveTrading/bitget/trading1"
  }
}
```

The directory must contain three UTF-8 files, each with one value:

```text
trading1/
  apikey
  secret_key
  passphrase
```

`passphrase` is the password chosen when creating the API key. The implementation
uses HMAC-SHA256 credentials. Credential values are not included in configuration
or logs. Keep the directory outside version control.

The venue reads the symbol's current margin mode and account position mode.
It supports both `one_way_mode` and `hedge_mode`, with one active position per
symbol. Configure leverage and margin mode in Bitget before starting the runner;
the venue does not change them. Broker leverage and commission settings should
match the exchange account. Use a dedicated account/symbol for each strategy:
existing positions or pending orders prevent a new entry.

## Execution behavior

- Quantities are base-asset units. Contract lot sizes, price ticks, minimum
  notional and maximum order sizes are validated before entry.
- Market entries wait for a terminal order state. Protection uses the actual
  average fill price, with mark-price triggers: position stop-loss at market and
  position take-profit at a limit price. Position protection follows remaining
  exposure after partial closes. A triggered limit take-profit can remain unfilled.
- Limit entries use GTC and support no attached protection, matching the current
  Binance venue restriction. Another entry is blocked while the order rests.
- Failed or uncertain entries are looked up by their unique client order ID.
  Recovery cancels any unfilled remainder, closes filled exposure, and verifies
  that the position is flat. Unconfirmed recovery raises an error; entry POSTs
  are never automatically retried.
- Explicit closes use reduce-only in one-way mode and `tradeSide=close` in hedge
  mode. Only this strategy's triggered protective limit orders are cancelled to
  release reserved position quantity. Remaining protection stays in place until
  the account confirms the position is flat.
- Strategy ownership is encoded in client order IDs. Plan history links triggered
  protective orders to their executable child IDs for fill attribution after a
  restart. Position opening time comes from Bitget's position creation timestamp.
- The private stream logs in, subscribes to orders, positions and trigger orders,
  sends text `ping` heartbeats, and reconnects on failure. REST fill pagination
  reconciles missed executions on connection and every 60 seconds while connected;
  trade IDs provide stable deduplication identities.
  Historical recovery is limited by Bitget's available API history.

## Verification

Run the offline contract tests from the repository root:

```bash
.venv/bin/python -m pytest -q trade/venue/live/bitget/test_bitget_venue.py
```

For a later check against the real account:

```bash
.venv/bin/python -m trade.venue.live.bitget.smoke_test \
  --key-path /home/chao/work/financial-ml-system/LiveTrading/bitget/trading1
```

The smoke command uses `read_only=True`, which blocks every non-GET REST request
and all order submission/cancellation methods. It checks contracts, account
mode, balances, positions, bid/ask, fill history, and private-stream subscriptions.
`--skip-websocket` checks only REST. It does not place a test trade or start a
strategy. Normal `requests` proxy environment variables can be used for REST.

## API references

- [Request signing](https://www.bitget.com/api-doc/classic/quickStart/intro)
- [Contract filters](https://www.bitget.com/api-doc/classic/contract/market/Get-All-Symbols-Contracts)
- [Order placement and position modes](https://www.bitget.com/api-doc/classic/contract/trade/Place-Order)
- [Position protection](https://www.bitget.com/api-doc/classic/contract/plan/Place-Tpsl-Order)
- [Trigger history and executable order IDs](https://www.bitget.com/api-doc/classic/contract/plan/orders-plan-history)
- [Private WebSocket authentication and heartbeats](https://www.bitget.com/api-doc/classic/quickStart/websocket-intro)
