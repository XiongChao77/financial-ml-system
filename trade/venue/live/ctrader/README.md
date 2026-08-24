# cTrader venue

Install the optional official SDK dependency:

```bash
pip install -e '.[ctrader]'
```

Register an Open API application in the cTrader portal, grant it trading access,
and place these UTF-8 text files in the configured `key_path` directory:

- `client_id`
- `client_secret`
- `access_token`
- `account_id` (optional when the token grants exactly one account)

The live runner also accepts `account_id` directly in the venue configuration.
Use `environment: "demo"` while validating a new setup and change it to `"live"`
only after the complete order lifecycle has been tested. `broker_symbol` is
optional and can be used when the broker's symbol differs from the market data
symbol, for example `DOGE/USD` versus `DOGEUSDT`.

The venue uses the strategy magic number as the cTrader position label. Position
queries and close requests only affect positions with that exact label and symbol.

`CTRADER_SYMBOL_MAP` contains the 32 cryptocurrency CFDs in the current FTMO
offering, mapped from Binance USDT market-data symbols to FTMO's cTrader USD
symbols. cTrader symbol availability is broker- and account-specific, so venue
initialization still verifies the mapped symbol against `ProtoOASymbolsListRes`.
Use `broker_symbol` to override the mapping for another broker.
