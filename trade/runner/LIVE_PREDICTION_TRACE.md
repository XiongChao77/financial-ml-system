# Live prediction trace

Enable one append-only CSV trace per feed group in the live runner configuration:

```json
{
  "prediction_trace": {
    "output_dir": "../../quant_output/live_prediction_traces"
  }
}
```

The output path is resolved relative to the live configuration file. Every
runner start creates a new timestamped CSV file. The initial feed cache is
written with `is_warmup=true`; each subsequently processed closed candle is
written with `is_warmup=false` after every strategy in the feed group has run.

Market columns use the canonical raw candle schema. Predictions use wide
strategy-specific columns:

```text
<strategy_id>__pred
<strategy_id>__pred_prob
<strategy_id>__net_score
```

The trace remains valid raw market data for batch inference. Extra trace and
prediction columns are ignored by feature generation.

To replay a single-feed trace through the live inference path and compare it
with the recorded predictions:

```bash
python -m trade.runner.live_prediction_replay \
  --config LiveTrading/live_config.json \
  --market-trace /path/to/feed_trace.csv \
  --output /tmp/replay_predictions.csv \
  --backtest-predictions /path/to/feed_trace.csv \
  --strategy-id STRATEGY_ID
```

The replay uses the `is_warmup` prefix to reconstruct the exact initial feed
cache. Prediction class values must match exactly; probability and net-score
values use the comparator's numeric tolerances.
