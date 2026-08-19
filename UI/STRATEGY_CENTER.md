# Strategy Center architecture

Strategy Center is one deployable application with a single frontend and a
single FastAPI backend. Business code remains separated by vertical module:

```text
UI/quant-ui                         one Vite SPA
UI/backend/main.py                  one FastAPI application
UI/backend/modules/experiments      experiment router and analysis service
UI/backend/modules/backtests        saved-backtest router and detail service
UI/backend/modules/labels           label router and preparation service
```

The browser uses same-origin `/api` requests. During development Vite proxies
those requests to port 8000. After `npm run build`, FastAPI serves the SPA from
`UI/quant-ui/dist` as part of the same deployment unit.

## Modules

- **Labels** is the default route. It loads the current `output/data`
  configuration and forward split, allows editing the complete `BaseDefine`
  configuration, and invokes `preparation.main` only after Generate is chosen.
  Generation writes to a staging directory and publishes the complete artifact
  set only after it has been read and validated successfully.
- **Backtests** reads the fixed standalone payload published by
  `trade/runner/backtest_runner.py`. It can also assemble the same detail DTO
  from a selected experiment's saved report and preparation artifacts without
  running inference or a backtest.
- **Experiments** recursively loads `reports.jsonl` from one or more selected
  folders, then provides generic filtering, grouping, comparison, equity, and
  a hand-off to Backtests.

All user-selected experiment paths and derived artifacts must remain below
`REPORTS_ROOT`. Dataset registrations are disposable in-memory references; a
backend restart requires loading the selected folders again.

Run one backend worker. Experiment dataset registrations and the Labels
generation lock are process-local by design; the documented Uvicorn command
uses one worker.

## Run

```bash
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000
cd UI/quant-ui
npm run dev
```

For a single-process deployment, build the frontend first and open port 8000:

```bash
cd UI/quant-ui
npm run build
cd ../..
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000
```
