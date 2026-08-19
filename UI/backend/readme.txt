Strategy Center uses one FastAPI application and one port.
Run exactly one backend worker because experiment datasets and label generation
coordination are maintained in process memory.

Development backend:
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000

Development frontend:
cd UI/quant-ui
npm install
npm run dev

Production-style local run:
cd UI/quant-ui
npm run build
cd ../..
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000

The FastAPI application serves the built frontend and all three API groups:
- /api/experiments
- /api/backtests
- /api/labels

Optional path overrides:
- REPORTS_ROOT=/path/to/quant_output/batch_experiments
- BACKTEST_FRONTEND_REPORT_PATH=/path/to/latest_backtest_report.json
- LABEL_OUTPUT_DIR=/path/to/preparation/output
