Strategy Center uses one Vite frontend and one FastAPI backend.

Development:

uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000
cd UI/quant-ui
npm install
npm run dev

Open http://localhost:5173. Vite proxies every same-origin /api request to
the backend on port 8000. Labels is the default page.

Single-process deployment:

cd UI/quant-ui
npm run build
cd ../..
uvicorn UI.backend.main:app --host 0.0.0.0 --port 8000

Open http://localhost:8000. FastAPI serves both the built SPA and its APIs.

Optional backend environment variables:
- REPORTS_ROOT
- BACKTEST_FRONTEND_REPORT_PATH
- LABEL_OUTPUT_DIR
