# Quant Trading System

An end-to-end quantitative trading system built around market data processing, feature engineering, machine-learning signal generation, backtesting, risk control, and visualization.

## Overview

This project is a quantitative trading system with a strong focus on AI/ML engineering. It is designed to explore how financial time-series data can be transformed into tradable signals and how those signals behave after they are passed through execution rules, holding logic, transaction costs, and risk controls.

The core workflow includes:

1. Downloading and preprocessing historical market data.
2. Building relative, volatility-aware, and market-state-aware features.
3. Training machine-learning models for long and short signal prediction.
4. Evaluating model outputs with both ML metrics and trading-oriented metrics.
5. Converting model predictions into strategy actions.
6. Running backtests with fees, drawdown, position logic, and risk constraints.
7. Visualizing model signals and strategy behavior through a local UI.

## Project Structure

```text
Quant/
├── data_process/          # Data download, cleaning, feature construction, preprocessing
├── model/                 # Model training, evaluation, feature selection, experiment scripts
│   ├── models/            # Model definitions
│   └── tasks/             # Training and experiment task definitions
├── trade/                 # Backtesting, market interface, strategy execution logic
│   ├── bt/                # Backtest-related modules
│   ├── market/            # Market data / exchange interface logic
│   └── strategy/          # Strategy rules, signal handling, position logic
├── experiment/            # Experiment comparison, reports, visualization, analysis scripts
├── UI/                    # Local UI for inspecting results and workflow outputs
│   ├── backend/           # Backend service
│   └── quant-ui/          # Frontend interface
├── utils/                 # Shared utilities
├── requirements.txt       # Python dependencies
└── Readme.md              # Project documentation
```

## Key Features

### Data Pipeline

The data pipeline handles historical market data collection, cleaning, alignment, and preprocessing.

Supported raw market fields include:

- `open`, `high`, `low`, `close`
- `volume`
- `number_of_trades`
- `quote_asset_volume`
- `taker_buy_base_volume`
- `taker_buy_quote_volume`

The pipeline is designed to reduce data leakage and alignment errors, which are especially important in time-series modeling and backtesting.

### Feature Engineering

The project builds market features from price, volume, volatility, candle structure, and technical indicators.

Feature directions include:

- Price-relative features.
- Volume and turnover features.
- Taker buy/sell pressure features.
- Volatility-normalized features.
- Candle body / wick structure features.
- KDJ / MFI / CFM-style technical features.
- Feature correlation and factor analysis.

A key design principle is that model inputs should rely on **relative information** rather than raw absolute values whenever possible. This improves robustness across assets, price levels, and market regimes.

### Scaling and Normalization

Financial time series are highly non-stationary, so preprocessing is treated as part of the modeling design rather than a mechanical step.

Explored preprocessing methods include:

- Relative change from a reference timestamp.
- Z-score normalization.
- Robust scaling.
- Ratio-based scaling.
- Log transformation.
- Rank / quantile normalization.
- Feature-group-aware normalization.

The project pays special attention to normalization mistakes that can damage signal quality, such as mixing feature means and base-feature standard deviations incorrectly, or using absolute thresholds that do not adapt to volatility.

### Machine-Learning Signal Models

The model layer is responsible for generating directional trading signals.

Current modeling directions include:

- Long-vs-other classification.
- Short-vs-other classification.
- Separate long and short models to reduce coupling.
- Feature selection and feature-combination experiments.
- Confidence-based signal filtering.
- Analysis of the gap between classification metrics and strategy performance.

The project separates long and short prediction because upward and downward movements often have different market mechanisms.

### Backtesting and Strategy Logic

The backtesting layer evaluates whether model outputs can become realistic trading behavior.

Focus areas include:

- Signal-driven entry logic.
- Fixed holding-period experiments.
- Minimum holding time plus signal refresh.
- Early exit on reverse signals.
- Path-dependent holding logic.
- Stop-loss and ATR-based risk control.
- Position sizing based on volatility.
- Fee-adjusted performance.
- Drawdown analysis.
- Long/short separated performance analysis.

The project treats backtesting as an engineering problem. Details such as timestamp alignment, prediction timing, fee assumptions, holding rules, and signal overwrite logic can completely change the result of an experiment.

### Local UI

The repository includes a local UI for inspecting experiments and strategy behavior.

The UI is used to:

- View backtest results.
- Compare experiment reports.
- Inspect model signals.
- Visualize whether signals match market movement.
- Support faster iteration during model and strategy development.

## Evaluation Philosophy

A major lesson from this project is that better ML metrics do not always produce better trading results.

The project uses standard classification metrics as diagnostic tools, but final strategy selection is based on metrics closer to trading objectives.

### ML Metrics

| Metric | Meaning |
| --- | --- |
| Precision | Among predicted positive samples, the proportion that is truly positive |
| Recall | Among true positive samples, the proportion correctly predicted |
| F1-score | Harmonic mean of precision and recall |
| Accuracy | Overall proportion of correct predictions |
| Macro Average | Equal-weighted average across classes |
| Weighted Average | Class-size-weighted average across classes |

### Trading-Oriented Metrics

More important evaluation metrics include:

- Total return.
- Fee-adjusted return.
- Maximum drawdown.
- Sharpe ratio.
- Profit factor.
- Win rate.
- Fee-adjusted win rate.
- Average profit / average loss.
- Trade count.
- Average holding period.
- Long/short separated returns.
- Regime-specific performance.
- Stability across assets and time windows.

The long-term direction is to select parameters and models using return-like evaluation metrics instead of relying only on loss, F1, or recall.

## Validation Methods

### Walk-Forward Analysis

Out-of-sample testing is used to simulate how a model would be updated over time without using future information.

### Cross-Asset Validation

A signal should not only work on a single asset by accident.

Example workflow:

```text
Optimize on ETH
Test on BTC, SOL, BNB, DOGE, or other assets
```

If a parameter only works on one asset and fails everywhere else, it is likely overfitted.

### Multi-Timeframe Validation

Signals can also be tested across different K-line periods to check whether they capture a robust market pattern or only fit one sampling interval.

## Engineering Challenges Addressed

This project contains several practical issues that often appear in AI-driven trading systems:

- Model precision and recall can improve while backtest performance gets worse.
- F1, accuracy, and recall may be negatively correlated with trading returns.
- Weighted precision may be more useful than general classification metrics in some cases.
- A model is not a strategy; signal generation and trade execution must be evaluated separately.
- Labels based on fixed absolute thresholds can fail under changing volatility.
- Volatility normalization is important for cross-asset and cross-regime adaptation.
- If both upward and downward barriers are touched during a label window, the label may become ambiguous.
- Wrong holding logic can accidentally produce good backtest results but fail in live-like conditions.
- Long and short confidence should not be mixed without careful interpretation.
- Index alignment bugs between prediction and label data can invalidate experiment results.

## Current Development Focus

- Designing loss functions closer to trading objectives.
- Selecting parameters using return-like evaluation metrics instead of only F1 or loss.
- Improving long/short separated modeling.
- Improving volatility-normalized labels.
- Reducing the gap between backtest results and live-like market behavior.
- Improving position sizing and single-trade risk control.
- Improving stop-loss logic and ATR reference periods.
- Adding better process visualization for signal inspection.
- Increasing data coverage and testing across assets.
- Improving the local UI for faster iteration.

## Environment

Python >= 3.10
pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly notebook jupyterlab ipykernel statsmodels xgboost lightgbm tqdm joblib requests beautifulsoup4 pytorch-ignite  colorlog backtrader pyarrow numba GitPython ignite
pip install MetaTrader5 pybit

For the frontend UI:

```bash
cd UI/quant-ui
npm install
```

## Example Workflow

### 1. Clone the repository

```bash
git clone https://github.com/XiongChao77/Quant.git
cd Quant
```

### 2. Download historical data

```bash
cd data_process
python download_binance_history.py
```

### 3. Prepare data

```bash
cd ../model
python preparation.py
```

### 4. Train model

```bash
python train_2model.py
```

Other experiment scripts may include:

```bash
python train_2head.py
python train_experiments.py
```

### 5. Evaluate model

```bash
python evaluate_test_set.py
```

### 6. Run backtest / simulation

```bash
cd ../trade
python simulation.py
```

### 7. Run local UI

Backend:

```bash
cd ..
uvicorn UI.backend.main:app --reload
```

Frontend:

```bash
cd UI/quant-ui
npm run dev
```

> Some entry points may change as the system evolves.

## Technical Highlights

This project demonstrates practical AI/ML engineering through:

- Financial time-series feature engineering.
- Non-stationary data preprocessing.
- Supervised learning for directional signal generation.
- Long/short separated modeling design.
- Experiment tracking and model evaluation.
- Backtesting with realistic trading constraints.
- Risk-control logic and drawdown analysis.
- Full pipeline implementation from data to model to strategy to visualization.

## Contribution Policy

This is a personal project and is not currently open to external code contributions.

## Disclaimer

This repository is for technical demonstration only.

It is not financial advice, investment advice, or a recommendation to trade. Quantitative trading involves substantial risk. Machine-learning models can overfit historical data, fail under regime changes, and produce misleading backtest results. Any live trading decision requires independent validation, realistic cost assumptions, strict risk control, and personal responsibility.

## License

No open-source license is currently specified. Unless a license is added, all rights are reserved by the author.




