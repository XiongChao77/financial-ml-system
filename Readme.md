# Machine Learning Driven Trading System
Train ML model to predict the trend of a single assert, make trade action based on model signals.


################################ environment ########################################
Python >= 3.10
pip install numpy scipy pandas scikit-learn matplotlib seaborn plotly notebook jupyterlab ipykernel statsmodels xgboost lightgbm tqdm joblib requests beautifulsoup4 pytorch-ignite  colorlog backtrader pyarrow numba GitPython ignite
pip install MetaTrader5 pybit

################################ how to use ########################################
*Download the data: 
    cd Quant/data_process
    python download_binance_history.py
*Data preprocess:   
    cd Quant/model
    python preparation.py
*Traning
    cd Quant/model
    python train.py
*Backtrade test
    cd Quant/trade
    python simulation.py
*UI
    backend:
        cd Quant
        uvicorn UI.backend.main:app --reload
    quant-ui:
        cd Quant\UI\quant-ui
        npm install
        npm run dev




