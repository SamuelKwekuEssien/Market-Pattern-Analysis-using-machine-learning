# Market-Pattern-Analysis-using-machine-learning
Market Pattern Analysis using machine learning

This project is a Python-based quantitative research workflow for testing whether a machine-learning model can predict short-term market direction using technical, candlestick, macro, and sentiment-style market features.

The main script, [pattern_ml_backtest_V4_sentiment.py](pattern_ml_backtest_V4_sentiment.py), downloads historical market data, engineers features, trains an XGBoost classifier, and runs a walk-forward backtest to evaluate predictive performance.

## What the project does

- Downloads OHLCV data from Yahoo Finance
- Builds technical indicators such as RSI, MACD, ATR, moving averages, volatility, and momentum
- Adds candlestick pattern features such as doji, hammer, shooting star, and engulfing patterns
- Incorporates macro-market features including VIX, DXY, SPY, Fed funds, yield curve, and CPI inflation data
- Trains a binary classifier to predict whether the next bar will be up or down
- Runs a walk-forward backtest and reports metrics such as win rate, drawdown, Sharpe ratio, and total return
- Saves a trained model artifact for later use

## Project file

- [pattern_ml_backtest_V4_sentiment.py](pattern_ml_backtest_V4_sentiment.py) — main training, feature engineering, backtest, and model export workflow

## Requirements

Install the required Python packages:

```bash
pip install yfinance pandas numpy scikit-learn xgboost fredapi joblib --break-system-packages
```

## Usage

Run the sentiment-enhanced backtest script:

```bash
python pattern_ml_backtest_V4_sentiment.py
```

This script will:

1. Load market data for the configured tickers
2. Build features from price and macro data
3. Train the model
4. Print backtest metrics and save the trained model file

## Notes

- The script is intended for research and experimentation, not live trading
- The workflow uses Yahoo Finance and FRED data, so internet access is required
- Model performance depends heavily on market regime, data quality, and feature tuning
- The script currently targets example tickers such as GC=F and EURUSD=X

## Disclaimer

This repository is for educational and research purposes only. It does not provide financial advice and should not be used as a standalone trading system without further validation.
