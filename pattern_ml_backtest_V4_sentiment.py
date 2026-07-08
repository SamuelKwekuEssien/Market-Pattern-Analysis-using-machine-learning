import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
from fredapi import Fred 
from xgboost import XGBClassifier

try:
    import yfinance as yf
except ImportError:
    yf = None  # Will raise a clear error at call time if missing

try:
    from fredapi import Fred
except ImportError:
    Fred = None
    FRED_API_KEY = "55050ce02081c3c6d66969b2ca000ad3"
fred = Fred(api_key="55050ce02081c3c6d66969b2ca000ad3")

# =========================================================
# 1. DATA LOADING
# =========================================================
def load_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance --break-system-packages")

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol is correct.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    
    available_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[available_cols].dropna()
    
    if "volume" not in df.columns:
        df["volume"] = 0
        
    df.index.name = "date"
    return df

def load_macro_data(period="5y"):
    vix = yf.download("^VIX", period=period, auto_adjust=True, progress=False)
    dxy = yf.download("DX-Y.NYB", period=period, auto_adjust=True, progress=False)
    spy = yf.download("SPY", period=period, auto_adjust=True, progress=False)

    macro = pd.DataFrame(index=vix.index)
    macro["vix_return"] = vix["Close"].pct_change()
    macro["dxy_return"] = dxy["Close"].pct_change(fill_method=None)
    macro["spy_return"] = spy["Close"].pct_change(fill_method=None)

    fedfunds = fred.get_series("FEDFUNDS")
    cpi      = fred.get_series("CPIAUCSL")
    yield10  = fred.get_series("DGS10")

    fedfunds.index = pd.to_datetime(fedfunds.index)
    cpi.index      = pd.to_datetime(cpi.index)
    yield10.index  = pd.to_datetime(yield10.index)

    # FIX: compute YoY on the monthly series before ffill — shift(12) = 12 months
    cpi_yoy_monthly = cpi / cpi.shift(12) - 1

    macro["fed_funds"]  = fedfunds
    macro["yield10"]    = yield10
    macro["cpi_yoy"]    = cpi_yoy_monthly

    macro = macro.sort_index()
    macro["fed_funds"]  = macro["fed_funds"].ffill()
    macro["yield10"]    = macro["yield10"].ffill()
    macro["cpi_yoy"]    = macro["cpi_yoy"].ffill()

    macro["fed_rate_change"] = macro["fed_funds"].diff()
    macro["yield10_change"]  = macro["yield10"].diff()

    return macro

# =========================================================
# 2. INDICATOR ENGINE CALCULATIONS
# =========================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.ffill().fillna(50) 


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Computes the raw MACD line, Signal line, and Histogram value."""
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Computes the wilder-normalized Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)
    
    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    
    # Bundle True Range arrays together and grab the element-wise maximum
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-coded candlestick pattern features (binary flags)."""
    body = df["close"] - df["open"]
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    df["body"] = body
    df["doji"] = (body.abs() <= 0.1 * rng).astype(int)
    df["hammer"] = ((lower_wick > 2 * body.abs()) & (upper_wick < body.abs())).astype(int)
    df["shooting_star"] = ((upper_wick > 2 * body.abs()) & (lower_wick < body.abs())).astype(int)

    df["bullish_engulfing"] = (
        (body.shift(1) < 0) & (body > 0) &
        (df["close"] > df["open"].shift(1)) &
        (df["open"] < df["close"].shift(1))
    ).astype(int)

    df["bearish_engulfing"] = (
        (body.shift(1) > 0) & (body < 0) &
        (df["open"] > df["close"].shift(1)) &
        (df["close"] < df["open"].shift(1))
    ).astype(int)

    return df


FEATURE_COLS = [

    # Technical
    "sma_ratio",
    "rsi",
    "volatility",
    "momentum_pct",
    "volume_change",
    "macd_hist_scaled",
    "atr_scaled",

    # Candles
    "doji",
    "hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",

    # Market sentiment
    "vix_return",
    "dxy_return",
    "spy_return",

    # Economic
    "fed_rate_change",
    "yield10_change",
    "cpi_yoy",

    # Regime
    "market_regime"
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators with proportional price scaling to prevent overfitting."""
    df = df.copy()
    macro = load_macro_data()
    lagged_macro = macro.shift(1)

    df = df.merge(
        lagged_macro ,
        left_index=True,
        right_index=True,
        how="left"
    )

    df = df.ffill()

    # Base Core Identifiers
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["market_regime"] = np.where(
        df["close"] > df["sma_200"],
        1,
        0
    )
    df["sma_ratio"] = df["sma_10"] / df["sma_50"]
    df["rsi"] = compute_rsi(df["close"], 14)
    df["volatility"] = df["close"].pct_change().rolling(20).std()
    
    # FIX 1: Turn absolute momentum into a percentage return vector
    df["momentum_pct"] = df["close"].pct_change(periods=10)
    
    # Calculate base indicators
    macd_l, signal_l, hist_l = compute_macd(df["close"])
    raw_atr = compute_atr(df)
    
    # FIX 2: Normalize absolute metrics by dividing by the closing price
    df["macd_hist_scaled"] = hist_l / df["close"]
    df["atr_scaled"] = raw_atr / df["close"]

    # Structural Volume Switch Tracker
    if df["volume"].sum() > 0:
        df["volume_change"] = df["volume"].pct_change().clip(-3, 3)
    else:
        df["volume_change"] = 0.0

    df = add_candle_patterns(df)

    # Future Returns Forecast Target (1 = Up day tomorrow, 0 = Flat/Down)
    df["future_return"] = df["close"].shift(-1) / df["close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)

    # Purge all infinite leaks or unhandled elements
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


# Added the new structural matrices straight into the pipeline array
FEATURE_COLS = [

    # Technical
    "sma_ratio",
    "rsi",
    "volatility",
    "momentum_pct",
    "volume_change",
    "macd_hist_scaled",
    "atr_scaled",

    # Candles
    "doji",
    "hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",

    # Market sentiment
    "vix_return",
    "dxy_return",
    "spy_return",

    # Economic
    "fed_rate_change",
    "yield10_change",
    "cpi_yoy",

    # Regime
    "market_regime"
]


# =========================================================
# 3. WALK-FORWARD BACKTEST
# =========================================================
FEATURE_COLS = [

    # Technical
    "sma_ratio",
    "rsi",
    "volatility",
    "momentum_pct",
    "volume_change",
    "macd_hist_scaled",
    "atr_scaled",

    # Candles
    "doji",
    "hammer",
    "shooting_star",
    "bullish_engulfing",
    "bearish_engulfing",

    # Market sentiment
    "vix_return",
    "dxy_return",
    "spy_return",

    # Economic
    "fed_rate_change",
    "yield10_change",
    "cpi_yoy",

    # Regime
    "market_regime"
]

def walk_forward_backtest(
    df: pd.DataFrame,
    train_window: int = 500,
    test_window: int = 60,
    threshold: float = 0.53,
):
    records = []
    start = 0

    while start + train_window + test_window <= len(df):
        train_df = df.iloc[start : start + train_window]
        
        # FIXED: Corrected the slice index to maintain a clean 60-bar testing window
        test_df = df.iloc[start + train_window : start + train_window + test_window]

        X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
        X_test = test_df[FEATURE_COLS]

        model = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        print("\n========== DEBUG ==========")
        print("Train shape:", X_train.shape)

        print("\nTarget counts:")
        print(y_train.value_counts())

        print("\nMissing values:")
        print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

        print("===========================\n")

        model.fit(X_train, y_train)

        proba_up = model.predict_proba(X_test)[:, 1]

        for i, idx in enumerate(test_df.index):
            p = proba_up[i]
            actual_return = test_df["future_return"].iloc[i]
            
            # Grab the current macro trend identifier for this specific day
            current_sma_ratio = test_df["sma_ratio"].iloc[i]

            if p >= threshold:
                signal = "buy"
                strategy_return = actual_return
            elif p <= (1 - threshold):
                # MACRO FILTER: If price is above the 50 SMA (ratio > 1), 
                # skip shorting/selling and stay flat to protect your capital.
                if current_sma_ratio > 1.0:
                    signal = "flat"
                    strategy_return = 0.0
                else:
                    signal = "sell"
                    strategy_return = -actual_return
            else:
                signal = "flat"
                strategy_return = 0.0

            records.append({
                "date": idx,
                "model_proba_up": p,
                "signal": signal,
                "actual_return": actual_return,
                "strategy_return": strategy_return,
            })
            
        start += test_window

    if len(records) == 0:
        raise ValueError("Not enough rows to generate a walk-forward window test.")

    results = pd.DataFrame(records).set_index("date")
    metrics = compute_metrics(results)
    return results, metrics

def compute_metrics(results: pd.DataFrame) -> dict:
    traded = results[results["signal"] != "flat"]
    if traded.empty:
        return {"note": "No trades were taken at this threshold."}

    wins = traded[traded["strategy_return"] > 0]
    losses = traded[traded["strategy_return"] <= 0]

    win_rate = len(wins) / len(traded)
    avg_win = wins["strategy_return"].mean() if len(wins) else 0.0
    avg_loss = losses["strategy_return"].mean() if len(losses) else 0.0

    equity_curve = (1 + traded["strategy_return"]).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()

    daily_std = traded["strategy_return"].std()
    sharpe = (traded["strategy_return"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    buy_hold_return = (1 + results["actual_return"]).prod() - 1
    strategy_total_return = equity_curve.iloc[-1] - 1 if len(equity_curve) else 0.0

    return {
        "total_signals": len(results),
        "trades_taken": len(traded),
        "trade_frequency": round(len(traded) / len(results), 3),
        "win_rate": round(win_rate, 3),
        "avg_win_pct": round(avg_win * 100, 3),
        "avg_loss_pct": round(avg_loss * 100, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "strategy_total_return_pct": round(strategy_total_return * 100, 2),
        "buy_and_hold_return_pct": round(buy_hold_return * 100, 2),
    }


# =========================================================
# 4. ENTRY POINT
# =========================================================
def run_study(ticker: str, period: str = "5y", interval: str = "1d", threshold: float = 0.53):
    print(f"Loading {ticker} ({period}, {interval}) from Yahoo Finance...")
    raw = load_data(ticker, period=period, interval=interval)
    print(f"Loaded {len(raw)} bars: {raw.index.date[0]} to {raw.index.date[-1]}")

    df = build_features(raw)
    print(f"{len(df)} bars usable after feature engineering")

    results, metrics = walk_forward_backtest(df, threshold=threshold)
    # Fit a final model on the entire dataset to save it for production
    final_model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
    X_final = df[FEATURE_COLS]
    y_final = df["target"]
    final_model.fit(X_final, y_final)

    # =====================================
    # FEATURE IMPORTANCE REPORT
    # =====================================
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": final_model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )
    print("\nTop Predictive Features")
    print(importance.head(15))
    save_path = r"c:\Users\Audit One\Desktop\personal staff\py_project\gb_model_GCF.pkl"
    joblib.dump(final_model, save_path)
    print(f"✅ Success! Production model file saved to: {save_path}")
    # Save the trained model and the feature list to files
    joblib.dump(final_model, f"gb_model_{ticker.replace('=','')}.pkl")
    print(f"Production model saved successfully as: gb_model_{ticker.replace('=','')}.pkl")

    print("\n--- Backtest Results ---")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title()}: {v}")


# if __name__ == "__main__":
#     run_study("GC=F", period="5y", interval="1d", threshold=0.53)
        
if __name__ == "__main__":
    # Gold futures and EUR/USD spot, Yahoo Finance tickers
    for ticker in ["GC=F", "EURUSD=X"]:
        print("\n" + "=" * 60)
        run_study(ticker, period="5y", interval="1d", threshold=0.55)

    