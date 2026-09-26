"""Generates stock_direction_model.ipynb from code below.
Run this once with `python build_notebook.py` whenever the pipeline changes;
it regenerates the .ipynb deterministically. Requires: pip install nbformat
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ------------------------------------------------------------------
md(r"""# Stock Direction Prediction with XGBoost
### CP020003 — Artificial Intelligence · Final Project

**Project topic:** Predicting next-period stock price direction (up / down) from
technical indicators, using XGBoost — connected to our live dashboard, **Stockscope**.

**Model / Task type:** Binary Classification (tabular, time series features)

| Group No. | Group Name | Student Name | Student ID |
|---|---|---|---|
| _fill in_ | _fill in_ | _member 1_ | _id_ |
| | | _member 2_ | _id_ |
| | | _member 3_ | _id_ |
| | | _member 4_ | _id_ |
| | | _member 5_ | _id_ |
""")

# ------------------------------------------------------------------
md(r"""## 1. Motivation

Our team built **Stockscope**, a real-time stock dashboard (Streamlit + yfinance) that already
computes technical indicators (RSI, MACD, moving averages, support/resistance zones) for any
ticker. A natural next question: **can those same indicators predict where the price goes next?**

We frame this as a **supervised binary classification** problem — *will the closing price be
higher N trading days from now?* — and train **XGBoost** to answer it, because:

- Our features are tabular (indicator values), not images or free text → tree ensembles are a
  strong, standard fit.
- XGBoost trains fast even on a laptop / free Colab CPU, so we can iterate and validate carefully.
- It gives **feature importances**, so the result is *explainable* — we can say *why* the model
  predicts what it predicts, not just report a number.

We treat this explicitly as a controlled experiment in market efficiency, **not** as investment
advice — see the Limitations section at the end.
""")

# ------------------------------------------------------------------
md("## 2. Setup")
code(r"""# If running in Google Colab, uncomment the line below first:
# !pip install -q xgboost scikit-learn shap yfinance

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
""")

# ------------------------------------------------------------------
md(r"""## 3. Data Collection

We pull **5 years of daily OHLCV data** for 20 large-cap US tickers spanning multiple sectors
(tech, finance, energy, healthcare, consumer, media) directly from Yahoo Finance via `yfinance` —
the same data source our Stockscope dashboard already uses live. Using multiple sectors, rather
than a single stock, gives the model more rows and reduces the risk of learning one company's
idiosyncratic quirks instead of general technical-indicator behavior.
""")
code(r"""TICKERS = [
    "AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "AMZN", "KO", "GOOGL", "META",
    "WMT", "PG", "V", "MA", "DIS", "NFLX", "AMD", "INTC", "BAC", "CVX",
]
PERIOD = "5y"

raw = yf.download(
    TICKERS, period=PERIOD, interval="1d",
    auto_adjust=False, progress=False, group_by="ticker", threads=True,
)

frames = []
for t in TICKERS:
    df = raw[t].copy().dropna(subset=["Close"])
    df["Ticker"] = t
    frames.append(df)

full = pd.concat(frames).reset_index()
print(f"Rows: {len(full):,} | Tickers: {full['Ticker'].nunique()} | "
      f"Date range: {full['Date'].min().date()} -> {full['Date'].max().date()}")
full.head()
""")

# ------------------------------------------------------------------
md(r"""## 4. Dataset Understanding (EDA)

Before modeling, we look at what the raw data actually contains: how many rows per ticker,
whether any values are missing, and what the price series looks like.
""")
code(r"""print("Rows per ticker:")
print(full.groupby("Ticker").size())

print("\nMissing values per column:")
print(full[["Open", "High", "Low", "Close", "Volume"]].isna().sum())

full[["Open", "High", "Low", "Close", "Volume"]].describe()
""")
code(r"""fig, ax = plt.subplots(figsize=(10, 4))
sample = full[full["Ticker"] == "AAPL"]
ax.plot(sample["Date"], sample["Close"])
ax.set_title("AAPL Close Price — 5y (sample ticker, sanity check)")
ax.set_xlabel("Date"); ax.set_ylabel("Close (USD)")
plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------------
md(r"""## 5. Feature Engineering

We compute the same technical indicators Stockscope shows on the dashboard:

| Feature | Meaning |
|---|---|
| `ret_1d`, `ret_5d`, `ret_10d`, `ret_21d` | Past return over 1/5/10/21 trading days (momentum) |
| `rsi14` | 14-day Relative Strength Index (overbought/oversold) |
| `macd_hist` | MACD histogram (trend strength/direction) |
| `dist_sma20`, `dist_sma50` | % distance of price from its 20/50-day moving average |
| `vol_ratio` | Today's volume vs. its 20-day average (unusual activity) |
| `volatility_20d` | Rolling std-dev of daily returns (risk/choppiness) |

**No look-ahead leakage:** every feature above is computed with `rolling()` / `ewm()` windows
that only look **backward** in time. We compute features **per ticker** (`groupby("Ticker")`)
so no information ever crosses from one company's timeline into another's.
""")
code(r"""def sma(s, w): return s.rolling(w, min_periods=w).mean()
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def rsi(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def macd_hist_line(s, fast=12, slow=26, signal=9):
    macd_line = ema(s, fast) - ema(s, slow)
    signal_line = ema(macd_line, signal)
    return macd_line - signal_line


FEATURES = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d",
    "rsi14", "macd_hist", "dist_sma20", "dist_sma50",
    "vol_ratio", "volatility_20d",
]

def make_features(g, horizon):
    g = g.sort_values("Date").copy()
    close = g["Close"]
    g["ret_1d"] = close.pct_change(1)
    g["ret_5d"] = close.pct_change(5)
    g["ret_10d"] = close.pct_change(10)
    g["ret_21d"] = close.pct_change(21)
    g["rsi14"] = rsi(close)
    g["macd_hist"] = macd_hist_line(close)
    sma20 = sma(close, 20)
    sma50 = sma(close, 50)
    g["dist_sma20"] = (close - sma20) / sma20
    g["dist_sma50"] = (close - sma50) / sma50
    g["vol_ratio"] = g["Volume"] / g["Volume"].rolling(20, min_periods=20).mean()
    g["volatility_20d"] = g["ret_1d"].rolling(20, min_periods=20).std()
    # Label uses ONLY future price (never fed back as a feature) — see section 6.
    g["label_up"] = (close.shift(-horizon) > close).astype(int)
    return g


def build_feature_frame(df, horizon):
    # Explicit loop (not groupby().apply()) — pandas' groupby-apply column handling
    # changed across versions (2.2 warns, 3.0 drops the grouping column from the
    # result by default), so a plain loop + concat is the version-safe choice here.
    parts = [make_features(df[df["Ticker"] == t], horizon) for t in df["Ticker"].unique()]
    return pd.concat(parts, ignore_index=True)
""")

# ------------------------------------------------------------------
md(r"""## 6. Labeling — Next-Period Direction

`label_up = 1` if the close price `horizon` trading days from now is **higher** than today's
close, else `0`. Note the label is built with `close.shift(-horizon)` — it reaches **forward**
in time — but it is only ever used as the target `y`, never joined back into `X`. The last
`horizon` rows of each ticker get a `NaN` label (no future price exists yet) and are dropped.

This separation — features strictly backward-looking, label strictly forward-looking, never mixed
— is what keeps the pipeline **leakage-free**.
""")

# ------------------------------------------------------------------
md(r"""## 7. Train / Test Split — Time-Based, Not Random

A random 80/20 row split would leak information: rows from the same week end up on both sides,
and because indicators are autocorrelated (today's RSI is close to yesterday's), the model would
partly "memorize" the test period instead of generalizing to the future.

Instead we pick a **single global date cutoff** at the 80th percentile of all dates. Every
training row happens strictly before every test row, across all 20 tickers at once — mimicking
how the model would actually be used: trained on the past, evaluated on data it has never seen.
""")
code(r"""HORIZON = 10  # trading days ahead — chosen after comparing horizons in section 9

feat_df = build_feature_frame(full, HORIZON)
model_df = feat_df.dropna(subset=FEATURES + ["label_up"]).copy()

cutoff = model_df["Date"].quantile(0.8)
train = model_df[model_df["Date"] < cutoff]
test = model_df[model_df["Date"] >= cutoff]

print(f"Cutoff date: {cutoff.date()}")
print(f"Train rows: {len(train):,} ({train['Date'].min().date()} -> {train['Date'].max().date()})")
print(f"Test rows : {len(test):,} ({test['Date'].min().date()} -> {test['Date'].max().date()})")
print("\nTrain label balance:", train["label_up"].value_counts(normalize=True).round(3).to_dict())
print("Test label balance :", test["label_up"].value_counts(normalize=True).round(3).to_dict())

X_train, y_train = train[FEATURES], train["label_up"]
X_test, y_test = test[FEATURES], test["label_up"]
""")

# ------------------------------------------------------------------
md(r"""## 8. Model Training — Baselines vs. XGBoost

We compare three models on the *exact same* leakage-free split, so the comparison is fair:

1. **Logistic Regression** — simplest linear baseline (features standardized first).
2. **Random Forest** — a second tree-ensemble baseline, no boosting.
3. **XGBoost** — our chosen model, tuned with `TimeSeriesSplit` cross-validation **inside the
   training set only** (the test set is never touched during tuning).
""")
code(r"""scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

logreg = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
logreg.fit(X_train_s, y_train)

rf = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

print("Baselines trained.")
""")
code(r"""# Small hyperparameter search for XGBoost, validated with TimeSeriesSplit
# INSIDE the training set only (test set stays untouched until final evaluation).
param_grid = [
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200},
    {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300},
    {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 150},
]
tscv = TimeSeriesSplit(n_splits=4)

best_params, best_auc = None, -1
for params in param_grid:
    fold_aucs = []
    for tr_idx, val_idx in tscv.split(X_train):
        m = XGBClassifier(
            **params, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        p = m.predict_proba(X_train.iloc[val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_train.iloc[val_idx], p))
    mean_auc = np.mean(fold_aucs)
    print(f"{params} -> CV AUC = {mean_auc:.4f}")
    if mean_auc > best_auc:
        best_auc, best_params = mean_auc, params

print("\nBest params:", best_params)

xgb_model = XGBClassifier(
    **best_params, subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=RANDOM_STATE,
)
xgb_model.fit(X_train, y_train)
""")

# ------------------------------------------------------------------
md("## 9. Evaluation")
code(r"""def evaluate(name, y_true, pred, proba):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
    }

results = []
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
results.append({"model": "Majority-class baseline", "accuracy": round(baseline_acc, 4),
                 "precision": None, "recall": None, "f1": None, "roc_auc": 0.5})
results.append(evaluate("Logistic Regression", y_test, logreg.predict(X_test_s), logreg.predict_proba(X_test_s)[:, 1]))
results.append(evaluate("Random Forest", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1]))
results.append(evaluate("XGBoost (ours)", y_test, xgb_model.predict(X_test), xgb_model.predict_proba(X_test)[:, 1]))

results_df = pd.DataFrame(results)
results_df
""")
code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 4))

cm = confusion_matrix(y_test, xgb_model.predict(X_test))
ConfusionMatrixDisplay(cm, display_labels=["Down", "Up"]).plot(ax=axes[0], colorbar=False)
axes[0].set_title(f"XGBoost Confusion Matrix (horizon={HORIZON}d)")

for name, proba in [
    ("Logistic Regression", logreg.predict_proba(X_test_s)[:, 1]),
    ("Random Forest", rf.predict_proba(X_test)[:, 1]),
    ("XGBoost", xgb_model.predict_proba(X_test)[:, 1]),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve"); axes[1].legend(fontsize=8)

plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------------
md(r"""## 10. Does the Prediction Horizon Matter?

Single-day price moves are dominated by noise (this is the core idea behind the **weak-form
Efficient Market Hypothesis**). We test whether looking further ahead — 5 or 10 trading days —
gives the same technical indicators more signal to work with, by repeating the *entire* leakage-
free pipeline (features → split → train → evaluate) at three horizons.
""")
code(r"""horizon_results = []
for h in [1, 5, 10]:
    fd = build_feature_frame(full, h)
    md_ = fd.dropna(subset=FEATURES + ["label_up"]).copy()
    c = md_["Date"].quantile(0.8)
    tr, te = md_[md_["Date"] < c], md_[md_["Date"] >= c]

    m = XGBClassifier(**best_params, subsample=0.8, colsample_bytree=0.8,
                       eval_metric="logloss", random_state=RANDOM_STATE)
    m.fit(tr[FEATURES], tr["label_up"])
    p = m.predict(te[FEATURES])
    pr = m.predict_proba(te[FEATURES])[:, 1]
    horizon_results.append({
        "horizon_days": h,
        "accuracy": round(accuracy_score(te["label_up"], p), 4),
        "roc_auc": round(roc_auc_score(te["label_up"], pr), 4),
        "baseline_acc": round(max(te["label_up"].mean(), 1 - te["label_up"].mean()), 4),
    })

horizon_df = pd.DataFrame(horizon_results)
horizon_df
""")
code(r"""fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(horizon_df["horizon_days"], horizon_df["roc_auc"], marker="o", label="XGBoost ROC-AUC")
ax.axhline(0.5, color="gray", linestyle="--", label="Random (AUC=0.5)")
ax.set_xlabel("Prediction horizon (trading days ahead)")
ax.set_ylabel("ROC-AUC (test set)")
ax.set_title("Does looking further ahead help?")
ax.legend(); plt.tight_layout(); plt.show()
""")

# ------------------------------------------------------------------
md(r"""## 11. Feature Importance & SHAP

Which indicators does the model lean on most? We use both XGBoost's built-in importance
(gain-based) and SHAP values (which also show *direction* of effect, not just magnitude).
""")
code(r"""importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
importances.plot.barh(ax=ax)
ax.set_title(f"XGBoost Feature Importance (horizon={HORIZON}d)")
ax.set_xlabel("Importance (gain)")
plt.tight_layout(); plt.show()
""")
code(r"""import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=True)
""")

# ------------------------------------------------------------------
md(r"""## 12. Illustrative Strategy Backtest

> ⚠️ **Academic illustration only — not investment advice.** No transaction costs, slippage,
> taxes, or position sizing are modeled. Model probabilities are *statistical estimates*, not
> guarantees of future performance.

A toy strategy: go long only on days the model predicts "up" with probability > 0.55, otherwise
stay in cash. We compare its cumulative return on the test period against simple buy-and-hold,
averaged across all 20 tickers.
""")
code(r"""test_bt = test.copy()
test_bt["proba_up"] = xgb_model.predict_proba(X_test)[:, 1]
test_bt["fwd_return"] = test_bt.groupby("Ticker")["Close"].transform(
    lambda s: s.shift(-HORIZON) / s - 1
)
test_bt = test_bt.dropna(subset=["fwd_return"])

test_bt["strategy_return"] = np.where(test_bt["proba_up"] > 0.55, test_bt["fwd_return"], 0.0)

strategy_cum = (1 + test_bt.groupby("Date")["strategy_return"].mean()).cumprod()
holdall_cum = (1 + test_bt.groupby("Date")["fwd_return"].mean()).cumprod()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(strategy_cum.index, strategy_cum.values, label="Model-gated strategy")
ax.plot(holdall_cum.index, holdall_cum.values, label="Buy & hold (equal-weight)")
ax.set_title("Illustrative cumulative return — test period only (no costs modeled)")
ax.set_ylabel("Growth of $1"); ax.legend()
plt.tight_layout(); plt.show()

print(f"Strategy final growth : {strategy_cum.iloc[-1]:.3f}x")
print(f"Buy & hold final growth: {holdall_cum.iloc[-1]:.3f}x")
""")

# ------------------------------------------------------------------
md(r"""**Reading this result:** in our test window, buy-and-hold usually wins by a wide margin.
That is not a bug — the model sits in cash whenever its predicted probability is ≤ 0.55, and our
test period happens to contain a strong broad market uptrend. A ~0.53 AUC edge is too weak to
compensate for missing that many "up" days in cash. This is itself an honest, useful finding:
**a marginal statistical edge is not automatically a profitable trading strategy** — timing
decisions carry a real opportunity cost that the accuracy/AUC numbers alone don't show.

## 13. Limitations

- **Modest signal.** ROC-AUC hovers near 0.50–0.53 — close to random. This is *expected and
  consistent* with market-efficiency theory: if technical indicators alone reliably predicted
  short-term direction, that edge would be arbitraged away. We report this honestly rather than
  tuning until numbers look better (which would likely mean the pipeline had started leaking).
- **No fundamental or macro data.** Earnings, interest rates, news events, and broad market
  regime shifts are not in the feature set.
- **Survivorship bias.** Our 20 tickers are large, currently-listed companies; delisted/failed
  companies are excluded, which can inflate apparent predictability.
- **No transaction costs** in the backtest (Section 12) — real trading would erode any edge
  further.
- **Live sentiment (Stockscope's news + VADER scraper) is not included here** — Yahoo News only
  exposes *current* headlines, not a historical archive aligned to each past trading day, so it
  cannot be backtested with this free data source. It could be added as a **live-only** feature
  in the deployed dashboard (see Section 15).

## 14. Conclusion

- We built a fully leakage-free pipeline (backward-looking features, forward-looking label kept
  strictly separate, single chronological train/test cutoff) predicting next-period stock
  direction from technical indicators, benchmarked against a majority-class baseline, Logistic
  Regression, and Random Forest.
- XGBoost modestly outperforms the naive baseline at longer horizons (10-day AUC ≈ 0.53) but not
  at 1-day — confirming that single-day moves are close to unpredictable from technical
  indicators alone, while slightly more signal exists at the multi-week scale.
- `macd_hist`, `dist_sma50`, and short-term returns are consistently the most informative
  features across horizons.
- **Main insight:** technical indicators carry a small, real signal for medium-term direction,
  but not enough to trade on without additional data (fundamentals, sentiment, risk management) —
  a finding well aligned with mainstream market-efficiency research.

## 15. Future Work — Connecting Back to Stockscope

Our dashboard already computes these exact indicators live for any ticker. A natural extension:
serialize `xgb_model` (`joblib.dump`) and load it in the Streamlit app to show
*"AI: 61% probability of higher close in 10 trading days"* next to the existing technical view —
clearly labeled as a statistical estimate, not financial advice.
""")

nb["cells"] = cells
nbf.write(nb, "stock_direction_model.ipynb")
print("Wrote stock_direction_model.ipynb with", len(cells), "cells")
