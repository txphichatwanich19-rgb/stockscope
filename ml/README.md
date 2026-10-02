# Stock Direction Prediction — CP020003 Final Project

**Model / Task type:** Binary Classification (XGBoost) — predicts whether a stock's close price
will be higher N trading days from now, using technical indicators.

## How to run

**Option A — Google Colab (recommended, free GPU/CPU, nothing to install locally):**
1. Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook
2. Upload `stock_direction_model.ipynb`
3. Run the first cell (uncomment the `!pip install` line), then Runtime → Run all

**Option B — locally:**
```bash
pip install -r requirements.txt
jupyter notebook stock_direction_model.ipynb
```

## What's inside the notebook

1. Data collection (5y daily OHLCV, 20 tickers, `yfinance`)
2. Dataset understanding / EDA
3. Feature engineering — technical indicators (RSI, MACD, SMA distance, volume ratio, volatility)
4. Labeling — next-period direction; unlabeled tail rows dropped (not silently labeled "down")
5. Chronological (not random) train/test split **with a label purge** (a training row is kept only if its
   label's target date is before the test cutoff) — the key anti-leakage design choice
6. Horizon + hyperparameters chosen with `TimeSeriesSplit` CV over unique dates, purged by target date,
   on the training set only
7. Model training — Logistic Regression & Random Forest baselines vs. XGBoost
8. Evaluation — accuracy / precision / recall / F1 / ROC-AUC with a block-bootstrap 95% CI
9. Multi-horizon comparison (1 / 5 / 10 days), reported with confidence intervals
10. Feature importance + SHAP
11. Illustrative backtest (clearly labeled as academic, not investment advice)
12. Limitations & conclusion

## Slides

`stock_direction_presentation.pptx` — 15-slide deck following the course's suggested format
(Motivation → Dataset → Methodology → Results → Impact & Q&A), built from the notebook's actual
numbers. Source/regeneration scripts are in `slides/`. Fill in the team name/members placeholders
on slides 1–2 before presenting.

## Before you present — fill these in

- [ ] Team info table at the top of the notebook, and on slides 1–2 of the deck
      (Group No. / Name / student names & IDs)
- [ ] Register the topic in the course's sign-up CSV: **Project Topic**, **Model/Task Type =
      "Binary Classification (XGBoost)"**
- [ ] The notebook downloads a **fixed data window** (2021-10-04 to 2026-10-01), so re-running it
      should reproduce the saved outputs and the slide numbers (small differences are possible if
      package versions or Yahoo's historical data change). Re-run it once before presenting.

## Headline result (for slides)

An honest negative result, not a high-accuracy claim: with free daily prices and standard
technical indicators, **this experiment finds no model (XGBoost, Random Forest, Logistic Regression) that beats a
coin flip** at any horizon tested — every ROC-AUC 95% confidence interval contains 0.5 — feature
importances sit in a narrow band, and the toy strategy trails buy-and-hold. That is consistent with
weak-form market efficiency but does not prove it; the project's value is a leakage-free pipeline
whose evaluation can be trusted. See the Limitations section in the notebook.

The numbers on slides 5 and 7–14 come from the notebook's saved Colab run (it also prints a
`PRESENTATION_RESULTS_JSON` line for exactly this purpose). If you re-run and any number changes,
update `slides/build_deck.js` and rebuild.

## About the notebook file

`stock_direction_model.ipynb` is the version edited and run in Colab (with saved outputs) and is the
source of truth. `build_notebook.py` is the earlier generator and is **outdated — do not run it**, it
would overwrite this notebook with an older version.
