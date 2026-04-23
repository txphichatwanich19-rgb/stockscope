# Stockscope

Real-time stock dashboard — technical analysis, news (with Thai translation), and rule-based signals.

## Features

- Candlestick chart with SMA / Bollinger Bands / RSI / MACD / Volume
- Company stats: market cap, P/E, dividend, 52-week range
- Yahoo Finance news, auto-translated to Thai
- Rule-based buy/sell signal (Bullish / Bearish / Neutral)
- 14 curated ticker categories — US mega cap, tech, finance, crypto, ETFs, Thai SET, and more

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Data: Yahoo Finance (delay ~1 min).
