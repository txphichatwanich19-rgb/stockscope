"""แดชบอร์ดหุ้นเรียลไทม์ — วิเคราะห์เทคนิค + ข่าว (แปลไทย) + ไอเดียลงทุน"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from deep_translator import GoogleTranslator
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sentiment_analyzer = SentimentIntensityAnalyzer()


def sentiment_score(text: str) -> float:
    """Return VADER compound score: -1 (very neg) to +1 (very pos)."""
    if not text:
        return 0.0
    try:
        return _sentiment_analyzer.polarity_scores(text)["compound"]
    except Exception:
        return 0.0


def sentiment_label(score: float) -> tuple[str, str]:
    """Returns (emoji+label, css class)."""
    if score >= 0.4:  return "🟢 บวกมาก", "up"
    if score >= 0.1:  return "🟢 บวก", "up"
    if score > -0.1:  return "⚪ กลาง", "flat"
    if score > -0.4:  return "🔴 ลบ", "down"
    return "🔴 ลบมาก", "down"

st.set_page_config(
    page_title="แดชบอร์ดหุ้น",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Global styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    .stApp { background: #f7f8fa; }

    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid rgba(15,23,42,0.06);
    }
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.25rem;
    }

    /* Hero card */
    .hero {
        background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 18px;
        padding: 1.6rem 1.9rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03), 0 8px 24px -8px rgba(15,23,42,0.06);
    }
    .hero .sym {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #64748b;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-weight: 500;
    }
    .hero .name {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.2rem 0 0.15rem 0;
        letter-spacing: -0.025em;
    }
    .hero .meta { color: #64748b; font-size: 0.85rem; }
    .hero .price {
        font-size: 2.8rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.035em;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .hero .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.86rem;
        margin-top: 0.6rem;
        font-variant-numeric: tabular-nums;
    }
    .chip.up    { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .chip.down  { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .chip.flat  { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

    /* Stat tiles */
    .tile {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
        transition: box-shadow 0.18s, border-color 0.18s;
    }
    .tile:hover {
        box-shadow: 0 2px 4px rgba(15,23,42,0.04), 0 8px 16px -8px rgba(15,23,42,0.08);
    }
    .tile .label {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .tile .value {
        color: #0f172a;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 0.4rem;
        letter-spacing: -0.015em;
        font-variant-numeric: tabular-nums;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 12px;
        padding: 0.3rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        color: #64748b;
        font-weight: 500;
        transition: background 0.15s, color 0.15s;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #0f172a; }
    .stTabs [aria-selected="true"] {
        background: #f1f5f9 !important;
        color: #0f172a !important;
        font-weight: 600;
    }

    /* News cards */
    .news-card {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.7rem;
        transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    }
    .news-card:hover {
        border-color: rgba(79,70,229,0.3);
        box-shadow: 0 4px 14px -4px rgba(79,70,229,0.12);
        transform: translateY(-1px);
    }
    .news-card .title { color: #0f172a; font-weight: 600; font-size: 1rem; line-height: 1.45; }
    .news-card .title a { color: #0f172a; text-decoration: none; }
    .news-card .title a:hover { color: #4f46e5; }
    .news-card .orig   { color: #94a3b8; font-size: 0.82rem; margin-top: 0.25rem; font-style: italic; }
    .news-card .meta   { color: #64748b; font-size: 0.78rem; margin-top: 0.4rem; }
    .news-card .summary{ color: #334155; font-size: 0.9rem; margin-top: 0.6rem; line-height: 1.6; }

    /* Verdict card */
    .verdict {
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        font-size: 1.45rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: -0.015em;
        margin-bottom: 1rem;
        border: 1px solid rgba(15,23,42,0.06);
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03), 0 8px 24px -8px rgba(15,23,42,0.06);
    }
    .verdict.bull { border-color: #a7f3d0; color: #047857; background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%); }
    .verdict.bear { border-color: #fecaca; color: #b91c1c; background: linear-gradient(180deg, #ffffff 0%, #fef2f2 100%); }
    .verdict.flat { border-color: #e2e8f0; color: #475569; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }

    /* Buttons */
    .stButton > button {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.1);
        color: #334155;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.15s;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    }
    .stButton > button:hover {
        background: #f8fafc;
        border-color: #4f46e5;
        color: #4f46e5;
        box-shadow: 0 2px 6px rgba(79,70,229,0.12);
    }

    /* Brand */
    .brand {
        display: flex; align-items: center; gap: 0.7rem;
        padding: 0.25rem 0 1.1rem 0;
        border-bottom: 1px solid rgba(15,23,42,0.06);
        margin-bottom: 1rem;
    }
    .brand .logo {
        width: 38px; height: 38px;
        border-radius: 11px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #e0e7ff;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.05rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.08), inset 0 1px 0 rgba(255,255,255,0.1);
    }
    .brand .name { font-weight: 700; font-size: 1.1rem; color: #0f172a; letter-spacing: -0.02em; }
    .brand .sub  { font-size: 0.7rem; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 500; }

    /* Generic mini-chip (used for sentiment + small badges) */
    .mini-chip {
        display: inline-flex; align-items: center;
        padding: 0.18rem 0.55rem; border-radius: 999px;
        font-size: 0.74rem; font-weight: 600;
        border: 1px solid transparent;
    }
    .mini-chip.up   { background: #f0fdf4; color: #166534; border-color: #d1fae5; }
    .mini-chip.down { background: #fef2f2; color: #991b1b; border-color: #fee2e2; }
    .mini-chip.flat { background: #f4f4f5; color: #52525b; border-color: #e4e4e7; }

    /* News overall sentiment */
    .sent-overall {
        display: flex; justify-content: space-between; align-items: center;
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    }
    .sent-overall-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
    }
    .sent-overall-score {
        display: flex; align-items: center; gap: 0.6rem;
    }
    .sent-counts {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Level tile */
    .level-tile {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 14px;
        padding: 0.95rem 1.05rem;
        height: 100%;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
        transition: box-shadow 0.18s;
    }
    .level-tile:hover { box-shadow: 0 2px 8px rgba(15,23,42,0.06); }
    .level-tile .role {
        font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: #64748b;
    }
    .level-tile .price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem; font-weight: 700;
        margin-top: 0.35rem; letter-spacing: -0.015em;
        font-variant-numeric: tabular-nums;
    }
    .level-tile .delta {
        font-size: 0.76rem; margin-top: 0.25rem; color: #64748b;
    }
    .level-tile.entry     { border-color: #a7f3d0; background: linear-gradient(180deg, #ffffff 0%, #f0fdf4 100%); }
    .level-tile.entry .price     { color: #047857; }
    .level-tile.stop      { border-color: #fecaca; background: linear-gradient(180deg, #ffffff 0%, #fef2f2 100%); }
    .level-tile.stop .price      { color: #b91c1c; }
    .level-tile.target    { border-color: #c7d2fe; background: linear-gradient(180deg, #ffffff 0%, #eef2ff 100%); }
    .level-tile.target .price    { color: #4338ca; }
    .level-tile.resistance { border-color: #fecaca; background: linear-gradient(180deg, #ffffff 0%, #fef2f2 100%); }
    .level-tile.resistance .price { color: #b91c1c; }
    .level-tile.current   { background: #f8fafc; }
    .level-tile.current .price   { color: #0f172a; }

    .rr-badge {
        display: inline-block;
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.08);
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    }
    .rr-badge b { color: #0f172a; font-weight: 700; }

    .pick-intro {
        color: #475569; font-size: 0.9rem; margin-bottom: 0.8rem;
    }
    .pick-disclaim {
        background: linear-gradient(180deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
        border-radius: 12px; padding: 0.75rem 0.95rem;
        color: #78350f; font-size: 0.85rem; margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(146,64,14,0.05);
    }

    /* Price ladder (above chart) */
    .lv-ladder {
        background: #ffffff;
        border: 1px solid rgba(15,23,42,0.06);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.7rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
    }
    .lv-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        padding: 0.35rem 0;
    }
    .lv-arrow {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        min-width: 145px;
        flex-shrink: 0;
    }
    .lv-row-up .lv-arrow { color: #b91c1c; }
    .lv-row-down .lv-arrow { color: #166534; }
    .lv-pills { display: flex; gap: 0.45rem; flex-wrap: wrap; }
    .lv-empty { font-size: 0.82rem; color: #94a3b8; font-style: italic; }

    .lv-pill {
        display: inline-flex; align-items: baseline; gap: 0.5rem;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        border: 1px solid;
        font-size: 0.85rem;
    }
    .lv-pill.lv-sell { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
    .lv-pill.lv-buy  { background: #f0fdf4; color: #166534; border-color: #a7f3d0; }
    .lv-pill.lv-stop { background: #fff7ed; color: #9a3412; border-color: #fed7aa; }
    .lv-pill .lv-role {
        font-weight: 600;
        font-size: 0.8rem;
    }
    .lv-pill .lv-price {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.92rem;
        font-variant-numeric: tabular-nums;
    }
    .lv-pill .lv-delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        opacity: 0.7;
        font-variant-numeric: tabular-nums;
    }

    /* Now row — prominent center */
    .lv-now {
        display: flex; align-items: center; gap: 0.8rem;
        padding: 0.7rem 0.85rem;
        margin: 0.4rem 0;
        background: linear-gradient(90deg, #f1f5f9 0%, #ffffff 50%, #f1f5f9 100%);
        border-top: 1px solid rgba(15,23,42,0.06);
        border-bottom: 1px solid rgba(15,23,42,0.06);
        border-radius: 8px;
    }
    .lv-now-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748b;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        min-width: 145px;
    }
    .lv-now-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01em;
        font-variant-numeric: tabular-nums;
    }
    .lv-now-tag {
        font-size: 0.7rem;
        font-weight: 700;
        color: #16a34a;
        background: #f0fdf4;
        padding: 0.2rem 0.5rem;
        border-radius: 999px;
        letter-spacing: 0.04em;
    }

    @media (max-width: 768px) {
        .lv-arrow, .lv-now-label { min-width: auto; width: 100%; }
        .lv-pill { padding: 0.35rem 0.65rem; font-size: 0.78rem; }
        .lv-pill .lv-price { font-size: 0.84rem; }
        .lv-pill .lv-role { font-size: 0.74rem; }
        .lv-pill .lv-delta { font-size: 0.68rem; }
        .lv-now-price { font-size: 1.2rem; }
    }

    /* Section heading */
    .section-h {
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748b;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin: 0.8rem 0 0.5rem 0;
    }

    /* Hide default streamlit footer */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }

    /* ========== Mobile (≤ 768px) ========== */
    @media (max-width: 768px) {
        /* Stack all column layouts vertically on mobile */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.5rem !important;
        }
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }

        /* Tighter page padding */
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.75rem !important;
        }

        /* Hero — smaller, less padding */
        .hero {
            padding: 1.1rem 1.2rem;
            border-radius: 14px;
        }
        .hero .name  { font-size: 1.3rem; }
        .hero .price { font-size: 2.1rem; }
        .hero .sym   { font-size: 0.72rem; }
        .hero .meta  { font-size: 0.78rem; }
        .hero .chip  { font-size: 0.8rem; padding: 0.28rem 0.65rem; }

        /* Stat tiles — slightly smaller */
        .tile { padding: 0.75rem 0.9rem; border-radius: 12px; }
        .tile .label { font-size: 0.66rem; }
        .tile .value { font-size: 1.1rem; }

        /* Tabs — horizontal scroll if needed, smaller text */
        .stTabs [data-baseweb="tab-list"] {
            overflow-x: auto;
            flex-wrap: nowrap;
            padding: 0.25rem;
            border-radius: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.45rem 0.8rem;
            font-size: 0.85rem;
            white-space: nowrap;
            flex-shrink: 0;
        }

        /* News card */
        .news-card { padding: 0.85rem 0.95rem; border-radius: 12px; }
        .news-card .title { font-size: 0.95rem; }
        .news-card .summary { font-size: 0.85rem; }

        /* Verdict */
        .verdict { padding: 1.1rem 1.2rem; font-size: 1.15rem; border-radius: 14px; }

        /* Level tile */
        .level-tile { padding: 0.8rem 0.95rem; border-radius: 12px; }
        .level-tile .price { font-size: 1.1rem; }

        /* Pick card */
        .pick-card { padding: 0.8rem 0.9rem; }
        .pick-card .price { font-size: 1.2rem; }

        /* TradingView widget — slightly shorter on mobile */
        .tradingview-widget-container { height: 520px !important; }
        [data-testid="stIFrame"] iframe { height: 540px !important; }
        iframe[title="streamlit_app"] { height: 540px !important; }

        /* Section heading bigger touch target spacing */
        .section-h { margin: 1rem 0 0.4rem 0; }

        /* Disclaim card */
        .pick-disclaim { font-size: 0.78rem; padding: 0.65rem 0.8rem; }
    }

    /* ========== Tablet (769-1024px) — 2-column compromise ========== */
    @media (min-width: 769px) and (max-width: 1024px) {
        .hero .price { font-size: 2.3rem; }
        .hero .name  { font-size: 1.5rem; }
        .tile .value { font-size: 1.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Indicators ----------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    return mid + n_std * std, mid, mid - n_std * std


def to_tv_symbol(yf_sym: str) -> str:
    """Convert Yahoo ticker → TradingView symbol format."""
    s = yf_sym.upper().strip()
    idx_map = {
        "^GSPC": "SP:SPX", "^IXIC": "NASDAQ:IXIC", "^DJI": "DJ:DJI",
        "^RUT": "TVC:RUT", "^VIX": "TVC:VIX", "^FTSE": "TVC:UKX",
        "^N225": "TVC:NI225", "^HSI": "TVC:HSI",
    }
    if s in idx_map:
        return idx_map[s]
    if s.endswith("-USD"):
        # Crypto — use Binance USDT pair (most liquid)
        base = s.replace("-USD", "")
        return f"BINANCE:{base}USDT"
    if s.endswith(".BK"):
        return f"SET:{s.replace('.BK', '')}"
    # Default: US stock — let TV auto-resolve exchange
    return s


def to_tv_interval(yf_interval: str) -> str:
    return {
        "1m": "1", "5m": "5", "15m": "15", "30m": "30",
        "1h": "60", "1d": "D", "1wk": "W", "1mo": "M",
    }.get(yf_interval, "D")


def compute_levels(df: pd.DataFrame, max_each: int = 3) -> dict:
    """Find support/resistance from local pivot highs/lows, cluster within 1.5%."""
    if len(df) < 20:
        return {"support": [], "resistance": [], "current": float(df["Close"].iloc[-1]) if len(df) else 0}
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    window = max(3, min(10, n // 40))
    res_piv, sup_piv = [], []
    for i in range(window, n - window):
        if all(highs[i] >= highs[i - k] and highs[i] >= highs[i + k] for k in range(1, window + 1)):
            res_piv.append(float(highs[i]))
        if all(lows[i] <= lows[i - k] and lows[i] <= lows[i + k] for k in range(1, window + 1)):
            sup_piv.append(float(lows[i]))

    def cluster(levels: list[float], tol: float = 0.015) -> list[float]:
        if not levels:
            return []
        levels = sorted(levels)
        groups = [[levels[0]]]
        for lvl in levels[1:]:
            if (lvl - groups[-1][-1]) / max(groups[-1][-1], 1e-9) < tol:
                groups[-1].append(lvl)
            else:
                groups.append([lvl])
        return [sum(g) / len(g) for g in groups]

    res_clust = cluster(res_piv)
    sup_clust = cluster(sup_piv)
    current = float(df["Close"].iloc[-1])
    resistance = sorted([x for x in res_clust if x > current * 1.002])[:max_each]
    support = sorted([x for x in sup_clust if x < current * 0.998], reverse=True)[:max_each]
    return {"support": support, "resistance": resistance, "current": current}


# ---------- Data fetch (cached) ----------
@st.cache_data(ttl=60, show_spinner=False)
def load_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def load_mini_batch(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Fetch 6-month daily data for a batch of tickers and compute summary stats + S/R."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            list(tickers), period="6mo", interval="1d",
            auto_adjust=False, progress=False, threads=True, group_by="ticker",
        )
    except Exception:
        return {}
    out: dict[str, dict] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            sub = raw[t] if multi else raw
            close = sub["Close"].dropna()
            if len(close) < 5:
                continue
            last = float(close.iloc[-1])

            def chg(days: int) -> float:
                if len(close) <= days:
                    return 0.0
                return (last / float(close.iloc[-days - 1]) - 1) * 100

            w1 = chg(5)
            m1 = chg(21)
            m3 = chg(63)
            r = float(rsi(close).iloc[-1])
            s20 = float(sma(close, 20).iloc[-1])
            s50 = float(sma(close, 50).iloc[-1]) if len(close) >= 30 else float("nan")
            if last > s20 and (np.isnan(s50) or last > s50) and r < 70:
                sig = "bull"
            elif last < s20 and (np.isnan(s50) or last < s50):
                sig = "bear"
            else:
                sig = "flat"

            # Compute nearest support / resistance from OHLC
            ohlc = sub.dropna(subset=["High", "Low", "Close"])
            s1 = r1 = None
            if len(ohlc) >= 30:
                lv = compute_levels(ohlc)
                if lv["support"]:
                    s1 = lv["support"][0]
                if lv["resistance"]:
                    r1 = lv["resistance"][0]

            out[t] = {
                "last": last, "w1": w1, "m1": m1, "m3": m3, "rsi": r, "sig": sig,
                "below_sma20": last < s20,
                "below_sma50": not np.isnan(s50) and last < s50,
                "s1": s1, "r1": r1,
            }
        except Exception:
            continue
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def load_info_batch(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Fetch fundamental info for multiple tickers in parallel. Cached 1 hour."""
    def _fetch(t: str) -> tuple[str, dict]:
        try:
            info = yf.Ticker(t).info or {}
            return t, {
                "name": info.get("longName") or info.get("shortName") or t,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "rev_growth": info.get("revenueGrowth"),         # % decimal
                "earnings_growth": info.get("earningsGrowth"),   # % decimal
                "profit_margin": info.get("profitMargins"),      # % decimal
                "roe": info.get("returnOnEquity"),
                "pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "rec_key": info.get("recommendationKey"),
                "rec_mean": info.get("recommendationMean"),      # 1 = strong buy, 5 = strong sell
                "rec_count": info.get("numberOfAnalystOpinions"),
                "target_mean": info.get("targetMeanPrice"),
                "current": info.get("currentPrice") or info.get("regularMarketPrice"),
            }
        except Exception:
            return t, {}

    if not tickers:
        return {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(_fetch, tickers))
    return dict(results)


def _rec_thai(key: str | None, mean: float | None) -> str:
    if mean is not None:
        if mean <= 1.5: return "🟢 Strong Buy"
        if mean <= 2.5: return "🟢 Buy"
        if mean <= 3.5: return "🟡 Hold"
        if mean <= 4.5: return "🔴 Sell"
        return "🔴 Strong Sell"
    if key:
        return {"strong_buy": "🟢 Strong Buy", "buy": "🟢 Buy",
                "hold": "🟡 Hold", "sell": "🔴 Sell",
                "strong_sell": "🔴 Strong Sell"}.get(key, "—")
    return "—"


@st.cache_data(ttl=3600, show_spinner=False)
def translate_th(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    try:
        # Google Translate รองรับ ~5000 ตัวอักษรต่อคำขอ
        chunks = [text[i : i + 4500] for i in range(0, len(text), 4500)]
        tr = GoogleTranslator(source="auto", target="th")
        return " ".join(tr.translate(c) for c in chunks if c.strip())
    except Exception:
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def load_news(ticker: str) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    items = []
    for n in raw:
        # yfinance recent versions nest under "content"
        c = n.get("content", n)
        title = c.get("title") or n.get("title")
        if not title:
            continue
        publisher = (
            (c.get("provider") or {}).get("displayName")
            if isinstance(c.get("provider"), dict)
            else c.get("publisher") or n.get("publisher")
        )
        link = None
        cl = c.get("canonicalUrl") or c.get("clickThroughUrl")
        if isinstance(cl, dict):
            link = cl.get("url")
        link = link or c.get("link") or n.get("link")
        ts = c.get("pubDate") or c.get("displayTime") or n.get("providerPublishTime")
        if isinstance(ts, (int, float)):
            when = datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
            try:
                when = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                when = None
        else:
            when = None
        summary = c.get("summary") or n.get("summary") or ""
        items.append(
            {"title": title, "publisher": publisher, "link": link, "when": when, "summary": summary}
        )
    return items


# ---------- Simple rule-based signal ----------
def build_signal(df: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    score = 0
    close = df["Close"]
    last = float(close.iloc[-1])

    s20 = sma(close, 20).iloc[-1]
    s50 = sma(close, 50).iloc[-1]
    s200 = sma(close, 200).iloc[-1] if len(close) >= 50 else np.nan

    if not np.isnan(s50) and not np.isnan(s200):
        if last > s50 > s200:
            score += 2
            reasons.append("ราคา > SMA50 > SMA200 → uptrend ระยะยาว")
        elif last < s50 < s200:
            score -= 2
            reasons.append("ราคา < SMA50 < SMA200 → downtrend ระยะยาว")

    if last > s20:
        score += 1
        reasons.append("ราคาอยู่เหนือ SMA20")
    else:
        score -= 1
        reasons.append("ราคาอยู่ต่ำกว่า SMA20")

    rsi_series = rsi(close)
    r = float(rsi_series.iloc[-1])
    if r < 30:
        score += 2
        reasons.append(f"RSI = {r:.1f} → oversold (มีโอกาสเด้ง)")
    elif r > 70:
        score -= 2
        reasons.append(f"RSI = {r:.1f} → overbought (มีโอกาสปรับฐาน)")
    else:
        reasons.append(f"RSI = {r:.1f} → กลาง")

    macd_line, signal_line, _ = macd(close)
    if len(macd_line) >= 2:
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            score += 2
            reasons.append("MACD เพิ่งตัดขึ้นเหนือ Signal → สัญญาณซื้อ")
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            score -= 2
            reasons.append("MACD เพิ่งตัดลงใต้ Signal → สัญญาณขาย")
        elif macd_line.iloc[-1] > signal_line.iloc[-1]:
            score += 1
            reasons.append("MACD อยู่เหนือ Signal")
        else:
            score -= 1
            reasons.append("MACD อยู่ใต้ Signal")

    if score >= 3:
        verdict = "🟢 โน้มเอียงฝั่งซื้อ (Bullish)"
    elif score <= -3:
        verdict = "🔴 โน้มเอียงฝั่งขาย (Bearish)"
    else:
        verdict = "🟡 ไม่มีสัญญาณชัดเจน (Neutral)"
    return verdict, reasons


# ---------- UI ----------
PICKS: dict[str, dict] = {
    "🌱 หุ้นเล็กน่าเติบโต": {
        "desc": "Small-mid cap growth · สแกนสดจาก universe 25 ตัว · เรียงตาม momentum 1 เดือน (RSI<75, เหนือ SMA50)",
        "mode": "auto_small_growth",
        "universe": [
            "IONQ", "RKLB", "ACHR", "JOBY", "SOUN", "HIMS", "SOFI", "RIOT",
            "MARA", "CLSK", "CIFR", "IREN", "BBAI", "LMND", "UPST", "AFRM",
            "PATH", "OPEN", "DKNG", "FUBO", "CHPT", "SERV", "NBIS", "CRCT",
            "TEM",
        ],
    },
    "🚀 หุ้นอนาคตไกล": {
        "desc": "Quality megatrend leaders · กรองด้วยกำไร/รายได้โต/นักวิเคราะห์แนะนำซื้อ · เรียงด้วยคะแนนรวม",
        "mode": "auto_future",
        "universe": [
            # AI / Cloud / Semi
            "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "AMD", "ARM", "ASML", "TSM",
            "AMAT", "KLAC", "LRCX", "MU", "QCOM", "ORCL", "CRM", "ADBE", "NOW", "PANW",
            "CRWD", "SNPS", "CDNS", "INTU",
            # Healthcare / Biotech leaders
            "LLY", "NVO", "UNH", "ISRG", "VRTX", "REGN", "MRK", "ABBV",
            # Financial leaders
            "V", "MA", "JPM", "GS", "BLK", "AXP",
            # Consumer / Platform leaders
            "COST", "WMT", "MCD", "NKE", "BKNG", "MELI", "SHOP", "NFLX", "DIS",
            # EV / Industrial
            "TSLA", "CAT", "DE", "HON",
        ],
    },
    "🎯 แตะโซนซื้อแล้ว": {
        "desc": "หุ้นที่ราคาตอนนี้อยู่ใกล้แนวรับ S1 (รัศมี ±2.5%) · พร้อมพิจารณาเข้าซื้อ",
        "mode": "auto_buyzone",
        "universe": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK-B",
            "AMD", "AVGO", "ASML", "TSM", "NFLX", "ORCL", "CRM", "ADBE",
            "PANW", "NOW", "ARM", "PLTR", "SMCI", "QCOM", "INTC",
            "IONQ", "RKLB", "ACHR", "SOFI", "HIMS", "RIOT", "COIN", "MSTR",
            "JPM", "BAC", "GS", "V", "MA", "BLK",
            "LLY", "UNH", "JNJ", "PFE", "MRK",
            "XOM", "CVX", "COP",
            "WMT", "COST", "MCD", "NKE", "SBUX",
            "BA", "F", "GM", "DIS",
            "BTC-USD", "ETH-USD", "SOL-USD",
        ],
    },
    "🚦 หุ้นซิ่ง": {
        "desc": "Momentum · สแกนสด · เรียงตาม % 1 สัปดาห์ล่าสุด",
        "mode": "momentum",
        "tickers": [
            "NVDA", "TSLA", "AMD", "PLTR", "MSTR", "COIN", "MARA",
            "SMCI", "ARM", "META", "NFLX", "AVGO", "CRWD", "SNOW",
            "IONQ", "RKLB", "SOFI", "HIMS",
        ],
    },
    "⚠️ ห้ามไปยุ่งตอนนี้": {
        "desc": "สแกนสด · ตัวที่ RSI > 75 (ร้อนเกิน) หรือราคาทะลุ SMA ลง",
        "mode": "avoid",
        "tickers": [
            "NVDA", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "PLTR", "AMD", "NFLX", "AVGO", "COIN", "MSTR", "SMCI",
            "ARM", "SNOW", "CRWD", "ORCL", "UBER", "LYFT",
        ],
    },
    "🎰 Option Plays (Call / Put)": {
        "desc": "หุ้น option liquid · สแกนสด · เรียงตาม volatility ล่าสุด (abs 1W %) · bias จากสัญญาณเทคนิค",
        "mode": "auto_options",
        "universe": [
            "SPY", "QQQ", "IWM", "DIA", "TSLA", "NVDA", "AAPL", "AMZN",
            "META", "MSFT", "GOOGL", "AMD", "NFLX", "BA", "COIN", "MSTR",
            "PLTR", "SMCI",
        ],
    },
}


CATEGORIES: dict[str, list[str]] = {
    "🔥 หุ้นยักษ์ใหญ่": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B"],
    "💻 เทคโนโลยี": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO"],
    "💰 การเงิน / ธนาคาร": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BLK", "C", "AXP"],
    "🏥 สุขภาพ / ยา": ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "⚡ พลังงาน": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG", "PSX", "MPC"],
    "🛒 สินค้าอุปโภคบริโภค": ["AMZN", "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT"],
    "🚗 รถยนต์ / EV": ["TSLA", "F", "GM", "RIVN", "LCID", "TM", "HMC", "STLA", "BYDDY"],
    "🎮 สื่อ / เกม": ["NFLX", "DIS", "SONY", "EA", "TTWO", "RBLX", "SPOT", "ROKU"],
    "✈️ สายการบิน / ท่องเที่ยว": ["DAL", "UAL", "AAL", "LUV", "BA", "BKNG", "ABNB", "MAR"],
    "🪙 คริปโต": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD"],
    "📊 ดัชนีตลาด": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI"],
    "🌐 กองทุน ETF": ["SPY", "QQQ", "VOO", "VTI", "IWM", "DIA", "ARKK", "GLD", "TLT"],
    "🇨🇳 หุ้นจีน (ADR)": ["BABA", "JD", "PDD", "NIO", "LI", "XPEV", "BIDU", "TME"],
    "🏦 หุ้นไทย (SET)": ["PTT.BK", "ADVANC.BK", "AOT.BK", "CPALL.BK", "KBANK.BK", "SCB.BK", "PTTEP.BK", "DELTA.BK"],
}

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"
if "category" not in st.session_state:
    st.session_state.category = "🔥 หุ้นยักษ์ใหญ่"

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="logo">📈</div>
            <div>
                <div class="name">แดชบอร์ดหุ้น</div>
                <div class="sub">เรียลไทม์ · ทั่วโลก</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-h">ค้นหาหุ้น</div>', unsafe_allow_html=True)
    ticker_input = st.text_input(
        "รหัสหุ้น",
        value=st.session_state.ticker,
        placeholder="AAPL, TSLA, BTC-USD, ^GSPC …",
        label_visibility="collapsed",
    )
    ticker = ticker_input.upper().strip()
    st.session_state.ticker = ticker

    st.markdown('<div class="section-h">หมวดหมู่</div>', unsafe_allow_html=True)

    pick_labels = [f"{name}" for name in PICKS.keys()]
    all_options = list(CATEGORIES.keys()) + ["──── 💡 ไอเดีย ────"] + pick_labels
    try:
        cat_index = all_options.index(st.session_state.category)
    except ValueError:
        cat_index = 0

    def _cat_disabled(opt: str) -> bool:
        return opt.startswith("──")

    category = st.selectbox(
        "Category",
        all_options,
        index=cat_index,
        label_visibility="collapsed",
    )
    if not _cat_disabled(category):
        st.session_state.category = category

    # Resolve to ticker list + optional metadata
    if category in CATEGORIES:
        tickers_in_cat = [(s, None, None) for s in CATEGORIES[category]]  # (sym, thesis, reason)
        show_meta = False
    elif category in PICKS:
        cfg = PICKS[category]
        mode = cfg["mode"]
        st.caption(cfg["desc"])

        universe = cfg.get("universe") or cfg.get("tickers") or []
        if isinstance(universe, dict):
            universe = list(universe.keys())

        with st.spinner("กำลังสแกน…"):
            mini = load_mini_batch(tuple(universe))

        tickers_in_cat = []
        if mode == "momentum":
            ranked = sorted([t for t in universe if t in mini],
                            key=lambda t: mini[t]["w1"], reverse=True)[:12]
            tickers_in_cat = [(t, f"1W {mini[t]['w1']:+.1f}% · RSI {mini[t]['rsi']:.0f}", None) for t in ranked]
        elif mode == "avoid":
            flagged = []
            for t in universe:
                d = mini.get(t)
                if not d:
                    continue
                if d["rsi"] > 75:
                    flagged.append((t, f"RSI {d['rsi']:.0f} · ร้อนเกิน", d["rsi"]))
                elif d["below_sma20"] and d["below_sma50"]:
                    flagged.append((t, "ใต้ SMA20 & 50 · trend พัง", d["rsi"]))
            flagged.sort(key=lambda x: x[2], reverse=True)
            tickers_in_cat = [(t, r, None) for t, r, _ in flagged[:8]]
            if not tickers_in_cat:
                st.info("ยังไม่มีตัวติดสัญญาณเตือน")
        elif mode == "auto_small_growth":
            with st.spinner("กำลังดึงข้อมูลพื้นฐาน…"):
                fund = load_info_batch(tuple(universe))
            scored = []
            for t, d in mini.items():
                if d["rsi"] >= 75 or (d["below_sma20"] and d["below_sma50"]):
                    continue
                f = fund.get(t, {})
                rev_g = f.get("rev_growth") or 0
                # For small caps: reward growth, accept negative margins (they're growing)
                fund_score = (rev_g * 80)
                analyst_score = (3.0 - (f.get("rec_mean") or 3.0)) * 12
                total = d["m1"] + fund_score + analyst_score
                scored.append((t, d, f, total))
            scored.sort(key=lambda x: x[3], reverse=True)
            tickers_in_cat = []
            for t, d, f, _ in scored[:12]:
                bits = [f"1M {d['m1']:+.1f}%"]
                if f.get("rev_growth") is not None:
                    bits.append(f"รายได้ {f['rev_growth']*100:+.0f}%")
                bits.append(f"RSI {d['rsi']:.0f}")
                if f.get("rec_mean"):
                    bits.append(_rec_thai(f.get("rec_key"), f.get("rec_mean")))
                tickers_in_cat.append((t, " · ".join(bits), None))
        elif mode == "auto_future":
            with st.spinner("กำลังดึงข้อมูลพื้นฐาน…"):
                fund = load_info_batch(tuple(universe))
            # Quality filter + composite score
            scored = []
            for t, d in mini.items():
                f = fund.get(t, {})
                rev_g = f.get("rev_growth") or 0
                margin = f.get("profit_margin") or 0
                rec_mean = f.get("rec_mean")  # 1=SB, 5=SS — lower is better
                # Quality gate: at least one of: rev growth > 3%, margin > 10%, or analyst Buy
                quality_pass = (
                    rev_g > 0.03 or margin > 0.10
                    or (rec_mean is not None and rec_mean <= 2.7)
                )
                if not quality_pass:
                    continue
                # Composite score
                perf_score = d["m3"]                                      # 3-month return %
                fund_score = (rev_g * 100) + (margin * 50)               # rewards growth + profitability
                analyst_score = (3.0 - rec_mean) * 15 if rec_mean else 0  # buy=+15, hold=0, sell=-15
                total = perf_score + fund_score + analyst_score
                scored.append((t, d, f, total))
            scored.sort(key=lambda x: x[3], reverse=True)
            tickers_in_cat = []
            for t, d, f, _score in scored[:12]:
                bits = [f"3M {d['m3']:+.1f}%"]
                if f.get("rev_growth") is not None:
                    bits.append(f"รายได้ {f['rev_growth']*100:+.0f}%")
                if f.get("profit_margin") is not None:
                    bits.append(f"กำไรสุทธิ {f['profit_margin']*100:.0f}%")
                if f.get("rec_mean"):
                    bits.append(_rec_thai(f.get("rec_key"), f.get("rec_mean")))
                if f.get("target_mean") and d.get("last"):
                    upside = (f["target_mean"] / d["last"] - 1) * 100
                    bits.append(f"🎯 เป้า {f['target_mean']:.0f} ({upside:+.0f}%)")
                tickers_in_cat.append((t, " · ".join(bits), None))
        elif mode == "auto_options":
            def _bias(sig: str) -> str:
                return {"bull": "📈 Call bias", "bear": "📉 Put bias", "flat": "⚪ Neutral"}[sig]
            candidates = sorted(mini.items(), key=lambda x: abs(x[1]["w1"]), reverse=True)
            tickers_in_cat = [
                (t, f"σ {abs(d['w1']):.1f}% · {_bias(d['sig'])}", None)
                for t, d in candidates[:12]
            ]
        elif mode == "auto_buyzone":
            # Stocks currently within ±2.5% of their nearest support (S1)
            candidates = []
            for t, d in mini.items():
                s1 = d.get("s1")
                if not s1 or not d.get("last"):
                    continue
                dist = (d["last"] / s1 - 1) * 100  # positive = above support, negative = below
                # In buy zone: -1% to +2.5% of S1 (touching or just above support)
                if -1.0 <= dist <= 2.5:
                    candidates.append((t, d, dist))
            candidates.sort(key=lambda x: abs(x[2]))  # closest to S1 first
            tickers_in_cat = [
                (t, f"S1 {d['s1']:,.2f} · ห่าง {dist:+.1f}% · RSI {d['rsi']:.0f}", None)
                for t, d, dist in candidates[:10]
            ]
            if not tickers_in_cat:
                st.info("ตอนนี้ยังไม่มีหุ้นที่ราคาเข้าโซนซื้อ (รัศมี ±2.5% ของ S1)")
    else:
        tickers_in_cat = []
        show_meta = False

    cols = st.columns(2)
    for i, (sym, meta, _) in enumerate(tickers_in_cat):
        display = sym.replace(".BK", "").replace("-USD", "")
        help_text = f"{sym}" + (f" · {meta}" if meta else "")
        if cols[i % 2].button(
            display,
            key=f"cat_{category}_{sym}",
            use_container_width=True,
            help=help_text,
        ):
            st.session_state.ticker = sym
            st.rerun()

    st.markdown('<div class="section-h">ช่วงเวลา</div>', unsafe_allow_html=True)
    period_labels = {
        "5d": "5 วัน", "1mo": "1 เดือน", "3mo": "3 เดือน", "6mo": "6 เดือน",
        "1y": "1 ปี", "2y": "2 ปี", "5y": "5 ปี", "10y": "10 ปี", "max": "ทั้งหมด",
    }
    period = st.selectbox(
        "ช่วงย้อนหลัง",
        list(period_labels.keys()),
        index=4,
        format_func=lambda x: period_labels[x],
        label_visibility="collapsed",
    )
    interval_labels = {
        "1m": "1 นาที", "5m": "5 นาที", "15m": "15 นาที", "30m": "30 นาที",
        "1h": "1 ชั่วโมง", "1d": "1 วัน", "1wk": "1 สัปดาห์", "1mo": "1 เดือน",
    }
    interval = st.selectbox(
        "ช่วงเวลาต่อแท่ง",
        list(interval_labels.keys()),
        index=5,
        format_func=lambda x: interval_labels[x],
        label_visibility="collapsed",
        help="1 นาที ใช้ได้กับข้อมูลย้อนหลังไม่เกิน 7 วัน",
    )

    st.markdown('<div class="section-h">ตัวเลือก</div>', unsafe_allow_html=True)
    translate_news = st.checkbox("🌐 แปลข่าวเป็นภาษาไทย", value=True)

    st.write("")
    if st.button("🔄 อัพเดตข้อมูล", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("ข้อมูลจาก Yahoo Finance · หน่วง ~1 นาที · แคช 60 วิ")

if not ticker:
    st.info("ใส่รหัสหุ้นทางซ้าย")
    st.stop()

with st.spinner(f"กำลังโหลด {ticker}…"):
    df = load_history(ticker, period, interval)
    info = load_info(ticker)
    news = load_news(ticker)

if df.empty:
    st.error(f"ไม่พบข้อมูลสำหรับ `{ticker}` — เช็ค ticker หรือเลือก interval/period ที่ Yahoo รองรับ")
    st.stop()

# Header
name = info.get("longName") or info.get("shortName") or ticker
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close
change = last_close - prev_close
pct = change / prev_close * 100 if prev_close else 0
currency = info.get("currency", "USD")

if change > 0:
    chip_cls, arrow = "up", "▲"
elif change < 0:
    chip_cls, arrow = "down", "▼"
else:
    chip_cls, arrow = "flat", "•"

meta_parts = [x for x in [info.get("exchange"), info.get("sector"), info.get("industry")] if x]
meta_line = " · ".join(meta_parts) if meta_parts else "—"

hero_left, hero_right = st.columns([1.5, 1])
with hero_left:
    st.markdown(
        f"""
        <div class="hero">
            <div class="sym">{ticker}</div>
            <div class="name">{name}</div>
            <div class="meta">{meta_line}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        f"""
        <div class="hero" style="text-align:right;">
            <div class="sym">ราคาล่าสุด · {currency}</div>
            <div class="price">{last_close:,.2f}</div>
            <div class="chip {chip_cls}">{arrow} {change:+.2f} ({pct:+.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Stat tiles
def tile(label: str, value: str) -> str:
    return f'<div class="tile"><div class="label">{label}</div><div class="value">{value}</div></div>'

vol_str = f"{int(df['Volume'].iloc[-1]):,}" if "Volume" in df and not pd.isna(df["Volume"].iloc[-1]) else "—"
rng_str = f"{df['Low'].min():,.2f} – {df['High'].max():,.2f}"
mcap = info.get("marketCap")
if mcap:
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if mcap >= div:
            mcap_str = f"{mcap / div:.2f}{unit}"
            break
    else:
        mcap_str = f"{mcap:,.0f}"
else:
    mcap_str = "—"
w52_hi = info.get("fiftyTwoWeekHigh")
w52_lo = info.get("fiftyTwoWeekLow")
w52_str = f"{w52_lo:,.2f} – {w52_hi:,.2f}" if w52_hi and w52_lo else "—"

t1, t2, t3, t4 = st.columns(4)
t1.markdown(tile("ปริมาณซื้อขาย", vol_str), unsafe_allow_html=True)
t2.markdown(tile("ช่วงราคา (ย้อนหลัง)", rng_str), unsafe_allow_html=True)
t3.markdown(tile("มูลค่าตลาด", mcap_str), unsafe_allow_html=True)
t4.markdown(tile("ช่วงราคา 52 สัปดาห์", w52_str), unsafe_allow_html=True)
st.write("")

tab_chart, tab_stats, tab_news, tab_signal = st.tabs(
    ["📊 กราฟเทคนิค", "📋 สถิติ", "📰 ข่าว", "🎯 สัญญาณสรุป"]
)

# ---------- Chart (TradingView widget) ----------
with tab_chart:
    # Quick levels banner above chart — 3 rows: sell zones / current / buy zones
    _lv = compute_levels(df)
    _sup = _lv["support"]
    _res = _lv["resistance"]
    _cur = _lv["current"]
    _stop_lv = None
    if len(_sup) >= 2:
        _stop_lv = _sup[1] * 0.98
    elif len(_sup) == 1:
        _stop_lv = _sup[0] * 0.96

    def _lv_pill(role: str, price: float | None, kind: str) -> str:
        if price is None:
            return ""
        delta = (price / _cur - 1) * 100
        delta_str = f"{delta:+.1f}%"
        return (
            f'<div class="lv-pill lv-{kind}">'
            f'<span class="lv-role">{role}</span>'
            f'<span class="lv-price">{price:,.2f}</span>'
            f'<span class="lv-delta">{delta_str}</span>'
            f'</div>'
        )

    # Sell zones row (above current price)
    sell_pills = ""
    for i, r in enumerate(_res, 1):
        sell_pills += _lv_pill(f"🔴 ขาย {i}", r, "sell")

    # Buy zones + SL row (below current price)
    buy_pills = ""
    for i, s in enumerate(_sup, 1):
        buy_pills += _lv_pill(f"🟢 ซื้อ {i}", s, "buy")
    buy_pills += _lv_pill("🛑 ตัดขาดทุน", _stop_lv, "stop")

    st.markdown(
        f"""
        <div class="lv-ladder">
            <div class="lv-row lv-row-up">
                <div class="lv-arrow">↑ ถ้าราคาขึ้นไปถึง</div>
                <div class="lv-pills">{sell_pills or '<span class="lv-empty">— ไม่มีแนวต้านในข้อมูลที่มี —</span>'}</div>
            </div>
            <div class="lv-now">
                <span class="lv-now-label">ราคาตอนนี้</span>
                <span class="lv-now-price">{_cur:,.2f}</span>
                <span class="lv-now-tag">● LIVE</span>
            </div>
            <div class="lv-row lv-row-down">
                <div class="lv-arrow">↓ ถ้าราคาลงมาถึง</div>
                <div class="lv-pills">{buy_pills or '<span class="lv-empty">— ไม่มีแนวรับในข้อมูลที่มี —</span>'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "📌 ทุกโซน = **เป้าหมายราคาที่ยังไม่ได้แตะ** · % คือระยะห่างจากราคาปัจจุบัน  ·  "
        "💡 อยากเห็นเส้นในกราฟ คลิกเครื่องมือ **Horizontal Line** ทางซ้ายของ TradingView → ป้อนราคา"
    )

    tv_symbol = to_tv_symbol(ticker)
    tv_interval = to_tv_interval(interval)
    tv_html = f"""
    <div class="tradingview-widget-container" style="height:720px;width:100%">
      <div id="tv_chart_container" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
          "autosize": true,
          "symbol": "{tv_symbol}",
          "interval": "{tv_interval}",
          "timezone": "Asia/Bangkok",
          "theme": "light",
          "style": "1",
          "locale": "th_TH",
          "toolbar_bg": "#ffffff",
          "enable_publishing": false,
          "withdateranges": true,
          "hide_side_toolbar": false,
          "allow_symbol_change": true,
          "save_image": true,
          "details": false,
          "calendar": false,
          "studies": [
              "MASimple@tv-basicstudies",
              "Volume@tv-basicstudies"
          ],
          "container_id": "tv_chart_container"
      }});
      </script>
    </div>
    """
    components.html(tv_html, height=740)
    st.caption(
        f"กราฟจาก TradingView · symbol = `{tv_symbol}` · "
        "เพิ่ม indicators (RSI / MACD / Bollinger / Fibonacci) ได้ที่ไอคอน fx ด้านบน · "
        "วาดเส้นแนวรับ/แนวต้านเองได้จากเครื่องมือซ้ายมือ"
    )

# ---------- Stats ----------
with tab_stats:
    def fmt(x, money=False, pct=False):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        if pct:
            return f"{x * 100:.2f}%" if abs(x) < 1 else f"{x:.2f}%"
        if money:
            for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
                if abs(x) >= div:
                    return f"{x / div:.2f}{unit}"
            return f"{x:,.2f}"
        return f"{x:,.2f}" if isinstance(x, (int, float)) else str(x)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("ราคาและขนาด")
        st.write(f"**มูลค่าตลาด:** {fmt(info.get('marketCap'), money=True)}")
        st.write(f"**สูงสุด 52 สัปดาห์:** {fmt(info.get('fiftyTwoWeekHigh'))}")
        st.write(f"**ต่ำสุด 52 สัปดาห์:** {fmt(info.get('fiftyTwoWeekLow'))}")
        st.write(f"**ปริมาณเฉลี่ย:** {fmt(info.get('averageVolume'), money=True)}")
        st.write(f"**ค่า Beta (ความผันผวน):** {fmt(info.get('beta'))}")
    with col2:
        st.subheader("มูลค่าหุ้น (Valuation)")
        st.write(f"**P/E (ย้อนหลัง):** {fmt(info.get('trailingPE'))}")
        st.write(f"**P/E (คาดการณ์):** {fmt(info.get('forwardPE'))}")
        st.write(f"**PEG (P/E ต่อการเติบโต):** {fmt(info.get('pegRatio'))}")
        st.write(f"**P/B (ราคาต่อมูลค่าทางบัญชี):** {fmt(info.get('priceToBook'))}")
        st.write(f"**EPS (กำไรต่อหุ้น 12 เดือนล่าสุด):** {fmt(info.get('trailingEps'))}")
    with col3:
        st.subheader("ปันผล & ความสามารถทำกำไร")
        st.write(f"**อัตราปันผล:** {fmt(info.get('dividendYield'), pct=True)}")
        st.write(f"**สัดส่วนการจ่ายปันผล:** {fmt(info.get('payoutRatio'), pct=True)}")
        st.write(f"**อัตรากำไรสุทธิ:** {fmt(info.get('profitMargins'), pct=True)}")
        st.write(f"**ROE (ผลตอบแทนผู้ถือหุ้น):** {fmt(info.get('returnOnEquity'), pct=True)}")
        st.write(f"**การเติบโตรายได้:** {fmt(info.get('revenueGrowth'), pct=True)}")

    # CEO + Key Officers section
    officers = info.get("companyOfficers", []) or []
    if officers:
        st.write("")
        st.subheader("👔 ผู้บริหาร")
        ceo = next(
            (o for o in officers if "CEO" in (o.get("title") or "").upper()
             or "CHIEF EXECUTIVE" in (o.get("title") or "").upper()),
            officers[0],
        )
        cur_year = datetime.now().year
        ceo_age = ceo.get("age") or (cur_year - ceo["yearBorn"] if ceo.get("yearBorn") else None)
        ceo_pay = ceo.get("totalPay")
        col_ceo, col_others = st.columns([1.2, 1])
        with col_ceo:
            st.markdown(f"**{ceo.get('name', '—')}**")
            st.caption(ceo.get("title", "—"))
            ceo_bits = []
            if ceo_age:
                ceo_bits.append(f"อายุ {ceo_age} ปี")
            if ceo_pay:
                ceo_bits.append(f"ค่าตอบแทน ${ceo_pay:,.0f}/ปี")
            if ceo_bits:
                st.write(" · ".join(ceo_bits))
        with col_others:
            others = [o for o in officers if o is not ceo][:4]
            if others:
                st.caption("**ผู้บริหารคนอื่น**")
                for o in others:
                    name = o.get("name", "—")
                    title = o.get("title", "—")
                    st.markdown(f"• {name} — _{title}_")

    desc = info.get("longBusinessSummary")
    if desc:
        st.write("")
        with st.expander("📄 เกี่ยวกับบริษัท"):
            st.write(desc)

# ---------- News ----------
with tab_news:
    st.caption("ข่าวจาก Yahoo Finance (รวมข่าวบริษัท, ข่าววิเคราะห์, และข่าวตลาดที่เกี่ยวข้อง)")
    if not news:
        st.info("ไม่พบข่าวล่าสุดสำหรับหุ้นตัวนี้")

    news_list = news[:25]
    # Sentiment (from English original — VADER works on English)
    for n in news_list:
        text = (n.get("title") or "") + ". " + (n.get("summary") or "")
        n["sentiment"] = sentiment_score(text)

    if news_list:
        avg_sent = sum(n["sentiment"] for n in news_list) / len(news_list)
        pos = sum(1 for n in news_list if n["sentiment"] >= 0.1)
        neg = sum(1 for n in news_list if n["sentiment"] <= -0.1)
        neu = len(news_list) - pos - neg
        label, cls = sentiment_label(avg_sent)
        st.markdown(
            f"""
            <div class="sent-overall">
                <div class="sent-overall-label">📊 ภาพรวมข่าวล่าสุด {len(news_list)} ข่าว</div>
                <div class="sent-overall-score">
                    <span class="mini-chip {cls}">{label}</span>
                    <span class="sent-counts">🟢 {pos} · ⚪ {neu} · 🔴 {neg}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if translate_news and news_list:
        with st.spinner("กำลังแปลข่าวเป็นภาษาไทย…"):
            for n in news_list:
                n["title_th"] = translate_th(n["title"])
                if n.get("summary"):
                    summary = n["summary"][:600]
                    n["summary_th"] = translate_th(summary)

    import html as _html
    for n in news_list:
        when_str = n["when"].astimezone().strftime("%Y-%m-%d %H:%M") if n["when"] else "—"
        pub = n.get("publisher") or "—"
        title = n["title"]
        link = n.get("link")
        title_th = n.get("title_th")

        display_title = title_th if (translate_news and title_th) else title
        orig_line = (
            f'<div class="orig">🇬🇧 {_html.escape(title)}</div>'
            if (translate_news and title_th and title_th != title)
            else ""
        )

        title_html = (
            f'<a href="{_html.escape(link)}" target="_blank">{_html.escape(display_title)}</a>'
            if link else _html.escape(display_title)
        )

        summary = n.get("summary") or ""
        summary_th = n.get("summary_th")
        shown_summary = ""
        if translate_news and summary_th:
            shown_summary = summary_th[:500] + ("…" if len(summary_th) > 500 else "")
        elif summary:
            shown_summary = summary[:400] + ("…" if len(summary) > 400 else "")
        summary_html = (
            f'<div class="summary">{_html.escape(shown_summary)}</div>' if shown_summary else ""
        )

        sent_label, sent_cls = sentiment_label(n.get("sentiment", 0))
        sent_chip = f'<span class="mini-chip {sent_cls}" style="font-size:0.7rem;padding:0.15rem 0.5rem;">{sent_label}</span>'

        st.markdown(
            f"""
            <div class="news-card">
                <div class="title">{title_html}</div>
                {orig_line}
                <div class="meta">📰 {_html.escape(pub)} · 🕐 {when_str} · {sent_chip}</div>
                {summary_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- Signal ----------
with tab_signal:
    verdict, reasons = build_signal(df)
    if "Bullish" in verdict:
        vcls = "bull"
    elif "Bearish" in verdict:
        vcls = "bear"
    else:
        vcls = "flat"
    st.markdown(f'<div class="verdict {vcls}">{verdict}</div>', unsafe_allow_html=True)

    # --- Entry / Exit levels ---
    levels = compute_levels(df)
    current = levels["current"]
    support = levels["support"]
    resistance = levels["resistance"]

    buy1 = support[0] if len(support) >= 1 else None
    buy2 = support[1] if len(support) >= 2 else None
    sell1 = resistance[0] if len(resistance) >= 1 else None
    sell2 = resistance[1] if len(resistance) >= 2 else None
    sell3 = resistance[2] if len(resistance) >= 3 else None
    stop_loss = buy2 * 0.98 if buy2 else (buy1 * 0.96 if buy1 else None)

    def level_tile(cls: str, role: str, price: float | None, sub: str = "") -> str:
        if price is None:
            return (
                f'<div class="level-tile {cls}"><div class="role">{role}</div>'
                f'<div class="price">—</div><div class="delta">ข้อมูลไม่พอ</div></div>'
            )
        delta_pct = (price / current - 1) * 100
        delta_str = f"{delta_pct:+.2f}% · {sub}" if sub else f"{delta_pct:+.2f}% จากราคาปัจจุบัน"
        return (
            f'<div class="level-tile {cls}"><div class="role">{role}</div>'
            f'<div class="price">{price:,.2f}</div><div class="delta">{delta_str}</div></div>'
        )

    st.markdown('<div class="section-h">🟢 แนวรับ — โซนซื้อ</div>', unsafe_allow_html=True)
    r1 = st.columns(4)
    r1[0].markdown(level_tile("current", "ราคาปัจจุบัน", current, "ตอนนี้"), unsafe_allow_html=True)
    r1[1].markdown(level_tile("entry", "🟢 โซนซื้อ 1", buy1, "แนวรับ S1 · ใกล้สุด"), unsafe_allow_html=True)
    r1[2].markdown(level_tile("entry", "🟢 โซนซื้อ 2", buy2, "แนวรับ S2 · ถัดลงไป"), unsafe_allow_html=True)
    r1[3].markdown(level_tile("stop", "🛑 ตัดขาดทุน", stop_loss, "~2% ใต้ S2" if buy2 else "~4% ใต้ S1"), unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-h">🔴 แนวต้าน — โซนขายทำกำไร</div>', unsafe_allow_html=True)
    r2 = st.columns(3)
    r2[0].markdown(level_tile("resistance", "🎯 โซนขาย 1", sell1, "แนวต้าน R1 · ใกล้สุด"), unsafe_allow_html=True)
    r2[1].markdown(level_tile("resistance", "🎯 โซนขาย 2", sell2, "แนวต้าน R2"), unsafe_allow_html=True)
    r2[2].markdown(level_tile("resistance", "🎯 โซนขาย 3", sell3, "แนวต้าน R3 · ไกลสุด"), unsafe_allow_html=True)

    # Risk/Reward
    if buy1 and stop_loss and sell1:
        risk = current - stop_loss
        reward = sell1 - current
        if risk > 0:
            rr = reward / risk
            rr_color = "#166534" if rr >= 2 else ("#a16207" if rr >= 1 else "#991b1b")
            st.markdown(
                f'<div style="margin-top:0.75rem;">'
                f'<span class="rr-badge">ความคุ้มค่า (ซื้อตอนนี้ → จุดขาย 1) · '
                f'<b style="color:{rr_color};">{rr:.2f} : 1</b></span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- All levels table ---
    if support or resistance:
        with st.expander("ดูแนวรับ/แนวต้านทั้งหมด"):
            rows = []
            for i, r in enumerate(resistance, 1):
                rows.append({"ประเภท": f"🔴 แนวต้าน R{i}", "ราคา": f"{r:,.2f}", "ห่างจากปัจจุบัน": f"{(r/current-1)*100:+.2f}%"})
            rows.append({"ประเภท": "⚪ ราคาปัจจุบัน", "ราคา": f"{current:,.2f}", "ห่างจากปัจจุบัน": "0.00%"})
            for i, s in enumerate(support, 1):
                rows.append({"ประเภท": f"🟢 แนวรับ S{i}", "ราคา": f"{s:,.2f}", "ห่างจากปัจจุบัน": f"{(s/current-1)*100:+.2f}%"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.write("")
    st.markdown('<div class="section-h">เหตุผลสัญญาณ</div>', unsafe_allow_html=True)
    reasons_html = "".join(
        f'<div class="tile" style="margin-bottom:0.5rem;"><div class="value" style="font-size:0.95rem;font-weight:500;">{r}</div></div>'
        for r in reasons
    )
    st.markdown(reasons_html, unsafe_allow_html=True)

    st.write("")
    st.warning(
        "⚠️ นี่เป็นสัญญาณ rule-based จาก indicator เท่านั้น ไม่ใช่คำแนะนำการลงทุน "
        "ควรพิจารณาปัจจัยพื้นฐาน ข่าว และการจัดการความเสี่ยงของตนเองประกอบ"
    )

