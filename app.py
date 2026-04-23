"""Real-time stock dashboard — technical analysis + news."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from deep_translator import GoogleTranslator
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Stock Dashboard",
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
    """Fetch 3-month daily data for a batch of tickers and compute summary stats."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            list(tickers), period="3mo", interval="1d",
            auto_adjust=False, progress=False, threads=True, group_by="ticker",
        )
    except Exception:
        return {}
    out: dict[str, dict] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            close = raw[t]["Close"].dropna() if multi else raw["Close"].dropna()
            if len(close) < 3:
                continue
            last = float(close.iloc[-1])
            def chg(days: int) -> float:
                if len(close) <= days:
                    return 0.0
                return (last / float(close.iloc[-days - 1]) - 1) * 100
            w1 = chg(5)
            m1 = chg(21)
            m3 = (last / float(close.iloc[0]) - 1) * 100
            r = float(rsi(close).iloc[-1])
            s20 = float(sma(close, 20).iloc[-1])
            s50 = float(sma(close, 50).iloc[-1]) if len(close) >= 30 else float("nan")
            if last > s20 and (np.isnan(s50) or last > s50) and r < 70:
                sig = "bull"
            elif last < s20 and (np.isnan(s50) or last < s50):
                sig = "bear"
            else:
                sig = "flat"
            out[t] = {"last": last, "w1": w1, "m1": m1, "m3": m3, "rsi": r, "sig": sig,
                      "below_sma20": last < s20, "below_sma50": not np.isnan(s50) and last < s50}
        except Exception:
            continue
    return out


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
        "desc": "Megatrend leaders · สแกนสดจาก universe 25 ตัว · เรียงตาม performance 3 เดือน",
        "mode": "auto_future",
        "universe": [
            "NVDA", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "TSM", "ASML",
            "AVGO", "AMD", "QCOM", "LLY", "NVO", "UNH", "V", "MA",
            "NOW", "CRM", "ADBE", "ORCL", "PANW", "NFLX", "ARM", "COST", "JPM",
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
    "🔥 Mega Cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B"],
    "💻 Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO"],
    "💰 Finance": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BLK", "C", "AXP"],
    "🏥 Healthcare": ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "⚡ Energy": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG", "PSX", "MPC"],
    "🛒 Consumer": ["AMZN", "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT"],
    "🚗 Auto / EV": ["TSLA", "F", "GM", "RIVN", "LCID", "TM", "HMC", "STLA", "BYDDY"],
    "🎮 Media / Gaming": ["NFLX", "DIS", "SONY", "EA", "TTWO", "RBLX", "SPOT", "ROKU"],
    "✈️ Airlines / Travel": ["DAL", "UAL", "AAL", "LUV", "BA", "BKNG", "ABNB", "MAR"],
    "🪙 Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD"],
    "📊 Indices": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI"],
    "🌐 ETFs": ["SPY", "QQQ", "VOO", "VTI", "IWM", "DIA", "ARKK", "GLD", "TLT"],
    "🇨🇳 China ADR": ["BABA", "JD", "PDD", "NIO", "LI", "XPEV", "BIDU", "TME"],
    "🏦 Thai ADR / SET": ["PTT.BK", "ADVANC.BK", "AOT.BK", "CPALL.BK", "KBANK.BK", "SCB.BK", "PTTEP.BK", "DELTA.BK"],
}

if "ticker" not in st.session_state:
    st.session_state.ticker = "AAPL"
if "category" not in st.session_state:
    st.session_state.category = "🔥 Mega Cap"

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="logo">📈</div>
            <div>
                <div class="name">Stockscope</div>
                <div class="sub">Real-time · Global</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-h">Ticker</div>', unsafe_allow_html=True)
    ticker_input = st.text_input(
        "Ticker",
        value=st.session_state.ticker,
        placeholder="AAPL, TSLA, BTC-USD, ^GSPC …",
        label_visibility="collapsed",
    )
    ticker = ticker_input.upper().strip()
    st.session_state.ticker = ticker

    st.markdown('<div class="section-h">Category</div>', unsafe_allow_html=True)

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
                            key=lambda t: mini[t]["w1"], reverse=True)[:8]
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
            candidates = [
                (t, d) for t, d in mini.items()
                if d["rsi"] < 75 and not (d["below_sma20"] and d["below_sma50"])
            ]
            candidates.sort(key=lambda x: x[1]["m1"], reverse=True)
            tickers_in_cat = [
                (t, f"1M {d['m1']:+.1f}% · RSI {d['rsi']:.0f}", None)
                for t, d in candidates[:8]
            ]
        elif mode == "auto_future":
            candidates = sorted(mini.items(), key=lambda x: x[1]["m3"], reverse=True)
            tickers_in_cat = [
                (t, f"3M {d['m3']:+.1f}% · 1M {d['m1']:+.1f}%", None)
                for t, d in candidates[:8]
            ]
        elif mode == "auto_options":
            def _bias(sig: str) -> str:
                return {"bull": "📈 Call bias", "bear": "📉 Put bias", "flat": "⚪ Neutral"}[sig]
            candidates = sorted(mini.items(), key=lambda x: abs(x[1]["w1"]), reverse=True)
            tickers_in_cat = [
                (t, f"σ {abs(d['w1']):.1f}% · {_bias(d['sig'])}", None)
                for t, d in candidates[:8]
            ]
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

    st.markdown('<div class="section-h">Timeframe</div>', unsafe_allow_html=True)
    period = st.selectbox(
        "Period", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
        index=4, label_visibility="collapsed",
    )
    interval = st.selectbox(
        "Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"],
        index=5, label_visibility="collapsed",
        help="1m ใช้ได้กับช่วง ≤7 วัน",
    )

    st.markdown('<div class="section-h">Indicators</div>', unsafe_allow_html=True)
    show_sma = st.checkbox("SMA 20 / 50 / 200", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=False)
    show_volume = st.checkbox("Volume", value=True)
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12, 26, 9)", value=True)

    st.markdown('<div class="section-h">Options</div>', unsafe_allow_html=True)
    translate_news = st.checkbox("🌐 แปลข่าวเป็นภาษาไทย", value=True)

    st.write("")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Yahoo Finance · delay ~1 นาที · cache 60s")

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
            <div class="sym">Last price · {currency}</div>
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
t1.markdown(tile("Volume", vol_str), unsafe_allow_html=True)
t2.markdown(tile("Range (period)", rng_str), unsafe_allow_html=True)
t3.markdown(tile("Market Cap", mcap_str), unsafe_allow_html=True)
t4.markdown(tile("52-Week Range", w52_str), unsafe_allow_html=True)
st.write("")

tab_chart, tab_stats, tab_news, tab_signal = st.tabs(
    ["📊 กราฟเทคนิค", "📋 สถิติ", "📰 ข่าว", "🎯 สัญญาณสรุป"]
)

# ---------- Chart ----------
with tab_chart:
    rows = 1 + int(show_volume) + int(show_rsi) + int(show_macd)
    heights = [0.55] + [0.15] * (rows - 1) if rows > 1 else [1.0]
    # normalize
    heights = [h / sum(heights) for h in heights]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#16a34a",
            decreasing_line_color="#dc2626",
            increasing_fillcolor="#16a34a",
            decreasing_fillcolor="#dc2626",
        ),
        row=1,
        col=1,
    )

    if show_sma:
        for w, color in [(20, "#d97706"), (50, "#2563eb"), (200, "#71717a")]:
            if len(df) >= 2:
                fig.add_trace(
                    go.Scatter(x=df.index, y=sma(df["Close"], w), name=f"SMA{w}", line=dict(width=1, color=color)),
                    row=1,
                    col=1,
                )

    if show_bb:
        upper, mid, lower = bollinger(df["Close"])
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="BB Upper", line=dict(width=1, color="rgba(113,113,122,0.5)")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="BB Lower", line=dict(width=1, color="rgba(113,113,122,0.5)"), fill="tonexty", fillcolor="rgba(113,113,122,0.06)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=mid, name="BB Mid", line=dict(width=1, dash="dot", color="rgba(113,113,122,0.7)")), row=1, col=1)

    levels = compute_levels(df)
    _support = levels["support"]
    _resistance = levels["resistance"]
    _current = levels["current"]

    # Resistance lines (Sell zones) — red, solid/dashed/dotted by proximity
    _res_styles = [
        ("solid",  2.0, "🔴 ขาย 1 (R1)"),
        ("dash",   1.6, "🔴 ขาย 2 (R2)"),
        ("dot",    1.3, "🔴 ขาย 3 (R3)"),
    ]
    for i, r in enumerate(_resistance):
        if i >= len(_res_styles):
            break
        style, width, label = _res_styles[i]
        fig.add_hline(
            y=r, line_dash=style, line_color="#dc2626", line_width=width, opacity=0.95,
            annotation_text=f" {label} · {r:,.2f} ",
            annotation_position="top left",
            annotation_font=dict(color="#ffffff", size=11, family="Inter"),
            annotation_bgcolor="#dc2626",
            annotation_bordercolor="#dc2626",
            row=1, col=1,
        )

    # Support lines (Buy zones) — green, solid/dashed/dotted
    _sup_styles = [
        ("solid",  2.0, "🟢 ซื้อ 1 (S1)"),
        ("dash",   1.6, "🟢 ซื้อ 2 (S2)"),
        ("dot",    1.3, "🟢 ซื้อ 3 (S3)"),
    ]
    for i, s in enumerate(_support):
        if i >= len(_sup_styles):
            break
        style, width, label = _sup_styles[i]
        fig.add_hline(
            y=s, line_dash=style, line_color="#15803d", line_width=width, opacity=0.95,
            annotation_text=f" {label} · {s:,.2f} ",
            annotation_position="bottom left",
            annotation_font=dict(color="#ffffff", size=11, family="Inter"),
            annotation_bgcolor="#15803d",
            annotation_bordercolor="#15803d",
            row=1, col=1,
        )

    # Stop Loss line — orange, dashed
    _stop = None
    if len(_support) >= 2:
        _stop = _support[1] * 0.98
    elif len(_support) == 1:
        _stop = _support[0] * 0.96
    if _stop is not None:
        fig.add_hline(
            y=_stop, line_dash="longdashdot", line_color="#ea580c", line_width=1.8, opacity=0.95,
            annotation_text=f" 🛑 Stop Loss · {_stop:,.2f} ",
            annotation_position="bottom left",
            annotation_font=dict(color="#ffffff", size=11, family="Inter"),
            annotation_bgcolor="#ea580c",
            annotation_bordercolor="#ea580c",
            row=1, col=1,
        )

    # Current price line — subtle gray
    fig.add_hline(
        y=_current, line_dash="dot", line_color="#52525b", line_width=1, opacity=0.6,
        annotation_text=f" ราคาปัจจุบัน · {_current:,.2f} ",
        annotation_position="top right",
        annotation_font=dict(color="#ffffff", size=11, family="Inter"),
        annotation_bgcolor="#52525b",
        annotation_bordercolor="#52525b",
        row=1, col=1,
    )

    r = 2
    if show_volume and "Volume" in df:
        colors = ["#16a34a" if c >= o else "#dc2626" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, opacity=0.6), row=r, col=1)
        fig.update_yaxes(title_text="Vol", row=r, col=1)
        r += 1

    if show_rsi:
        r_series = rsi(df["Close"])
        fig.add_trace(go.Scatter(x=df.index, y=r_series, name="RSI", line=dict(color="#6366f1", width=1.3)), row=r, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", line_width=1, row=r, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", line_width=1, row=r, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=r, col=1)
        r += 1

    if show_macd:
        macd_line, signal_line, hist = macd(df["Close"])
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD", line=dict(color="#2563eb", width=1.3)), row=r, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, name="Signal", line=dict(color="#d97706", width=1.3)), row=r, col=1)
        hist_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in hist]
        fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist", marker_color=hist_colors, opacity=0.7), row=r, col=1)
        fig.update_yaxes(title_text="MACD", row=r, col=1)

    fig.update_layout(
        height=720,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white",
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#475569", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0)", font=dict(size=11, color="#475569"),
        ),
    )
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

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
        st.write(f"**Market Cap:** {fmt(info.get('marketCap'), money=True)}")
        st.write(f"**52W High:** {fmt(info.get('fiftyTwoWeekHigh'))}")
        st.write(f"**52W Low:** {fmt(info.get('fiftyTwoWeekLow'))}")
        st.write(f"**Avg Volume:** {fmt(info.get('averageVolume'), money=True)}")
        st.write(f"**Beta:** {fmt(info.get('beta'))}")
    with col2:
        st.subheader("Valuation")
        st.write(f"**P/E (trailing):** {fmt(info.get('trailingPE'))}")
        st.write(f"**P/E (forward):** {fmt(info.get('forwardPE'))}")
        st.write(f"**PEG:** {fmt(info.get('pegRatio'))}")
        st.write(f"**P/B:** {fmt(info.get('priceToBook'))}")
        st.write(f"**EPS (TTM):** {fmt(info.get('trailingEps'))}")
    with col3:
        st.subheader("ปันผล & กำไร")
        st.write(f"**Dividend Yield:** {fmt(info.get('dividendYield'), pct=True)}")
        st.write(f"**Payout Ratio:** {fmt(info.get('payoutRatio'), pct=True)}")
        st.write(f"**Profit Margin:** {fmt(info.get('profitMargins'), pct=True)}")
        st.write(f"**ROE:** {fmt(info.get('returnOnEquity'), pct=True)}")
        st.write(f"**Revenue Growth:** {fmt(info.get('revenueGrowth'), pct=True)}")

    desc = info.get("longBusinessSummary")
    if desc:
        with st.expander("เกี่ยวกับบริษัท"):
            st.write(desc)

# ---------- News ----------
with tab_news:
    st.caption("ข่าวจาก Yahoo Finance (รวมข่าวบริษัท, ข่าววิเคราะห์, และข่าวตลาดที่เกี่ยวข้อง)")
    if not news:
        st.info("ไม่พบข่าวล่าสุดสำหรับหุ้นตัวนี้")

    news_list = news[:25]
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

        st.markdown(
            f"""
            <div class="news-card">
                <div class="title">{title_html}</div>
                {orig_line}
                <div class="meta">📰 {_html.escape(pub)} · 🕐 {when_str}</div>
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
    r1[1].markdown(level_tile("entry", "🟢 Buy Zone 1", buy1, "แนวรับ S1 · ใกล้สุด"), unsafe_allow_html=True)
    r1[2].markdown(level_tile("entry", "🟢 Buy Zone 2", buy2, "แนวรับ S2 · ถัดลงไป"), unsafe_allow_html=True)
    r1[3].markdown(level_tile("stop", "🛑 Stop Loss", stop_loss, "~2% ใต้ S2" if buy2 else "~4% ใต้ S1"), unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-h">🔴 แนวต้าน — โซนขาย / Take Profit</div>', unsafe_allow_html=True)
    r2 = st.columns(3)
    r2[0].markdown(level_tile("resistance", "🎯 Sell Zone 1", sell1, "แนวต้าน R1 · ใกล้สุด"), unsafe_allow_html=True)
    r2[1].markdown(level_tile("resistance", "🎯 Sell Zone 2", sell2, "แนวต้าน R2"), unsafe_allow_html=True)
    r2[2].markdown(level_tile("resistance", "🎯 Sell Zone 3", sell3, "แนวต้าน R3 · ไกลสุด"), unsafe_allow_html=True)

    # Risk/Reward
    if buy1 and stop_loss and sell1:
        risk = current - stop_loss
        reward = sell1 - current
        if risk > 0:
            rr = reward / risk
            rr_color = "#166534" if rr >= 2 else ("#a16207" if rr >= 1 else "#991b1b")
            st.markdown(
                f'<div style="margin-top:0.75rem;">'
                f'<span class="rr-badge">Risk/Reward (ซื้อราคานี้ → Sell 1) · '
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

