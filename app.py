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


@st.cache_data(ttl=3600, show_spinner=False)
def ai_summarize_news(api_key: str, ticker: str, news_signature: str, news_payload: str) -> str:
    """Cache key includes news signature so we re-summarize when news changes."""
    if not api_key or not news_payload:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": (
                    f"ฉันมีข่าวเกี่ยวกับหุ้น {ticker} ดังนี้:\n\n{news_payload}\n\n"
                    "ช่วยวิเคราะห์เป็นภาษาไทยให้ครบทุกข้อ:\n"
                    "1. **สรุปภาพรวม** (2-3 ประโยค)\n"
                    "2. **ประเด็นสำคัญ 3-5 ข้อ** (bullet)\n"
                    "3. **Sentiment โดยรวม** (บวก/ลบ/กลาง) พร้อมเหตุผลสั้นๆ\n"
                    "4. **สิ่งที่ควรจับตา** (บอกว่าควรตามดูอะไรต่อ)\n\n"
                    "ใช้ภาษาไทยที่อ่านง่าย ตรงไปตรงมา ไม่ต้องเกริ่นนำ"
                ),
            }],
        )
        return msg.content[0].text
    except Exception as e:
        return f"❌ Error: {str(e)[:200]}"


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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Playfair+Display:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(1200px 600px at 80% -5%, rgba(212,175,122,0.07), transparent 65%),
            radial-gradient(800px 400px at -5% 100%, rgba(212,175,122,0.04), transparent 55%),
            #0c0d10;
    }

    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem !important; max-width: 1380px; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #131419 0%, #0e0f13 100%);
        border-right: 1px solid rgba(212,175,122,0.08);
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .hero, .tile, .pos52, .news-card, .level-tile, .verdict,
    .macro-tile, .sector-cell, .mover-cell, .sent-overall, .lv-ladder {
        animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Global typography — warm cream on near-black */
    body, .stMarkdown, p, label, span, div { color: #ebe4d8; }
    h1, h2, h3, h4, h5, h6, .stMarkdown h2, .stMarkdown h3 {
        color: #f5efe1;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }
    .stCaption, [data-testid="stCaptionContainer"] { color: #8a8275 !important; }

    /* Hairline divider lines */
    hr, [data-testid="stMarkdownContainer"] hr {
        border-color: rgba(212,175,122,0.15) !important;
        opacity: 1;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid rgba(15,23,42,0.06);
    }
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.25rem;
    }

    /* Hero card — editorial dark with warm champagne accent */
    .hero {
        background:
            linear-gradient(135deg, rgba(212,175,122,0.04) 0%, transparent 60%),
            linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.15);
        border-radius: 4px;
        padding: 2rem 2.25rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.4s;
    }
    .hero::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(212,175,122,0.6) 50%, transparent 100%);
    }
    .hero:hover { border-color: rgba(212,175,122,0.35); }
    .hero .sym {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #d4af7a;
        letter-spacing: 0.3em;
        text-transform: uppercase;
        font-weight: 500;
    }
    .hero .name {
        font-family: 'Playfair Display', serif !important;
        font-size: 2.4rem;
        font-weight: 600;
        color: #f5efe1;
        margin: 0.4rem 0 0.4rem 0;
        letter-spacing: -0.02em;
        line-height: 1.05;
    }
    .hero .meta {
        color: #8a8275;
        font-size: 0.78rem;
        font-weight: 400;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .hero .price {
        font-family: 'Playfair Display', serif !important;
        font-size: 4rem;
        font-weight: 600;
        color: #f5efe1;
        letter-spacing: -0.04em;
        line-height: 1;
        font-variant-numeric: tabular-nums;
    }
    .hero .hero-thb {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8a8275;
        font-weight: 400;
        margin-top: 0.6rem;
        letter-spacing: 0.04em;
    }
    .hero .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.9rem;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.02em;
    }
    .chip.up    {
        background: rgba(132,204,22,0.08); color: #a3e635;
        border: 1px solid rgba(132,204,22,0.35);
    }
    .chip.down  {
        background: rgba(220,38,38,0.08); color: #f87171;
        border: 1px solid rgba(220,38,38,0.35);
    }
    .chip.flat  {
        background: rgba(212,175,122,0.06); color: #d4af7a;
        border: 1px solid rgba(212,175,122,0.25);
    }

    /* Market session badge + extended hours */
    .session-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        padding: 0.2rem 0.6rem;
        border-radius: 2px;
        margin-top: 0.6rem;
        margin-left: 0.4rem;
        border: 1px solid transparent;
    }
    .session-open   { background: rgba(132,204,22,0.06); color: #a3e635; border-color: rgba(132,204,22,0.3); }
    .session-pre    { background: rgba(212,175,122,0.06); color: #d4af7a; border-color: rgba(212,175,122,0.3); }
    .session-post   { background: rgba(196,148,255,0.06); color: #c4b5fd; border-color: rgba(196,148,255,0.3); }
    .session-closed { background: rgba(138,130,117,0.06); color: #8a8275; border-color: rgba(138,130,117,0.3); }

    .ext-hours {
        margin-top: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px dashed rgba(212,175,122,0.18);
        text-align: right;
    }
    .ext-hours .ext-head {
        font-size: 0.68rem;
        font-weight: 500;
        color: #8a8275;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }
    .ext-hours .ext-row {
        display: flex;
        justify-content: flex-end;
        align-items: baseline;
        gap: 0.7rem;
    }
    .ext-hours .ext-price {
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem;
        font-weight: 600;
        color: #ebe4d8;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .ext-hours .ext-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.22rem 0.55rem;
        border-radius: 2px;
        border: 1px solid transparent;
        font-variant-numeric: tabular-nums;
    }
    .ext-hours .ext-chip.up   { background: rgba(132,204,22,0.06); color: #a3e635; border-color: rgba(132,204,22,0.3); }
    .ext-hours .ext-chip.down { background: rgba(220,38,38,0.06); color: #f87171; border-color: rgba(220,38,38,0.3); }

    /* Stat tiles — editorial cards */
    .tile {
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.1);
        border-radius: 4px;
        padding: 1.2rem 1.3rem;
        height: 100%;
        transition: all 0.3s;
        position: relative;
    }
    .tile:hover {
        border-color: rgba(212,175,122,0.3);
        transform: translateY(-2px);
    }
    .tile .label {
        color: #8a8275;
        font-size: 0.66rem;
        font-weight: 500;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .tile .value {
        color: #f5efe1;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.7rem;
        font-weight: 600;
        margin-top: 0.5rem;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }

    /* Tabs — minimalist underline style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border: none;
        border-bottom: 1px solid rgba(212,175,122,0.12);
        border-radius: 0;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 0.8rem 1.4rem;
        color: #8a8275;
        font-weight: 500;
        font-size: 0.92rem;
        letter-spacing: 0.04em;
        transition: all 0.25s;
        border-bottom: 2px solid transparent;
        margin-bottom: -1px;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #ebe4d8; }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #d4af7a !important;
        font-weight: 600;
        border-bottom: 2px solid #d4af7a !important;
    }

    /* News cards — editorial */
    .news-card {
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.08);
        border-radius: 4px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 0.7rem;
        transition: all 0.3s;
    }
    .news-card:hover {
        border-color: rgba(212,175,122,0.3);
        transform: translateY(-1px);
    }
    .news-card .title {
        font-family: 'Playfair Display', serif !important;
        color: #f5efe1;
        font-weight: 600;
        font-size: 1.1rem;
        line-height: 1.4;
        letter-spacing: -0.01em;
    }
    .news-card .title a { color: #f5efe1; text-decoration: none; }
    .news-card .title a:hover { color: #d4af7a; }
    .news-card .orig   { color: #6b6557; font-size: 0.78rem; margin-top: 0.3rem; font-style: italic; }
    .news-card .meta   { color: #8a8275; font-size: 0.74rem; margin-top: 0.5rem; letter-spacing: 0.04em; }
    .news-card .summary{ color: #c9c2b3; font-size: 0.88rem; margin-top: 0.65rem; line-height: 1.65; }

    /* Verdict card — editorial */
    .verdict {
        border-radius: 4px;
        padding: 1.8rem 2rem;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.65rem;
        font-weight: 600;
        text-align: center;
        letter-spacing: -0.015em;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.15);
        position: relative;
    }
    .verdict::before {
        content: '';
        position: absolute; top: 0; left: 50%; transform: translateX(-50%);
        width: 60%; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212,175,122,0.4), transparent);
    }
    .verdict.bull { color: #a3e635; border-color: rgba(132,204,22,0.3); }
    .verdict.bear { color: #f87171; border-color: rgba(220,38,38,0.3); }
    .verdict.flat { color: #d4af7a; border-color: rgba(212,175,122,0.3); }

    /* Buttons — minimalist editorial */
    .stButton > button {
        background: transparent;
        border: 1px solid rgba(212,175,122,0.2);
        color: #ebe4d8;
        font-weight: 500;
        font-size: 0.88rem;
        border-radius: 2px;
        letter-spacing: 0.04em;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        background: rgba(212,175,122,0.06);
        border-color: #d4af7a;
        color: #d4af7a;
    }

    /* Brand — refined editorial */
    .brand {
        display: flex; align-items: center; gap: 0.85rem;
        padding: 0.4rem 0 1.3rem 0;
        border-bottom: 1px solid rgba(212,175,122,0.12);
        margin-bottom: 1.1rem;
    }
    .brand .logo {
        width: 42px; height: 42px;
        border-radius: 2px;
        background: linear-gradient(135deg, #d4af7a 0%, #b8915e 100%);
        color: #15171c;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
    }
    .brand .name {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600;
        font-size: 1.35rem;
        color: #f5efe1;
        letter-spacing: -0.015em;
        line-height: 1;
    }
    .brand .sub  {
        font-size: 0.62rem;
        color: #d4af7a;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    /* Macro market bar (top of page) */
    .macro-bar {
        display: grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.85rem;
    }
    .macro-tile {
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.08);
        border-radius: 4px;
        padding: 0.75rem 0.95rem;
        transition: all 0.25s;
    }
    .macro-tile:hover {
        border-color: rgba(212,175,122,0.3);
        transform: translateY(-1px);
    }
    .macro-label {
        font-size: 0.62rem; font-weight: 500;
        color: #8a8275; letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .macro-row {
        display: flex; justify-content: space-between; align-items: baseline;
        gap: 0.5rem; margin-top: 0.35rem;
    }
    .macro-price {
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem; font-weight: 600;
        color: #f5efe1;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.01em;
    }
    .macro-chip {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 500;
        padding: 0.15rem 0.5rem; border-radius: 2px;
        font-variant-numeric: tabular-nums;
        border: 1px solid transparent;
    }
    .macro-chip.up   { background: rgba(132,204,22,0.06); color: #a3e635; border-color: rgba(132,204,22,0.25); }
    .macro-chip.down { background: rgba(220,38,38,0.06); color: #f87171; border-color: rgba(220,38,38,0.25); }

    @media (max-width: 768px) {
        .macro-bar { grid-template-columns: repeat(2, 1fr); gap: 0.4rem; }
        .macro-price { font-size: 0.88rem; }
        .macro-chip { font-size: 0.68rem; }
    }

    /* Top movers grid */
    .mover-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.45rem;
        margin-bottom: 0.55rem;
    }
    .mover-cell {
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.08);
        border-radius: 4px;
        padding: 0.85rem 1rem;
        transition: all 0.25s;
    }
    .mover-cell:hover {
        border-color: rgba(212,175,122,0.3);
        transform: translateY(-1px);
    }
    .mover-cell.mover-up   { border-left: 2px solid #a3e635; }
    .mover-cell.mover-down { border-left: 2px solid #f87171; }
    .mover-sym {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500; font-size: 0.85rem;
        color: #f5efe1;
        letter-spacing: 0.04em;
    }
    .mover-pct {
        font-family: 'Playfair Display', serif;
        font-weight: 600; font-size: 1.15rem;
        margin-top: 0.3rem;
        font-variant-numeric: tabular-nums;
    }
    .mover-up .mover-pct   { color: #a3e635; }
    .mover-down .mover-pct { color: #f87171; }
    .mover-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        color: #8a8275;
        margin-top: 0.15rem;
        font-variant-numeric: tabular-nums;
    }
    @media (max-width: 768px) {
        .mover-grid { grid-template-columns: repeat(2, 1fr); }
    }

    /* Sector heatmap grid */
    .sector-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .sector-cell {
        border: 1px solid;
        border-radius: 4px;
        padding: 0.95rem 1.1rem;
        transition: all 0.25s;
    }
    .sector-cell:hover { transform: translateY(-2px); }
    .sector-sym {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        font-size: 0.85rem;
        color: #f5efe1;
        letter-spacing: 0.05em;
    }
    .sector-name {
        font-size: 0.7rem;
        color: #8a8275;
        margin-top: 0.15rem;
    }
    .sector-pct {
        font-family: 'Playfair Display', serif;
        font-weight: 600;
        font-size: 1.35rem;
        margin-top: 0.4rem;
        letter-spacing: -0.01em;
        font-variant-numeric: tabular-nums;
    }
    .sector-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #8a8275;
        margin-top: 0.15rem;
        font-variant-numeric: tabular-nums;
    }
    @media (max-width: 768px) {
        .sector-grid { grid-template-columns: repeat(2, 1fr); }
    }

    /* 52-week position gauge */
    .pos52 {
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.1);
        border-radius: 4px;
        padding: 1.05rem 1.25rem;
        margin-bottom: 0.7rem;
    }
    .pos52-head {
        display: flex; justify-content: space-between;
        align-items: baseline; margin-bottom: 0.7rem;
    }
    .pos52-title {
        font-size: 0.7rem; font-weight: 500;
        color: #8a8275; letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .pos52-pct {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500; font-size: 0.88rem;
        color: #d4af7a;
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.04em;
    }
    .pos52-track {
        position: relative; height: 2px;
        background: linear-gradient(90deg, rgba(220,38,38,0.6) 0%, rgba(212,175,122,0.5) 50%, rgba(132,204,22,0.6) 100%);
        border-radius: 0;
        overflow: visible;
    }
    .pos52-marker {
        position: absolute; top: 50%; transform: translate(-50%, -50%);
        width: 14px; height: 14px; border-radius: 50%;
        background: #d4af7a;
        border: 2px solid #15171c;
        box-shadow: 0 0 0 1px #d4af7a;
    }
    .pos52-ends {
        display: flex; justify-content: space-between;
        margin-top: 0.7rem; font-size: 0.74rem; color: #8a8275;
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }

    /* Generic mini-chip */
    .mini-chip {
        display: inline-flex; align-items: center;
        padding: 0.2rem 0.6rem; border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem; font-weight: 500;
        border: 1px solid transparent;
        letter-spacing: 0.02em;
    }
    .mini-chip.up   { background: rgba(132,204,22,0.06); color: #a3e635; border-color: rgba(132,204,22,0.3); }
    .mini-chip.down { background: rgba(220,38,38,0.06); color: #f87171; border-color: rgba(220,38,38,0.3); }
    .mini-chip.flat { background: rgba(212,175,122,0.06); color: #d4af7a; border-color: rgba(212,175,122,0.25); }

    /* News overall sentiment */
    .sent-overall {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.1);
        border-radius: 4px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .sent-overall-label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #c9c2b3;
        letter-spacing: 0.04em;
    }
    .sent-overall-score {
        display: flex; align-items: center; gap: 0.6rem;
    }
    .sent-counts {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8a8275;
    }

    /* Level tile — editorial */
    .level-tile {
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.1);
        border-radius: 4px;
        padding: 1.1rem 1.2rem;
        height: 100%;
        transition: all 0.25s;
    }
    .level-tile:hover { transform: translateY(-2px); border-color: rgba(212,175,122,0.3); }
    .level-tile .role {
        font-size: 0.66rem; font-weight: 500;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: #8a8275;
    }
    .level-tile .price {
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem; font-weight: 600;
        margin-top: 0.5rem; letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .level-tile .delta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem; margin-top: 0.3rem; color: #8a8275;
    }
    .level-tile.entry  { border-left: 2px solid #a3e635; }
    .level-tile.entry .price { color: #a3e635; }
    .level-tile.stop   { border-left: 2px solid #fb923c; }
    .level-tile.stop .price  { color: #fb923c; }
    .level-tile.target { border-left: 2px solid #d4af7a; }
    .level-tile.target .price { color: #d4af7a; }
    .level-tile.resistance { border-left: 2px solid #f87171; }
    .level-tile.resistance .price { color: #f87171; }
    .level-tile.current { border-left: 2px solid #d4af7a; background: linear-gradient(180deg, rgba(212,175,122,0.05) 0%, #15171c 100%); }
    .level-tile.current .price { color: #f5efe1; }

    .rr-badge {
        display: inline-block;
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.25);
        border-radius: 2px;
        padding: 0.4rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #c9c2b3;
        font-weight: 500;
        letter-spacing: 0.04em;
    }
    .rr-badge b { color: #d4af7a; font-weight: 600; }

    .pick-intro {
        color: #8a8275; font-size: 0.88rem; margin-bottom: 0.8rem;
        font-style: italic;
    }
    .pick-disclaim {
        background: linear-gradient(180deg, rgba(212,175,122,0.05) 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.2);
        border-left: 2px solid #d4af7a;
        border-radius: 4px; padding: 0.85rem 1.1rem;
        color: #c9c2b3; font-size: 0.85rem; margin-bottom: 1rem;
    }

    /* Price ladder (above chart) */
    .lv-ladder {
        background: linear-gradient(180deg, #181a20 0%, #15171c 100%);
        border: 1px solid rgba(212,175,122,0.1);
        border-radius: 4px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.7rem;
    }
    .lv-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        padding: 0.45rem 0;
    }
    .lv-arrow {
        font-size: 0.72rem;
        font-weight: 500;
        color: #8a8275;
        min-width: 145px;
        flex-shrink: 0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .lv-row-up .lv-arrow { color: #f87171; }
    .lv-row-down .lv-arrow { color: #a3e635; }
    .lv-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .lv-empty { font-size: 0.8rem; color: #6b6557; font-style: italic; }

    .lv-pill {
        display: inline-flex; align-items: baseline; gap: 0.5rem;
        padding: 0.4rem 0.85rem;
        border-radius: 2px;
        border: 1px solid;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .lv-pill.lv-sell { background: rgba(220,38,38,0.06); color: #f87171; border-color: rgba(220,38,38,0.3); }
    .lv-pill.lv-buy  { background: rgba(132,204,22,0.06); color: #a3e635; border-color: rgba(132,204,22,0.3); }
    .lv-pill.lv-stop { background: rgba(251,146,60,0.06); color: #fb923c; border-color: rgba(251,146,60,0.3); }
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

    /* Now row — editorial center */
    .lv-now {
        display: flex; align-items: center; gap: 1rem;
        padding: 0.95rem 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(90deg, rgba(212,175,122,0.04) 0%, rgba(212,175,122,0.08) 50%, rgba(212,175,122,0.04) 100%);
        border-top: 1px solid rgba(212,175,122,0.2);
        border-bottom: 1px solid rgba(212,175,122,0.2);
        border-radius: 0;
    }
    .lv-now-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #d4af7a;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        min-width: 145px;
    }
    .lv-now-price {
        font-family: 'Playfair Display', serif;
        font-size: 1.85rem;
        font-weight: 600;
        color: #f5efe1;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .lv-now-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 500;
        color: #a3e635;
        background: rgba(132,204,22,0.08);
        border: 1px solid rgba(132,204,22,0.3);
        padding: 0.25rem 0.65rem;
        border-radius: 2px;
        letter-spacing: 0.1em;
    }

    @media (max-width: 768px) {
        .lv-arrow, .lv-now-label { min-width: auto; width: 100%; }
        .lv-pill { padding: 0.35rem 0.65rem; font-size: 0.78rem; }
        .lv-pill .lv-price { font-size: 0.84rem; }
        .lv-pill .lv-role { font-size: 0.74rem; }
        .lv-pill .lv-delta { font-size: 0.68rem; }
        .lv-now-price { font-size: 1.2rem; }
    }

    /* Section heading — editorial */
    .section-h {
        font-size: 0.66rem;
        font-weight: 500;
        color: #d4af7a;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        margin: 1rem 0 0.6rem 0;
        position: relative;
        padding-left: 0.85rem;
    }
    .section-h::before {
        content: '';
        position: absolute; left: 0; top: 50%;
        width: 4px; height: 4px;
        background: #d4af7a;
        transform: translateY(-50%);
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


def get_market_session(info: dict) -> dict:
    """Returns market state + extended hours info if available.
    Market states: REGULAR / PRE / POST / POSTPOST / PREPRE / CLOSED
    """
    state = (info.get("marketState") or "").upper()

    state_label = {
        "REGULAR":  ("🟢 ตลาดเปิด", "open"),
        "PRE":      ("🌅 Pre-Market", "pre"),
        "PREPRE":   ("🌅 Pre-Market (early)", "pre"),
        "POST":     ("🌙 After Hours", "post"),
        "POSTPOST": ("🌙 After Hours (late)", "post"),
        "CLOSED":   ("⚪ ตลาดปิด", "closed"),
    }.get(state, ("⚪ —", "closed"))

    result = {"state": state, "label": state_label[0], "css": state_label[1], "ext": None}

    if state in ("PRE", "PREPRE"):
        p = info.get("preMarketPrice")
        if p:
            result["ext"] = {
                "session": "Pre-Market",
                "icon": "🌅",
                "price": p,
                "change": info.get("preMarketChange"),
                "change_pct": info.get("preMarketChangePercent"),
                "time": info.get("preMarketTime"),
            }
    elif state in ("POST", "POSTPOST"):
        p = info.get("postMarketPrice")
        if p:
            result["ext"] = {
                "session": "After Hours",
                "icon": "🌙",
                "price": p,
                "change": info.get("postMarketChange"),
                "change_pct": info.get("postMarketChangePercent"),
                "time": info.get("postMarketTime"),
            }
    elif state == "CLOSED":
        # Show last post-market if available
        p = info.get("postMarketPrice")
        if p:
            result["ext"] = {
                "session": "Last After Hours",
                "icon": "🌙",
                "price": p,
                "change": info.get("postMarketChange"),
                "change_pct": info.get("postMarketChangePercent"),
                "time": info.get("postMarketTime"),
            }
    return result


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


@st.cache_data(ttl=600, show_spinner=False)
def load_usdthb() -> float | None:
    try:
        h = yf.Ticker("THB=X").history(period="1d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None


@st.cache_data(ttl=120, show_spinner=False)
def load_premarket_movers(tickers: tuple[str, ...]) -> list[dict]:
    """Fetch pre/post-market price changes via yfinance.Ticker.info in parallel."""
    if not tickers:
        return []
    def _fetch(t: str) -> dict | None:
        try:
            info = yf.Ticker(t).info or {}
            state = (info.get("marketState") or "").upper()
            session, p, c, pct = None, None, None, None
            if state in ("PRE", "PREPRE") and info.get("preMarketPrice"):
                session = "Pre"
                p = info["preMarketPrice"]
                c = info.get("preMarketChange")
                pct = info.get("preMarketChangePercent")
            elif state in ("POST", "POSTPOST", "CLOSED") and info.get("postMarketPrice"):
                session = "After"
                p = info["postMarketPrice"]
                c = info.get("postMarketChange")
                pct = info.get("postMarketChangePercent")
            if session is None or p is None or pct is None:
                return None
            return {"sym": t, "session": session, "price": p,
                    "change": c, "pct": pct, "state": state}
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=12) as ex:
        results = [r for r in ex.map(_fetch, tickers) if r is not None]
    return results


@st.cache_data(ttl=180, show_spinner=False)
def load_daily_movers(tickers: tuple[str, ...]) -> list[dict]:
    """Return list of {sym, last, d1} sorted ready to use."""
    if not tickers:
        return []
    try:
        raw = yf.download(list(tickers), period="5d", interval="1d",
                          progress=False, threads=True, auto_adjust=False)
    except Exception:
        return []
    out = []
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            close = raw["Close"][t].dropna() if multi else raw["Close"].dropna()
            if len(close) >= 2:
                last = float(close.iloc[-1])
                d1 = (last / float(close.iloc[-2]) - 1) * 100
                out.append({"sym": t, "last": last, "d1": d1})
        except Exception:
            continue
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_sector_heatmap() -> list[dict]:
    """11 SPDR sector ETFs + their 1D / 5D / 1M change."""
    sectors = {
        "XLK": "เทคโนโลยี", "XLF": "การเงิน", "XLV": "สุขภาพ",
        "XLE": "พลังงาน", "XLI": "อุตสาหกรรม", "XLY": "อุปโภคบริโภค (สมัครใจ)",
        "XLP": "อุปโภคบริโภค (จำเป็น)", "XLU": "สาธารณูปโภค", "XLB": "วัตถุดิบ",
        "XLRE": "อสังหา", "XLC": "สื่อสาร",
    }
    try:
        raw = yf.download(list(sectors.keys()), period="1mo", interval="1d",
                          progress=False, threads=True, auto_adjust=False)
    except Exception:
        return []
    out = []
    for s, name in sectors.items():
        try:
            close = raw["Close"][s].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"].dropna()
            if len(close) < 2:
                continue
            last = float(close.iloc[-1])
            d1 = (last / float(close.iloc[-2]) - 1) * 100 if len(close) >= 2 else 0
            d5 = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
            d20 = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0
            out.append({"sym": s, "name": name, "last": last, "d1": d1, "d5": d5, "d20": d20})
        except Exception:
            continue
    return out


@st.cache_data(ttl=300, show_spinner=False)
def load_macro() -> dict:
    """Fetch major index/yield/dollar levels for sidebar."""
    syms = ["^GSPC", "^IXIC", "^VIX", "^TNX", "^SET.BK", "THB=X"]
    labels = {
        "^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^VIX": "VIX",
        "^TNX": "10Y Yield (US)", "^SET.BK": "SET Index", "THB=X": "USD/THB",
    }
    out = {}
    try:
        raw = yf.download(syms, period="5d", interval="1d",
                          progress=False, threads=True, auto_adjust=False)
    except Exception:
        return {}
    for s in syms:
        try:
            close = raw["Close"][s].dropna() if isinstance(raw.columns, pd.MultiIndex) else raw["Close"].dropna()
            if len(close) >= 2:
                last = float(close.iloc[-1])
                pct = (last / float(close.iloc[-2]) - 1) * 100
                out[s] = {"label": labels[s], "last": last, "pct": pct}
        except Exception:
            continue
    return out


def compute_risk_metrics(close: pd.Series) -> dict:
    if len(close) < 30:
        return {}
    returns = close.pct_change().dropna()
    if returns.empty:
        return {}
    vol_annual = float(returns.std() * (252 ** 0.5) * 100)
    annual_return = float(returns.mean() * 252 * 100)
    sharpe = annual_return / vol_annual if vol_annual else 0
    rolling_max = close.cummax()
    drawdown = (close - rolling_max) / rolling_max * 100
    max_dd = float(drawdown.min())
    return {
        "vol": vol_annual,
        "max_dd": max_dd,
        "ann_return": annual_return,
        "sharpe": sharpe,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_quarterly_financials(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).quarterly_income_stmt
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_insider_trades(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).insider_transactions
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_upgrades(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).upgrades_downgrades
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def load_earnings_dates(ticker: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).earnings_dates
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _classify_insider(text: str, transaction: str = "") -> str:
    """Returns 'buy', 'sell', 'gift', or 'other'."""
    t = (text or "") + " " + (transaction or "")
    t = t.lower()
    if any(w in t for w in ["sale", "sold", "sell"]):
        return "sell"
    if any(w in t for w in ["purchase", "buy", "bought", "acquired"]):
        return "buy"
    if "gift" in t:
        return "gift"
    return "other"


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
    "💎 หุ้นปันผลสูง": {
        "desc": "หุ้นปันผล Yield > 3% และจ่ายปันผลแบบยั่งยืน (Payout < 85%) · เรียงตาม Yield สูงสุด",
        "mode": "auto_dividend",
        "universe": [
            "VZ", "T", "MO", "PFE", "IBM", "XOM", "CVX", "KO", "PEP", "MMM",
            "JNJ", "MRK", "ABBV", "BMY", "CSCO", "JPM", "BAC", "WFC", "USB",
            "GE", "CAT", "MCD", "WBA", "TGT", "AMGN", "TXN", "AVGO",
            "O", "SPG", "AMT", "PSA", "WELL",
            "ED", "DUK", "SO", "NEE", "AEP",
        ],
    },
    "💸 ผู้บริหารซื้อหุ้นเอง (Insider Buy)": {
        "desc": "หุ้นที่ผู้บริหาร/Director ซื้อหุ้นบริษัทใน 6 เดือนล่าสุด · สัญญาณเชื่อมั่นจากคนใน",
        "mode": "auto_insider_buy",
        "universe": [
            "AAPL", "MSFT", "NVDA", "META", "TSLA", "AMZN", "GOOGL",
            "AMD", "AVGO", "PLTR", "SMCI", "SOFI", "HIMS", "MARA", "RIOT",
            "COIN", "MSTR", "IONQ", "RKLB", "ACHR", "RIVN", "LCID",
            "JPM", "BAC", "WFC", "GS",
            "LLY", "UNH", "PFE", "JNJ", "MRK",
            "XOM", "CVX", "BA", "F", "GM",
        ],
    },
    "📉 หุ้น Oversold (RSI ต่ำ น่าเด้ง)": {
        "desc": "หุ้น quality ที่ RSI < 35 (ขายมากเกินไป) · contrarian play · เรียง RSI ต่ำสุด",
        "mode": "auto_oversold",
        "universe": [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "AMD", "AVGO", "QCOM", "TSM", "ASML", "ARM",
            "ORCL", "CRM", "ADBE", "NOW", "PANW", "CRWD",
            "JPM", "BAC", "GS", "V", "MA", "BLK",
            "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV",
            "WMT", "COST", "MCD", "NKE", "SBUX", "DIS",
            "XOM", "CVX", "BA",
            "VZ", "T", "PFE",
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
    "☁️ Cloud / SaaS": ["CRM", "NOW", "SNOW", "MDB", "DDOG", "NET", "OKTA", "PLTR", "WDAY", "ZS"],
    "🔐 ไซเบอร์ซีเคียวริตี้": ["CRWD", "PANW", "FTNT", "ZS", "S", "TENB", "QLYS", "CHKP"],
    "🤖 ชิป / Semiconductor": ["NVDA", "AMD", "AVGO", "TSM", "ASML", "AMAT", "KLAC", "LRCX", "MU", "QCOM", "ARM", "INTC"],
    "💰 การเงิน / ธนาคาร": ["JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "BLK", "C", "AXP"],
    "🏥 สุขภาพ / ยา": ["JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "🧬 Biotech": ["MRNA", "BNTX", "REGN", "VRTX", "GILD", "ALNY", "BMRN", "INCY", "BIIB"],
    "🛡️ กลาโหม / การทหาร": ["LMT", "RTX", "NOC", "GD", "BA", "LHX", "HII", "TDG"],
    "⚡ พลังงาน": ["XOM", "CVX", "COP", "SLB", "OXY", "EOG", "PSX", "MPC"],
    "🌱 พลังงานสะอาด": ["TSLA", "ENPH", "FSLR", "NEE", "BE", "PLUG", "RUN", "SEDG"],
    "⛏️ ทอง / เหมืองแร่": ["GLD", "GOLD", "NEM", "AEM", "FCX", "WPM", "AU", "KGC"],
    "🏠 อสังหา / REIT": ["AMT", "PLD", "EQIX", "WELL", "CCI", "PSA", "O", "SPG", "VICI"],
    "📡 โทรคมนาคม": ["T", "VZ", "TMUS", "CMCSA", "CHTR"],
    "🛒 สินค้าอุปโภคบริโภค": ["AMZN", "WMT", "COST", "PG", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT"],
    "🍔 ร้านอาหาร / Food": ["MCD", "SBUX", "CMG", "YUM", "DPZ", "QSR", "WEN", "SHAK"],
    "🎲 คาสิโน / Gambling": ["LVS", "MGM", "WYNN", "DKNG", "PENN", "CZR"],
    "🚗 รถยนต์ / EV": ["TSLA", "F", "GM", "RIVN", "LCID", "TM", "HMC", "STLA", "BYDDY"],
    "🎮 สื่อ / เกม": ["NFLX", "DIS", "SONY", "EA", "TTWO", "RBLX", "SPOT", "ROKU"],
    "✈️ สายการบิน / ท่องเที่ยว": ["DAL", "UAL", "AAL", "LUV", "BA", "BKNG", "ABNB", "MAR"],
    "🪙 คริปโต": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD"],
    "📊 ดัชนีตลาด": ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^FTSE", "^N225", "^HSI"],
    "🌐 กองทุน ETF": ["SPY", "QQQ", "VOO", "VTI", "IWM", "DIA", "ARKK", "GLD", "TLT"],
    "🇨🇳 หุ้นจีน (ADR)": ["BABA", "JD", "PDD", "NIO", "LI", "XPEV", "BIDU", "TME"],
    "🇯🇵 หุ้นญี่ปุ่น (ADR)": ["TM", "SONY", "HMC", "NTT", "MUFG", "SMFG", "TAK"],
    "🇮🇳 หุ้นอินเดีย (ADR)": ["INFY", "WIT", "HDB", "IBN", "TTM"],
    "🇪🇺 หุ้นยุโรป (ADR)": ["ASML", "SAP", "NVO", "NVS", "SHEL", "RIO", "BP", "UL"],
    "🏦 หุ้นไทย (SET)": ["PTT.BK", "ADVANC.BK", "AOT.BK", "CPALL.BK", "KBANK.BK", "SCB.BK", "PTTEP.BK", "DELTA.BK"],
}

if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "category" not in st.session_state:
    st.session_state.category = "🔥 หุ้นยักษ์ใหญ่"
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "page" not in st.session_state:
    st.session_state.page = "🌐 ภาพรวมตลาด"

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="logo">S</div>
            <div>
                <div class="name">Stockscope</div>
                <div class="sub">Real-time · Global</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Page navigation
    PAGES = ["🌐 ภาพรวมตลาด", "📊 ดูหุ้น", "🔍 สแกนหุ้น"]
    page = st.radio(
        "หน้า", PAGES,
        index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown('<div class="section-h">ค้นหาหุ้น</div>', unsafe_allow_html=True)
    if page == "🌐 ภาพรวมตลาด":
        initial_val = ""
        placeholder_txt = "พิมพ์ ticker → กระโดดไปดูหุ้น"
    else:
        initial_val = st.session_state.ticker
        placeholder_txt = "AAPL, TSLA, BTC-USD, ^GSPC …"
    ticker_input = st.text_input(
        "รหัสหุ้น",
        value=initial_val,
        placeholder=placeholder_txt,
        label_visibility="collapsed",
        key=f"ticker_input_{page}",
    )
    typed = ticker_input.upper().strip()
    if typed and typed != st.session_state.ticker:
        st.session_state.ticker = typed
        if page == "🌐 ภาพรวมตลาด":
            st.session_state.page = "📊 ดูหุ้น"
            st.rerun()
    ticker = st.session_state.ticker

    # Watchlist controls
    in_watchlist = ticker in st.session_state.watchlist
    wc1, wc2 = st.columns([1, 1])
    if not in_watchlist:
        if wc1.button("⭐ เพิ่มใน Watchlist", use_container_width=True, key="add_wl"):
            if ticker and ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
                st.rerun()
    else:
        if wc1.button("★ อยู่ใน Watchlist", use_container_width=True, key="remove_wl"):
            st.session_state.watchlist.remove(ticker)
            st.rerun()
    if st.session_state.watchlist and wc2.button("🗑️ ล้าง", use_container_width=True, key="clear_wl"):
        st.session_state.watchlist = []
        st.rerun()

    if st.session_state.watchlist:
        st.markdown('<div class="section-h">⭐ Watchlist</div>', unsafe_allow_html=True)
        wl_cols = st.columns(2)
        for i, sym in enumerate(st.session_state.watchlist):
            display = sym.replace(".BK", "").replace("-USD", "")
            if wl_cols[i % 2].button(display, key=f"wl_{sym}", use_container_width=True):
                st.session_state.ticker = sym
                st.rerun()

    # Defaults (used when sidebar controls are hidden on non-stock pages)
    period = "1y"
    interval = "1d"
    translate_news = True

if page != "📊 ดูหุ้น":
    with st.sidebar:
        st.write("")
        if st.button("🔄 อัพเดตข้อมูล", use_container_width=True, key="refresh_global"):
            st.cache_data.clear()
            st.rerun()
        st.caption("ข้อมูลจาก Yahoo Finance · แคช 60 วิ ถึง 1 ชม.")

if page == "📊 ดูหุ้น":
  with st.sidebar:
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
        elif mode == "auto_dividend":
            with st.spinner("กำลังดึงข้อมูลปันผล…"):
                fund = load_info_batch(tuple(universe))
            candidates = []
            for t in universe:
                f = fund.get(t, {})
                yld = f.get("rev_growth")  # placeholder; we'll use info dict directly below
                # Re-fetch direct from yfinance info would be too slow; use what's in fund
                # info-level yields aren't in load_info_batch — need to extend OR fetch dividendYield separately
                pass
            # Fetch fresh info containing dividendYield (not in load_info_batch by default)
            from yfinance import Ticker as _T
            for t in universe:
                try:
                    inf = _T(t).info or {}
                    yld = inf.get("dividendYield")
                    payout = inf.get("payoutRatio") or 0
                    if not yld or yld <= 0:
                        continue
                    # yfinance returns dividendYield as decimal (0.03 = 3%) or sometimes as percent
                    yld_pct = yld * 100 if yld < 1 else yld
                    if yld_pct < 3.0:
                        continue
                    if payout and payout > 0.85:
                        continue
                    last = mini.get(t, {}).get("last")
                    candidates.append((t, yld_pct, payout, last))
                except Exception:
                    continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            tickers_in_cat = []
            for t, yld, payout, last in candidates[:12]:
                bits = [f"Yield {yld:.2f}%"]
                if payout:
                    bits.append(f"จ่าย {payout*100:.0f}% ของกำไร")
                if last:
                    bits.append(f"ราคา {last:,.2f}")
                tickers_in_cat.append((t, " · ".join(bits), None))
            if not tickers_in_cat:
                st.info("ไม่พบหุ้นปันผลที่ผ่านเกณฑ์")

        elif mode == "auto_insider_buy":
            with st.spinner("กำลังสแกน insider transactions…"):
                from datetime import timedelta
                cutoff_date = datetime.now().date() - timedelta(days=180)
                scored = []
                def _scan(t: str):
                    try:
                        idf = yf.Ticker(t).insider_transactions
                        if idf is None or idf.empty:
                            return t, 0, 0, 0
                        buy_val = 0; buy_count = 0; sell_count = 0
                        for _, row in idf.iterrows():
                            try:
                                d = pd.to_datetime(row.get("Start Date")).date()
                                if d < cutoff_date:
                                    continue
                                kind = _classify_insider(row.get("Text", ""), row.get("Transaction", ""))
                                if kind == "buy":
                                    buy_val += row.get("Value", 0) or 0
                                    buy_count += 1
                                elif kind == "sell":
                                    sell_count += 1
                            except Exception:
                                pass
                        return t, buy_val, buy_count, sell_count
                    except Exception:
                        return t, 0, 0, 0
                with ThreadPoolExecutor(max_workers=10) as ex:
                    for t, buy_val, buy_c, sell_c in ex.map(_scan, universe):
                        if buy_c > 0 and buy_c >= sell_c:  # net buying or balanced
                            scored.append((t, buy_val, buy_c, sell_c))
            scored.sort(key=lambda x: x[1], reverse=True)
            tickers_in_cat = []
            for t, buy_val, buy_c, sell_c in scored[:10]:
                if buy_val > 0:
                    bits = [f"ซื้อ {buy_c} ครั้ง · ${buy_val/1e6:.2f}M"]
                else:
                    bits = [f"ซื้อ {buy_c} ครั้ง"]
                if sell_c > 0:
                    bits.append(f"ขาย {sell_c} ครั้ง")
                tickers_in_cat.append((t, " · ".join(bits), None))
            if not tickers_in_cat:
                st.info("ไม่พบหุ้นที่มี insider ซื้อใน 6 เดือนล่าสุด")

        elif mode == "auto_oversold":
            candidates = [
                (t, d) for t, d in mini.items()
                if d["rsi"] < 35
            ]
            candidates.sort(key=lambda x: x[1]["rsi"])
            tickers_in_cat = [
                (t, f"RSI {d['rsi']:.0f} · 1M {d['m1']:+.1f}% · 3M {d['m3']:+.1f}%", None)
                for t, d in candidates[:12]
            ]
            if not tickers_in_cat:
                st.info("ไม่มีหุ้นไหน Oversold ตอนนี้ (RSI < 35)")

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

    with st.expander("🤖 AI สรุปข่าว (Claude)"):
        st.caption("ใส่ Anthropic API key เพื่อให้ AI สรุปข่าวเป็นภาษาไทย")
        api_key_input = st.text_input(
            "Anthropic API Key",
            type="password",
            value=st.session_state.get("anthropic_key", ""),
            placeholder="sk-ant-…",
            label_visibility="collapsed",
        )
        if api_key_input != st.session_state.get("anthropic_key"):
            st.session_state["anthropic_key"] = api_key_input
        st.caption(
            "ขอ key ฟรีที่ [console.anthropic.com](https://console.anthropic.com) · "
            "$5 ฟรีตอนสมัคร · ใช้ ~1฿ ต่อการสรุป"
        )

    st.write("")
    if st.button("🔄 อัพเดตข้อมูล", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("ข้อมูลจาก Yahoo Finance · หน่วง ~1 นาที · แคช 60 วิ")

# Macro market overview (always shown — every page)
macro = load_macro()
if macro:
    macro_html = '<div class="macro-bar">'
    for sym, d in macro.items():
        cls = "up" if d["pct"] >= 0 else "down"
        arrow = "▲" if d["pct"] >= 0 else "▼"
        macro_html += (
            f'<div class="macro-tile">'
            f'<div class="macro-label">{d["label"]}</div>'
            f'<div class="macro-row">'
            f'<span class="macro-price">{d["last"]:,.2f}</span>'
            f'<span class="macro-chip {cls}">{arrow} {d["pct"]:+.2f}%</span>'
            f'</div></div>'
        )
    macro_html += "</div>"
    st.markdown(macro_html, unsafe_allow_html=True)

# ========== PAGE: สแกนหุ้น ==========
if st.session_state.page == "🔍 สแกนหุ้น":
    st.markdown("## 🔍 สแกนหุ้น (Stock Screener)")
    st.caption("กรองหุ้นจากทั่วทุกหมวดด้วยเกณฑ์ที่ตั้งเอง · ตั้งค่า filter แล้วกด **เริ่มสแกน**")
    sc_top = st.columns([1.5, 1.5, 1])
    sc_universe = sc_top[0].selectbox(
        "🌐 Universe",
        ["ทุกหมวดรวมกัน"] + list(CATEGORIES.keys()),
        key="sc_universe",
    )
    sc_sort = sc_top[1].selectbox(
        "🔢 เรียงโดย",
        [
            "Market Cap (สูง→ต่ำ)",
            "P/E (ต่ำ→สูง)",
            "Revenue Growth (สูง→ต่ำ)",
            "Performance 3 เดือน (สูง→ต่ำ)",
            "Performance 1 เดือน (สูง→ต่ำ)",
            "RSI (ต่ำ→สูง · oversold ก่อน)",
            "Profit Margin (สูง→ต่ำ)",
        ],
        key="sc_sort",
    )

    f1, f2, f3, f4 = st.columns(4)
    pe_max = f1.number_input("P/E ≤", min_value=0, max_value=500, value=100, step=5, key="sc_pe")
    growth_min = f2.number_input("Revenue Growth ≥ (%)", min_value=-50, max_value=200, value=-50, step=5, key="sc_growth")
    margin_min = f3.number_input("Profit Margin ≥ (%)", min_value=-100, max_value=80, value=-100, step=5, key="sc_margin")
    perf_min = f4.number_input("3M Performance ≥ (%)", min_value=-90, max_value=300, value=-90, step=5, key="sc_perf")

    rsi_range = st.slider("RSI ในช่วง", 0, 100, (0, 100), key="sc_rsi")
    only_buy_rated = st.checkbox("เฉพาะที่นักวิเคราะห์ Buy / Strong Buy (rec_mean ≤ 2.5)", value=False, key="sc_buy")

    if st.button("🔍 เริ่มสแกน", use_container_width=True, key="sc_scan"):
        if sc_universe == "ทุกหมวดรวมกัน":
            scan_tickers = set()
            for tks in CATEGORIES.values():
                scan_tickers.update(tks)
        else:
            scan_tickers = set(CATEGORIES[sc_universe])
        scan_tickers = tuple(sorted(scan_tickers))

        with st.spinner(f"กำลังสแกน {len(scan_tickers)} หุ้น…"):
            mini_scan = load_mini_batch(scan_tickers)
            fund_scan = load_info_batch(scan_tickers)

        results = []
        for t in scan_tickers:
            d = mini_scan.get(t, {})
            f = fund_scan.get(t, {})
            if not d or not d.get("last"):
                continue
            pe = f.get("pe")
            rev_g = f.get("rev_growth")
            margin = f.get("profit_margin")
            rec_mean = f.get("rec_mean")
            rsi_v = d.get("rsi")
            m3 = d.get("m3")

            if pe and pe > pe_max:
                continue
            if rev_g is not None and rev_g * 100 < growth_min:
                continue
            if margin is not None and margin * 100 < margin_min:
                continue
            if m3 is not None and m3 < perf_min:
                continue
            if rsi_v is not None and (rsi_v < rsi_range[0] or rsi_v > rsi_range[1]):
                continue
            if only_buy_rated and (rec_mean is None or rec_mean > 2.5):
                continue

            results.append({
                "t": t, "d": d, "f": f,
                "mcap": f.get("market_cap") or 0,
                "pe": pe, "rev_g": rev_g, "margin": margin,
                "rsi": rsi_v, "m1": d.get("m1"), "m3": m3,
                "rec": rec_mean,
            })

        # Sort
        sort_map = {
            "Market Cap (สูง→ต่ำ)": (lambda r: r["mcap"] or 0, True),
            "P/E (ต่ำ→สูง)": (lambda r: r["pe"] if r["pe"] else 99999, False),
            "Revenue Growth (สูง→ต่ำ)": (lambda r: r["rev_g"] or -999, True),
            "Performance 3 เดือน (สูง→ต่ำ)": (lambda r: r["m3"] or -999, True),
            "Performance 1 เดือน (สูง→ต่ำ)": (lambda r: r["m1"] or -999, True),
            "RSI (ต่ำ→สูง · oversold ก่อน)": (lambda r: r["rsi"] or 100, False),
            "Profit Margin (สูง→ต่ำ)": (lambda r: r["margin"] or -999, True),
        }
        key_fn, reverse = sort_map[sc_sort]
        results.sort(key=key_fn, reverse=reverse)

        st.success(f"พบ {len(results)} หุ้น (แสดง {min(50, len(results))} อันดับแรก)")

        def _mcap_fmt(x):
            if not x: return "—"
            for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
                if x >= div: return f"{x/div:.1f}{unit}"
            return f"{x:,.0f}"

        table = []
        for r in results[:50]:
            table.append({
                "Ticker": r["t"],
                "ราคา": f"{r['d']['last']:,.2f}",
                "Mcap": _mcap_fmt(r["mcap"]),
                "P/E": f"{r['pe']:.1f}" if r["pe"] else "—",
                "Rev%": f"{r['rev_g']*100:+.0f}" if r["rev_g"] is not None else "—",
                "Margin%": f"{r['margin']*100:.0f}" if r["margin"] is not None else "—",
                "RSI": f"{r['rsi']:.0f}" if r["rsi"] is not None else "—",
                "1M%": f"{r['m1']:+.1f}" if r["m1"] is not None else "—",
                "3M%": f"{r['m3']:+.1f}" if r["m3"] is not None else "—",
                "Analyst": _rec_thai(None, r["rec"]) if r["rec"] else "—",
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        # Quick-pick buttons for top 8
        if results:
            st.caption("คลิกเพื่อดูรายละเอียดเต็ม:")
            pick_cols = st.columns(8)
            for i, r in enumerate(results[:8]):
                if pick_cols[i].button(r["t"], key=f"sc_pick_{r['t']}", use_container_width=True):
                    st.session_state.ticker = r["t"]
                    st.rerun()

    st.stop()

# ========== PAGE: ภาพรวมตลาด ==========
if st.session_state.page == "🌐 ภาพรวมตลาด":
    st.markdown("## 🌐 ภาพรวมตลาด US")
    st.caption("ดูสถานะ sector ทั้งหมด · คลิก ETF เพื่อเข้าดูรายละเอียดเต็ม")
    st.write("")
    sectors = load_sector_heatmap()
    if sectors:
        # Sort by 1-day change descending for at-a-glance
        sectors_sorted = sorted(sectors, key=lambda x: x["d1"], reverse=True)

        def _sec_color(pct: float) -> tuple[str, str]:
            # Returns (border_color, bg) for dark editorial theme
            if pct >= 2:    return "rgba(132,204,22,0.5)", "linear-gradient(180deg, rgba(132,204,22,0.10) 0%, #15171c 100%)"
            if pct >= 1:    return "rgba(132,204,22,0.35)", "linear-gradient(180deg, rgba(132,204,22,0.06) 0%, #15171c 100%)"
            if pct >= 0.3:  return "rgba(132,204,22,0.2)", "linear-gradient(180deg, rgba(132,204,22,0.03) 0%, #15171c 100%)"
            if pct > -0.3:  return "rgba(212,175,122,0.15)", "#15171c"
            if pct > -1:    return "rgba(220,38,38,0.2)", "linear-gradient(180deg, rgba(220,38,38,0.03) 0%, #15171c 100%)"
            if pct > -2:    return "rgba(220,38,38,0.35)", "linear-gradient(180deg, rgba(220,38,38,0.06) 0%, #15171c 100%)"
            return "rgba(220,38,38,0.5)", "linear-gradient(180deg, rgba(220,38,38,0.10) 0%, #15171c 100%)"

        st.caption("คลิกชื่อ ETF เพื่อดูรายละเอียดเต็ม · สี = % เปลี่ยนแปลงวันนี้")
        hm_html = '<div class="sector-grid">'
        for s in sectors_sorted:
            color, bg = _sec_color(s["d1"])
            hm_html += (
                f'<div class="sector-cell" style="background:{bg};border-color:{color};">'
                f'<div class="sector-sym">{s["sym"]}</div>'
                f'<div class="sector-name">{s["name"]}</div>'
                f'<div class="sector-pct" style="color:{color};">{s["d1"]:+.2f}%</div>'
                f'<div class="sector-sub">5D {s["d5"]:+.1f}% · 1M {s["d20"]:+.1f}%</div>'
                f'</div>'
            )
        hm_html += "</div>"
        st.markdown(hm_html, unsafe_allow_html=True)

        # Quick-pick buttons
        sec_cols = st.columns(6)
        for i, s in enumerate(sectors_sorted[:6]):
            if sec_cols[i].button(s["sym"], key=f"sec_{s['sym']}", use_container_width=True):
                st.session_state.ticker = s["sym"]
                st.rerun()
    else:
        st.info("ไม่สามารถโหลดข้อมูล sector ได้")

    # === Top Daily Movers ===
    st.write("")
    st.markdown("### 📈 Top Movers วันนี้")
    POPULAR_UNIVERSE = (
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
        "AMD", "QCOM", "TSM", "ASML", "ARM", "ORCL", "CRM", "ADBE", "NOW",
        "PANW", "CRWD", "SNOW", "PLTR", "SMCI", "NFLX", "DIS",
        "JPM", "BAC", "GS", "V", "MA",
        "LLY", "UNH", "JNJ", "PFE",
        "XOM", "CVX", "WMT", "COST", "MCD",
        "COIN", "MSTR", "IONQ", "RKLB", "HIMS", "SOFI",
        "BTC-USD", "ETH-USD",
    )
    with st.spinner("กำลังโหลด Top Movers…"):
        movers = load_daily_movers(POPULAR_UNIVERSE)
    if movers:
        gainers = sorted(movers, key=lambda x: x["d1"], reverse=True)[:8]
        losers = sorted(movers, key=lambda x: x["d1"])[:8]

        def _mover_html(items: list[dict], cls: str) -> str:
            html = '<div class="mover-grid">'
            for m in items:
                arrow = "▲" if m["d1"] >= 0 else "▼"
                html += (
                    f'<div class="mover-cell mover-{cls}">'
                    f'<div class="mover-sym">{m["sym"]}</div>'
                    f'<div class="mover-pct">{arrow} {m["d1"]:+.2f}%</div>'
                    f'<div class="mover-price">{m["last"]:,.2f}</div>'
                    f'</div>'
                )
            html += "</div>"
            return html

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**🟢 ราคาขึ้นมากที่สุด**")
            st.markdown(_mover_html(gainers, "up"), unsafe_allow_html=True)
        with mc2:
            st.markdown("**🔴 ราคาลงมากที่สุด**")
            st.markdown(_mover_html(losers, "down"), unsafe_allow_html=True)

        # Quick-pick buttons for gainers
        st.write("")
        st.caption("คลิกเพื่อดูรายละเอียดเต็ม:")
        pick_g = st.columns(8)
        for i, m in enumerate(gainers):
            if pick_g[i].button(m["sym"], key=f"mover_g_{m['sym']}", use_container_width=True):
                st.session_state.ticker = m["sym"]
                st.session_state.page = "📊 ดูหุ้น"
                st.rerun()

    # === Pre-Market / After-Hours Movers ===
    st.write("")
    st.markdown("### 🌅 Pre-Market / 🌙 After-Hours")
    st.caption("ดูราคาหุ้นช่วง **ก่อน/หลังตลาดเปิด** · สหรัฐฯ pre-market 16:00–20:30 ไทย · after-hours 03:00–07:00 ไทย")
    with st.spinner("กำลังดึงข้อมูล Pre/After-Hours…"):
        pm_data = load_premarket_movers(POPULAR_UNIVERSE)
    if pm_data:
        pm_gainers = sorted([m for m in pm_data if (m["pct"] or 0) > 0],
                            key=lambda x: x["pct"], reverse=True)[:8]
        pm_losers = sorted([m for m in pm_data if (m["pct"] or 0) < 0],
                           key=lambda x: x["pct"])[:8]

        def _pm_html(items: list[dict], cls: str) -> str:
            html = '<div class="mover-grid">'
            for m in items:
                arrow = "▲" if (m["pct"] or 0) >= 0 else "▼"
                icon = "🌅" if m["session"] == "Pre" else "🌙"
                html += (
                    f'<div class="mover-cell mover-{cls}">'
                    f'<div class="mover-sym">{icon} {m["sym"]}</div>'
                    f'<div class="mover-pct">{arrow} {m["pct"]:+.2f}%</div>'
                    f'<div class="mover-price">{m["price"]:,.2f}</div>'
                    f'</div>'
                )
            html += "</div>"
            return html

        if not pm_gainers and not pm_losers:
            st.info("ตอนนี้ยังไม่มีการเคลื่อนไหวใน Pre/After-Hours (อาจเป็นช่วงตลาดเปิดหรือดึก ๆ)")
        else:
            pm1, pm2 = st.columns(2)
            if pm_gainers:
                with pm1:
                    st.markdown("**🟢 ขึ้นมากสุด (นอกเวลา)**")
                    st.markdown(_pm_html(pm_gainers, "up"), unsafe_allow_html=True)
            if pm_losers:
                with pm2:
                    st.markdown("**🔴 ลงมากสุด (นอกเวลา)**")
                    st.markdown(_pm_html(pm_losers, "down"), unsafe_allow_html=True)

            # Quick-pick for top pre-market gainers
            all_pm = (pm_gainers + pm_losers)[:8]
            if all_pm:
                st.caption("คลิกเพื่อดูรายละเอียดเต็ม:")
                pm_cols = st.columns(min(8, len(all_pm)))
                for i, m in enumerate(all_pm):
                    if pm_cols[i].button(m["sym"], key=f"pm_pick_{m['sym']}", use_container_width=True):
                        st.session_state.ticker = m["sym"]
                        st.session_state.page = "📊 ดูหุ้น"
                        st.rerun()
    else:
        st.info("ตอนนี้ไม่มีข้อมูล Pre/After-Hours สำหรับหุ้นในรายการ (อาจเป็นช่วงเวลาทำการปกติ)")

    # === Watchlist summary ===
    if st.session_state.watchlist:
        st.write("")
        st.markdown("### ⭐ Watchlist ของคุณ")
        with st.spinner("กำลังโหลด watchlist…"):
            wl_data = load_daily_movers(tuple(st.session_state.watchlist))
        if wl_data:
            wl_html = '<div class="mover-grid">'
            for m in wl_data:
                cls = "up" if m["d1"] >= 0 else "down"
                arrow = "▲" if m["d1"] >= 0 else "▼"
                wl_html += (
                    f'<div class="mover-cell mover-{cls}">'
                    f'<div class="mover-sym">{m["sym"]}</div>'
                    f'<div class="mover-pct">{arrow} {m["d1"]:+.2f}%</div>'
                    f'<div class="mover-price">{m["last"]:,.2f}</div>'
                    f'</div>'
                )
            wl_html += "</div>"
            st.markdown(wl_html, unsafe_allow_html=True)
            # Quick-pick
            wl_cols = st.columns(min(8, len(wl_data)))
            for i, m in enumerate(wl_data[:8]):
                if wl_cols[i].button(m["sym"], key=f"wl_pick_{m['sym']}", use_container_width=True):
                    st.session_state.ticker = m["sym"]
                    st.session_state.page = "📊 ดูหุ้น"
                    st.rerun()
    else:
        st.write("")
        st.info("⭐ ยังไม่มี Watchlist · ไปดูหุ้นรายตัวแล้วกดเพิ่ม จะเห็นสรุปที่นี่")

    st.stop()

# ========== PAGE: ดูหุ้น (default) ==========
if not ticker:
    st.markdown("## 🔍 เลือกหุ้นที่อยากดู")
    st.caption("ค้นหาในช่อง sidebar หรือเลือกจากปุ่มด้านล่าง · หรือกลับไป **🌐 ภาพรวมตลาด** เพื่อดู Top Movers / Sector")
    st.write("")
    st.markdown("**🔥 หุ้นยอดนิยม**")
    popular = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"]
    pop_cols = st.columns(4)
    for i, sym in enumerate(popular):
        if pop_cols[i % 4].button(sym, key=f"empty_pop_{sym}", use_container_width=True):
            st.session_state.ticker = sym
            st.rerun()

    st.write("")
    st.markdown("**🪙 คริปโต**")
    crypto = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"]
    cr_cols = st.columns(4)
    for i, sym in enumerate(crypto):
        if cr_cols[i].button(sym.replace("-USD", ""), key=f"empty_cr_{sym}", use_container_width=True):
            st.session_state.ticker = sym
            st.rerun()

    st.write("")
    st.markdown("**🏦 หุ้นไทย (SET)**")
    thai = ["PTT.BK", "KBANK.BK", "SCB.BK", "AOT.BK", "CPALL.BK", "ADVANC.BK", "DELTA.BK", "PTTEP.BK"]
    th_cols = st.columns(4)
    for i, sym in enumerate(thai):
        if th_cols[i % 4].button(sym.replace(".BK", ""), key=f"empty_th_{sym}", use_container_width=True):
            st.session_state.ticker = sym
            st.rerun()

    if st.session_state.watchlist:
        st.write("")
        st.markdown("**⭐ Watchlist ของคุณ**")
        wl_cols = st.columns(4)
        for i, sym in enumerate(st.session_state.watchlist[:8]):
            display = sym.replace(".BK", "").replace("-USD", "")
            if wl_cols[i % 4].button(display, key=f"empty_wl_{sym}", use_container_width=True):
                st.session_state.ticker = sym
                st.rerun()

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
    thb_html = ""
    if currency == "USD":
        thb_rate = load_usdthb()
        if thb_rate:
            thb_val = last_close * thb_rate
            thb_html = f'<div class="hero-thb">≈ ฿{thb_val:,.2f} บาท</div>'

    # Market session + extended hours
    session = get_market_session(info)
    session_badge = f'<div class="session-badge session-{session["css"]}">{session["label"]}</div>'
    ext_html = ""
    if session["ext"]:
        e = session["ext"]
        ep = e["price"]; ec = e["change"] or 0; epct = e["change_pct"] or 0
        ext_cls = "up" if ec >= 0 else "down"
        ext_arrow = "▲" if ec >= 0 else "▼"
        ext_html = f"""
        <div class="ext-hours">
            <div class="ext-head">{e['icon']} {e['session']}</div>
            <div class="ext-row">
                <span class="ext-price">{ep:,.2f}</span>
                <span class="ext-chip {ext_cls}">{ext_arrow} {ec:+.2f} ({epct:+.2f}%)</span>
            </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="hero" style="text-align:right;">
            <div class="sym">ราคาล่าสุด · {currency}</div>
            <div class="price">{last_close:,.2f}</div>
            {thb_html}
            <div class="chip {chip_cls}">{arrow} {change:+.2f} ({pct:+.2f}%)</div>
            {session_badge}
            {ext_html}
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
          "theme": "dark",
          "style": "1",
          "locale": "th_TH",
          "toolbar_bg": "#15171c",
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
    # === 52-Week Position Gauge ===
    w52_hi = info.get("fiftyTwoWeekHigh")
    w52_lo = info.get("fiftyTwoWeekLow")
    if w52_hi and w52_lo and w52_hi > w52_lo:
        pos_pct = (last_close - w52_lo) / (w52_hi - w52_lo) * 100
        pos_pct = max(0, min(100, pos_pct))
        if pos_pct >= 80:
            zone = "ใกล้จุดสูงสุด — ระวัง valuation"
        elif pos_pct >= 60:
            zone = "ช่วงบน — momentum ดี"
        elif pos_pct >= 40:
            zone = "ช่วงกลาง"
        elif pos_pct >= 20:
            zone = "ช่วงล่าง — น่าสนใจถ้าพื้นฐานดี"
        else:
            zone = "ใกล้จุดต่ำสุด — รอ confirm reversal"
        st.markdown(
            f"""
            <div class="pos52">
                <div class="pos52-head">
                    <span class="pos52-title">📍 ตำแหน่งราคาใน 52 สัปดาห์</span>
                    <span class="pos52-pct">{pos_pct:.0f}% · {zone}</span>
                </div>
                <div class="pos52-track">
                    <div class="pos52-marker" style="left:{pos_pct}%;"></div>
                </div>
                <div class="pos52-ends">
                    <span>ต่ำสุด {w52_lo:,.2f}</span>
                    <span>ราคาตอนนี้ {last_close:,.2f}</span>
                    <span>สูงสุด {w52_hi:,.2f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # === Risk Metrics ===
    risk = compute_risk_metrics(df["Close"])
    if risk:
        st.markdown('<div class="section-h">⚠️ ความเสี่ยง & ผลตอบแทน (ในช่วงข้อมูล)</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        beta_val = info.get("beta")
        rc1.markdown(
            tile("Beta (vs ตลาด)", f"{beta_val:.2f}" if beta_val else "—"),
            unsafe_allow_html=True,
        )
        rc2.markdown(
            tile("ความผันผวน/ปี", f"{risk['vol']:.1f}%"),
            unsafe_allow_html=True,
        )
        rc3.markdown(
            tile("Max Drawdown", f"{risk['max_dd']:.1f}%"),
            unsafe_allow_html=True,
        )
        rc4.markdown(
            tile("Sharpe Ratio", f"{risk['sharpe']:.2f}"),
            unsafe_allow_html=True,
        )
        st.caption(
            "📖 **Beta** > 1 = ผันผวนกว่าตลาด · **ผันผวน/ปี** = standard deviation รายปี · "
            "**Max Drawdown** = ขาดทุนหนักสุดจาก peak · **Sharpe** = ผลตอบแทนเทียบความเสี่ยง (>1 ดี, >2 ยอดเยี่ยม)"
        )
        st.write("")

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

    # === Quarterly Revenue & Earnings chart ===
    qf = load_quarterly_financials(ticker)
    if not qf.empty:
        try:
            rows = ["Total Revenue", "Net Income"]
            available = [r for r in rows if r in qf.index]
            if available:
                st.write("")
                st.subheader("📊 รายได้ & กำไรรายไตรมาส (4 ไตรมาสล่าสุด)")
                cols_q = list(qf.columns)[:4][::-1]  # oldest → newest
                date_labels = [d.strftime("%Y Q%q").replace("Q%q", f"Q{((d.month-1)//3)+1}") for d in cols_q]

                fig_q = go.Figure()
                if "Total Revenue" in qf.index:
                    rev_vals = [float(qf.at["Total Revenue", c]) / 1e9 for c in cols_q]
                    fig_q.add_trace(go.Bar(
                        name="รายได้", x=date_labels, y=rev_vals,
                        marker_color="#d4af7a",
                        text=[f"${v:.1f}B" for v in rev_vals], textposition="outside",
                        textfont=dict(color="#c9c2b3"),
                    ))
                if "Net Income" in qf.index:
                    ni_vals = [float(qf.at["Net Income", c]) / 1e9 for c in cols_q]
                    fig_q.add_trace(go.Bar(
                        name="กำไรสุทธิ", x=date_labels, y=ni_vals,
                        marker_color="#a3e635",
                        text=[f"${v:.1f}B" for v in ni_vals], textposition="outside",
                        textfont=dict(color="#c9c2b3"),
                    ))

                fig_q.update_layout(
                    barmode="group", height=320,
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title="พันล้านดอลลาร์ (USD)",
                    font=dict(family="Inter, sans-serif", color="#8a8275", size=11),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                )
                fig_q.update_xaxes(gridcolor="rgba(212,175,122,0.05)", color="#8a8275")
                fig_q.update_yaxes(gridcolor="rgba(212,175,122,0.05)", color="#8a8275", zerolinecolor="rgba(212,175,122,0.1)")
                st.plotly_chart(fig_q, use_container_width=True,
                                config={"displaylogo": False})

                # Growth annotations
                if "Total Revenue" in qf.index and len(cols_q) >= 2:
                    last_rev = float(qf.at["Total Revenue", cols_q[-1]])
                    prev_rev = float(qf.at["Total Revenue", cols_q[-2]])
                    qoq = (last_rev / prev_rev - 1) * 100 if prev_rev else 0
                    cap_parts = [f"📈 รายได้ QoQ: **{qoq:+.1f}%**"]
                    if len(cols_q) >= 4:
                        yoy_rev = float(qf.at["Total Revenue", cols_q[0]])
                        yoy = (last_rev / yoy_rev - 1) * 100 if yoy_rev else 0
                        cap_parts.append(f"YoY (4Q): **{yoy:+.1f}%**")
                    st.caption(" · ".join(cap_parts))
        except Exception:
            pass

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

    # === Insider Transactions ===
    insider_df = load_insider_trades(ticker)
    if not insider_df.empty:
        st.write("")
        st.subheader("👥 การซื้อขายของคนใน (Insider Transactions)")
        # Classify and aggregate
        rows_3mo = []
        cutoff = datetime.now().date()
        for _, row in insider_df.iterrows():
            try:
                date_val = pd.to_datetime(row.get("Start Date")).date()
                days_ago = (cutoff - date_val).days
                if days_ago > 180:
                    continue
                kind = _classify_insider(row.get("Text", ""), row.get("Transaction", ""))
                rows_3mo.append({
                    "date": date_val, "name": row.get("Insider", "—"),
                    "position": row.get("Position", "—"),
                    "kind": kind, "shares": row.get("Shares", 0),
                    "value": row.get("Value", 0) or 0,
                    "text": row.get("Text", "—"),
                })
            except Exception:
                continue

        if rows_3mo:
            buys = [r for r in rows_3mo if r["kind"] == "buy"]
            sells = [r for r in rows_3mo if r["kind"] == "sell"]
            buy_val = sum(r["value"] or 0 for r in buys)
            sell_val = sum(r["value"] or 0 for r in sells)

            sum_col1, sum_col2, sum_col3 = st.columns(3)
            sum_col1.markdown(
                tile(
                    "ซื้อใน 6 เดือน",
                    f"{len(buys)} ครั้ง · ${buy_val/1e6:.1f}M" if buy_val else f"{len(buys)} ครั้ง",
                ),
                unsafe_allow_html=True,
            )
            sum_col2.markdown(
                tile(
                    "ขายใน 6 เดือน",
                    f"{len(sells)} ครั้ง · ${sell_val/1e6:.1f}M" if sell_val else f"{len(sells)} ครั้ง",
                ),
                unsafe_allow_html=True,
            )
            net = buy_val - sell_val
            signal = "🟢 Net buying" if net > 0 else ("🔴 Net selling" if net < 0 else "⚪ Balanced")
            sum_col3.markdown(tile("สรุป", signal), unsafe_allow_html=True)

            st.write("")
            with st.expander(f"ดูรายการทั้งหมด {len(rows_3mo)} รายการ (6 เดือนล่าสุด)"):
                tbl = pd.DataFrame([
                    {
                        "วันที่": r["date"],
                        "ประเภท": {"buy": "🟢 ซื้อ", "sell": "🔴 ขาย", "gift": "🎁 โอน", "other": "—"}[r["kind"]],
                        "ผู้บริหาร": r["name"],
                        "ตำแหน่ง": r["position"],
                        "จำนวนหุ้น": f"{int(r['shares']):,}" if r["shares"] else "—",
                        "มูลค่า": f"${r['value']:,.0f}" if r["value"] else "—",
                        "รายละเอียด": r["text"][:60],
                    } for r in rows_3mo
                ])
                st.dataframe(tbl, use_container_width=True, hide_index=True)

    # === Earnings ===
    ed = load_earnings_dates(ticker)
    if not ed.empty:
        st.write("")
        st.subheader("📅 ผลประกอบการ (Earnings)")
        now = pd.Timestamp.now(tz="UTC")
        ed_indexed = ed.copy()
        # Try to find the next upcoming (NaN Reported EPS = future)
        future = ed_indexed[ed_indexed["Reported EPS"].isna()]
        past = ed_indexed[ed_indexed["Reported EPS"].notna()].head(4)

        ec1, ec2 = st.columns([1, 2])
        with ec1:
            if not future.empty:
                next_date = future.index[-1]  # closest future
                try:
                    days_left = (next_date.tz_convert("UTC") - now).days
                except Exception:
                    days_left = None
                est = future.iloc[-1].get("EPS Estimate")
                st.markdown(f"**📅 ประกาศผลครั้งต่อไป**")
                st.markdown(f"### {next_date.strftime('%d %b %Y')}")
                if days_left is not None and days_left >= 0:
                    st.caption(f"อีก {days_left} วัน")
                if est and not pd.isna(est):
                    st.caption(f"คาดการณ์ EPS: **{est:.2f}**")
            else:
                st.caption("ไม่มีข้อมูลวันประกาศครั้งต่อไป")
        with ec2:
            if not past.empty:
                st.markdown("**ผลย้อนหลัง 4 ไตรมาส**")
                hist_rows = []
                beats = 0
                for date, row in past.iterrows():
                    surprise = row.get("Surprise(%)")
                    if surprise is None or pd.isna(surprise):
                        continue
                    icon = "🟢" if surprise > 0 else ("🔴" if surprise < 0 else "⚪")
                    if surprise > 0: beats += 1
                    hist_rows.append({
                        "ไตรมาส": date.strftime("%Y-%m"),
                        "Estimate": f"{row['EPS Estimate']:.2f}" if not pd.isna(row['EPS Estimate']) else "—",
                        "Reported": f"{row['Reported EPS']:.2f}",
                        "Surprise": f"{icon} {surprise:+.1f}%",
                    })
                if hist_rows:
                    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
                    streak = f"{beats}/{len(hist_rows)}"
                    st.caption(f"📊 ชนะ Estimate {streak} ไตรมาส")

    # === Analyst upgrades / downgrades ===
    ud = load_upgrades(ticker)
    if not ud.empty:
        st.write("")
        st.subheader("📈 การปรับเป้าราคา / เรตติ้งโดยนักวิเคราะห์")
        recent = ud.head(8)
        ud_rows = []
        for date, row in recent.iterrows():
            action = row.get("priceTargetAction", "")
            icon = "🟢" if "Raise" in str(action) else ("🔴" if "Lower" in str(action) else "⚪")
            from_grade = row.get("FromGrade", "")
            to_grade = row.get("ToGrade", "")
            grade_change = f"{to_grade}" if from_grade == to_grade else f"{from_grade} → {to_grade}"
            cur_pt = row.get("currentPriceTarget")
            prior_pt = row.get("priorPriceTarget")
            pt_str = f"${cur_pt:.0f}" if cur_pt and not pd.isna(cur_pt) else "—"
            if prior_pt and cur_pt and prior_pt != cur_pt and not pd.isna(prior_pt):
                pt_str = f"${prior_pt:.0f} → ${cur_pt:.0f}"
            try:
                date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            except Exception:
                date_str = str(date)
            ud_rows.append({
                "วันที่": date_str,
                "สถาบัน": row.get("Firm", "—"),
                "การกระทำ": f"{icon} {action or '—'}",
                "เรตติ้ง": grade_change,
                "เป้าราคา": pt_str,
            })
        st.dataframe(pd.DataFrame(ud_rows), use_container_width=True, hide_index=True)

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

        # AI summary
        api_key = st.session_state.get("anthropic_key", "")
        if api_key:
            ai_col1, ai_col2 = st.columns([3, 1])
            ai_col1.caption("💡 ให้ AI วิเคราะห์ข่าวทั้งหมดและสรุปเป็นภาษาไทย")
            if ai_col2.button("🤖 สรุปด้วย AI", use_container_width=True, key="ai_summary_btn"):
                with st.spinner("AI กำลังอ่านและสรุป…"):
                    # Build payload from first 15 news
                    items = []
                    for n in news_list[:15]:
                        ts = n["when"].strftime("%Y-%m-%d") if n.get("when") else "—"
                        items.append(f"- [{ts}] {n.get('title', '')}: {(n.get('summary') or '')[:300]}")
                    payload = "\n".join(items)
                    # Signature based on titles
                    import hashlib
                    sig = hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]
                    summary = ai_summarize_news(api_key, ticker, sig, payload)
                if summary:
                    with st.container(border=True):
                        st.markdown("**🤖 AI วิเคราะห์ข่าว** (Claude Haiku)")
                        st.markdown(summary)

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

