# Stockscope — Handoff Document
Complete context for the next AI to continue this project.

---

## 🎯 Project Overview

**Stockscope** — Real-time stock analysis dashboard for Thai retail investors.
Built with Streamlit + Python. Covers US stocks, Thai SET, crypto, ETFs, indices.

**Owner**: Thai user, non-technical/beginner developer.
**Target users**: Thai retail investors who want an all-in-one analysis tool in Thai language.
**Business model**: Free public dashboard now → plans for membership system later.

---

## 🔗 Links

- **Live app**: https://stockscopeproinw.streamlit.app
- **GitHub repo**: https://github.com/txphichatwanich19-rgb/stockscope (PUBLIC — free Streamlit Cloud requires it)
- **Hosting**: Streamlit Community Cloud (free tier)
- **Local dev path**: `C:\Users\NBODT\Desktop\หุ้น\`

---

## 📁 File Structure

```
หุ้น/
├── app.py                    # Main Streamlit app (~3500+ lines, single file)
├── requirements.txt          # Python dependencies
├── README.md                 # Basic setup docs
├── HANDOFF.md                # This file
├── run.bat                   # Windows quick-launch
├── make_promo.py             # Promo image generator (PIL)
├── promo_square.png          # IG post promo
├── promo_story.png           # IG/TikTok story promo
├── promo_banner.png          # FB/Twitter banner promo
├── .streamlit/
│   └── config.toml           # Streamlit theme + config
├── .gitignore                # excludes .env, .claude, __pycache__
└── streamlit.log             # local run logs (ignored)
```

---

## 🛠 Tech Stack

- **Frontend/UI**: Streamlit 1.56+
- **Charts**: TradingView Advanced Chart Widget (embedded via components.html)
- **Data**: yfinance (free, no API key — Yahoo Finance)
- **Translation**: deep-translator (free, Google Translate scrape)
- **Sentiment**: vaderSentiment (offline, English)
- **AI (optional)**: anthropic SDK — Claude Haiku 4.5 (user provides own key)
- **Fonts**: Inter + Anuphan (Thai) + JetBrains Mono (numbers)
- **Python**: 3.14 local, Streamlit Cloud uses 3.12

---

## ✅ Features (all implemented)

### Pages (sidebar nav)
1. **🌐 ภาพรวมตลาด** (Market Overview) — default landing
   - Macro bar: S&P 500 · Nasdaq · VIX · 10Y Yield · SET Index · USD/THB
   - Sector Heatmap (11 US sector ETFs)
   - Top Movers today (gainers/losers)
   - Pre-Market / After-Hours movers
   - Watchlist summary
2. **📊 ดูหุ้น** (Stock Detail) — per-ticker view
   - Hero card (name, price, change, market session badge, extended hours)
   - Stat tiles (Volume, Range, Market Cap, 52W)
   - 4 tabs: Chart / Stats / News / Signal
3. **🔍 สแกนหุ้น** (Screener) — filter stocks by P/E, growth, RSI, etc.

### Tabs (in ดูหุ้น page)
- **📊 กราฟเทคนิค**: TradingView widget + Price Ladder (S/R zones with STATUS: TESTED/BROKEN/APPROACHING/PENDING)
- **📋 สถิติ**: 52W gauge, Risk Metrics, Quarterly financials chart, Officers/CEO, Insider transactions, Analyst ratings, Earnings history, About company
- **📰 ข่าว**: Yahoo Finance news + Thai translation + VADER sentiment + optional AI summary (Claude)
- **🎯 สัญญาณสรุป**: Bull/Bear verdict, Buy Zones 1/2 + Stop Loss + Sell Zones 1/2/3 (with status), Risk/Reward, all-levels expander

### Sidebar
- Page nav (radio)
- Ticker search (empty on market page, filled on stock page)
- Watchlist quick links
- Categories (28 total: 14 sector + 14 country/theme picks including auto-screeners)
- Timeframe (period/interval)
- Options (translate news, AI key input)
- Refresh button

### Categories (in dropdown)
**Static sectors** (14): Mega Cap, Tech, Cloud/SaaS, Cyber, Semi, Finance, Healthcare, Biotech, Defense, Energy, Clean Energy, Gold/Mining, REIT, Telecom, Consumer, Food, Casino, Auto/EV, Media/Games, Airlines, Crypto, Indices, ETFs, China ADR, Japan ADR, India ADR, Europe ADR, Thai SET

**Auto-screener picks** (6, marked with 💡):
- 🌱 หุ้นเล็กน่าเติบโต (auto: small-mid cap growth)
- 🚀 หุ้นอนาคตไกล (auto: megatrend leaders + fundamentals filter)
- 🎯 แตะโซนซื้อแล้ว (auto: within ±2.5% of S1)
- 🚦 หุ้นซิ่ง (auto: top momentum 1W)
- ⚠️ ห้ามไปยุ่งตอนนี้ (auto: RSI>75 or broken trend)
- 🎰 Option Plays (auto: liquid options + volatility)
- 💎 หุ้นปันผลสูง (auto: yield >3%, sustainable payout)
- 💸 Insider Buying (auto: CEO/officers buying in 6mo)
- 📉 หุ้น Oversold (auto: RSI <35)

### Data caching (TTLs)
- `load_history`: 60s
- `load_info`: 300s
- `load_news`: 300s
- `load_macro`: 300s
- `load_mini_batch`: 600s (multi-ticker screener data)
- `load_info_batch`: 3600s (fundamentals, uses ThreadPoolExecutor)
- `load_insider_trades`, `load_earnings_history`: 900s
- `load_quarterly_financials`: 3600s
- `translate_th`: 3600s
- `load_premarket_movers`: 60s
- `load_daily_movers`: 180s
- `ai_summarize_news`: 3600s
- **Parallel pre-warm** on ticker load — fires 6 fetches concurrently

---

## 🎨 Current UI/UX Style: "Professional Dark Flat"

**IMPORTANT — the user has strong opinions on this, learned through many iterations:**

### DO ✅
- **Dark background**: `#0b0f1a` (near-black navy)
- **Cards**: solid `#141a26`, border `#1e2532`
- **Text**: `#f1f5f9` primary, `#94a3b8` secondary, `#64748b` muted
- **Accent**: `#3b82f6` (professional blue, single)
- **Up (green)**: `#10b981` (emerald)
- **Down (red)**: `#ef4444`
- **Small corner radius** (4-8px)
- **Left-border color coding** for cards
- **Bloomberg-style tab underlines**
- **Inter** (Latin) + **Anuphan** (Thai) + **JetBrains Mono** (numbers)
- **Monospace tabular-nums** for all prices
- **UPPERCASE English** labels for zone status (TESTED/BROKEN/etc)
- **Flat solid backgrounds** — no gradients on cards
- **Semantic color usage only** (up/down state, not decorative)

### DON'T ❌
- ~~Chartreuse/lime green neon~~
- ~~Aurora gradient backgrounds~~
- ~~Glassmorphism / backdrop-filter blur~~
- ~~Glow shadows~~
- ~~Serif/decorative fonts~~ (Playfair, Instrument Serif, Sora)
- ~~Gradient text~~
- ~~Pulse/glow animations~~
- ~~Large corner radius (16px+)~~
- ~~Purple/pink accents~~
- ~~Emoji as primary UI element (only as status indicators)~~

### User's aesthetic keywords (from feedback)
- ✅ "Professional", "Bloomberg style", "clean", "premium"
- ❌ "เหมือนเด็กเล่น" (childish), "ฉูดฉาด" (flashy), "เชย" (dated), "ไม่เริ่ด" (not chic)

**Reference apps user liked**: Bloomberg Terminal, Interactive Brokers Pro, Public.com

---

## 🚫 Things User Doesn't Want (Explicitly Rejected)

1. **Paid AI features** — user cancelled AI CEO analysis when reminded it costs API tokens
   - AI news summary is kept (optional, requires user's own key)
2. **Login/auth system for now** — "เอาไว้ก่อน" (leave for later)
3. **Migrating hosting to paid** ($7 Render/Railway) — considered but not decided yet
4. **Neon/flashy design** — repeatedly rejected
5. **Complex signup flows** — wants dashboard to stay accessible without registration

---

## 🎯 Recent Feature: Zone Status Tracking

**Latest big feature** (2026-09-16, commit 357d46d):

Each S/R zone now has a **status** based on 30-day price history:
- **TESTED · Nd** — price low/high touched zone N days ago, still holds
- **BROKEN · Nd** — price closed beyond zone (broken)
- **APPROACHING** — current price within 3% of zone
- **PENDING** — waiting for price to reach zone

Colors:
- TESTED support/BROKEN resistance = green `#10b981`
- BROKEN support/TESTED resistance = red `#ef4444`
- APPROACHING = amber `#f59e0b`
- PENDING = slate `#64748b`

Shown in:
- Level tiles in Signal tab (`.zone-status` badge)
- Price Ladder pills in Chart tab (`.lv-status` inline tag)

Logic in `zone_status(df, price, kind, lookback=30, tol=0.008)` function.

---

## 📝 Common Commands

### Local dev
```bash
cd "C:\Users\NBODT\Desktop\หุ้น"
python -m streamlit run app.py
# or double-click run.bat
```

### Deploy
```bash
git add -A
git commit -m "your message"
git push
# Streamlit Cloud auto-redeploys in ~2 min
```

### Regenerate promo images
```bash
python make_promo.py
```

### Emergency test
```bash
python -c "import ast; ast.parse(open('app.py',encoding='utf-8').read()); print('syntax OK')"
```

---

## 🐛 Known Issues / Watch-outs

1. **Streamlit Cloud sleep**: free tier sleeps after 15min inactivity → first user gets slow load (30-60s)
2. **yfinance rate limits**: heavy scanning may throttle → mitigated with caching + `threads=False` on downloads
3. **deep-translator fragility**: scrapes Google Translate — may break with heavy use
4. **Material Icons rendering**: had bug where icon names showed as text (e.g., "arrow_right") — fixed by explicit font import + CSS protection selectors
5. **TradingView widget doesn't allow custom line overlays** — that's why we use Price Ladder pills above the chart instead
6. **Session state resets on browser close** — no persistent user data (watchlist etc. lost)
7. **Thai text with PIL** requires Leelawadee UI font (Windows) — used in `make_promo.py`

---

## 🚀 Roadmap (Discussed But Not Built)

- **Membership system** (user mentioned wanting later)
  - Preferred path: Supabase Auth ($0 to start)
  - Would enable: persistent watchlist, per-user notes, admin analysis broadcast
- **Admin analyst mode** — admin publishes S/R zones + commentary, users see read-only
  - Discussed 3 approaches; user was considering "Option 3 hybrid" (JSON in repo + admin panel)
- **Migrate to private hosting** — Render ($7/mo) or Railway (~$5) to hide source code
  - User asked about this but hasn't decided
- **Save drawings on chart** — currently users must login to TradingView themselves to persist
  - Alternative: rebuild with Lightweight Charts (Apache 2.0) — 3-5 days work
- **Commercial licensing cleanup** — currently uses TradingView widget + yfinance + Google Translate (all restricted for commercial use). Need paid alternatives if selling.

---

## 💬 Communication Style with User

- **Language**: Thai (user is Thai). Uses casual/informal register.
- **Prefers**: Concise, direct explanations with tradeoffs. Bullet lists. Tables for comparison.
- **Dislikes**: Long-winded explanations, over-engineering, being upsold on paid stuff
- **Iteration pattern**: User will look at result → give brief feedback ("ยัง...", "ไม่...", "อยาก...") → expect immediate iteration
- **Screenshots**: User often sends screenshots to point out issues instead of describing in words

---

## 🔑 Secrets & Config

- **No secrets committed to repo** ✅
- **User's Anthropic API key**: stored only in browser session state (never on server)
- **`.streamlit/secrets.toml`**: not used (feature commented out)
- **`.streamlit/config.toml`**: theme config (base=dark, primary=#3b82f6, bg=#0b0f1a)

---

## 📋 Handoff Prompt for New AI

Copy this to a new AI session to onboard them instantly:

> I have a Streamlit stock dashboard project called "Stockscope" at
> `C:\Users\NBODT\Desktop\หุ้น\`. It's deployed at
> https://stockscopeproinw.streamlit.app and lives on GitHub at
> https://github.com/txphichatwanich19-rgb/stockscope.
>
> Please read `HANDOFF.md` in the project root first — it contains full context
> including tech stack, features built, UI style guide, what the user rejects,
> and known issues.
>
> The main file is `app.py` (~3500 lines, single-file Streamlit app).
> Everything is in Thai. User prefers professional Bloomberg-style dark UI.
> User doesn't want to spend money on API calls.
>
> After reading HANDOFF.md, ask me what I want to work on next.

---

_Last updated: 2026-09-16 · Last commit: `adfad2b` (Remove AI CEO feature)_
_Written by: Claude (Claude Code session)_
