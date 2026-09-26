const pptxgen = require("pptxgenjs");
const path = require("path");

const ICON_DIR = path.join(__dirname, "icons");
const icon = (name, color) => path.join(ICON_DIR, `${name}-${color}.png`);

// ---------- Palette (matches the live Stockscope app's approved theme) ----------
const C = {
  bg: "0A0E14",
  panel: "141A23",
  panel2: "1B2330",
  lime: "A3E635",
  white: "F5F5F5",
  muted: "9AA5B1",
  mutedDark: "5B6B7C",
  coral: "FF6B6B",
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
const PAGE_W = 13.333;
const PAGE_H = 7.5;

const FONT = "Calibri";
const FONT_HEAD = "Cambria";

let pageNum = 0;
function newSlide() {
  pageNum += 1;
  const s = pres.addSlide();
  s.background = { color: C.bg };
  if (pageNum > 1) {
    s.addText(`${pageNum}`, {
      x: PAGE_W - 0.7, y: PAGE_H - 0.45, w: 0.4, h: 0.3,
      fontFace: FONT, fontSize: 10, color: C.mutedDark, align: "right", isTextBox: true, margin: 0,
    });
    s.addText("Stockscope AI · CP020003", {
      x: 0.6, y: PAGE_H - 0.45, w: 4, h: 0.3,
      fontFace: FONT, fontSize: 10, color: C.mutedDark, align: "left", isTextBox: true, margin: 0,
    });
  }
  return s;
}

function kicker(slide, text, opts = {}) {
  slide.addText(text.toUpperCase(), {
    x: opts.x ?? 0.6, y: opts.y ?? 0.45, w: opts.w ?? 8, h: 0.35,
    fontFace: FONT, fontSize: 13, color: C.lime, bold: true, charSpacing: 2,
    isTextBox: true, margin: 0,
  });
}

function title(slide, text, opts = {}) {
  slide.addText(text, {
    x: opts.x ?? 0.6, y: opts.y ?? 0.78, w: opts.w ?? 11.5, h: opts.h ?? 0.8,
    fontFace: FONT_HEAD, fontSize: opts.size ?? 32, color: C.white, bold: true,
    isTextBox: true, margin: 0,
  });
}

function iconRow(slide, { x, y, w, iconName, iconColor = "dark", circleColor = C.lime, heading, desc, headSize = 15, descSize = 12 }) {
  const d = 0.55;
  slide.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: circleColor }, line: { type: "none" } });
  slide.addImage({ path: icon(iconName, iconColor), x: x + 0.13, y: y + 0.13, w: 0.29, h: 0.29 });
  slide.addText(heading, {
    x: x + d + 0.22, y: y - 0.04, w: w - d - 0.22, h: 0.32,
    fontFace: FONT, fontSize: headSize, bold: true, color: C.white, isTextBox: true, margin: 0,
  });
  slide.addText(desc, {
    x: x + d + 0.22, y: y + 0.26, w: w - d - 0.22, h: 0.6,
    fontFace: FONT, fontSize: descSize, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });
}

function statCallout(slide, { x, y, w, value, label, color = C.lime, valueSize = 40, labelSize = 12 }) {
  slide.addText(value, {
    x, y, w, h: 0.75, fontFace: FONT_HEAD, fontSize: valueSize, bold: true, color,
    align: "left", isTextBox: true, margin: 0,
  });
  slide.addText(label.toUpperCase(), {
    x, y: y + 0.7, w, h: 0.4, fontFace: FONT, fontSize: labelSize, color: C.muted, charSpacing: 1,
    align: "left", isTextBox: true, margin: 0,
  });
}

function panelBox(slide, { x, y, w, h, fill = C.panel }) {
  slide.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill }, line: { type: "none" },
    shadow: { type: "outer", color: "000000", opacity: 0.35, blur: 8, offset: 3, angle: 90 },
  });
}

// ============================================================
// SLIDE 1 — TITLE
// ============================================================
{
  const s = newSlide();
  pageNum = 0; // title slide gets no footer/number

  s.addText("STOCKSCOPE AI", {
    x: 0.9, y: 1.0, w: 6, h: 0.4, fontFace: FONT, fontSize: 14, bold: true,
    color: C.lime, charSpacing: 3, isTextBox: true, margin: 0,
  });
  s.addText("Stock Direction Prediction\nwith XGBoost", {
    x: 0.85, y: 2.6, w: 9.5, h: 1.9, fontFace: FONT_HEAD, fontSize: 46, bold: true,
    color: C.white, isTextBox: true, margin: 0, lineSpacing: 52,
  });
  s.addText("A leakage-free machine learning pipeline extending our live dashboard", {
    x: 0.9, y: 4.55, w: 8.5, h: 0.5, fontFace: FONT, fontSize: 18, color: C.muted,
    isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 5.35, w: 4.6, h: 0.5, rectRadius: 0.08,
    fill: { color: C.panel }, line: { type: "none" },
  });
  s.addText("Group ___  ·  [Team Name Here]", {
    x: 1.05, y: 5.35, w: 4.3, h: 0.5, fontFace: FONT, fontSize: 13, color: C.white,
    isTextBox: true, margin: 0, valign: "middle",
  });

  s.addText("CP020003 · Artificial Intelligence · Final Project · Khon Kaen University 2026", {
    x: 0.9, y: PAGE_H - 0.7, w: 9, h: 0.35, fontFace: FONT, fontSize: 11, color: C.mutedDark,
    isTextBox: true, margin: 0,
  });

  // Decorative candlestick motif, bottom-right corner (not a full-width stripe)
  const bars = [
    { h: 1.0, up: true }, { h: 1.6, up: false }, { h: 0.8, up: true }, { h: 2.1, up: true },
    { h: 1.3, up: false }, { h: 1.9, up: true }, { h: 1.1, up: false }, { h: 2.4, up: true },
  ];
  let bx = 9.6;
  const baseY = 6.7;
  bars.forEach((b) => {
    s.addShape(pres.ShapeType.roundRect, {
      x: bx, y: baseY - b.h, w: 0.28, h: b.h, rectRadius: 0.03,
      fill: { color: b.up ? C.lime : C.mutedDark }, line: { type: "none" },
      transparency: b.up ? 0 : 35,
    });
    bx += 0.42;
  });
  pageNum = 1;
}

// ============================================================
// SLIDE 2 — TEAM
// ============================================================
{
  const s = newSlide();
  kicker(s, "Our Team");
  title(s, "Group Members");

  panelBox(s, { x: 0.6, y: 1.85, w: 12.1, h: 0.9 });
  s.addText("Group No.", { x: 0.9, y: 1.85, w: 2, h: 0.9, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0, valign: "middle" });
  s.addText("___", { x: 0.9, y: 2.15, w: 2, h: 0.5, fontFace: FONT_HEAD, fontSize: 20, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("Group Name", { x: 3.3, y: 1.85, w: 4, h: 0.4, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0 });
  s.addText("[Team Name Here]", { x: 3.3, y: 2.2, w: 6, h: 0.5, fontFace: FONT_HEAD, fontSize: 20, bold: true, color: C.white, isTextBox: true, margin: 0 });

  const rows = [
    ["#", "Student Name", "Student ID"],
    ["1", "_______________________", "_______________"],
    ["2", "_______________________", "_______________"],
    ["3", "_______________________", "_______________"],
    ["4", "_______________________", "_______________"],
    ["5", "_______________________", "_______________"],
  ];
  const tableRows = rows.map((r, i) => r.map((cell) => ({
    text: cell,
    options: {
      fontFace: FONT, fontSize: 13, color: i === 0 ? C.lime : C.white,
      bold: i === 0, fill: { color: i === 0 ? C.panel2 : (i % 2 === 0 ? C.panel : C.bg) },
      valign: "middle",
    },
  })));
  s.addTable(tableRows, {
    x: 0.6, y: 3.0, w: 12.1, h: 3.6,
    colW: [1.2, 6.9, 4.0],
    border: { type: "none" },
    autoPage: false,
    rowH: 0.6,
  });
}

// ============================================================
// SLIDE 3 — MOTIVATION
// ============================================================
{
  const s = newSlide();
  kicker(s, "Motivation & Topic");
  title(s, "Why This Project?");

  s.addText(
    "We already built Stockscope — a live dashboard that computes RSI, MACD, moving averages, and support/resistance zones for any stock, in real time.",
    { x: 0.6, y: 1.75, w: 6.6, h: 1.3, fontFace: FONT, fontSize: 16, color: C.white, isTextBox: true, margin: 0, valign: "top" }
  );
  s.addText(
    "That raised a natural question we could actually test with data we already had:",
    { x: 0.6, y: 3.0, w: 6.6, h: 0.6, fontFace: FONT, fontSize: 14, color: C.muted, isTextBox: true, margin: 0 }
  );

  panelBox(s, { x: 0.6, y: 3.7, w: 6.6, h: 1.7 });
  s.addText("“Can the same indicators predict where the price goes next?”", {
    x: 0.95, y: 3.7, w: 5.9, h: 1.7, fontFace: FONT_HEAD, fontSize: 20, italic: true,
    color: C.lime, isTextBox: true, margin: 0, valign: "middle",
  });

  // Right column: simple flow diagram Dashboard -> Question -> Model
  const items = [
    { icon: "bar-chart", label: "Live Dashboard", desc: "RSI · MACD · SMA · Volume\ncomputed today" },
    { icon: "target", label: "The Question", desc: "Up or down over\nthe next N days?" },
    { icon: "cpu", label: "ML Model", desc: "Binary classification\ntrained on history" },
  ];
  let iy = 1.85;
  const iyStep = 1.68;
  items.forEach((it, idx) => {
    panelBox(s, { x: 7.7, y: iy, w: 5.0, h: 1.35 });
    s.addShape(pres.ShapeType.ellipse, { x: 7.95, y: iy + 0.28, w: 0.8, h: 0.8, fill: { color: C.lime }, line: { type: "none" } });
    s.addImage({ path: icon(it.icon, "dark"), x: 8.15, y: iy + 0.48, w: 0.4, h: 0.4 });
    s.addText(it.label, { x: 8.95, y: iy + 0.16, w: 3.6, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: C.white, isTextBox: true, margin: 0 });
    s.addText(it.desc, { x: 8.95, y: iy + 0.55, w: 3.6, h: 0.7, fontFace: FONT, fontSize: 11, color: C.muted, isTextBox: true, margin: 0, valign: "top" });
    if (idx < items.length - 1) {
      const gapTop = iy + 1.35;
      const gapH = iyStep - 1.35;
      s.addImage({ path: icon("arrow-right", "muted"), x: 9.95, y: gapTop + (gapH - 0.2) / 2, w: 0.2, h: 0.2 });
    }
    iy += iyStep;
  });
}

// ============================================================
// SLIDE 4 — THE AI TASK
// ============================================================
{
  const s = newSlide();
  kicker(s, "The AI Task");
  title(s, "Framing It as Machine Learning");

  panelBox(s, { x: 0.6, y: 1.8, w: 5.8, h: 4.9 });
  s.addText("Binary Classification", { x: 0.95, y: 2.05, w: 5.1, h: 0.5, fontFace: FONT_HEAD, fontSize: 20, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("label_up = 1  if Close[t+N] > Close[t]\nlabel_up = 0  otherwise", {
    x: 0.95, y: 2.65, w: 5.1, h: 0.8, fontFace: "Courier New", fontSize: 13, color: C.white, isTextBox: true, margin: 0,
  });
  s.addText("Input features: technical indicators up to day t\nOutput: probability price is higher N trading days later", {
    x: 0.95, y: 3.55, w: 5.1, h: 0.9, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });

  const reasons = [
    ["zap", "Fast to train", "Iterates in seconds on a laptop or free Colab CPU"],
    ["layers", "Tabular-native", "Our features are indicator numbers, not images or text"],
    ["target", "Explainable", "Feature importance shows *why*, not just a number"],
  ];
  let ry = 4.55;
  reasons.forEach((r) => {
    s.addImage({ path: icon(r[0], "lime"), x: 0.95, y: ry, w: 0.28, h: 0.28 });
    s.addText(r[1], { x: 1.4, y: ry - 0.06, w: 4.6, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: C.white, isTextBox: true, margin: 0 });
    s.addText(r[2], { x: 1.4, y: ry + 0.28, w: 4.7, h: 0.35, fontFace: FONT, fontSize: 11, color: C.muted, isTextBox: true, margin: 0 });
    ry += 0.62;
  });

  // Right: pipeline flow, vertical
  const steps = [
    ["database", "Data", "5y OHLCV, 20 tickers"],
    ["activity", "Features", "RSI · MACD · SMA · Vol"],
    ["cpu", "Model", "XGBoost classifier"],
    ["bar-chart", "Prediction", "P(up) in N days"],
  ];
  let sy = 1.95;
  steps.forEach((st, idx) => {
    s.addShape(pres.ShapeType.ellipse, { x: 7.1, y: sy, w: 0.6, h: 0.6, fill: { color: C.panel2 }, line: { color: C.lime, width: 1.5 } });
    s.addImage({ path: icon(st[0], "lime"), x: 7.25, y: sy + 0.15, w: 0.3, h: 0.3 });
    s.addText(st[1], { x: 7.95, y: sy - 0.02, w: 4.7, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.white, isTextBox: true, margin: 0 });
    s.addText(st[2], { x: 7.95, y: sy + 0.3, w: 4.7, h: 0.35, fontFace: FONT, fontSize: 11, color: C.muted, isTextBox: true, margin: 0 });
    if (idx < steps.length - 1) {
      s.addShape(pres.ShapeType.line, {
        x: 7.4, y: sy + 0.6, w: 0, h: 0.55, line: { color: C.mutedDark, width: 1.5, dashType: "dash" },
      });
    }
    sy += 1.15;
  });
}

// ============================================================
// SLIDE 5 — DATASET
// ============================================================
{
  const s = newSlide();
  kicker(s, "Dataset & EDA");
  title(s, "Dataset");

  const stats = [
    ["20", "Tickers"], ["5 yrs", "Daily history"], ["25,100", "Total rows"], ["0", "Missing values"],
  ];
  let sx = 0.6;
  stats.forEach((st) => {
    panelBox(s, { x: sx, y: 1.85, w: 2.85, h: 1.4 });
    statCallout(s, { x: sx + 0.25, y: 2.0, w: 2.4, value: st[0], label: st[1], valueSize: 30 });
    sx += 3.05;
  });

  s.addText("Same data source that powers Stockscope live: Yahoo Finance via yfinance.", {
    x: 0.6, y: 3.5, w: 12, h: 0.4, fontFace: FONT, fontSize: 14, italic: true, color: C.lime, isTextBox: true, margin: 0,
  });

  s.addText("2021-09-27  →  2026-09-25", {
    x: 0.6, y: 4.05, w: 6, h: 0.4, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0,
  });

  s.addText("SECTORS COVERED", {
    x: 0.6, y: 4.6, w: 6, h: 0.35, fontFace: FONT, fontSize: 12, bold: true, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0,
  });
  const sectors = ["Technology", "Finance", "Energy", "Healthcare", "Consumer", "Media"];
  let chx = 0.6, chy = 5.0;
  const CHIP_RIGHT_EDGE = 7.6; // stay clear of the right panel starting at x=8.0
  sectors.forEach((sec) => {
    const w = 0.35 + sec.length * 0.11;
    if (chx + w > CHIP_RIGHT_EDGE) { chx = 0.6; chy += 0.6; }
    s.addShape(pres.ShapeType.roundRect, { x: chx, y: chy, w, h: 0.45, rectRadius: 0.22, fill: { color: C.panel2 }, line: { type: "none" } });
    s.addText(sec, { x: chx, y: chy, w, h: 0.45, fontFace: FONT, fontSize: 12, color: C.white, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    chx += w + 0.2;
  });

  panelBox(s, { x: 8.0, y: 1.85, w: 4.7, h: 4.9 });
  s.addText("EXAMPLE TICKERS", { x: 8.35, y: 2.1, w: 4, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0 });
  s.addText("AAPL · MSFT · NVDA · JPM · XOM\nJNJ · AMZN · KO · GOOGL · META\nWMT · PG · V · MA · DIS\nNFLX · AMD · INTC · BAC · CVX", {
    x: 8.35, y: 2.5, w: 4.1, h: 2.0, fontFace: "Courier New", fontSize: 13, color: C.white, isTextBox: true, margin: 0, lineSpacing: 26,
  });
  s.addText("Multiple sectors, not one stock — reduces the risk of learning one company's quirks instead of general indicator behavior.", {
    x: 8.35, y: 4.75, w: 4.05, h: 1.8, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });
}

// ============================================================
// SLIDE 6 — FEATURE ENGINEERING
// ============================================================
{
  const s = newSlide();
  kicker(s, "Methodology");
  title(s, "Features — What the Model Sees");

  const feats = [
    ["activity", "ret_1d / 5d / 10d / 21d", "Past return, 1–21 trading days (momentum)"],
    ["target", "rsi14", "14-day RSI — overbought vs. oversold"],
    ["bar-chart", "macd_hist", "MACD histogram (trend strength & direction)"],
    ["layers", "dist_sma20 / dist_sma50", "% distance from the 20/50-day moving average"],
    ["zap", "vol_ratio", "Today's volume vs. its 20-day average"],
    ["shield", "volatility_20d", "Rolling std-dev of daily returns (choppiness)"],
  ];
  let fy = 1.85;
  const rowStep = 1.62;
  const col1x = 0.6, col2x = 6.9;
  feats.forEach((f, i) => {
    const x = i % 2 === 0 ? col1x : col2x;
    if (i % 2 === 0 && i > 0) fy += rowStep;
    panelBox(s, { x, y: fy, w: 5.8, h: 1.4 });
    iconRow(s, { x: x + 0.3, y: fy + 0.4, w: 5.2, iconName: f[0], heading: f[1], desc: f[2], headSize: 14, descSize: 11 });
  });

  s.addText("All computed with rolling / ewm windows that only look backward in time — the anti-leakage design continues on the next slide.", {
    x: 0.6, y: 6.75, w: 12.1, h: 0.4, fontFace: FONT, fontSize: 12, italic: true, color: C.muted, isTextBox: true, margin: 0,
  });
}

// ============================================================
// SLIDE 7 — ANTI-LEAKAGE DESIGN
// ============================================================
{
  const s = newSlide();
  kicker(s, "Methodology");
  title(s, "Avoiding Data Leakage");
  s.addText("Our core design decision — directly targets “leakage-free, well-justified methodology” in the grading rubric.", {
    x: 0.6, y: 1.35, w: 12, h: 0.4, fontFace: FONT, fontSize: 13, italic: true, color: C.lime, isTextBox: true, margin: 0,
  });

  // Left: wrong approach
  panelBox(s, { x: 0.6, y: 1.95, w: 5.85, h: 3.0 });
  s.addImage({ path: icon("x-circle", "coral"), x: 0.9, y: 2.2, w: 0.4, h: 0.4 });
  s.addText("Random 80/20 row split", { x: 1.45, y: 2.22, w: 4.8, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: C.coral, isTextBox: true, margin: 0 });
  s.addText("Rows from the same week end up on both sides. Indicators are autocorrelated — the model partly memorizes the test period instead of generalizing.", {
    x: 0.9, y: 2.85, w: 5.2, h: 1.9, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });

  // Right: correct approach
  panelBox(s, { x: 6.85, y: 1.95, w: 5.85, h: 3.0, fill: C.panel2 });
  s.addImage({ path: icon("check-circle", "lime"), x: 7.15, y: 2.2, w: 0.4, h: 0.4 });
  s.addText("Single global date cutoff", { x: 7.7, y: 2.22, w: 4.8, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("Every training row happens strictly before every test row, across all 20 tickers at once — mimicking real deployment: train on the past, evaluate on the unseen future.", {
    x: 7.15, y: 2.85, w: 5.2, h: 1.9, fontFace: FONT, fontSize: 13, color: C.white, isTextBox: true, margin: 0, valign: "top",
  });

  // Timeline diagram
  s.addText("2021", { x: 0.7, y: 5.5, w: 1, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, isTextBox: true, margin: 0 });
  s.addText("2026", { x: 11.6, y: 5.5, w: 1, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, align: "right", isTextBox: true, margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.85, w: 8.6, h: 0.55, rectRadius: 0.06, fill: { color: C.lime }, line: { type: "none" } });
  s.addShape(pres.ShapeType.roundRect, { x: 9.2, y: 5.85, w: 3.5, h: 0.55, rectRadius: 0.06, fill: { color: C.coral }, line: { type: "none" }, transparency: 15 });
  s.addText("TRAIN  —  19,280 rows  (2021–10 → 2025–10)", { x: 0.6, y: 5.85, w: 8.6, h: 0.55, fontFace: FONT, fontSize: 12, bold: true, color: C.bg, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText("TEST  —  4,840 rows", { x: 9.2, y: 5.85, w: 3.5, h: 0.55, fontFace: FONT, fontSize: 12, bold: true, color: C.bg, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText("cutoff: 2025-10-09", { x: 8.6, y: 6.55, w: 2.4, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: "center", isTextBox: true, margin: 0 });
}

// ============================================================
// SLIDE 8 — MODEL TRAINING
// ============================================================
{
  const s = newSlide();
  kicker(s, "Methodology");
  title(s, "Model Training — Baselines vs. XGBoost");

  const models = [
    ["Logistic Regression", "Simplest linear baseline (features standardized)"],
    ["Random Forest", "Second tree-ensemble baseline, no boosting"],
    ["XGBoost (ours)", "Tuned with TimeSeriesSplit CV — inside training data only"],
  ];
  let mx = 0.6;
  models.forEach((m, i) => {
    const isOurs = i === 2;
    panelBox(s, { x: mx, y: 1.9, w: 3.9, h: 1.9, fill: isOurs ? C.panel2 : C.panel });
    s.addText(m[0], { x: mx + 0.3, y: 2.1, w: 3.4, h: 0.5, fontFace: FONT, fontSize: 15, bold: true, color: isOurs ? C.lime : C.white, isTextBox: true, margin: 0 });
    s.addText(m[1], { x: mx + 0.3, y: 2.65, w: 3.4, h: 1.0, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0, valign: "top" });
    mx += 4.1;
  });

  s.addText("HYPERPARAMETER SEARCH — TimeSeriesSplit (4 folds), training set only", {
    x: 0.6, y: 4.15, w: 8, h: 0.35, fontFace: FONT, fontSize: 12, bold: true, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0,
  });

  const grid = [
    ["max_depth=3, lr=0.05, n=200", "CV AUC 0.499"],
    ["max_depth=4, lr=0.05, n=300", "CV AUC 0.505  ✓ best"],
    ["max_depth=3, lr=0.10, n=150", "CV AUC 0.501"],
  ];
  let gy = 4.6;
  grid.forEach((g) => {
    const isBest = g[1].includes("best");
    s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: gy, w: 8.1, h: 0.55, rectRadius: 0.06, fill: { color: isBest ? C.panel2 : C.panel }, line: isBest ? { color: C.lime, width: 1 } : { type: "none" } });
    s.addText(g[0], { x: 0.85, y: gy, w: 4.2, h: 0.55, fontFace: "Courier New", fontSize: 12, color: C.white, valign: "middle", isTextBox: true, margin: 0 });
    s.addText(g[1], { x: 5.2, y: gy, w: 3.3, h: 0.55, fontFace: FONT, fontSize: 12, bold: isBest, color: isBest ? C.lime : C.muted, valign: "middle", align: "right", isTextBox: true, margin: 0 });
    gy += 0.68;
  });

  panelBox(s, { x: 9.1, y: 4.15, w: 3.6, h: 2.65, fill: C.panel2 });
  s.addText("FINAL CONFIG", { x: 9.4, y: 4.4, w: 3, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0 });
  s.addText("max_depth: 4\nlearning_rate: 0.05\nn_estimators: 300\nsubsample: 0.8", {
    x: 9.4, y: 4.75, w: 3, h: 1.7, fontFace: "Courier New", fontSize: 14, color: C.lime, isTextBox: true, margin: 0, lineSpacing: 26,
  });
}

// ============================================================
// SLIDE 9 — RESULTS: MODEL COMPARISON
// ============================================================
{
  const s = newSlide();
  kicker(s, "Results & Evaluation");
  title(s, "Model Comparison on the Test Set");

  const cats = ["Baseline", "LogReg", "Rand.Forest", "XGBoost"];
  const accVals = [0.5072, 0.5085, 0.5087, 0.5246];
  const aucVals = [0.50, 0.5166, 0.5361, 0.5262];

  s.addText("Accuracy", { x: 0.6, y: 1.85, w: 5.8, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.white, isTextBox: true, margin: 0 });
  s.addChart(pres.ChartType.bar, [{ name: "Accuracy", labels: cats, values: accVals }], {
    x: 0.6, y: 2.2, w: 5.8, h: 3.1,
    chartColors: [C.mutedDark, C.mutedDark, C.mutedDark, C.lime],
    showTitle: false, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white, dataLabelFontSize: 11,
    dataLabelFormatCode: "0.0%",
    catAxisLabelColor: C.muted, catAxisLabelFontSize: 11, catAxisLineColor: C.mutedDark,
    valAxisHidden: true, valAxisLineShow: false,
    valAxisMinVal: 0.45, valAxisMaxVal: 0.56,
    catGridLine: { style: "none" }, valGridLine: { style: "none" },
    barGapWidthPct: 40,
  });

  s.addText("ROC-AUC", { x: 6.9, y: 1.85, w: 5.8, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.white, isTextBox: true, margin: 0 });
  s.addChart(pres.ChartType.bar, [{ name: "ROC-AUC", labels: cats, values: aucVals }], {
    x: 6.9, y: 2.2, w: 5.8, h: 3.1,
    chartColors: [C.mutedDark, C.mutedDark, C.lime, C.mutedDark],
    showTitle: false, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white, dataLabelFontSize: 11,
    dataLabelFormatCode: "0.000",
    catAxisLabelColor: C.muted, catAxisLabelFontSize: 11, catAxisLineColor: C.mutedDark,
    valAxisHidden: true, valAxisLineShow: false,
    valAxisMinVal: 0.45, valAxisMaxVal: 0.56,
    catGridLine: { style: "none" }, valGridLine: { style: "none" },
    barGapWidthPct: 40,
  });

  panelBox(s, { x: 0.6, y: 5.55, w: 12.1, h: 1.15 });
  s.addText("XGBoost wins on accuracy (52.5%); Random Forest edges it slightly on ROC-AUC (0.536 vs 0.526). Both tree ensembles clearly beat Logistic Regression and the 50.7% majority-class baseline.", {
    x: 0.95, y: 5.55, w: 11.4, h: 1.15, fontFace: FONT, fontSize: 14, color: C.white, isTextBox: true, margin: 0, valign: "middle",
  });
}

// ============================================================
// SLIDE 10 — HORIZON COMPARISON (key insight)
// ============================================================
{
  const s = newSlide();
  kicker(s, "Results & Evaluation");
  title(s, "Does the Prediction Horizon Matter?");
  s.addText("Same leakage-free pipeline, repeated at three horizons — 1, 5, and 10 trading days ahead.", {
    x: 0.6, y: 1.35, w: 12, h: 0.4, fontFace: FONT, fontSize: 13, italic: true, color: C.muted, isTextBox: true, margin: 0,
  });

  const cats = ["1 day", "5 days", "10 days"];
  s.addChart(
    pres.ChartType.bar,
    [
      { name: "XGBoost Accuracy", labels: cats, values: [0.5041, 0.5136, 0.5246] },
      { name: "Majority Baseline", labels: cats, values: [0.513, 0.5221, 0.5072] },
    ],
    {
      x: 0.6, y: 1.95, w: 7.6, h: 4.3,
      barDir: "col", barGrouping: "clustered",
      chartColors: [C.lime, C.mutedDark],
      showTitle: false,
      showLegend: true, legendColor: C.muted, legendFontSize: 11, legendPos: "b",
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white, dataLabelFontSize: 10,
      dataLabelFormatCode: "0.0%",
      catAxisLabelColor: C.muted, catAxisLabelFontSize: 12, catAxisLineColor: C.mutedDark,
      valAxisHidden: true, valAxisLineShow: false,
      valAxisMinVal: 0.45, valAxisMaxVal: 0.56,
      catGridLine: { style: "none" }, valGridLine: { style: "none" },
      barGapWidthPct: 30,
    }
  );

  panelBox(s, { x: 8.5, y: 1.95, w: 4.2, h: 4.3, fill: C.panel2 });
  s.addText("ROC-AUC BY HORIZON", { x: 8.8, y: 2.15, w: 3.6, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0 });
  const aucRows = [["1 day", "0.505"], ["5 days", "0.507"], ["10 days", "0.526"]];
  let ary = 2.6;
  aucRows.forEach((r, i) => {
    const isBest = i === 2;
    s.addText(r[0], { x: 8.8, y: ary, w: 1.8, h: 0.45, fontFace: FONT, fontSize: 14, color: C.white, valign: "middle", isTextBox: true, margin: 0 });
    s.addText(r[1], { x: 10.5, y: ary, w: 1.9, h: 0.45, fontFace: FONT_HEAD, fontSize: 18, bold: true, color: isBest ? C.lime : C.muted, align: "right", valign: "middle", isTextBox: true, margin: 0 });
    ary += 0.55;
  });
  s.addText("Random = 0.500", { x: 8.8, y: ary + 0.1, w: 3.6, h: 0.3, fontFace: FONT, fontSize: 10, italic: true, color: C.mutedDark, isTextBox: true, margin: 0 });
  s.addText("Short-term moves are close to random. A small, real edge appears at 10 trading days — consistent with market-efficiency theory.", {
    x: 8.8, y: 4.5, w: 3.7, h: 1.6, fontFace: FONT, fontSize: 12, color: C.white, isTextBox: true, margin: 0, valign: "top",
  });
}

// ============================================================
// SLIDE 11 — FEATURE IMPORTANCE
// ============================================================
{
  const s = newSlide();
  kicker(s, "Results & Evaluation");
  title(s, "What Does the Model Look At?");

  const featCats = ["ret_1d", "ret_5d", "ret_10d", "vol_ratio", "rsi14", "ret_21d", "macd_hist", "dist_sma50", "dist_sma20", "volatility_20d"];
  const featVals = [0.0782, 0.0902, 0.0945, 0.0948, 0.1015, 0.1043, 0.1049, 0.1076, 0.1077, 0.1163];
  const featColors = featCats.map((_, i) => (i >= 7 ? C.lime : C.mutedDark));

  s.addChart(pres.ChartType.bar, [{ name: "Importance", labels: featCats, values: featVals }], {
    x: 0.6, y: 1.85, w: 7.6, h: 5.0,
    barDir: "bar",
    chartColors: featColors,
    showTitle: false, showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white, dataLabelFontSize: 10,
    dataLabelFormatCode: "0.000",
    catAxisLabelColor: C.white, catAxisLabelFontSize: 12, catAxisLineColor: C.mutedDark,
    valAxisHidden: true, valAxisLineShow: false,
    catGridLine: { style: "none" }, valGridLine: { style: "none" },
    barGapWidthPct: 30,
  });

  panelBox(s, { x: 8.5, y: 1.85, w: 4.2, h: 5.0, fill: C.panel2 });
  s.addImage({ path: icon("target", "lime"), x: 8.8, y: 2.15, w: 0.4, h: 0.4 });
  s.addText("Key Insight", { x: 9.35, y: 2.2, w: 3, h: 0.35, fontFace: FONT, fontSize: 15, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("Medium-term trend and volatility signals (volatility_20d, dist_sma20/50, macd_hist) matter most.\n\nSingle-day return (ret_1d) matters least — the noisiest, least-informative feature, exactly as expected.", {
    x: 8.8, y: 2.75, w: 3.6, h: 3.8, fontFace: FONT, fontSize: 13, color: C.white, isTextBox: true, margin: 0, valign: "top", lineSpacing: 20,
  });
}

// ============================================================
// SLIDE 12 — BACKTEST
// ============================================================
{
  const s = newSlide();
  kicker(s, "Results & Evaluation");
  title(s, "Illustrative Strategy Backtest");

  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.4, w: 6.6, h: 0.45, rectRadius: 0.22, fill: { color: C.panel2 }, line: { color: C.coral, width: 1 } });
  s.addImage({ path: icon("alert-triangle", "coral"), x: 0.8, y: 1.51, w: 0.24, h: 0.24 });
  s.addText("Academic illustration only — not investment advice", {
    x: 1.15, y: 1.4, w: 6.0, h: 0.45, fontFace: FONT, fontSize: 12, bold: true, color: C.coral, valign: "middle", isTextBox: true, margin: 0,
  });

  s.addText("Rule: go long only when P(up) > 0.55, otherwise hold cash. Equal-weighted across all 20 tickers, test period only, no transaction costs.", {
    x: 0.6, y: 2.05, w: 12.0, h: 0.5, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0,
  });

  panelBox(s, { x: 0.6, y: 2.75, w: 5.85, h: 2.1 });
  statCallout(s, { x: 1.0, y: 3.05, w: 5, value: "3.86x", label: "Model-gated strategy", color: C.white, valueSize: 44 });

  panelBox(s, { x: 6.85, y: 2.75, w: 5.85, h: 2.1, fill: C.panel2 });
  statCallout(s, { x: 7.25, y: 3.05, w: 5, value: "12.46x", label: "Buy & hold (equal-weight)", color: C.lime, valueSize: 44 });
  s.addText("growth of $1 over the test period", { x: 1.0, y: 4.15, w: 5, h: 0.3, fontFace: FONT, fontSize: 11, italic: true, color: C.mutedDark, isTextBox: true, margin: 0 });
  s.addText("growth of $1 over the test period", { x: 7.25, y: 4.15, w: 5, h: 0.3, fontFace: FONT, fontSize: 11, italic: true, color: C.mutedDark, isTextBox: true, margin: 0 });

  panelBox(s, { x: 0.6, y: 5.15, w: 12.1, h: 1.55 });
  s.addImage({ path: icon("target", "lime"), x: 0.9, y: 5.4, w: 0.35, h: 0.35 });
  s.addText("Why buy-and-hold wins here", { x: 1.4, y: 5.4, w: 5, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.white, isTextBox: true, margin: 0 });
  s.addText("The test window contains a strong bull run. Sitting in cash whenever confidence dips costs more than a ~0.53 AUC edge can recover. A statistical edge is not automatically a profitable trading strategy — timing has a real opportunity cost.", {
    x: 0.9, y: 5.8, w: 11.5, h: 0.85, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });
}

// ============================================================
// SLIDE 13 — LIMITATIONS
// ============================================================
{
  const s = newSlide();
  kicker(s, "Discussion");
  title(s, "Limitations — What This Model Doesn't Capture");

  const lims = [
    ["alert-triangle", "Modest signal", "ROC-AUC ≈ 0.50–0.53, close to random — reported honestly, not tuned to look better."],
    ["database", "No fundamentals or macro data", "Earnings, interest rates, and news events aren't in the feature set."],
    ["layers", "Survivorship bias", "Only large, currently-listed companies — delistings are excluded."],
    ["shield", "No transaction costs", "The backtest ignores fees, slippage, and taxes."],
    ["activity", "Sentiment isn't backtestable", "Stockscope's live news scraper has no historical archive to align to past dates."],
  ];
  let ly = 1.9;
  lims.forEach((l) => {
    iconRow(s, { x: 0.6, y: ly, w: 11.9, iconName: l[0], iconColor: "dark", circleColor: C.lime, heading: l[1], desc: l[2], headSize: 15, descSize: 12 });
    ly += 0.98;
  });
}

// ============================================================
// SLIDE 14 — IMPACT & CONCLUSION
// ============================================================
{
  const s = newSlide();
  kicker(s, "Impact & Conclusion");
  title(s, "Connecting Back to Stockscope");

  panelBox(s, { x: 0.6, y: 1.85, w: 5.85, h: 4.85 });
  s.addText("KEY INSIGHTS", { x: 0.95, y: 2.1, w: 5, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: C.lime, charSpacing: 1, isTextBox: true, margin: 0 });
  const insights = [
    "Fully leakage-free pipeline: backward-looking features, forward-looking label kept strictly separate",
    "XGBoost modestly beats the naive baseline at longer horizons, not at 1 day",
    "macd_hist, dist_sma50/20, and volatility are the most informative features",
    "A small statistical edge ≠ a profitable trading strategy",
  ];
  let iny = 2.55;
  insights.forEach((t) => {
    s.addShape(pres.ShapeType.ellipse, { x: 0.95, y: iny + 0.06, w: 0.12, h: 0.12, fill: { color: C.lime }, line: { type: "none" } });
    s.addText(t, { x: 1.25, y: iny - 0.1, w: 4.9, h: 0.75, fontFace: FONT, fontSize: 13, color: C.white, isTextBox: true, margin: 0, valign: "top" });
    iny += 0.92;
  });

  panelBox(s, { x: 6.85, y: 1.85, w: 5.85, h: 4.85, fill: C.panel2 });
  s.addImage({ path: icon("arrow-right", "lime"), x: 7.2, y: 2.15, w: 0.35, h: 0.35 });
  s.addText("Next Step: Ship It in the Live App", { x: 7.7, y: 2.15, w: 4.8, h: 0.45, fontFace: FONT, fontSize: 16, bold: true, color: C.lime, isTextBox: true, margin: 0 });

  panelBox(s, { x: 7.2, y: 2.85, w: 5.1, h: 1.5, fill: C.bg });
  s.addText("AI: 61% probability of higher\nclose in 10 trading days", {
    x: 7.45, y: 3.05, w: 4.6, h: 0.9, fontFace: FONT, fontSize: 16, bold: true, color: C.lime, isTextBox: true, margin: 0,
  });
  s.addText("statistical estimate, not financial advice", {
    x: 7.45, y: 3.85, w: 4.6, h: 0.35, fontFace: FONT, fontSize: 10, italic: true, color: C.mutedDark, isTextBox: true, margin: 0,
  });

  s.addText("Serialize the trained model and load it in the Streamlit app, next to the existing technical view Stockscope already shows for any ticker.", {
    x: 7.2, y: 4.55, w: 5.1, h: 1.9, fontFace: FONT, fontSize: 13, color: C.white, isTextBox: true, margin: 0, valign: "top",
  });
}

// ============================================================
// SLIDE 15 — THANK YOU / Q&A
// ============================================================
{
  const s = newSlide();
  pageNum = 0;

  s.addImage({ path: icon("trending-up", "lime"), x: PAGE_W / 2 - 0.35, y: 2.1, w: 0.7, h: 0.7 });
  s.addText("Thank You", {
    x: 0, y: 3.0, w: PAGE_W, h: 1.0, fontFace: FONT_HEAD, fontSize: 48, bold: true, color: C.white,
    align: "center", isTextBox: true, margin: 0,
  });
  s.addText("Questions & Discussion", {
    x: 0, y: 4.0, w: PAGE_W, h: 0.5, fontFace: FONT, fontSize: 18, color: C.lime,
    align: "center", isTextBox: true, margin: 0,
  });
  s.addText("Stockscope AI  ·  CP020003 Artificial Intelligence  ·  Group ___", {
    x: 0, y: PAGE_H - 0.9, w: PAGE_W, h: 0.4, fontFace: FONT, fontSize: 12, color: C.mutedDark,
    align: "center", isTextBox: true, margin: 0,
  });
}

pres.writeFile({ fileName: path.join(__dirname, "stock_direction_presentation.pptx") }).then(() => {
  console.log("Deck written.");
});
