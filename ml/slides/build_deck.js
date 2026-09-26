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

// Leelawadee UI ships with Windows 10/11 and renders both Thai and Latin text cleanly.
const FONT = "Leelawadee UI";
const FONT_HEAD = "Leelawadee UI";

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
  s.addText("ทำนายทิศทางหุ้น\nด้วย XGBoost", {
    x: 0.85, y: 2.6, w: 9.5, h: 1.9, fontFace: FONT_HEAD, fontSize: 46, bold: true,
    color: C.white, isTextBox: true, margin: 0, lineSpacing: 56,
  });
  s.addText("ไปป์ไลน์ Machine Learning แบบไม่รั่วไหลข้อมูล ต่อยอดจากแดชบอร์ดที่ใช้งานจริง", {
    x: 0.9, y: 4.55, w: 9.2, h: 0.5, fontFace: FONT, fontSize: 17, color: C.muted,
    isTextBox: true, margin: 0,
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 0.9, y: 5.35, w: 4.6, h: 0.5, rectRadius: 0.08,
    fill: { color: C.panel }, line: { type: "none" },
  });
  s.addText("กลุ่ม ___  ·  [ชื่อทีมที่นี่]", {
    x: 1.05, y: 5.35, w: 4.3, h: 0.5, fontFace: FONT, fontSize: 13, color: C.white,
    isTextBox: true, margin: 0, valign: "middle",
  });

  s.addText("CP020003 · ปัญญาประดิษฐ์ · โปรเจกต์จบภาคการศึกษา · มหาวิทยาลัยขอนแก่น 2026", {
    x: 0.9, y: PAGE_H - 0.7, w: 10, h: 0.35, fontFace: FONT, fontSize: 11, color: C.mutedDark,
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
  kicker(s, "ทีมของเรา");
  title(s, "สมาชิกในกลุ่ม");

  panelBox(s, { x: 0.6, y: 1.85, w: 12.1, h: 0.9 });
  s.addText("กลุ่มที่", { x: 0.9, y: 1.85, w: 2, h: 0.9, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0, valign: "middle" });
  s.addText("___", { x: 0.9, y: 2.15, w: 2, h: 0.5, fontFace: FONT_HEAD, fontSize: 20, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("ชื่อกลุ่ม", { x: 3.3, y: 1.85, w: 4, h: 0.4, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0 });
  s.addText("[ชื่อทีมที่นี่]", { x: 3.3, y: 2.2, w: 6, h: 0.5, fontFace: FONT_HEAD, fontSize: 20, bold: true, color: C.white, isTextBox: true, margin: 0 });

  const rows = [
    ["#", "ชื่อนักศึกษา", "รหัสนักศึกษา"],
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
  kicker(s, "ที่มาและหัวข้อ");
  title(s, "ทำไมถึงทำโปรเจกต์นี้?");

  s.addText(
    "เราสร้าง Stockscope ไว้แล้ว — แดชบอร์ดที่คำนวณ RSI, MACD, ค่าเฉลี่ยเคลื่อนที่ และโซนแนวรับ-แนวต้าน ให้หุ้นทุกตัวแบบเรียลไทม์",
    { x: 0.6, y: 1.75, w: 6.6, h: 1.3, fontFace: FONT, fontSize: 16, color: C.white, isTextBox: true, margin: 0, valign: "top" }
  );
  s.addText(
    "คำถามที่ตามมาคือสิ่งที่เราทดสอบได้จริงด้วยข้อมูลที่มีอยู่แล้ว:",
    { x: 0.6, y: 3.0, w: 6.6, h: 0.6, fontFace: FONT, fontSize: 14, color: C.muted, isTextBox: true, margin: 0 }
  );

  panelBox(s, { x: 0.6, y: 3.7, w: 6.6, h: 1.7 });
  s.addText("“อินดิเคเตอร์เดียวกันนี้ ทำนายทิศทางราคาถัดไปได้ไหม?”", {
    x: 0.95, y: 3.7, w: 5.9, h: 1.7, fontFace: FONT_HEAD, fontSize: 19, italic: true,
    color: C.lime, isTextBox: true, margin: 0, valign: "middle",
  });

  // Right column: simple flow diagram Dashboard -> Question -> Model
  const items = [
    { icon: "bar-chart", label: "แดชบอร์ดเรียลไทม์", desc: "RSI · MACD · SMA · Volume\nคำนวณแบบวันต่อวัน" },
    { icon: "target", label: "คำถามหลัก", desc: "ขึ้นหรือลง\nในอีก N วันข้างหน้า?" },
    { icon: "cpu", label: "โมเดล ML", desc: "จำแนกสองกลุ่ม\nเทรนจากข้อมูลย้อนหลัง" },
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
  kicker(s, "โจทย์ AI");
  title(s, "มองเป็นปัญหา Machine Learning");

  panelBox(s, { x: 0.6, y: 1.8, w: 5.8, h: 4.9 });
  s.addText("การจำแนกแบบสองกลุ่ม", { x: 0.95, y: 2.05, w: 5.1, h: 0.5, fontFace: FONT_HEAD, fontSize: 19, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("label_up = 1  if Close[t+N] > Close[t]\nlabel_up = 0  otherwise", {
    x: 0.95, y: 2.65, w: 5.1, h: 0.8, fontFace: "Courier New", fontSize: 13, color: C.white, isTextBox: true, margin: 0,
  });
  s.addText("Input: อินดิเคเตอร์ทางเทคนิคจนถึงวันที่ t\nOutput: ความน่าจะเป็นที่ราคาจะสูงขึ้นในอีก N วัน", {
    x: 0.95, y: 3.55, w: 5.1, h: 0.9, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });

  const reasons = [
    ["zap", "เทรนเร็ว", "รันจบในไม่กี่วินาที บนโน้ตบุ๊กหรือ Colab ฟรี"],
    ["layers", "เหมาะกับข้อมูลตาราง", "ฟีเจอร์ของเราคือตัวเลขอินดิเคเตอร์ ไม่ใช่รูปภาพหรือข้อความ"],
    ["target", "อธิบายได้", "Feature Importance บอกได้ว่า “ทำไม” ไม่ใช่แค่ตัวเลข"],
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
    ["database", "ข้อมูล", "OHLCV 5 ปี, 20 หุ้น"],
    ["activity", "ฟีเจอร์", "RSI · MACD · SMA · Vol"],
    ["cpu", "โมเดล", "XGBoost classifier"],
    ["bar-chart", "ผลทำนาย", "P(ขึ้น) ใน N วัน"],
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
  kicker(s, "ชุดข้อมูลและ EDA");
  title(s, "ชุดข้อมูล (Dataset)");

  const stats = [
    ["20", "จำนวนหุ้น"], ["5 ปี", "ข้อมูลรายวัน"], ["25,100", "แถวข้อมูลทั้งหมด"], ["0", "ข้อมูลขาดหาย"],
  ];
  let sx = 0.6;
  stats.forEach((st) => {
    panelBox(s, { x: sx, y: 1.85, w: 2.85, h: 1.4 });
    statCallout(s, { x: sx + 0.25, y: 2.0, w: 2.4, value: st[0], label: st[1], valueSize: 30 });
    sx += 3.05;
  });

  s.addText("แหล่งข้อมูลเดียวกับที่ Stockscope ใช้งานจริง: Yahoo Finance ผ่าน yfinance", {
    x: 0.6, y: 3.5, w: 12, h: 0.4, fontFace: FONT, fontSize: 14, italic: true, color: C.lime, isTextBox: true, margin: 0,
  });

  s.addText("ช่วงข้อมูล: 2021-09-27  →  2026-09-25", {
    x: 0.6, y: 4.05, w: 7, h: 0.4, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0,
  });

  s.addText("กลุ่มอุตสาหกรรมที่ครอบคลุม", {
    x: 0.6, y: 4.6, w: 6, h: 0.35, fontFace: FONT, fontSize: 12, bold: true, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0,
  });
  const sectors = ["เทคโนโลยี", "การเงิน", "พลังงาน", "สุขภาพ", "สินค้าอุปโภคบริโภค", "สื่อ/บันเทิง"];
  let chx = 0.6, chy = 5.0;
  const CHIP_RIGHT_EDGE = 7.6; // stay clear of the right panel starting at x=8.0
  sectors.forEach((sec) => {
    const w = 0.5 + sec.length * 0.13;
    if (chx + w > CHIP_RIGHT_EDGE) { chx = 0.6; chy += 0.6; }
    s.addShape(pres.ShapeType.roundRect, { x: chx, y: chy, w, h: 0.45, rectRadius: 0.22, fill: { color: C.panel2 }, line: { type: "none" } });
    s.addText(sec, { x: chx, y: chy, w, h: 0.45, fontFace: FONT, fontSize: 12, color: C.white, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    chx += w + 0.2;
  });

  panelBox(s, { x: 8.0, y: 1.85, w: 4.7, h: 4.9 });
  s.addText("ตัวอย่างหุ้นที่ใช้", { x: 8.35, y: 2.1, w: 4, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0 });
  s.addText("AAPL · MSFT · NVDA · JPM · XOM\nJNJ · AMZN · KO · GOOGL · META\nWMT · PG · V · MA · DIS\nNFLX · AMD · INTC · BAC · CVX", {
    x: 8.35, y: 2.5, w: 4.1, h: 2.0, fontFace: "Courier New", fontSize: 13, color: C.white, isTextBox: true, margin: 0, lineSpacing: 26,
  });
  s.addText("หลายอุตสาหกรรม ไม่ใช่หุ้นเดียว — ลดความเสี่ยงที่โมเดลจะเรียนรู้พฤติกรรมเฉพาะของบริษัทเดียว แทนที่จะเป็นพฤติกรรมทั่วไปของอินดิเคเตอร์", {
    x: 8.35, y: 4.75, w: 4.05, h: 1.8, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });
}

// ============================================================
// SLIDE 6 — FEATURE ENGINEERING
// ============================================================
{
  const s = newSlide();
  kicker(s, "ระเบียบวิธี");
  title(s, "ฟีเจอร์ — สิ่งที่โมเดลมองเห็น");

  const feats = [
    ["activity", "ret_1d / 5d / 10d / 21d", "ผลตอบแทนย้อนหลัง 1–21 วันทำการ (โมเมนตัม)"],
    ["target", "rsi14", "RSI 14 วัน — ซื้อมากไปหรือขายมากไป"],
    ["bar-chart", "macd_hist", "MACD Histogram (ความแรง & ทิศทางเทรนด์)"],
    ["layers", "dist_sma20 / dist_sma50", "% ระยะห่างจากเส้นค่าเฉลี่ย 20/50 วัน"],
    ["zap", "vol_ratio", "ปริมาณซื้อขายวันนี้ เทียบค่าเฉลี่ย 20 วัน"],
    ["shield", "volatility_20d", "ส่วนเบี่ยงเบนมาตรฐานผลตอบแทนรายวัน (ความผันผวน)"],
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

  s.addText("คำนวณด้วยหน้าต่างข้อมูลย้อนหลังเท่านั้น (rolling / ewm) — เรื่องการออกแบบป้องกันข้อมูลรั่วไหลต่อในสไลด์ถัดไป", {
    x: 0.6, y: 6.75, w: 12.1, h: 0.4, fontFace: FONT, fontSize: 12, italic: true, color: C.muted, isTextBox: true, margin: 0,
  });
}

// ============================================================
// SLIDE 7 — ANTI-LEAKAGE DESIGN
// ============================================================
{
  const s = newSlide();
  kicker(s, "ระเบียบวิธี");
  title(s, "ป้องกันข้อมูลรั่วไหล (Data Leakage)");
  s.addText("การตัดสินใจออกแบบที่สำคัญที่สุด — ตรงกับเกณฑ์ “ระเบียบวิธีไม่รั่วไหลข้อมูล มีเหตุผลรองรับ” ในเกณฑ์การให้คะแนน", {
    x: 0.6, y: 1.35, w: 12.1, h: 0.4, fontFace: FONT, fontSize: 13, italic: true, color: C.lime, isTextBox: true, margin: 0,
  });

  // Left: wrong approach
  panelBox(s, { x: 0.6, y: 1.95, w: 5.85, h: 3.0 });
  s.addImage({ path: icon("x-circle", "coral"), x: 0.9, y: 2.2, w: 0.4, h: 0.4 });
  s.addText("แบ่งข้อมูลแบบสุ่ม 80/20", { x: 1.45, y: 2.22, w: 4.8, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: C.coral, isTextBox: true, margin: 0 });
  s.addText("ข้อมูลสัปดาห์เดียวกันหลุดไปอยู่ทั้งฝั่ง train และ test อินดิเคเตอร์มีความสัมพันธ์กันในตัวเอง (autocorrelated) — โมเดลจะจำช่วง test แทนที่จะเรียนรู้ภาพรวม", {
    x: 0.9, y: 2.85, w: 5.2, h: 1.9, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });

  // Right: correct approach
  panelBox(s, { x: 6.85, y: 1.95, w: 5.85, h: 3.0, fill: C.panel2 });
  s.addImage({ path: icon("check-circle", "lime"), x: 7.15, y: 2.2, w: 0.4, h: 0.4 });
  s.addText("ตัดข้อมูลตามวันที่เดียวทั้งชุด", { x: 7.7, y: 2.22, w: 4.8, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("ทุกแถวของ train เกิดก่อนทุกแถวของ test เสมอ ทั้ง 20 หุ้นพร้อมกัน — เลียนแบบการใช้งานจริง: เทรนจากอดีต ทดสอบกับอนาคตที่ยังไม่เคยเห็น", {
    x: 7.15, y: 2.85, w: 5.2, h: 1.9, fontFace: FONT, fontSize: 13, color: C.white, isTextBox: true, margin: 0, valign: "top",
  });

  // Timeline diagram
  s.addText("2021", { x: 0.7, y: 5.5, w: 1, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, isTextBox: true, margin: 0 });
  s.addText("2026", { x: 11.6, y: 5.5, w: 1, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, align: "right", isTextBox: true, margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.85, w: 8.6, h: 0.55, rectRadius: 0.06, fill: { color: C.lime }, line: { type: "none" } });
  s.addShape(pres.ShapeType.roundRect, { x: 9.2, y: 5.85, w: 3.5, h: 0.55, rectRadius: 0.06, fill: { color: C.coral }, line: { type: "none" }, transparency: 15 });
  s.addText("TRAIN  —  19,280 แถว  (ต.ค. 2021 → ต.ค. 2025)", { x: 0.6, y: 5.85, w: 8.6, h: 0.55, fontFace: FONT, fontSize: 12, bold: true, color: C.bg, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText("TEST  —  4,840 แถว", { x: 9.2, y: 5.85, w: 3.5, h: 0.55, fontFace: FONT, fontSize: 12, bold: true, color: C.bg, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  s.addText("cutoff: 2025-10-09", { x: 8.6, y: 6.55, w: 2.4, h: 0.3, fontFace: FONT, fontSize: 10, color: C.muted, align: "center", isTextBox: true, margin: 0 });
}

// ============================================================
// SLIDE 8 — MODEL TRAINING
// ============================================================
{
  const s = newSlide();
  kicker(s, "ระเบียบวิธี");
  title(s, "การเทรนโมเดล — Baseline เทียบ XGBoost", { size: 30 });

  const models = [
    ["Logistic Regression", "โมเดลเชิงเส้นพื้นฐานที่สุด (standardize ฟีเจอร์ก่อน)"],
    ["Random Forest", "โมเดล tree-ensemble อีกตัว ไม่มี boosting"],
    ["XGBoost (ของเรา)", "ปรับจูนด้วย TimeSeriesSplit CV — ใช้เฉพาะชุด train"],
  ];
  let mx = 0.6;
  models.forEach((m, i) => {
    const isOurs = i === 2;
    panelBox(s, { x: mx, y: 1.9, w: 3.9, h: 1.9, fill: isOurs ? C.panel2 : C.panel });
    s.addText(m[0], { x: mx + 0.3, y: 2.1, w: 3.4, h: 0.5, fontFace: FONT, fontSize: 15, bold: true, color: isOurs ? C.lime : C.white, isTextBox: true, margin: 0 });
    s.addText(m[1], { x: mx + 0.3, y: 2.65, w: 3.4, h: 1.0, fontFace: FONT, fontSize: 12, color: C.muted, isTextBox: true, margin: 0, valign: "top" });
    mx += 4.1;
  });

  s.addText("ค้นหาค่า HYPERPARAMETER — TimeSeriesSplit (4 folds) ใช้เฉพาะชุด train", {
    x: 0.6, y: 4.15, w: 8, h: 0.35, fontFace: FONT, fontSize: 12, bold: true, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0,
  });

  const grid = [
    ["max_depth=3, lr=0.05, n=200", "CV AUC 0.499"],
    ["max_depth=4, lr=0.05, n=300", "CV AUC 0.505  ✓ ดีที่สุด"],
    ["max_depth=3, lr=0.10, n=150", "CV AUC 0.501"],
  ];
  let gy = 4.6;
  grid.forEach((g) => {
    const isBest = g[1].includes("ดีที่สุด");
    s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: gy, w: 8.1, h: 0.55, rectRadius: 0.06, fill: { color: isBest ? C.panel2 : C.panel }, line: isBest ? { color: C.lime, width: 1 } : { type: "none" } });
    s.addText(g[0], { x: 0.85, y: gy, w: 4.2, h: 0.55, fontFace: "Courier New", fontSize: 12, color: C.white, valign: "middle", isTextBox: true, margin: 0 });
    s.addText(g[1], { x: 5.2, y: gy, w: 3.3, h: 0.55, fontFace: FONT, fontSize: 12, bold: isBest, color: isBest ? C.lime : C.muted, valign: "middle", align: "right", isTextBox: true, margin: 0 });
    gy += 0.68;
  });

  panelBox(s, { x: 9.1, y: 4.15, w: 3.6, h: 2.65, fill: C.panel2 });
  s.addText("ค่าที่ใช้จริง", { x: 9.4, y: 4.4, w: 3, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0 });
  s.addText("max_depth: 4\nlearning_rate: 0.05\nn_estimators: 300\nsubsample: 0.8", {
    x: 9.4, y: 4.75, w: 3, h: 1.7, fontFace: "Courier New", fontSize: 14, color: C.lime, isTextBox: true, margin: 0, lineSpacing: 26,
  });
}

// ============================================================
// SLIDE 9 — RESULTS: MODEL COMPARISON
// ============================================================
{
  const s = newSlide();
  kicker(s, "ผลลัพธ์และการประเมิน");
  title(s, "เปรียบเทียบโมเดลบนชุดทดสอบ");

  const cats = ["Baseline", "LogReg", "Rand.Forest", "XGBoost"];
  const accVals = [0.5072, 0.5085, 0.5087, 0.5246];
  const aucVals = [0.50, 0.5166, 0.5361, 0.5262];

  s.addText("ความแม่นยำ (Accuracy)", { x: 0.6, y: 1.85, w: 5.8, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.white, isTextBox: true, margin: 0 });
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
  s.addText("XGBoost แม่นยำที่สุด (52.5%) ส่วน Random Forest ทำ ROC-AUC ได้ดีกว่าเล็กน้อย (0.536 เทียบ 0.526) ทั้งสองโมเดลแบบ tree-ensemble ชนะ Logistic Regression และ baseline ที่ทายกลุ่มมากสุดอย่างเดียว (50.7%) อย่างชัดเจน", {
    x: 0.95, y: 5.55, w: 11.4, h: 1.15, fontFace: FONT, fontSize: 14, color: C.white, isTextBox: true, margin: 0, valign: "middle",
  });
}

// ============================================================
// SLIDE 10 — HORIZON COMPARISON (key insight)
// ============================================================
{
  const s = newSlide();
  kicker(s, "ผลลัพธ์และการประเมิน");
  title(s, "ระยะเวลาทำนายมีผลไหม?");
  s.addText("ใช้ไปป์ไลน์เดียวกัน ทดสอบ 3 ระยะเวลา — 1, 5 และ 10 วันทำการข้างหน้า", {
    x: 0.6, y: 1.35, w: 12, h: 0.4, fontFace: FONT, fontSize: 13, italic: true, color: C.muted, isTextBox: true, margin: 0,
  });

  const cats = ["1 วัน", "5 วัน", "10 วัน"];
  s.addChart(
    pres.ChartType.bar,
    [
      { name: "ความแม่นยำ XGBoost", labels: cats, values: [0.5041, 0.5136, 0.5246] },
      { name: "Baseline (ทายกลุ่มมากสุด)", labels: cats, values: [0.513, 0.5221, 0.5072] },
    ],
    {
      x: 0.6, y: 1.95, w: 7.6, h: 4.3,
      barDir: "col", barGrouping: "clustered",
      chartColors: [C.lime, C.mutedDark],
      showTitle: false,
      showLegend: true, legendColor: C.muted, legendFontSize: 11, legendPos: "b", legendFontFace: FONT,
      showValue: true, dataLabelPosition: "outEnd", dataLabelColor: C.white, dataLabelFontSize: 10, dataLabelFontFace: FONT,
      dataLabelFormatCode: "0.0%",
      catAxisLabelColor: C.muted, catAxisLabelFontSize: 12, catAxisLineColor: C.mutedDark, catAxisLabelFontFace: FONT,
      valAxisHidden: true, valAxisLineShow: false,
      valAxisMinVal: 0.45, valAxisMaxVal: 0.56,
      catGridLine: { style: "none" }, valGridLine: { style: "none" },
      barGapWidthPct: 30,
    }
  );

  panelBox(s, { x: 8.5, y: 1.95, w: 4.2, h: 4.3, fill: C.panel2 });
  s.addText("ROC-AUC ตามระยะเวลา", { x: 8.8, y: 2.15, w: 3.6, h: 0.3, fontFace: FONT, fontSize: 11, color: C.muted, charSpacing: 1, isTextBox: true, margin: 0 });
  const aucRows = [["1 วัน", "0.505"], ["5 วัน", "0.507"], ["10 วัน", "0.526"]];
  let ary = 2.6;
  aucRows.forEach((r, i) => {
    const isBest = i === 2;
    s.addText(r[0], { x: 8.8, y: ary, w: 1.8, h: 0.45, fontFace: FONT, fontSize: 14, color: C.white, valign: "middle", isTextBox: true, margin: 0 });
    s.addText(r[1], { x: 10.5, y: ary, w: 1.9, h: 0.45, fontFace: FONT_HEAD, fontSize: 18, bold: true, color: isBest ? C.lime : C.muted, align: "right", valign: "middle", isTextBox: true, margin: 0 });
    ary += 0.55;
  });
  s.addText("สุ่มล้วนๆ = 0.500", { x: 8.8, y: ary + 0.1, w: 3.6, h: 0.3, fontFace: FONT, fontSize: 10, italic: true, color: C.mutedDark, isTextBox: true, margin: 0 });
  s.addText("การเคลื่อนไหวระยะสั้นใกล้เคียงการสุ่ม แต่เริ่มมีความได้เปรียบเล็กน้อยที่ระยะ 10 วัน — สอดคล้องกับทฤษฎีตลาดมีประสิทธิภาพ (Market Efficiency)", {
    x: 8.8, y: 4.5, w: 3.7, h: 1.6, fontFace: FONT, fontSize: 12, color: C.white, isTextBox: true, margin: 0, valign: "top",
  });
}

// ============================================================
// SLIDE 11 — FEATURE IMPORTANCE
// ============================================================
{
  const s = newSlide();
  kicker(s, "ผลลัพธ์และการประเมิน");
  title(s, "โมเดลให้ความสำคัญกับอะไรบ้าง?");

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
  s.addText("ข้อค้นพบสำคัญ", { x: 9.35, y: 2.2, w: 3, h: 0.35, fontFace: FONT, fontSize: 15, bold: true, color: C.lime, isTextBox: true, margin: 0 });
  s.addText("สัญญาณเทรนด์ระยะกลางและความผันผวน (volatility_20d, dist_sma20/50, macd_hist) สำคัญที่สุด\n\nผลตอบแทนรายวัน (ret_1d) สำคัญน้อยที่สุด — เป็นฟีเจอร์ที่มี noise มากและให้ข้อมูลน้อยที่สุด ตรงตามที่คาดไว้", {
    x: 8.8, y: 2.75, w: 3.6, h: 3.8, fontFace: FONT, fontSize: 13, color: C.white, isTextBox: true, margin: 0, valign: "top", lineSpacing: 20,
  });
}

// ============================================================
// SLIDE 12 — BACKTEST
// ============================================================
{
  const s = newSlide();
  kicker(s, "ผลลัพธ์และการประเมิน");
  title(s, "ทดสอบกลยุทธ์ย้อนหลัง (เพื่อการศึกษา)", { size: 28 });

  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 1.4, w: 6.6, h: 0.45, rectRadius: 0.22, fill: { color: C.panel2 }, line: { color: C.coral, width: 1 } });
  s.addImage({ path: icon("alert-triangle", "coral"), x: 0.8, y: 1.51, w: 0.24, h: 0.24 });
  s.addText("เพื่อการศึกษาเท่านั้น — ไม่ใช่คำแนะนำการลงทุน", {
    x: 1.15, y: 1.4, w: 6.0, h: 0.45, fontFace: FONT, fontSize: 12, bold: true, color: C.coral, valign: "middle", isTextBox: true, margin: 0,
  });

  s.addText("กติกา: ซื้อเมื่อ P(ขึ้น) > 0.55 เท่านั้น นอกนั้นถือเงินสด น้ำหนักเท่ากันทั้ง 20 หุ้น เฉพาะช่วง test ไม่รวมค่าธรรมเนียม", {
    x: 0.6, y: 2.05, w: 12.0, h: 0.5, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0,
  });

  panelBox(s, { x: 0.6, y: 2.75, w: 5.85, h: 2.1 });
  statCallout(s, { x: 1.0, y: 3.05, w: 5, value: "3.86x", label: "กลยุทธ์ตามสัญญาณโมเดล", color: C.white, valueSize: 44 });

  panelBox(s, { x: 6.85, y: 2.75, w: 5.85, h: 2.1, fill: C.panel2 });
  statCallout(s, { x: 7.25, y: 3.05, w: 5, value: "12.46x", label: "ซื้อแล้วถือ (น้ำหนักเท่ากัน)", color: C.lime, valueSize: 44 });
  s.addText("มูลค่าจาก $1 ตลอดช่วง test", { x: 1.0, y: 4.15, w: 5, h: 0.3, fontFace: FONT, fontSize: 11, italic: true, color: C.mutedDark, isTextBox: true, margin: 0 });
  s.addText("มูลค่าจาก $1 ตลอดช่วง test", { x: 7.25, y: 4.15, w: 5, h: 0.3, fontFace: FONT, fontSize: 11, italic: true, color: C.mutedDark, isTextBox: true, margin: 0 });

  panelBox(s, { x: 0.6, y: 5.15, w: 12.1, h: 1.55 });
  s.addImage({ path: icon("target", "lime"), x: 0.9, y: 5.4, w: 0.35, h: 0.35 });
  s.addText("ทำไมซื้อแล้วถือถึงชนะ", { x: 1.4, y: 5.4, w: 5, h: 0.35, fontFace: FONT, fontSize: 14, bold: true, color: C.white, isTextBox: true, margin: 0 });
  s.addText("ช่วง test เป็นตลาดขาขึ้นแรง การถือเงินสดทุกครั้งที่โมเดลไม่มั่นใจ ทำให้เสียโอกาสมากกว่าความได้เปรียบเล็กน้อยจาก AUC ~0.53 จะชดเชยได้ — ความได้เปรียบทางสถิติไม่ได้แปลว่าจะทำกำไรได้จริงเสมอไป การจับจังหวะตลาดมีต้นทุนค่าเสียโอกาสจริง", {
    x: 0.9, y: 5.8, w: 11.5, h: 0.85, fontFace: FONT, fontSize: 13, color: C.muted, isTextBox: true, margin: 0, valign: "top",
  });
}

// ============================================================
// SLIDE 13 — LIMITATIONS
// ============================================================
{
  const s = newSlide();
  kicker(s, "อภิปราย");
  title(s, "ข้อจำกัด — สิ่งที่โมเดลนี้ยังทำไม่ได้", { size: 30 });

  const lims = [
    ["alert-triangle", "สัญญาณยังไม่ชัดเจนมาก", "ROC-AUC ≈ 0.50–0.53 ใกล้เคียงการสุ่ม — รายงานตามจริง ไม่ได้ปรับแต่งให้ดูดีเกินจริง"],
    ["database", "ไม่มีข้อมูลปัจจัยพื้นฐานหรือเศรษฐกิจมหภาค", "ไม่มีผลประกอบการ อัตราดอกเบี้ย หรือข่าวสารอยู่ในชุดฟีเจอร์"],
    ["layers", "Survivorship Bias", "ใช้เฉพาะบริษัทใหญ่ที่ยังจดทะเบียนอยู่ — ไม่รวมหุ้นที่ถูกถอดออกจากตลาด"],
    ["shield", "ไม่รวมต้นทุนการซื้อขาย", "การทดสอบย้อนหลังไม่ได้รวมค่าธรรมเนียม ส่วนต่างราคา หรือภาษี"],
    ["activity", "ทดสอบย้อนหลังด้าน Sentiment ไม่ได้", "ระบบดึงข่าวของ Stockscope ไม่มีข้อมูลข่าวย้อนหลังให้จับคู่กับวันที่ในอดีต"],
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
  kicker(s, "ผลกระทบและบทสรุป");
  title(s, "เชื่อมโยงกลับสู่ Stockscope");

  panelBox(s, { x: 0.6, y: 1.85, w: 5.85, h: 4.85 });
  s.addText("ข้อค้นพบสำคัญ", { x: 0.95, y: 2.1, w: 5, h: 0.3, fontFace: FONT, fontSize: 12, bold: true, color: C.lime, charSpacing: 1, isTextBox: true, margin: 0 });
  const insights = [
    "ไปป์ไลน์ไม่รั่วไหลข้อมูลอย่างสมบูรณ์: ฟีเจอร์มองย้อนหลัง label มองไปข้างหน้า แยกกันเด็ดขาด",
    "XGBoost ชนะ baseline เล็กน้อยที่ระยะยาว แต่ไม่ชนะที่ระยะ 1 วัน",
    "macd_hist, dist_sma50/20 และ volatility เป็นฟีเจอร์ที่ให้ข้อมูลมากที่สุด",
    "ความได้เปรียบทางสถิติเล็กน้อย ≠ กลยุทธ์ที่ทำกำไรได้จริง",
  ];
  let iny = 2.55;
  insights.forEach((t) => {
    s.addShape(pres.ShapeType.ellipse, { x: 0.95, y: iny + 0.06, w: 0.12, h: 0.12, fill: { color: C.lime }, line: { type: "none" } });
    s.addText(t, { x: 1.25, y: iny - 0.1, w: 4.9, h: 0.75, fontFace: FONT, fontSize: 13, color: C.white, isTextBox: true, margin: 0, valign: "top" });
    iny += 0.92;
  });

  panelBox(s, { x: 6.85, y: 1.85, w: 5.85, h: 4.85, fill: C.panel2 });
  s.addImage({ path: icon("arrow-right", "lime"), x: 7.2, y: 2.15, w: 0.35, h: 0.35 });
  s.addText("ขั้นต่อไป: นำไปใช้งานจริงในแอป", { x: 7.7, y: 2.15, w: 4.8, h: 0.45, fontFace: FONT, fontSize: 16, bold: true, color: C.lime, isTextBox: true, margin: 0 });

  panelBox(s, { x: 7.2, y: 2.85, w: 5.1, h: 1.5, fill: C.bg });
  s.addText("AI: มีโอกาส 61% ที่ราคาจะปิดสูงขึ้น\nในอีก 10 วันทำการ", {
    x: 7.45, y: 3.05, w: 4.6, h: 0.9, fontFace: FONT, fontSize: 16, bold: true, color: C.lime, isTextBox: true, margin: 0,
  });
  s.addText("เป็นการประมาณค่าทางสถิติ ไม่ใช่คำแนะนำทางการเงิน", {
    x: 7.45, y: 3.85, w: 4.6, h: 0.35, fontFace: FONT, fontSize: 10, italic: true, color: C.mutedDark, isTextBox: true, margin: 0,
  });

  s.addText("บันทึกโมเดลที่เทรนแล้วและโหลดเข้าแอป Streamlit วางคู่กับมุมมองทางเทคนิคที่ Stockscope แสดงอยู่แล้วสำหรับทุกหุ้น", {
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
  s.addText("ขอบคุณ", {
    x: 0, y: 3.0, w: PAGE_W, h: 1.0, fontFace: FONT_HEAD, fontSize: 48, bold: true, color: C.white,
    align: "center", isTextBox: true, margin: 0,
  });
  s.addText("ถาม-ตอบ (Q&A)", {
    x: 0, y: 4.0, w: PAGE_W, h: 0.5, fontFace: FONT, fontSize: 18, color: C.lime,
    align: "center", isTextBox: true, margin: 0,
  });
  s.addText("Stockscope AI  ·  CP020003 ปัญญาประดิษฐ์  ·  กลุ่ม ___", {
    x: 0, y: PAGE_H - 0.9, w: PAGE_W, h: 0.4, fontFace: FONT, fontSize: 12, color: C.mutedDark,
    align: "center", isTextBox: true, margin: 0,
  });
}

pres.writeFile({ fileName: path.join(__dirname, "stock_direction_presentation.pptx") }).then(() => {
  console.log("Deck written.");
});
