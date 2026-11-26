// Support/resistance, zone status, the rule-based signal and risk metrics (ported from app.py).
import { macd, rsi, sma } from "./indicators";

export type OHLC = { time: string; open: number; high: number; low: number; close: number; volume: number };

export type Levels = { support: number[]; resistance: number[]; current: number };

export function computeLevels(bars: OHLC[], maxEach = 3): Levels {
  const n = bars.length;
  const current = n ? bars[n - 1].close : 0;
  if (n < 20) return { support: [], resistance: [], current };
  const window = Math.max(3, Math.min(10, Math.floor(n / 40)));
  const res: number[] = [];
  const sup: number[] = [];
  for (let i = window; i < n - window; i++) {
    let isHigh = true;
    let isLow = true;
    for (let k = 1; k <= window; k++) {
      if (!(bars[i].high >= bars[i - k].high && bars[i].high >= bars[i + k].high)) isHigh = false;
      if (!(bars[i].low <= bars[i - k].low && bars[i].low <= bars[i + k].low)) isLow = false;
    }
    if (isHigh) res.push(bars[i].high);
    if (isLow) sup.push(bars[i].low);
  }
  const cluster = (levels: number[], tol = 0.015) => {
    if (!levels.length) return [];
    const s = [...levels].sort((a, b) => a - b);
    const groups: number[][] = [[s[0]]];
    for (const l of s.slice(1)) {
      const last = groups[groups.length - 1];
      const ref = Math.max(last[last.length - 1], 1e-9);
      if ((l - ref) / ref < tol) last.push(l);
      else groups.push([l]);
    }
    return groups.map((g) => g.reduce((a, b) => a + b, 0) / g.length);
  };
  const resistance = cluster(res).filter((x) => x > current * 1.002).sort((a, b) => a - b).slice(0, maxEach);
  const support = cluster(sup).filter((x) => x < current * 0.998).sort((a, b) => b - a).slice(0, maxEach);
  return { support, resistance, current };
}

export type ZoneKey = "touched" | "broken" | "approaching" | "waiting";
export type ZoneStatus = { key: ZoneKey; label: string; daysAgo: number | null };

export function zoneStatus(bars: OHLC[], zone: number, kind: "support" | "resistance", lookback = 30, tol = 0.008): ZoneStatus {
  if (bars.length < 2) return { key: "waiting", label: "—", daysAgo: null };
  const current = bars[bars.length - 1].close;
  const recent = bars.slice(-lookback);
  let broken: number | null = null;
  let touched: number | null = null;
  for (let i = recent.length - 1; i >= 0; i--) {
    const b = recent[i];
    if (kind === "support") {
      if (b.low <= zone * (1 - tol) && b.close < zone * (1 - tol) && broken === null) broken = i;
      if (b.low <= zone * (1 + tol) && touched === null) touched = i;
    } else {
      if (b.high >= zone * (1 + tol) && b.close > zone * (1 + tol) && broken === null) broken = i;
      if (b.high >= zone * (1 - tol) && touched === null) touched = i;
    }
  }
  const arrow = kind === "support" ? "⬇" : "⬆";
  const age = (i: number) => recent.length - 1 - i;
  if (broken !== null) return { key: "broken", label: age(broken) > 0 ? `${arrow} BROKEN · ${age(broken)}d` : `${arrow} BROKEN`, daysAgo: age(broken) };
  if (touched !== null) return { key: "touched", label: age(touched) > 0 ? `◉ TESTED · ${age(touched)}d` : "◉ TESTED", daysAgo: age(touched) };
  const near = kind === "support" ? current < zone * 1.03 : current > zone * 0.97;
  return near ? { key: "approaching", label: "● APPROACHING", daysAgo: null } : { key: "waiting", label: "○ PENDING", daysAgo: null };
}

export type Signal = { verdict: "bull" | "bear" | "flat"; verdictLabel: string; score: number; reasons: string[] };

export function buildSignal(bars: OHLC[]): Signal {
  const close = bars.map((b) => b.close);
  const last = close[close.length - 1];
  const s20 = sma(close, 20, false).at(-1) as number;
  const s50 = sma(close, 50, false).at(-1) as number;
  const s200 = close.length >= 50 ? (sma(close, 200, false).at(-1) as number) : NaN;
  const reasons: string[] = [];
  let score = 0;

  if (!Number.isNaN(s50) && !Number.isNaN(s200)) {
    if (last > s50 && s50 > s200) { score += 2; reasons.push("ราคา > SMA50 > SMA200 → uptrend ระยะยาว"); }
    else if (last < s50 && s50 < s200) { score -= 2; reasons.push("ราคา < SMA50 < SMA200 → downtrend ระยะยาว"); }
  }
  if (last > s20) { score += 1; reasons.push("ราคาอยู่เหนือ SMA20"); }
  else { score -= 1; reasons.push("ราคาอยู่ต่ำกว่า SMA20"); }

  const r = rsi(close).at(-1) as number;
  if (r < 30) { score += 2; reasons.push(`RSI = ${r.toFixed(1)} → oversold (มีโอกาสเด้ง)`); }
  else if (r > 70) { score -= 2; reasons.push(`RSI = ${r.toFixed(1)} → overbought (มีโอกาสปรับฐาน)`); }
  else reasons.push(`RSI = ${r.toFixed(1)} → กลาง`);

  const { line, signal } = macd(close);
  const n = line.length;
  if (n >= 2) {
    const up = line[n - 1] > signal[n - 1];
    const wasUp = line[n - 2] > signal[n - 2];
    if (up && !wasUp) { score += 2; reasons.push("MACD เพิ่งตัดขึ้นเหนือ Signal → สัญญาณซื้อ"); }
    else if (!up && line[n - 1] < signal[n - 1] && wasUp) { score -= 2; reasons.push("MACD เพิ่งตัดลงใต้ Signal → สัญญาณขาย"); }
    else if (up) { score += 1; reasons.push("MACD อยู่เหนือ Signal"); }
    else { score -= 1; reasons.push("MACD อยู่ใต้ Signal"); }
  }

  if (score >= 3) return { verdict: "bull", verdictLabel: "โน้มเอียงฝั่งซื้อ (Bullish)", score, reasons };
  if (score <= -3) return { verdict: "bear", verdictLabel: "โน้มเอียงฝั่งขาย (Bearish)", score, reasons };
  return { verdict: "flat", verdictLabel: "ไม่มีสัญญาณชัดเจน (Neutral)", score, reasons };
}

export type Risk = { vol: number; annualReturn: number; maxDrawdown: number; sharpe: number | null };

/** Annualised with sqrt(252) like the original — only meaningful for daily bars. */
export function riskMetrics(bars: OHLC[]): Risk | null {
  if (bars.length < 30) return null;
  const rets: number[] = [];
  for (let i = 1; i < bars.length; i++) rets.push(bars[i].close / bars[i - 1].close - 1);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / (rets.length - 1));
  const vol = sd * Math.sqrt(252) * 100;
  const annualReturn = mean * 252 * 100;
  let peak = -Infinity;
  let mdd = 0;
  for (const b of bars) {
    peak = Math.max(peak, b.close);
    mdd = Math.min(mdd, ((b.close - peak) / peak) * 100);
  }
  return { vol, annualReturn, maxDrawdown: mdd, sharpe: vol > 0 ? annualReturn / vol : null };
}

/** Light summary used by the scanner and the idea lists. */
export type Mini = {
  last: number;
  w1: number;
  m1: number;
  m3: number;
  rsi: number;
  sig: "bull" | "bear" | "flat";
  belowSma20: boolean;
  belowSma50: boolean;
  s1: number | null;
};

export function miniStats(bars: OHLC[]): Mini | null {
  if (bars.length < 5) return null;
  const close = bars.map((b) => b.close);
  const last = close[close.length - 1];
  const chg = (d: number) => (close.length > d ? (last / close[close.length - 1 - d] - 1) * 100 : 0);
  const r = rsi(close).at(-1) as number;
  const s20 = sma(close, 20, false).at(-1) as number;
  const s50 = close.length >= 30 ? (sma(close, 50, false).at(-1) as number) : NaN;
  let sig: Mini["sig"] = "flat";
  if (last > s20 && (Number.isNaN(s50) || last > s50) && r < 70) sig = "bull";
  else if (last < s20 && (Number.isNaN(s50) || last < s50)) sig = "bear";
  const lv = bars.length >= 30 ? computeLevels(bars) : null;
  return {
    last, w1: chg(5), m1: chg(21), m3: chg(63), rsi: r, sig,
    belowSma20: last < s20,
    belowSma50: !Number.isNaN(s50) && last < s50,
    s1: lv?.support[0] ?? null,
  };
}
