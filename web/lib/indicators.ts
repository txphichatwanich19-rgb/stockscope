// Indicator maths, ported from the Streamlit app (pandas rolling/ewm semantics).

/** Rolling mean; like pandas min_periods=1 when `strict` is false. */
export function sma(values: number[], window: number, strict = true): (number | null)[] {
  const out: (number | null)[] = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    const n = Math.min(i + 1, window);
    out.push(strict && i < window - 1 ? null : sum / n);
  }
  return out;
}

/** Exponential moving average with pandas ewm(span, adjust=False). */
export function ema(values: number[], span: number): number[] {
  const a = 2 / (span + 1);
  const out: number[] = [];
  for (let i = 0; i < values.length; i++) out.push(i === 0 ? values[0] : a * values[i] + (1 - a) * out[i - 1]);
  return out;
}

export function macd(values: number[], fast = 12, slow = 26, signal = 9) {
  const slowEma = ema(values, slow);
  const line = ema(values, fast).map((v, i) => v - slowEma[i]);
  const sig = ema(line, signal);
  return { line, signal: sig, hist: line.map((v, i) => v - sig[i]) };
}

/** Wilder RSI (ewm alpha = 1/period, adjust=False). Defined from the second bar on. */
export function rsi(values: number[], period = 14): number[] {
  const out: number[] = [50];
  const a = 1 / period;
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    const gain = Math.max(d, 0);
    const loss = Math.max(-d, 0);
    avgGain = i === 1 ? gain : a * gain + (1 - a) * avgGain;
    avgLoss = i === 1 ? loss : a * loss + (1 - a) * avgLoss;
    out.push(avgLoss === 0 ? (avgGain === 0 ? 50 : 100) : 100 - 100 / (1 + avgGain / avgLoss));
  }
  return out;
}
