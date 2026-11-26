// Chart period / bar-size options (same choices as the Streamlit sidebar). Safe to import anywhere.
export const PERIODS = [
  { key: "5d", label: "5 วัน", days: 5 },
  { key: "1mo", label: "1 เดือน", days: 31 },
  { key: "3mo", label: "3 เดือน", days: 92 },
  { key: "6mo", label: "6 เดือน", days: 183 },
  { key: "1y", label: "1 ปี", days: 366 },
  { key: "2y", label: "2 ปี", days: 731 },
  { key: "5y", label: "5 ปี", days: 1827 },
  { key: "10y", label: "10 ปี", days: 3653 },
  { key: "max", label: "ทั้งหมด", days: 0 },
] as const;
export type PeriodKey = (typeof PERIODS)[number]["key"];

export const INTERVALS = [
  { key: "1m", label: "1 นาที", maxDays: 7 },
  { key: "5m", label: "5 นาที", maxDays: 59 },
  { key: "15m", label: "15 นาที", maxDays: 59 },
  { key: "30m", label: "30 นาที", maxDays: 59 },
  { key: "1h", label: "1 ชั่วโมง", maxDays: 729 },
  { key: "1d", label: "1 วัน", maxDays: Infinity },
  { key: "1wk", label: "1 สัปดาห์", maxDays: Infinity },
  { key: "1mo", label: "1 เดือน", maxDays: Infinity },
] as const;
export type IntervalKey = (typeof INTERVALS)[number]["key"];

export const DEFAULT_PERIOD: PeriodKey = "1y";
export const DEFAULT_INTERVAL: IntervalKey = "1d";

export const isPeriod = (v: string | undefined): v is PeriodKey => PERIODS.some((p) => p.key === v);
export const isInterval = (v: string | undefined): v is IntervalKey => INTERVALS.some((i) => i.key === v);
export const isIntraday = (i: IntervalKey) => i === "1m" || i === "5m" || i === "15m" || i === "30m" || i === "1h";

/** Days of history to request, clamped to what Yahoo allows for that bar size. */
export function lookbackDays(period: PeriodKey, interval: IntervalKey): number {
  const want = PERIODS.find((p) => p.key === period)!.days || 365 * 40;
  const cap = INTERVALS.find((i) => i.key === interval)!.maxDays;
  return Math.min(want, cap);
}
