/* eslint-disable @typescript-eslint/no-explicit-any -- Yahoo payloads are loosely shaped; every field is guarded by num()/iso() */
import "server-only";
import YahooFinance from "yahoo-finance2";
import { cacheLife } from "next/cache";
import { miniStats, type Mini, type OHLC } from "./analysis";
import { isIntraday, lookbackDays, type IntervalKey, type PeriodKey } from "./ranges";

// Yahoo Finance is an unofficial source: quotes can be delayed and the endpoints can change.
const yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });
const LENIENT = { validateResult: false } as const;

// With validation off the library types its results as `unknown`; Yahoo's payloads are loosely
// shaped anyway, so read them as plain objects and guard every field with num()/iso().
const qsum = (symbol: string, modules: string[]): Promise<any> => (yf.quoteSummary as any)(symbol, { modules }, LENIENT);
const qquote = (symbols: string[]): Promise<any[]> => (yf.quote as any)(symbols, {}, LENIENT);
const qchart = (symbol: string, opts: object): Promise<any> => (yf.chart as any)(symbol, opts, LENIENT);
const qseries = (symbol: string, opts: object): Promise<any[]> => (yf.fundamentalsTimeSeries as any)(symbol, opts, LENIENT);

export type Quote = {
  symbol: string;
  name: string;
  price: number | null;
  change: number | null;
  changePct: number | null;
  currency: string;
  exchange: string;
  marketState: string; // REGULAR, PRE, POST, CLOSED...
  marketTime: string | null; // ISO
  postPrice: number | null;
  postChange: number | null;
  postChangePct: number | null;
  prePrice: number | null;
  preChange: number | null;
  preChangePct: number | null;
  volume: number | null;
  marketCap: number | null;
  dayLow: number | null;
  dayHigh: number | null;
  low52: number | null;
  high52: number | null;
  pe: number | null;
  forwardPe: number | null;
  dividendYield: number | null; // percent
  beta: number | null;
  eps: number | null;
  priceToBook: number | null;
  avgVolume: number | null;
};

export type Bar = OHLC;

export type Profile = {
  sector: string | null;
  industry: string | null;
  website: string | null;
  summary: string | null;
  employees: number | null;
  officers: { name: string; title: string; age: number | null; pay: number | null }[];
};

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const iso = (v: unknown): string | null => (v instanceof Date ? v.toISOString() : null);

/** Run `fn` over `items` with at most `limit` in flight (Yahoo throttles big bursts). */
async function pool<T, R>(items: T[], limit: number, fn: (x: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const i = next++;
        out[i] = await fn(items[i]);
      }
    }),
  );
  return out;
}
function toQuote(q: any): Quote | null {
  if (!q || num(q.regularMarketPrice) === null) return null;
  return {
    symbol: q.symbol,
    name: q.longName || q.shortName || q.symbol,
    price: num(q.regularMarketPrice),
    change: num(q.regularMarketChange),
    changePct: num(q.regularMarketChangePercent),
    currency: q.currency || "USD",
    exchange: q.fullExchangeName || q.exchange || "",
    marketState: q.marketState || "",
    marketTime: iso(q.regularMarketTime),
    postPrice: num(q.postMarketPrice),
    postChange: num(q.postMarketChange),
    postChangePct: num(q.postMarketChangePercent),
    prePrice: num(q.preMarketPrice),
    preChange: num(q.preMarketChange),
    preChangePct: num(q.preMarketChangePercent),
    volume: num(q.regularMarketVolume),
    marketCap: num(q.marketCap),
    dayLow: num(q.regularMarketDayLow),
    dayHigh: num(q.regularMarketDayHigh),
    low52: num(q.fiftyTwoWeekLow),
    high52: num(q.fiftyTwoWeekHigh),
    pe: num(q.trailingPE),
    forwardPe: num(q.forwardPE),
    dividendYield: num(q.trailingAnnualDividendYield) !== null ? (q.trailingAnnualDividendYield as number) * 100 : null,
    beta: null,
    eps: num(q.epsTrailingTwelveMonths),
    priceToBook: num(q.priceToBook),
    avgVolume: num(q.averageDailyVolume3Month),
  };
}

async function fetchQuotes(symbols: string[]): Promise<Quote[]> {
  if (!symbols.length) return [];
  try {
    const raw = await qquote(symbols);
    const bySym = new Map(raw.map((q) => [q.symbol, toQuote(q)]));
    // keep the requested order, drop symbols Yahoo did not return
    return symbols.map((s) => bySym.get(s)).filter((q): q is Quote => !!q);
  } catch {
    return [];
  }
}

async function fetchHistory(symbol: string, period: PeriodKey, interval: IntervalKey): Promise<Bar[]> {
  return fetchBars(symbol, lookbackDays(period, interval), interval);
}

async function fetchBars(symbol: string, days: number, interval: IntervalKey): Promise<Bar[]> {
  try {
    const c = await qchart(symbol, { period1: new Date(Date.now() - days * 864e5), interval });
    return c.quotes
      .filter((b: any) => b.open != null && b.high != null && b.low != null && b.close != null)
      .map((b: any) => ({
        time: isIntraday(interval) ? b.date.toISOString() : b.date.toISOString().slice(0, 10),
        open: b.open as number,
        high: b.high as number,
        low: b.low as number,
        close: b.close as number,
        volume: b.volume ?? 0,
      }))
      // the chart needs strictly increasing, unique dates
      .filter((b: any, i: any, a: any) => i === 0 || b.time > a[i - 1].time);
  } catch {
    return [];
  }
}

async function fetchProfile(symbol: string): Promise<Profile | null> {
  try {
    const s = await qsum(symbol, ["assetProfile"]);
    const p = s.assetProfile;
    if (!p) return null;
    return {
      sector: p.sector || null,
      industry: p.industry || null,
      website: p.website || null,
      summary: p.longBusinessSummary || null,
      employees: num(p.fullTimeEmployees),
      officers: (p.companyOfficers ?? []).map((o: any) => ({
        name: o.name,
        title: o.title,
        age: num(o.age),
        pay: num(o.totalPay),
      })),
    };
  } catch {
    return null;
  }
}

// ---------- Fundamentals / stats tab ----------
export type Stats = {
  beta: number | null;
  peg: number | null;
  payoutRatio: number | null;
  profitMargin: number | null;
  roe: number | null;
  revenueGrowth: number | null;
  dividendYield: number | null; // percent
  recMean: number | null;
  targetMean: number | null;
  analystCount: number | null;
};

async function fetchStats(symbol: string): Promise<Stats | null> {
  try {
    const s = await qsum(symbol, ["summaryDetail", "defaultKeyStatistics", "financialData"]);
    const d = s.summaryDetail;
    const k = s.defaultKeyStatistics;
    const f = s.financialData;
    return {
      beta: num(d?.beta) ?? num(k?.beta),
      peg: num(k?.pegRatio),
      payoutRatio: num(d?.payoutRatio),
      profitMargin: num(f?.profitMargins) ?? num(k?.profitMargins),
      roe: num(f?.returnOnEquity),
      revenueGrowth: num(f?.revenueGrowth),
      dividendYield: num(d?.dividendYield) !== null ? (d?.dividendYield as number) * 100 : null,
      recMean: num(f?.recommendationMean),
      targetMean: num(f?.targetMeanPrice),
      analystCount: num(f?.numberOfAnalystOpinions),
    };
  } catch {
    return null;
  }
}

export type QuarterRow = { date: string; revenue: number | null; netIncome: number | null };

async function fetchQuarterly(symbol: string): Promise<QuarterRow[]> {
  try {
    const rows = await qseries(symbol, { period1: new Date(Date.now() - 800 * 864e5), type: "quarterly", module: "financials" });
    return rows
      .map((r: any) => ({ date: (r.date as Date).toISOString().slice(0, 10), revenue: num(r.totalRevenue), netIncome: num(r.netIncome) }))
      .filter((r) => r.revenue !== null)
      .sort((a, b) => a.date.localeCompare(b.date));
  } catch {
    return [];
  }
}

export type Insider = {
  date: string;
  kind: "buy" | "sell" | "gift" | "other";
  name: string;
  position: string;
  shares: number | null;
  value: number | null;
  text: string;
};

export function classifyInsider(text: string): Insider["kind"] {
  const t = text.toLowerCase();
  if (/sale|sold|sell/.test(t)) return "sell";
  if (/purchase|buy|bought|acquired/.test(t)) return "buy";
  if (/gift/.test(t)) return "gift";
  return "other";
}

async function fetchInsiders(symbol: string): Promise<Insider[]> {
  try {
    const s = await qsum(symbol, ["insiderTransactions"]);
    const cutoff = Date.now() - 180 * 864e5;
    return (s.insiderTransactions?.transactions ?? [])
      .filter((t: any) => t.startDate instanceof Date && t.startDate.getTime() >= cutoff)
      .map((t: any) => ({
        date: (t.startDate as Date).toISOString().slice(0, 10),
        kind: classifyInsider(`${t.transactionText ?? ""} ${(t as { ownership?: string }).ownership ?? ""}`),
        name: t.filerName ?? "",
        position: t.filerRelation ?? "",
        shares: num(t.shares),
        value: num(t.value),
        text: t.transactionText ?? "",
      }))
      .sort((a: any, b: any) => b.date.localeCompare(a.date));
  } catch {
    return [];
  }
}

export type EarningsInfo = {
  nextDate: string | null;
  nextEpsEstimate: number | null;
  history: { quarter: string; estimate: number | null; actual: number | null; surprisePct: number | null }[];
};

async function fetchEarnings(symbol: string): Promise<EarningsInfo> {
  try {
    const s = await qsum(symbol, ["earningsHistory", "calendarEvents"]);
    const e = s.calendarEvents?.earnings;
    const next = e?.earningsDate?.find((d: any) => d.getTime() > Date.now()) ?? null;
    return {
      nextDate: next ? next.toISOString() : null,
      nextEpsEstimate: num(e?.earningsAverage),
      history: (s.earningsHistory?.history ?? [])
        .map((h: any) => ({
          quarter: h.quarter instanceof Date ? h.quarter.toISOString().slice(0, 7) : "",
          estimate: num(h.epsEstimate),
          actual: num(h.epsActual),
          surprisePct: num(h.surprisePercent) !== null ? (h.surprisePercent as number) * 100 : null,
        }))
        .sort((a: any, b: any) => b.quarter.localeCompare(a.quarter))
        .slice(0, 4),
    };
  } catch {
    return { nextDate: null, nextEpsEstimate: null, history: [] };
  }
}

export type Upgrade = { date: string; firm: string; action: string; fromGrade: string; toGrade: string; targetAction: string };

async function fetchUpgrades(symbol: string): Promise<Upgrade[]> {
  try {
    const s = await qsum(symbol, ["upgradeDowngradeHistory"]);
    return (s.upgradeDowngradeHistory?.history ?? []).slice(0, 8).map((h: any) => ({
      date: h.epochGradeDate instanceof Date ? h.epochGradeDate.toISOString().slice(0, 10) : "",
      firm: h.firm ?? "",
      action: h.action ?? "",
      fromGrade: h.fromGrade ?? "",
      toGrade: h.toGrade ?? "",
      targetAction: (h as { priceTargetAction?: string }).priceTargetAction ?? "",
    }));
  } catch {
    return [];
  }
}

// ---------- Scanner / ideas ----------
export type Fund = {
  name: string;
  marketCap: number | null;
  revGrowth: number | null; // decimal
  profitMargin: number | null; // decimal
  recMean: number | null;
  targetMean: number | null;
  pe: number | null;
};

async function fetchMiniOne(symbol: string): Promise<Mini | null> {
  return miniStats(await fetchBars(symbol, 190, "1d"));
}

type FinOne = { revGrowth: number | null; profitMargin: number | null; recMean: number | null; targetMean: number | null };
async function fetchFinOne(symbol: string): Promise<FinOne> {
  try {
    const f = (await qsum(symbol, ["financialData"])).financialData;
    return { revGrowth: num(f?.revenueGrowth), profitMargin: num(f?.profitMargins), recMean: num(f?.recommendationMean), targetMean: num(f?.targetMeanPrice) };
  } catch {
    return { revGrowth: null, profitMargin: null, recMean: null, targetMean: null };
  }
}

export type InsiderScan = { buyValue: number; buyCount: number; sellCount: number };

async function fetchInsiderScanOne(symbol: string): Promise<InsiderScan> {
  const tx = await fetchInsiders(symbol);
  let buyValue = 0, buyCount = 0, sellCount = 0;
  for (const t of tx) {
    if (t.kind === "buy") { buyCount++; buyValue += t.value ?? 0; }
    else if (t.kind === "sell") sellCount++;
  }
  return { buyValue, buyCount, sellCount };
}

type Div = { yieldPct: number; payout: number | null } | null;
async function fetchDividendOne(symbol: string): Promise<Div> {
  try {
    const d = (await qsum(symbol, ["summaryDetail"])).summaryDetail;
    const y = num(d?.dividendYield);
    return y ? { yieldPct: y * 100, payout: num(d?.payoutRatio) } : null;
  } catch {
    return null;
  }
}

// ---------- Market overview ----------
export type SectorRow = { symbol: string; name: string; d1: number; d5: number; d20: number };

export const SECTORS: [string, string][] = [
  ["XLK", "เทคโนโลยี"], ["XLF", "การเงิน"], ["XLV", "สุขภาพ"], ["XLE", "พลังงาน"], ["XLI", "อุตสาหกรรม"],
  ["XLY", "อุปโภคบริโภค (สมัครใจ)"], ["XLP", "อุปโภคบริโภค (จำเป็น)"], ["XLU", "สาธารณูปโภค"],
  ["XLB", "วัตถุดิบ"], ["XLRE", "อสังหา"], ["XLC", "สื่อสาร"],
];

async function fetchSectors(): Promise<SectorRow[]> {
  const rows = await pool(SECTORS, 6, async ([symbol, name]) => {
    const bars = await fetchBars(symbol, 45, "1d");
    if (bars.length < 2) return null;
    const c = bars.map((b) => b.close);
    const last = c[c.length - 1];
    const back = (d: number) => (c.length > d ? (last / c[c.length - 1 - d] - 1) * 100 : 0);
    return { symbol, name, d1: back(1), d5: back(5), d20: back(20) };
  });
  return rows.filter((r): r is SectorRow => r !== null).sort((a, b) => b.d1 - a.d1);
}

// ---------- Cached accessors ----------
// Prices refresh in the background about once a minute; slower-moving data is cached longer.
export async function getQuote(symbol: string) {
  "use cache";
  cacheLife({ stale: 30, revalidate: 60, expire: 3600 });
  return (await fetchQuotes([symbol]))[0] ?? null;
}

export async function getQuotes(symbols: string[]) {
  "use cache";
  cacheLife({ stale: 30, revalidate: 60, expire: 3600 });
  return fetchQuotes(symbols);
}

export async function getHistory(symbol: string, period: PeriodKey, interval: IntervalKey) {
  "use cache";
  cacheLife({ stale: 30, revalidate: 60, expire: 86400 });
  return fetchHistory(symbol, period, interval);
}

export async function getProfile(symbol: string) {
  "use cache";
  cacheLife("days");
  return fetchProfile(symbol);
}

export async function getStats(symbol: string) {
  "use cache";
  cacheLife("hours");
  return fetchStats(symbol);
}

export async function getQuarterly(symbol: string) {
  "use cache";
  cacheLife("hours");
  return fetchQuarterly(symbol);
}

export async function getInsiders(symbol: string) {
  "use cache";
  cacheLife({ stale: 300, revalidate: 900, expire: 86400 });
  return fetchInsiders(symbol);
}

export async function getEarnings(symbol: string) {
  "use cache";
  cacheLife("hours");
  return fetchEarnings(symbol);
}

export async function getUpgrades(symbol: string) {
  "use cache";
  cacheLife("hours");
  return fetchUpgrades(symbol);
}

// Per-symbol caches, so a second scan over an overlapping universe reuses earlier work.
async function miniOne(symbol: string) {
  "use cache";
  cacheLife({ stale: 120, revalidate: 600, expire: 3600 });
  return fetchMiniOne(symbol);
}
async function finOne(symbol: string) {
  "use cache";
  cacheLife("hours");
  return fetchFinOne(symbol);
}
async function insiderOne(symbol: string) {
  "use cache";
  cacheLife({ stale: 300, revalidate: 900, expire: 86400 });
  return fetchInsiderScanOne(symbol);
}
async function dividendOne(symbol: string) {
  "use cache";
  cacheLife("hours");
  return fetchDividendOne(symbol);
}

export async function getMini(symbols: string[]): Promise<Record<string, Mini>> {
  const rows = await pool(symbols, 8, async (s) => [s, await miniOne(s)] as const);
  return Object.fromEntries(rows.filter((r): r is readonly [string, Mini] => r[1] !== null));
}

export async function getFunds(symbols: string[]): Promise<Record<string, Fund>> {
  const [quotes, fin] = await Promise.all([getQuotes(symbols), pool(symbols, 8, finOne)]);
  const q = new Map(quotes.map((x) => [x.symbol, x]));
  const out: Record<string, Fund> = {};
  symbols.forEach((s, i) => {
    const qq = q.get(s);
    out[s] = { name: qq?.name ?? s, marketCap: qq?.marketCap ?? null, pe: qq?.pe ?? null, ...fin[i] };
  });
  return out;
}

export async function getInsiderScan(symbols: string[]): Promise<Record<string, InsiderScan>> {
  const rows = await pool(symbols, 8, insiderOne);
  return Object.fromEntries(symbols.map((s, i) => [s, rows[i]]));
}

export async function getDividends(symbols: string[]): Promise<Record<string, { yieldPct: number; payout: number | null }>> {
  const rows = await pool(symbols, 8, dividendOne);
  return Object.fromEntries(symbols.flatMap((s, i) => (rows[i] ? [[s, rows[i] as NonNullable<Div>]] : [])));
}

export async function getSectors() {
  "use cache";
  cacheLife({ stale: 120, revalidate: 300, expire: 3600 });
  return fetchSectors();
}
