import "server-only";
import { CATEGORIES, PICKS } from "./universe";
import {
  getDividends, getFunds, getInsiderScan, getMini, getQuotes, type Fund,
} from "./yahoo";
import type { Mini } from "./analysis";

export type PickRow = { symbol: string; meta: string };
export type PickResult = { rows: PickRow[]; empty: string };

const sgn = (v: number, d = 1) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}%`;
const recTh = (mean: number | null) =>
  mean === null ? "" : mean <= 1.5 ? "🟢 Strong Buy" : mean <= 2.5 ? "🟢 Buy" : mean <= 3.5 ? "🟡 Hold" : mean <= 4.5 ? "🔴 Sell" : "🔴 Strong Sell";

/** Same selection rules as the Streamlit "ไอเดีย" lists. */
export async function runPick(id: string): Promise<PickResult | null> {
  const cfg = PICKS.find((p) => p.id === id);
  if (!cfg) return null;
  const U = cfg.universe;

  switch (cfg.mode) {
    case "momentum": {
      const mini = await getMini(U);
      const rows = Object.entries(mini).sort((a, b) => b[1].w1 - a[1].w1).slice(0, 12)
        .map(([s, d]) => ({ symbol: s, meta: `1W ${sgn(d.w1)} · RSI ${d.rsi.toFixed(0)}` }));
      return { rows, empty: "ไม่มีข้อมูล" };
    }
    case "avoid": {
      const mini = await getMini(U);
      const flagged: { symbol: string; meta: string; rsi: number }[] = [];
      for (const s of U) {
        const d = mini[s];
        if (!d) continue;
        if (d.rsi > 75) flagged.push({ symbol: s, meta: `RSI ${d.rsi.toFixed(0)} · ร้อนเกิน`, rsi: d.rsi });
        else if (d.belowSma20 && d.belowSma50) flagged.push({ symbol: s, meta: "ใต้ SMA20 & 50 · trend พัง", rsi: d.rsi });
      }
      flagged.sort((a, b) => b.rsi - a.rsi);
      return { rows: flagged.slice(0, 8), empty: "ยังไม่มีตัวติดสัญญาณเตือน" };
    }
    case "auto_small_growth": {
      const [mini, fund] = await Promise.all([getMini(U), getFunds(U)]);
      const scored = Object.entries(mini)
        .filter(([, d]) => !(d.rsi >= 75 || (d.belowSma20 && d.belowSma50)))
        .map(([s, d]) => {
          const f = fund[s];
          return { s, d, f, score: d.m1 + (f?.revGrowth ?? 0) * 80 + (3 - (f?.recMean ?? 3)) * 12 };
        })
        .sort((a, b) => b.score - a.score).slice(0, 12);
      return {
        rows: scored.map(({ s, d, f }) => ({
          symbol: s,
          meta: [`1M ${sgn(d.m1)}`, f.revGrowth !== null && `รายได้ ${sgn(f.revGrowth * 100, 0)}`, `RSI ${d.rsi.toFixed(0)}`, recTh(f.recMean)].filter(Boolean).join(" · "),
        })),
        empty: "ไม่มีหุ้นที่ผ่านเกณฑ์",
      };
    }
    case "auto_future": {
      const [mini, fund] = await Promise.all([getMini(U), getFunds(U)]);
      const scored = Object.entries(mini)
        .map(([s, d]) => ({ s, d, f: fund[s] as Fund }))
        .filter(({ f }) => (f.revGrowth ?? 0) > 0.03 || (f.profitMargin ?? 0) > 0.1 || (f.recMean !== null && f.recMean <= 2.7))
        .map((x) => ({
          ...x,
          score: x.d.m3 + (x.f.revGrowth ?? 0) * 100 + (x.f.profitMargin ?? 0) * 50 + (x.f.recMean !== null ? (3 - x.f.recMean) * 15 : 0),
        }))
        .sort((a, b) => b.score - a.score).slice(0, 12);
      return {
        rows: scored.map(({ s, d, f }) => ({
          symbol: s,
          meta: [
            `3M ${sgn(d.m3)}`,
            f.revGrowth !== null && `รายได้ ${sgn(f.revGrowth * 100, 0)}`,
            f.profitMargin !== null && `กำไรสุทธิ ${(f.profitMargin * 100).toFixed(0)}%`,
            recTh(f.recMean),
            f.targetMean !== null && d.last ? `🎯 เป้า ${f.targetMean.toFixed(0)} (${sgn((f.targetMean / d.last - 1) * 100, 0)})` : "",
          ].filter(Boolean).join(" · "),
        })),
        empty: "ไม่มีหุ้นที่ผ่านเกณฑ์",
      };
    }
    case "auto_buyzone": {
      const mini = await getMini(U);
      const rows = Object.entries(mini)
        .filter(([, d]) => d.s1 && d.last)
        .map(([s, d]) => ({ s, d, dist: (d.last / (d.s1 as number) - 1) * 100 }))
        .filter((x) => x.dist >= -1 && x.dist <= 2.5)
        .sort((a, b) => Math.abs(a.dist) - Math.abs(b.dist)).slice(0, 10)
        .map(({ s, d, dist }) => ({ symbol: s, meta: `S1 ${(d.s1 as number).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} · ห่าง ${sgn(dist)} · RSI ${d.rsi.toFixed(0)}` }));
      return { rows, empty: "ตอนนี้ยังไม่มีหุ้นที่ราคาเข้าโซนซื้อ (ใกล้แนวรับ S1 ตั้งแต่ −1% ถึง +2.5%)" };
    }
    case "auto_oversold": {
      const mini = await getMini(U);
      const rows = Object.entries(mini).filter(([, d]) => d.rsi < 35).sort((a, b) => a[1].rsi - b[1].rsi).slice(0, 12)
        .map(([s, d]) => ({ symbol: s, meta: `RSI ${d.rsi.toFixed(0)} · 1M ${sgn(d.m1)} · 3M ${sgn(d.m3)}` }));
      return { rows, empty: "ไม่มีหุ้นไหน Oversold ตอนนี้ (RSI < 35)" };
    }
    case "auto_options": {
      const mini = await getMini(U);
      const bias = (d: Mini) => (d.sig === "bull" ? "📈 Call bias" : d.sig === "bear" ? "📉 Put bias" : "⚪ Neutral");
      const rows = Object.entries(mini).sort((a, b) => Math.abs(b[1].w1) - Math.abs(a[1].w1)).slice(0, 12)
        .map(([s, d]) => ({ symbol: s, meta: `σ ${Math.abs(d.w1).toFixed(1)}% · ${bias(d)}` }));
      return { rows, empty: "ไม่มีข้อมูล" };
    }
    case "auto_dividend": {
      const div = await getDividends(U);
      const picked = Object.entries(div)
        .filter(([, d]) => d.yieldPct >= 3 && !(d.payout !== null && d.payout > 0.85))
        .sort((a, b) => b[1].yieldPct - a[1].yieldPct).slice(0, 12);
      const quotes = await getQuotes(picked.map(([s]) => s));
      const price = new Map(quotes.map((q) => [q.symbol, q.price]));
      return {
        rows: picked.map(([s, d]) => ({
          symbol: s,
          meta: [`Yield ${d.yieldPct.toFixed(2)}%`, d.payout ? `จ่าย ${(d.payout * 100).toFixed(0)}% ของกำไร` : "", price.get(s) ? `ราคา ${(price.get(s) as number).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : ""].filter(Boolean).join(" · "),
        })),
        empty: "ไม่พบหุ้นปันผลที่ผ่านเกณฑ์",
      };
    }
    case "auto_insider_buy": {
      const scan = await getInsiderScan(U);
      const rows = Object.entries(scan)
        .filter(([, d]) => d.buyCount > 0 && d.buyCount >= d.sellCount)
        .sort((a, b) => b[1].buyValue - a[1].buyValue).slice(0, 10)
        .map(([s, d]) => ({
          symbol: s,
          meta: [d.buyValue > 0 ? `ซื้อ ${d.buyCount} ครั้ง · $${(d.buyValue / 1e6).toFixed(2)}M` : `ซื้อ ${d.buyCount} ครั้ง`, d.sellCount > 0 ? `ขาย ${d.sellCount} ครั้ง` : ""].filter(Boolean).join(" · "),
        }));
      return { rows, empty: "ไม่พบหุ้นที่มี insider ซื้อใน 6 เดือนล่าสุด" };
    }
  }
}

export { CATEGORIES, PICKS };
