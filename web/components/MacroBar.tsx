import { getQuotes } from "@/lib/yahoo";
import { direction, fmtPct, fmtPrice } from "@/lib/format";

const MACRO: [string, string][] = [
  ["^GSPC", "S&P 500"], ["^IXIC", "Nasdaq"], ["^VIX", "VIX"], ["^TNX", "10Y Yield (US)"], ["^SET.BK", "SET Index"], ["THB=X", "USD/THB"],
];

export default async function MacroBar() {
  const quotes = await getQuotes(MACRO.map(([s]) => s));
  const by = new Map(quotes.map((q) => [q.symbol, q]));
  return (
    <div className="macro" aria-label="ตัวชี้วัดตลาด">
      {MACRO.map(([sym, label]) => {
        const q = by.get(sym);
        if (!q) return null;
        const dir = direction(q.changePct);
        return (
          <div className="card macro-tile" key={sym}>
            <span className="label">{label}</span>
            <span className="v num glow-text">{fmtPrice(q.price)}</span>
            <span className={`num ${dir}`} style={{ fontSize: 12 }}>{dir === "up" ? "▲" : dir === "down" ? "▼" : "•"} {fmtPct(q.changePct)}</span>
          </div>
        );
      })}
    </div>
  );
}
