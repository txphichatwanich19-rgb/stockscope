import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import MacroStrip from "@/components/MacroStrip";
import { CATEGORIES } from "@/lib/universe";
import { getFunds, getMini } from "@/lib/yahoo";
import { fmtBig, fmtPrice } from "@/lib/format";

export const metadata: Metadata = { title: "สแกนหุ้น · Stockscope" };

type SP = { u?: string; sort?: string; pe?: string; rg?: string; pm?: string; m3?: string; rmin?: string; rmax?: string; buy?: string; run?: string };

const SORTS: Record<string, string> = {
  mcap: "Market Cap (สูง→ต่ำ)",
  pe: "P/E (ต่ำ→สูง)",
  rg: "Revenue Growth (สูง→ต่ำ)",
  m3: "Performance 3 เดือน (สูง→ต่ำ)",
  m1: "Performance 1 เดือน (สูง→ต่ำ)",
  rsi: "RSI (ต่ำ→สูง · oversold ก่อน)",
  pm: "Profit Margin (สูง→ต่ำ)",
};
const ALL = [...new Set(CATEGORIES.flatMap((c) => c.symbols))].sort();
const toNum = (v: string | undefined, d: number) => (v !== undefined && v !== "" && Number.isFinite(Number(v)) ? Number(v) : d);
const rec = (m: number | null) => (m === null ? "—" : m <= 1.5 ? "🟢 Strong Buy" : m <= 2.5 ? "🟢 Buy" : m <= 3.5 ? "🟡 Hold" : m <= 4.5 ? "🔴 Sell" : "🔴 Strong Sell");

export default function ScanPage({ searchParams }: PageProps<"/scan">) {
  return (
    <>
      <section className="section" style={{ marginTop: 40 }}>
        <div className="kicker">STOCK SCREENER</div>
        <h1 className="glow-text" style={{ fontSize: 34, margin: "10px 0" }}>สแกนหุ้น</h1>
        <p className="muted-note">กรองหุ้นจากทั่วทุกหมวดด้วยเกณฑ์ที่ตั้งเอง · ตั้งค่า filter แล้วกด “เริ่มสแกน” (ถ้าเลือกทุกหมวด ใช้เวลาครั้งแรกประมาณครึ่งนาทีถึงหนึ่งนาที ครั้งต่อไปเร็วขึ้นเพราะเก็บแคชไว้)</p>
        <Suspense fallback={null}><Form sp={searchParams} /></Suspense>
      </section>
      <Suspense fallback={<p className="empty-note">กำลังตั้งค่า…</p>}>
        <Results sp={searchParams} />
      </Suspense>
      <MacroStrip />
    </>
  );
}

async function Form({ sp }: { sp: Promise<SP> }) {
  const p = await sp;
  return (
    <form className="card scan-form" action="/scan" method="get">
      <input type="hidden" name="run" value="1" />
      <label>🌐 Universe
        <select name="u" defaultValue={p.u ?? "all"}>
          <option value="all">ทุกหมวดรวมกัน ({ALL.length} ตัว)</option>
          {CATEGORIES.map((c) => <option key={c.title} value={c.title}>{c.title}</option>)}
        </select></label>
      <label>🔢 เรียงโดย
        <select name="sort" defaultValue={p.sort ?? "mcap"}>
          {Object.entries(SORTS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select></label>
      <label>P/E ≤<input name="pe" type="number" step="5" min="0" max="500" defaultValue={p.pe ?? "100"} /></label>
      <label>Revenue Growth ≥ (%)<input name="rg" type="number" min="-50" max="200" defaultValue={p.rg ?? "-50"} /></label>
      <label>Profit Margin ≥ (%)<input name="pm" type="number" min="-100" max="80" defaultValue={p.pm ?? "-100"} /></label>
      <label>3M Performance ≥ (%)<input name="m3" type="number" min="-90" max="300" defaultValue={p.m3 ?? "-90"} /></label>
      <label>RSI ตั้งแต่<input name="rmin" type="number" min="0" max="100" defaultValue={p.rmin ?? "0"} /></label>
      <label>RSI ถึง<input name="rmax" type="number" min="0" max="100" defaultValue={p.rmax ?? "100"} /></label>
      <label className="check full"><input type="checkbox" name="buy" value="1" defaultChecked={p.buy === "1"} />เฉพาะที่นักวิเคราะห์ Buy / Strong Buy (คะแนนเฉลี่ย ≤ 2.5)</label>
      <div className="full"><button className="btn" type="submit">🔍 เริ่มสแกน</button></div>
    </form>
  );
}

async function Results({ sp }: { sp: Promise<SP> }) {
  const p = await sp;
  if (p.run !== "1") return null;
  const universe = p.u && p.u !== "all" ? (CATEGORIES.find((c) => c.title === p.u)?.symbols ?? ALL) : ALL;
  const f = { pe: toNum(p.pe, 100), rg: toNum(p.rg, -50), pm: toNum(p.pm, -100), m3: toNum(p.m3, -90), rmin: toNum(p.rmin, 0), rmax: toNum(p.rmax, 100), buy: p.buy === "1" };
  const sort = p.sort && p.sort in SORTS ? p.sort : "mcap";

  const [mini, fund] = await Promise.all([getMini(universe), getFunds(universe)]);
  const rows = Object.entries(mini)
    .map(([s, d]) => ({ s, d, f: fund[s] }))
    .filter(({ d, f: fu }) => {
      if (!d.last) return false;
      if (fu.pe && fu.pe > f.pe) return false;
      if (fu.revGrowth !== null && fu.revGrowth * 100 < f.rg) return false;
      if (fu.profitMargin !== null && fu.profitMargin * 100 < f.pm) return false;
      if (d.m3 < f.m3) return false;
      if (d.rsi < f.rmin || d.rsi > f.rmax) return false;
      if (f.buy && (fu.recMean === null || fu.recMean > 2.5)) return false;
      return true;
    });
  const key: Record<string, (r: (typeof rows)[number]) => number> = {
    mcap: (r) => -(r.f.marketCap ?? 0), pe: (r) => r.f.pe ?? 99999, rg: (r) => -(r.f.revGrowth ?? -999),
    m3: (r) => -r.d.m3, m1: (r) => -r.d.m1, rsi: (r) => r.d.rsi, pm: (r) => -(r.f.profitMargin ?? -999),
  };
  rows.sort((a, b) => key[sort](a) - key[sort](b));
  const shown = rows.slice(0, 50);
  const sgn = (v: number, d = 1) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;

  return (
    <section className="section">
      <p className="rr">✅ พบ {rows.length} หุ้น (แสดง {shown.length} อันดับแรก) จากที่สแกนได้ {Object.keys(mini).length} ตัว</p>
      {shown.length > 0 && (
        <div className="card tbl-box" style={{ padding: "8px 8px" }}>
          <div className="tbl-wrap">
            <table>
              <thead><tr><th>Ticker</th><th>ราคา</th><th>Mcap</th><th>P/E</th><th>Rev%</th><th>Margin%</th><th>RSI</th><th>1M%</th><th>3M%</th><th>Analyst</th></tr></thead>
              <tbody>
                {shown.map(({ s, d, f: fu }) => (
                  <tr key={s}>
                    <td><Link href={`/stock/${encodeURIComponent(s)}`} style={{ textDecoration: "underline", textUnderlineOffset: 3 }}>{s.replace(".BK", "").replace("-USD", "")}</Link></td>
                    <td className="num">{fmtPrice(d.last)}</td>
                    <td className="num">{fmtBig(fu.marketCap)}</td>
                    <td className="num">{fu.pe ? fu.pe.toFixed(1) : "—"}</td>
                    <td className="num">{fu.revGrowth !== null ? sgn(fu.revGrowth * 100, 0) : "—"}</td>
                    <td className="num">{fu.profitMargin !== null ? (fu.profitMargin * 100).toFixed(0) : "—"}</td>
                    <td className="num">{d.rsi.toFixed(0)}</td>
                    <td className={`num ${d.m1 >= 0 ? "up" : "down"}`}>{sgn(d.m1)}</td>
                    <td className={`num ${d.m3 >= 0 ? "up" : "down"}`}>{sgn(d.m3)}</td>
                    <td>{rec(fu.recMean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <p className="muted-note" style={{ marginTop: 12 }}>
        หุ้นที่ไม่มีค่า P/E, รายได้ หรือกำไรสุทธิ จะไม่ถูกตัดออกด้วยเกณฑ์นั้น (ยกเว้นเกณฑ์ “นักวิเคราะห์ Buy” ที่ต้องมีข้อมูล) · เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน
      </p>
    </section>
  );
}
