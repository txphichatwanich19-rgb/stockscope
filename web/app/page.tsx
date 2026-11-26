import Link from "next/link";
import { connection } from "next/server";
import { Suspense } from "react";
import MacroStrip from "@/components/MacroStrip";
import SearchBox from "@/components/SearchBox";
import { WatchlistSummary } from "@/components/Watchlist";
import { SESSION_HINT, STATE_LABEL, sessionWindowsThai, usSession } from "@/lib/market";
import { POPULAR_UNIVERSE } from "@/lib/universe";
import { getQuote, getQuotes, getSectors, type Quote } from "@/lib/yahoo";
import { direction, displaySymbol, fmtPct, fmtPrice, fmtSigned } from "@/lib/format";

const GROUPS: { title: string; symbols: string[] }[] = [
  { title: "หุ้นยักษ์ใหญ่สหรัฐ", symbols: ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO"] },
  { title: "หุ้นไทย (SET)", symbols: ["PTT.BK", "CPALL.BK", "KBANK.BK", "ADVANC.BK", "AOT.BK", "DELTA.BK"] },
  { title: "คริปโต", symbols: ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"] },
];
const colsFor = (n: number) => (n % 4 === 0 ? 4 : 3);

function TickerCard({ q }: { q: Quote }) {
  const dir = direction(q.changePct);
  return (
    <Link href={`/stock/${encodeURIComponent(q.symbol)}`} className="card ticker-card">
      <div className="row">
        <span className="sym">{displaySymbol(q.symbol)}</span>
        <span className={`num ${dir}`}>{fmtPct(q.changePct)}</span>
      </div>
      <span className="name">{q.name}</span>
      <span className="num glow-text" style={{ fontSize: 20 }}>
        {fmtPrice(q.price)} <span className="unit">{q.currency}</span>
      </span>
    </Link>
  );
}

async function Group({ title, symbols }: { title: string; symbols: string[] }) {
  const quotes = await getQuotes(symbols);
  return (
    <section className="section">
      <div className="section-head"><h2>{title}</h2><span className="count">{quotes.length} รายการ</span></div>
      {quotes.length ? (
        <div className="ticker-grid" style={{ ["--cols" as string]: colsFor(quotes.length) }}>
          {quotes.map((q) => <TickerCard q={q} key={q.symbol} />)}
        </div>
      ) : <p style={{ color: "var(--muted)" }}>ดึงราคาไม่สำเร็จในขณะนี้</p>}
    </section>
  );
}

function Skeleton({ n, title }: { n: number; title?: string }) {
  return (
    <section className="section" aria-busy>
      {title && <div className="section-head"><h2>{title}</h2></div>}
      <div className="ticker-grid" style={{ ["--cols" as string]: colsFor(n) }}>
        {Array.from({ length: n }, (_, i) => <div className="card ticker-card skeleton" key={i} />)}
      </div>
    </section>
  );
}

// ---------- Sector heatmap ----------
async function Sectors() {
  const rows = await getSectors();
  if (!rows.length) return <p className="muted-note">ไม่สามารถโหลดข้อมูล sector ได้</p>;
  const tone = (d: number) =>
    d >= 1 ? "rgba(92,255,122,.9)" : d >= 0.3 ? "rgba(92,255,122,.55)" : d > -0.3 ? "rgba(255,255,255,.25)" : d > -1 ? "rgba(208,106,106,.55)" : "rgba(208,106,106,.9)";
  return (
    <>
      <div className="sector-grid">
        {rows.map((r) => (
          <Link key={r.symbol} href={`/stock/${r.symbol}`} className="card sector-cell" style={{ borderLeftColor: tone(r.d1) }}>
            <span className="sym" style={{ fontFamily: "var(--font-sans)", fontWeight: 600, letterSpacing: ".08em" }}>{r.symbol}</span>
            <span className="nm">{r.name}</span>
            <span className={`d1 num ${direction(r.d1)}`}>{fmtPct(r.d1)}</span>
            <span className="sub num">5D {fmtPct(r.d5)} · 1M {fmtPct(r.d20)}</span>
          </Link>
        ))}
      </div>
      <p className="muted-note">คลิก ETF เพื่อดูรายละเอียดเต็ม · สี = % เปลี่ยนแปลงวันนี้</p>
    </>
  );
}

// ---------- Top movers ----------
function MoverRow({ q, pct, price }: { q: string; pct: number; price: number | null }) {
  const dir = direction(pct);
  return (
    <Link href={`/stock/${encodeURIComponent(q)}`} className="card mover">
      <span className="s">{displaySymbol(q)}</span>
      <span className={`num ${dir}`}>{dir === "up" ? "▲" : "▼"} {fmtPct(pct)}</span>
      <span className="p num">{fmtPrice(price)}</span>
    </Link>
  );
}

async function TopMovers() {
  const quotes = (await getQuotes(POPULAR_UNIVERSE)).filter((q) => q.changePct !== null);
  const sorted = [...quotes].sort((a, b) => (b.changePct as number) - (a.changePct as number));
  const up = sorted.slice(0, 8);
  const down = sorted.slice(-8).reverse();
  return (
    <div className="movers">
      <div><h3 className="h-sm" style={{ marginTop: 0 }}>🟢 ราคาขึ้นมากที่สุด</h3>
        <div className="mover-list">{up.map((q) => <MoverRow key={q.symbol} q={q.symbol} pct={q.changePct as number} price={q.price} />)}</div></div>
      <div><h3 className="h-sm" style={{ marginTop: 0 }}>🔴 ราคาลงมากที่สุด</h3>
        <div className="mover-list">{down.map((q) => <MoverRow key={q.symbol} q={q.symbol} pct={q.changePct as number} price={q.price} />)}</div></div>
    </div>
  );
}

// ---------- Pre-market / after-hours ----------
type Ext = { symbol: string; label: "Pre" | "After"; price: number; change: number | null; pct: number };

function extOf(q: Quote): Ext | null {
  const pre = q.prePrice !== null && q.preChangePct !== null ? { label: "Pre" as const, price: q.prePrice, change: q.preChange, pct: q.preChangePct } : null;
  const post = q.postPrice !== null && q.postChangePct !== null ? { label: "After" as const, price: q.postPrice, change: q.postChange, pct: q.postChangePct } : null;
  const inPre = q.marketState === "PRE" || q.marketState === "PREPRE";
  const inPost = q.marketState === "POST" || q.marketState === "POSTPOST";
  const pick = inPre && pre ? pre : inPost && post ? post : pre ?? post;
  return pick ? { symbol: q.symbol, ...pick } : null;
}

async function ExtendedHours({ sp }: { sp: Promise<{ pm?: string }> }) {
  await connection(); // the US session depends on the current time
  const pmRaw = ((await sp).pm ?? "").trim().toUpperCase();
  const session = usSession();
  const hero = /^[A-Z0-9.^=-]{1,15}$/.test(pmRaw) ? await getQuote(pmRaw) : null;
  const heroExt = hero ? extOf(hero) : null;
  const all = (await getQuotes(POPULAR_UNIVERSE)).map(extOf).filter((e): e is Ext => e !== null);
  const gain = all.filter((e) => e.pct > 0).sort((a, b) => b.pct - a.pct).slice(0, 8);
  const lose = all.filter((e) => e.pct < 0).sort((a, b) => a.pct - b.pct).slice(0, 8);
  const preN = all.filter((e) => e.label === "Pre").length;
  const postN = all.length - preN;

  return (
    <>
      <p className="muted-note">{SESSION_HINT[session]} · {sessionWindowsThai()}</p>
      <form className="pm-form" action="/" method="get">
        <input name="pm" defaultValue={pmRaw} placeholder="พิมพ์ AAPL, NVDA, TSLA, BTC-USD … แล้วกด Enter" aria-label="ค้นหา Pre/After-Hours" />
        {pmRaw && <Link href="/" className="link-btn">เคลียร์ค้นหา</Link>}
      </form>

      {pmRaw && (hero && heroExt ? (
        <div className="card pm-hero">
          <span className="kicker">{heroExt.label === "Pre" ? "🌅 Pre-Market" : "🌙 After-Hours"} · {hero.symbol}</span>
          <span className="num glow-text" style={{ fontSize: 36 }}>{fmtPrice(heroExt.price)} <span className="unit">{hero.currency}</span></span>
          <span className={`num ${direction(heroExt.pct)}`}>{fmtSigned(heroExt.change)} ({fmtPct(heroExt.pct)})</span>
          <span className="muted-note">สถานะตลาด: {STATE_LABEL[hero.marketState] ?? hero.marketState}</span>
          <Link href={`/stock/${hero.symbol}`} className="link-btn">📊 ดูรายละเอียดเต็มของ {hero.symbol} →</Link>
        </div>
      ) : (
        <p className="warn-box">ไม่พบข้อมูล Pre/After-Hours สำหรับ {pmRaw} · อาจเป็นเพราะ: ticker ไม่ถูกต้อง, ไม่ใช่หุ้น US, หรือยังไม่มีคนเทรดในช่วงนี้</p>
      ))}

      {all.length ? (
        <>
          <p className="muted-note">📊 พบ {all.length} หุ้น · {preN && postN ? `🌅 Pre ${preN} · 🌙 After ${postN}` : preN ? "🌅 Pre-Market" : "🌙 After-Hours"}</p>
          <div className="movers">
            <div><h3 className="h-sm" style={{ marginTop: 0 }}>🟢 ราคาขึ้นมากที่สุด</h3><div className="mover-list">{gain.length ? gain.map((e) => <MoverRow key={e.symbol} q={e.symbol} pct={e.pct} price={e.price} />) : <span className="muted-note">ไม่มี</span>}</div></div>
            <div><h3 className="h-sm" style={{ marginTop: 0 }}>🔴 ราคาลงมากที่สุด</h3><div className="mover-list">{lose.length ? lose.map((e) => <MoverRow key={e.symbol} q={e.symbol} pct={e.pct} price={e.price} />) : <span className="muted-note">ไม่มี</span>}</div></div>
          </div>
        </>
      ) : (
        <p className="empty-note">ตอนนี้ไม่มีข้อมูล Pre/After-Hours — ถ้ายังไม่มีเลย แปลว่าตอนนี้เป็นช่วงตลาดปกติ / ดึก / วันหยุด</p>
      )}
    </>
  );
}

export default function Home({ searchParams }: PageProps<"/">) {
  return (
    <>
      <section className="hero-home">
        <div className="rings" aria-hidden />
        <div className="kicker" style={{ position: "relative" }}>Market terminal · US · SET · Crypto</div>
        <h1 className="glow-text hero-lines">
          <span>มองตลาดให้ไกลกว่าราคา</span>
          <span>เข้าใจสินทรัพย์ให้มากกว่าที่เคย</span>
        </h1>
        <SearchBox big />
        <div className="hint">กด Enter เพื่อเปิดหน้าหุ้น</div>
      </section>

      <MacroStrip />

      <section className="section">
        <div className="section-head"><h2>⭐ Watchlist ของคุณ</h2></div>
        <WatchlistSummary />
      </section>

      {GROUPS.map((g) => (
        <Suspense key={g.title} fallback={<Skeleton title={g.title} n={g.symbols.length} />}>
          <Group {...g} />
        </Suspense>
      ))}

      <section className="section">
        <div className="section-head"><h2>ภาพรวมตลาด US · Sector</h2><span className="count">SECTOR HEATMAP</span></div>
        <Suspense fallback={<Skeleton n={8} />}><Sectors /></Suspense>
      </section>

      <section className="section">
        <div className="section-head"><h2>Top Movers วันนี้</h2><span className="count">DAILY LEADERS</span></div>
        <Suspense fallback={<Skeleton n={8} />}><TopMovers /></Suspense>
      </section>

      <section className="section">
        <div className="section-head"><h2>Pre-Market / After-Hours</h2><span className="count">EXTENDED HOURS</span></div>
        <Suspense fallback={<Skeleton n={8} />}>
          <ExtendedHours sp={searchParams} />
        </Suspense>
      </section>

      <p className="source-note">
        ข้อมูลจาก Yahoo Finance อาจหน่วงตามเงื่อนไขของแต่ละตลาด และเก็บแคชไว้ประมาณ 1 นาที เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน
      </p>
    </>
  );
}
