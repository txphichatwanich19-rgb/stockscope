import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import MacroStrip from "@/components/MacroStrip";
import { runPick } from "@/lib/picks";
import { CATEGORIES, PICKS } from "@/lib/universe";
import { getQuotes } from "@/lib/yahoo";
import { direction, displaySymbol, fmtPct, fmtPrice } from "@/lib/format";

export const metadata: Metadata = { title: "หมวดหมู่และไอเดียลงทุน · Stockscope" };

type SP = { c?: string; p?: string };

export default function BrowsePage({ searchParams }: PageProps<"/browse">) {
  return (
    <>
    <section className="section" style={{ marginTop: 40 }}>
      <div className="kicker">CATEGORIES · IDEAS</div>
      <h1 className="glow-text" style={{ fontSize: 34, margin: "10px 0" }}>หมวดหมู่และไอเดียลงทุน</h1>
      <Suspense fallback={<p className="empty-note">กำลังโหลด…</p>}>
        <Body sp={searchParams} />
      </Suspense>
    </section>
    <MacroStrip />
    </>
  );
}

async function Body({ sp }: { sp: Promise<SP> }) {
  const p = await sp;
  const pick = PICKS.find((x) => x.id === p.p);
  const cat = CATEGORIES.find((c) => c.title === p.c) ?? (pick ? undefined : CATEGORIES[0]);

  return (
    <>
      <h2 className="h-sm">หมวดหมู่</h2>
      <div className="cat-chips">
        {CATEGORIES.map((c) => <Link key={c.title} href={`/browse?c=${encodeURIComponent(c.title)}`} className={cat?.title === c.title ? "on" : ""}>{c.title}</Link>)}
      </div>
      <h2 className="h-sm">💡 ไอเดีย</h2>
      <div className="cat-chips">
        {PICKS.map((x) => <Link key={x.id} href={`/browse?p=${x.id}`} className={pick?.id === x.id ? "on" : ""}>{x.title}</Link>)}
      </div>

      {cat && <Suspense fallback={<p className="empty-note">กำลังโหลดราคา…</p>}><CategoryView symbols={cat.symbols} /></Suspense>}
      {pick && (
        <>
          <p className="muted-note">{pick.desc}</p>
          <Suspense fallback={<p className="empty-note">กำลังสแกน {pick.universe.length} ตัว… ครั้งแรกอาจใช้เวลาสักครู่ ครั้งต่อไปเร็วขึ้น</p>}>
            <PickView id={pick.id} />
          </Suspense>
        </>
      )}
    </>
  );
}

async function CategoryView({ symbols }: { symbols: string[] }) {
  const quotes = await getQuotes(symbols);
  if (!quotes.length) return <p className="empty-note">ดึงราคาไม่สำเร็จในขณะนี้</p>;
  return (
    <div className="ticker-grid" style={{ ["--cols" as string]: 4 }}>
      {quotes.map((q) => (
        <Link key={q.symbol} href={`/stock/${encodeURIComponent(q.symbol)}`} className="card ticker-card">
          <div className="row"><span className="sym">{displaySymbol(q.symbol)}</span><span className={`num ${direction(q.changePct)}`}>{fmtPct(q.changePct)}</span></div>
          <span className="name">{q.name}</span>
          <span className="num glow-text" style={{ fontSize: 20 }}>{fmtPrice(q.price)} <span className="unit">{q.currency}</span></span>
        </Link>
      ))}
    </div>
  );
}

async function PickView({ id }: { id: string }) {
  const r = await runPick(id);
  if (!r) return null;
  if (!r.rows.length) return <p className="empty-note">{r.empty}</p>;
  return (
    <>
      <div className="pick-grid">
        {r.rows.map((row) => (
          <Link key={row.symbol} href={`/stock/${encodeURIComponent(row.symbol)}`} className="card pick">
            <span className="s">{displaySymbol(row.symbol)}</span>
            <span className="m">{row.meta}</span>
          </Link>
        ))}
      </div>
      <p className="muted-note" style={{ marginTop: 14 }}>ผลจากการสแกนด้วยกฎอย่างง่าย ไม่ได้ทดสอบย้อนหลังว่าให้ผลตอบแทนจริง · เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน</p>
    </>
  );
}
