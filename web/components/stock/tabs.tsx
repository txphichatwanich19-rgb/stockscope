import Link from "next/link";
import { connection } from "next/server";
import PriceChart from "@/components/PriceChart";
import NewsAI from "@/components/stock/NewsAI";
import {
  buildSignal, computeLevels, riskMetrics, zoneStatus, type OHLC, type ZoneStatus,
} from "@/lib/analysis";
import {
  daysUntil, direction, fmtBig, fmtDateTh, fmtDateTimeTh, fmtMoneyUsd, fmtPct, fmtPlainPct, fmtPrice, recLabel,
} from "@/lib/format";
import {
  getEarnings, getInsiders, getProfile, getQuarterly, getStats, getUpgrades, type Quote,
} from "@/lib/yahoo";
import { getNews, sentimentLabel, sentimentScore, translateTexts } from "@/lib/news";

const tone = (kind: "support" | "resistance", s: ZoneStatus) =>
  s.key === "approaching" ? "warn" : s.key === "waiting" ? "dim"
    : kind === "support" ? (s.key === "touched" ? "good" : "bad")
    : (s.key === "touched" ? "bad" : "good");

// ------------------------------------------------------------------ chart
function Pill({ role, price, current, kind, bars }: { role: string; price: number; current: number; kind: "support" | "resistance"; bars: OHLC[] }) {
  const st = zoneStatus(bars, price, kind);
  const delta = (price / current - 1) * 100;
  return (
    <div className="pill">
      <span className="pill-role">{role}</span>
      <span className="num pill-price">{fmtPrice(price)}</span>
      <span className="num pill-delta">{delta >= 0 ? "+" : ""}{delta.toFixed(1)}%</span>
      <span className={`zone tone-${tone(kind, st)}`}>{st.label}</span>
    </div>
  );
}

export function ChartTab({ symbol, bars, intraday }: { symbol: string; bars: OHLC[]; intraday: boolean }) {
  const lv = computeLevels(bars);
  const stop = lv.support[1] ? lv.support[1] * 0.98 : lv.support[0] ? lv.support[0] * 0.96 : null;
  return (
    <>
      <div className="card chart-card">
        <div className="chart-top">
          <div className="legend">
            <span><i style={{ background: "rgba(255,255,255,0.85)" }} />SMA 20</span>
            <span><i style={{ background: "rgba(255,255,255,0.3)" }} />SMA 50</span>
            <span>แท่งล่าง = ปริมาณซื้อขาย</span>
          </div>
          <a className="link-btn" target="_blank" rel="noreferrer"
            href={`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(toTv(symbol))}`}>
            เปิดใน TradingView ↗
          </a>
        </div>
        <PriceChart bars={bars} />
      </div>

      <div className="card ladder">
        <div className="ladder-title">เป้าราคาถัดไป <span className="muted-note">· ทุกโซนคือราคาที่ยังไม่ถูกแตะ · % คือระยะห่างจากราคาปัจจุบัน</span></div>
        <div className="ladder-row">
          <div className="ladder-h">↑ ถ้าราคาขึ้นไปถึง</div>
          <div className="pills">
            {lv.resistance.length ? [...lv.resistance].reverse().map((p, i, a) => (
              <Pill key={p} role={`🔴 ขาย ${a.length - i}`} price={p} current={lv.current} kind="resistance" bars={bars} />
            )) : <span className="muted-note">— ไม่มีแนวต้านในข้อมูลที่มี —</span>}
          </div>
        </div>
        <div className="ladder-now"><span>ราคาตอนนี้</span><span className="num glow-text">{fmtPrice(lv.current)}</span><span className="live-dot">● {intraday ? "INTRADAY" : "LAST"}</span></div>
        <div className="ladder-row">
          <div className="ladder-h">↓ ถ้าราคาลงมาถึง</div>
          <div className="pills">
            {lv.support.map((p, i) => <Pill key={p} role={`🟢 ซื้อ ${i + 1}`} price={p} current={lv.current} kind="support" bars={bars} />)}
            {stop !== null && <Pill role="🛑 ตัดขาดทุน" price={stop} current={lv.current} kind="support" bars={bars} />}
            {!lv.support.length && <span className="muted-note">— ไม่มีแนวรับในข้อมูลที่มี —</span>}
          </div>
        </div>
      </div>
    </>
  );
}

function toTv(sym: string): string {
  const idx: Record<string, string> = { "^GSPC": "SP:SPX", "^IXIC": "NASDAQ:IXIC", "^DJI": "DJ:DJI", "^RUT": "TVC:RUT", "^VIX": "TVC:VIX", "^FTSE": "TVC:UKX", "^N225": "TVC:NI225", "^HSI": "TVC:HSI" };
  if (idx[sym]) return idx[sym];
  if (sym.endsWith("-USD")) return `BINANCE:${sym.replace("-USD", "")}USDT`;
  if (sym.endsWith(".BK")) return `SET:${sym.replace(".BK", "")}`;
  return sym;
}

// ------------------------------------------------------------------ stats
function Row({ k, v }: { k: string; v: string }) {
  return <div className="kv"><span>{k}</span><span className="num">{v}</span></div>;
}

function QuarterBars({ rows }: { rows: { date: string; revenue: number | null; netIncome: number | null }[] }) {
  const last4 = rows.slice(-4);
  const max = Math.max(...last4.flatMap((r) => [r.revenue ?? 0, Math.abs(r.netIncome ?? 0)]), 1);
  const label = (d: string) => `${d.slice(0, 4)} Q${Math.ceil(Number(d.slice(5, 7)) / 3)}`;
  return (
    <div className="qbars" role="img" aria-label="รายได้และกำไรสุทธิรายไตรมาส">
      {last4.map((r) => (
        <div className="qgroup" key={r.date}>
          <div className="qpair">
            <div className="qbar rev" style={{ height: `${((r.revenue ?? 0) / max) * 100}%` }}><em className="num">{((r.revenue ?? 0) / 1e9).toFixed(1)}B</em></div>
            <div className="qbar ni" style={{ height: `${(Math.abs(r.netIncome ?? 0) / max) * 100}%` }}><em className="num">{((r.netIncome ?? 0) / 1e9).toFixed(1)}B</em></div>
          </div>
          <span className="qlabel">{label(r.date)}</span>
        </div>
      ))}
      <div className="qlegend"><span><i className="rev" />รายได้</span><span><i className="ni" />กำไรสุทธิ</span><span>หน่วย: พันล้านดอลลาร์ (USD)</span></div>
    </div>
  );
}

export async function StatsTab({ symbol, quote, bars }: { symbol: string; quote: Quote; bars: OHLC[] }) {
  await connection(); // "days until earnings" depends on the current time
  const [stats, profile, quarterly, insiders, earnings, upgrades] = await Promise.all([
    getStats(symbol), getProfile(symbol), getQuarterly(symbol), getInsiders(symbol), getEarnings(symbol), getUpgrades(symbol),
  ]);
  const risk = riskMetrics(bars);
  const last = quote.price ?? 0;
  const gauge = quote.low52 !== null && quote.high52 !== null && quote.high52 > quote.low52
    ? Math.min(100, Math.max(0, ((last - quote.low52) / (quote.high52 - quote.low52)) * 100)) : null;
  const zone = gauge === null ? "" : gauge >= 80 ? "ใกล้จุดสูงสุด — ระวัง valuation" : gauge >= 60 ? "ช่วงบน — momentum ดี" : gauge >= 40 ? "ช่วงกลาง" : gauge >= 20 ? "ช่วงล่าง — น่าสนใจถ้าพื้นฐานดี" : "ใกล้จุดต่ำสุด — รอ confirm reversal";
  const pct = (v: number | null) => (v === null ? "—" : `${(v * 100).toFixed(2)}%`);

  const q = quarterly;
  const qoq = q.length >= 2 && q[q.length - 2].revenue ? ((q[q.length - 1].revenue! / q[q.length - 2].revenue!) - 1) * 100 : null;
  const yoy = q.length >= 5 && q[q.length - 5].revenue ? ((q[q.length - 1].revenue! / q[q.length - 5].revenue!) - 1) * 100 : null;

  const buys = insiders.filter((i) => i.kind === "buy");
  const sells = insiders.filter((i) => i.kind === "sell");
  const buyVal = buys.reduce((a, b) => a + (b.value ?? 0), 0);
  const sellVal = sells.reduce((a, b) => a + (b.value ?? 0), 0);
  const net = buyVal > sellVal ? "🟢 Net buying" : sellVal > buyVal ? "🔴 Net selling" : "⚪ Balanced";
  const kindIcon = { buy: "🟢 ซื้อ", sell: "🔴 ขาย", gift: "🎁 โอน", other: "—" } as const;

  const ceo = profile?.officers.find((o) => /CEO|CHIEF EXECUTIVE/i.test(o.title)) ?? profile?.officers[0];
  const others = (profile?.officers ?? []).filter((o) => o !== ceo).slice(0, 4);
  const daysTo = earnings.nextDate ? daysUntil(earnings.nextDate) : null;
  const beats = earnings.history.filter((h) => (h.surprisePct ?? 0) > 0).length;
  const withSurprise = earnings.history.filter((h) => h.surprisePct !== null).length;

  return (
    <>
      {gauge !== null && (
        <div className="card gauge">
          <div className="gauge-top"><span>📍 ตำแหน่งราคาใน 52 สัปดาห์</span><span className="num">{gauge.toFixed(0)}% · {zone}</span></div>
          <div className="gauge-bar"><i style={{ left: `${gauge}%` }} /></div>
          <div className="gauge-ends num"><span>ต่ำสุด {fmtPrice(quote.low52)}</span><span>ราคาตอนนี้ {fmtPrice(last)}</span><span>สูงสุด {fmtPrice(quote.high52)}</span></div>
        </div>
      )}

      {risk && (
        <section className="section">
          <h2 className="h-sm">⚠️ ความเสี่ยงและผลตอบแทน (ในช่วงข้อมูล)</h2>
          <div className="stats">
            <div className="card stat"><div className="label">Beta (vs ตลาด)</div><div className="value num">{stats?.beta != null ? stats.beta.toFixed(2) : "—"}</div></div>
            <div className="card stat"><div className="label">ความผันผวน/ปี</div><div className="value num">{fmtPlainPct(risk.vol)}</div></div>
            <div className="card stat"><div className="label">Max Drawdown</div><div className="value num">{fmtPlainPct(risk.maxDrawdown)}</div></div>
            <div className="card stat"><div className="label">Sharpe Ratio</div><div className="value num">{risk.sharpe === null ? "—" : risk.sharpe.toFixed(2)}</div></div>
          </div>
          <p className="muted-note">📖 Beta &gt; 1 = ผันผวนกว่าตลาด · ผันผวน/ปี = standard deviation รายปี · Max Drawdown = ขาดทุนหนักสุดจากจุดสูงสุด · Sharpe = ผลตอบแทนเทียบความเสี่ยง (&gt;1 ดี, &gt;2 ยอดเยี่ยม) ไม่รวมอัตราดอกเบี้ยไร้ความเสี่ยง และคำนวณแบบรายปีจากแท่งรายวัน (ถ้าเลือกแท่งช่วงอื่นตัวเลขจะคลาดเคลื่อน)</p>
        </section>
      )}

      <section className="section three">
        <div className="card kvbox"><h3>ราคาและขนาด</h3>
          <Row k="มูลค่าตลาด" v={fmtBig(quote.marketCap)} />
          <Row k="สูงสุด 52 สัปดาห์" v={fmtPrice(quote.high52)} />
          <Row k="ต่ำสุด 52 สัปดาห์" v={fmtPrice(quote.low52)} />
          <Row k="ปริมาณเฉลี่ย" v={fmtBig(quote.avgVolume)} />
          <Row k="ค่า Beta (ความผันผวน)" v={stats?.beta != null ? stats.beta.toFixed(2) : "—"} />
        </div>
        <div className="card kvbox"><h3>มูลค่าหุ้น (Valuation)</h3>
          <Row k="P/E (ย้อนหลัง)" v={quote.pe !== null ? quote.pe.toFixed(2) : "—"} />
          <Row k="P/E (คาดการณ์)" v={quote.forwardPe !== null ? quote.forwardPe.toFixed(2) : "—"} />
          <Row k="PEG (P/E ต่อการเติบโต)" v={stats?.peg != null ? stats.peg.toFixed(2) : "—"} />
          <Row k="P/B (ราคาต่อมูลค่าทางบัญชี)" v={quote.priceToBook !== null ? quote.priceToBook.toFixed(2) : "—"} />
          <Row k="EPS (กำไรต่อหุ้น 12 เดือน)" v={quote.eps !== null ? quote.eps.toFixed(2) : "—"} />
        </div>
        <div className="card kvbox"><h3>ปันผลและความสามารถทำกำไร</h3>
          <Row k="อัตราปันผล" v={stats?.dividendYield != null ? `${stats.dividendYield.toFixed(2)}%` : "—"} />
          <Row k="สัดส่วนการจ่ายปันผล" v={pct(stats?.payoutRatio ?? null)} />
          <Row k="อัตรากำไรสุทธิ" v={pct(stats?.profitMargin ?? null)} />
          <Row k="ROE (ผลตอบแทนผู้ถือหุ้น)" v={pct(stats?.roe ?? null)} />
          <Row k="การเติบโตรายได้" v={pct(stats?.revenueGrowth ?? null)} />
        </div>
      </section>

      {stats?.recMean != null && (
        <div className="card kvbox wide">
          <h3>มุมมองนักวิเคราะห์</h3>
          <Row k="คำแนะนำเฉลี่ย (1 = Strong Buy, 5 = Strong Sell)" v={`${stats.recMean.toFixed(2)} · ${recLabel(stats.recMean)}`} />
          <Row k="ราคาเป้าหมายเฉลี่ย" v={stats.targetMean != null ? `${fmtPrice(stats.targetMean)}${last ? ` (${fmtPct((stats.targetMean / last - 1) * 100)})` : ""}` : "—"} />
          <Row k="จำนวนนักวิเคราะห์" v={stats.analystCount != null ? String(stats.analystCount) : "—"} />
        </div>
      )}

      {q.length >= 2 && (
        <section className="section">
          <h2 className="h-sm">📊 รายได้และกำไรรายไตรมาส (4 ไตรมาสล่าสุด)</h2>
          <div className="card"><QuarterBars rows={q} /></div>
          <p className="muted-note">
            📈 รายได้ QoQ: <b className={direction(qoq)}>{qoq === null ? "—" : fmtPct(qoq)}</b>
            {yoy !== null && <> · YoY: <b className={direction(yoy)}>{fmtPct(yoy)}</b></>}
          </p>
        </section>
      )}

      {profile && profile.officers.length > 0 && ceo && (
        <section className="section">
          <h2 className="h-sm">👔 ผู้บริหาร</h2>
          <div className="two">
            <div className="card kvbox"><div className="officer">{ceo.name}</div><div className="muted-note">{ceo.title}</div>
              <div className="muted-note">{ceo.age !== null && `อายุ ${ceo.age} ปี`}{ceo.age !== null && ceo.pay !== null && " · "}{ceo.pay !== null && `ค่าตอบแทน $${ceo.pay.toLocaleString("en-US")}/ปี`}</div></div>
            <div className="card kvbox"><h3>ผู้บริหารคนอื่น</h3>{others.map((o) => <div className="kv" key={o.name}><span>{o.name}</span><span className="muted-note">{o.title}</span></div>)}</div>
          </div>
        </section>
      )}

      <section className="section">
        <h2 className="h-sm">👥 การซื้อขายของคนใน (Insider Transactions) 6 เดือน</h2>
        {insiders.length ? (
          <>
            <div className="stats three-cols">
              <div className="card stat"><div className="label">ซื้อ</div><div className="value num">{buys.length} ครั้ง · {fmtMoneyUsd(buyVal)}</div></div>
              <div className="card stat"><div className="label">ขาย</div><div className="value num">{sells.length} ครั้ง · {fmtMoneyUsd(sellVal)}</div></div>
              <div className="card stat"><div className="label">สรุป</div><div className="value">{net}</div></div>
            </div>
            <details className="card tbl-box"><summary>ดูรายการทั้งหมด {insiders.length} รายการ (6 เดือนล่าสุด)</summary>
              <div className="tbl-wrap"><table><thead><tr><th>วันที่</th><th>ประเภท</th><th>ผู้บริหาร</th><th>ตำแหน่ง</th><th>จำนวนหุ้น</th><th>มูลค่า</th><th>รายละเอียด</th></tr></thead>
                <tbody>{insiders.map((i, n) => <tr key={n}><td>{i.date}</td><td>{kindIcon[i.kind]}</td><td>{i.name}</td><td>{i.position}</td><td className="num">{i.shares?.toLocaleString("en-US") ?? "—"}</td><td className="num">{i.value ? `$${i.value.toLocaleString("en-US")}` : "—"}</td><td>{i.text.slice(0, 60)}</td></tr>)}</tbody></table></div>
            </details>
          </>
        ) : <p className="muted-note">ไม่มีรายการ insider ในช่วง 6 เดือนล่าสุด (หรือไม่มีข้อมูลสำหรับหุ้นนี้)</p>}
      </section>

      <section className="section">
        <h2 className="h-sm">📅 ผลประกอบการ (Earnings)</h2>
        <div className="two">
          <div className="card kvbox">
            <div className="label">ประกาศผลครั้งต่อไป</div>
            {earnings.nextDate ? (<><div className="big-date">{fmtDateTh(earnings.nextDate)}</div>
              <div className="muted-note">{daysTo !== null && `อีก ${daysTo} วัน`}{earnings.nextEpsEstimate !== null && ` · คาดการณ์ EPS: ${earnings.nextEpsEstimate.toFixed(2)}`}</div></>)
              : <div className="muted-note">ไม่มีข้อมูลวันประกาศครั้งต่อไป</div>}
          </div>
          <div className="card kvbox">
            <h3>ผลย้อนหลัง 4 ไตรมาส</h3>
            {earnings.history.length ? earnings.history.map((h) => (
              <div className="kv" key={h.quarter}><span>{h.quarter}</span>
                <span className="num">{h.estimate?.toFixed(2) ?? "—"} → {h.actual?.toFixed(2) ?? "—"} <b className={direction(h.surprisePct)}>{h.surprisePct === null ? "" : `${h.surprisePct > 0 ? "🟢" : h.surprisePct < 0 ? "🔴" : "⚪"} ${fmtPct(h.surprisePct)}`}</b></span></div>
            )) : <div className="muted-note">ไม่มีข้อมูล</div>}
            {withSurprise > 0 && <div className="muted-note">📊 ชนะ Estimate {beats}/{withSurprise} ไตรมาส</div>}
          </div>
        </div>
      </section>

      {upgrades.length > 0 && (
        <section className="section">
          <h2 className="h-sm">📈 การปรับเป้าราคา / เรตติ้งโดยนักวิเคราะห์</h2>
          <div className="card tbl-box"><div className="tbl-wrap"><table><thead><tr><th>วันที่</th><th>สถาบัน</th><th>การกระทำ</th><th>เรตติ้ง</th></tr></thead>
            <tbody>{upgrades.map((u, n) => (
              <tr key={n}><td>{u.date}</td><td>{u.firm}</td>
                <td>{u.targetAction.includes("Raise") ? "🟢" : u.targetAction.includes("Lower") ? "🔴" : "⚪"} {u.targetAction || u.action}</td>
                <td>{u.fromGrade && u.fromGrade !== u.toGrade ? `${u.fromGrade} → ${u.toGrade}` : u.toGrade}</td></tr>))}</tbody></table></div></div>
        </section>
      )}

      {profile?.summary && (
        <details className="card about"><summary>📄 เกี่ยวกับบริษัท</summary><p>{profile.summary}</p></details>
      )}
    </>
  );
}

// ------------------------------------------------------------------ news
export async function NewsTab({ symbol, translate, hrefToggle }: { symbol: string; translate: boolean; hrefToggle: string }) {
  const news = (await getNews(symbol)).slice(0, 25);
  if (!news.length) {
    return <p className="empty-note">ไม่พบข่าวล่าสุดสำหรับหุ้นตัวนี้{symbol.endsWith(".BK") && " (Yahoo Finance มีข่าวหุ้นไทยน้อยมาก)"}</p>;
  }
  const scored = news.map((n) => ({ ...n, score: sentimentScore(`${n.title}. ${n.summary}`) }));
  const avg = scored.reduce((a, b) => a + b.score, 0) / scored.length;
  const pos = scored.filter((n) => n.score >= 0.1).length;
  const neg = scored.filter((n) => n.score <= -0.1).length;
  const overall = sentimentLabel(avg);

  let th: string[] | null = null;
  let thSummary: string[] | null = null;
  if (translate) {
    try {
      th = await translateTexts(scored.map((n) => n.title));
      const withSum = scored.filter((n) => n.summary);
      if (withSum.length) {
        const t = await translateTexts(withSum.map((n) => n.summary.slice(0, 450)));
        thSummary = scored.map((n) => (n.summary ? t[withSum.indexOf(n)] ?? "" : ""));
      }
    } catch { th = null; }
  }

  return (
    <>
      <div className="card sent-bar">
        <span>📊 ภาพรวมข่าวล่าสุด {scored.length} ข่าว</span>
        <span className={`chip ${overall.cls}`}>{overall.label}</span>
        <span className="num muted-note">🟢 {pos} · ⚪ {scored.length - pos - neg} · 🔴 {neg}</span>
        <Link href={hrefToggle} className="link-btn" scroll={false}>{translate ? "แสดงต้นฉบับอังกฤษ" : "🌐 แปลเป็นภาษาไทย"}</Link>
      </div>
      {translate && th === null && <p className="muted-note">ตอนนี้บริการแปลฟรีไม่ตอบสนอง จึงแสดงข่าวเป็นภาษาอังกฤษ ลองใหม่อีกครั้งในอีกสักครู่</p>}

      <NewsAI ticker={symbol} items={scored.slice(0, 15).map((n) => ({ date: (n.time ?? "").slice(0, 10), title: n.title, summary: n.summary }))} />

      <div className="news-list">
        {scored.map((n, i) => {
          const s = sentimentLabel(n.score);
          const title = th?.[i] || n.title;
          const showOrig = th?.[i] && th[i] !== n.title;
          const sum = (thSummary?.[i] || n.summary || "").slice(0, 500);
          return (
            <article className="card news-card" key={i}>
              <h3 className="title">{n.link ? <a href={n.link} target="_blank" rel="noreferrer">{title}</a> : title}</h3>
              {showOrig && <div className="orig">🇬🇧 {n.title}</div>}
              <div className="meta">📰 {n.publisher} · 🕐 {fmtDateTimeTh(n.time)} · <span className={`chip sm ${s.cls}`}>{s.label}</span></div>
              {sum && <p className="summary">{sum}</p>}
            </article>
          );
        })}
      </div>
    </>
  );
}

// ------------------------------------------------------------------ signal
function LevelTile({ role, price, sub, current, kind, bars, cls }: { role: string; price: number | null; sub: string; current: number; kind: "support" | "resistance"; bars: OHLC[]; cls: string }) {
  if (price === null) return <div className={`card level ${cls}`}><div className="role">{role}</div><div className="price num">—</div><div className="delta">ข้อมูลไม่พอ</div></div>;
  const st = cls === "current" ? null : zoneStatus(bars, price, kind);
  const d = (price / current - 1) * 100;
  return (
    <div className={`card level ${cls}`}>
      <div className="role">{role}</div>
      <div className="price num">{fmtPrice(price)}</div>
      <div className="delta num">{cls === "current" ? sub : `${d >= 0 ? "+" : ""}${d.toFixed(2)}% · ${sub}`}</div>
      {st && <span className={`zone tone-${tone(kind, st)}`}>{st.label}</span>}
    </div>
  );
}

export function SignalTab({ bars }: { bars: OHLC[] }) {
  const sig = buildSignal(bars);
  const lv = computeLevels(bars);
  const [s1, s2] = [lv.support[0] ?? null, lv.support[1] ?? null];
  const stop = s2 ? s2 * 0.98 : s1 ? s1 * 0.96 : null;
  const [r1, r2, r3] = [lv.resistance[0] ?? null, lv.resistance[1] ?? null, lv.resistance[2] ?? null];
  const risk = stop !== null ? lv.current - stop : 0;
  const rr = s1 && stop !== null && r1 && risk > 0 ? (r1 - lv.current) / risk : null;
  const rows: [string, string][] = [
    ...lv.resistance.slice().reverse().map((p, i, a): [string, string] => [`🔴 แนวต้าน R${a.length - i}`, p.toString()]),
    ["⚪ ราคาปัจจุบัน", lv.current.toString()],
    ...lv.support.map((p, i): [string, string] => [`🟢 แนวรับ S${i + 1}`, p.toString()]),
  ];

  return (
    <>
      <div className={`card verdict ${sig.verdict}`}>
        {sig.verdict === "bull" ? "🟢" : sig.verdict === "bear" ? "🔴" : "🟡"} {sig.verdictLabel}
      </div>

      <h2 className="h-sm">🟢 แนวรับ — โซนซื้อ</h2>
      <div className="levels four">
        <LevelTile role="ราคาปัจจุบัน" price={lv.current} sub="ตอนนี้" current={lv.current} kind="support" bars={bars} cls="current" />
        <LevelTile role="🟢 โซนซื้อ 1" price={s1} sub="แนวรับ S1" current={lv.current} kind="support" bars={bars} cls="entry" />
        <LevelTile role="🟢 โซนซื้อ 2" price={s2} sub="แนวรับ S2" current={lv.current} kind="support" bars={bars} cls="entry" />
        <LevelTile role="🛑 ตัดขาดทุน" price={stop} sub={s2 ? "~2% ใต้ S2" : "~4% ใต้ S1"} current={lv.current} kind="support" bars={bars} cls="stop" />
      </div>
      <h2 className="h-sm">🔴 แนวต้าน — โซนขายทำกำไร</h2>
      <div className="levels three">
        <LevelTile role="🎯 โซนขาย 1" price={r1} sub="แนวต้าน R1" current={lv.current} kind="resistance" bars={bars} cls="res" />
        <LevelTile role="🎯 โซนขาย 2" price={r2} sub="แนวต้าน R2" current={lv.current} kind="resistance" bars={bars} cls="res" />
        <LevelTile role="🎯 โซนขาย 3" price={r3} sub="แนวต้าน R3" current={lv.current} kind="resistance" bars={bars} cls="res" />
      </div>
      {rr !== null && <p className="rr">ความคุ้มค่า (ซื้อตอนนี้ → จุดขาย 1) · <b className={`num ${rr >= 2 ? "up" : rr >= 1 ? "flat" : "down"}`}>{rr.toFixed(2)} : 1</b></p>}

      <details className="card tbl-box"><summary>ดูแนวรับ/แนวต้านทั้งหมด</summary>
        <div className="tbl-wrap"><table><thead><tr><th>ประเภท</th><th>ราคา</th><th>ห่างจากปัจจุบัน</th></tr></thead>
          <tbody>{rows.map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{fmtPrice(Number(v))}</td><td className="num">{k.includes("ปัจจุบัน") ? "0.00%" : fmtPct((Number(v) / lv.current - 1) * 100)}</td></tr>)}</tbody></table></div>
      </details>

      <h2 className="h-sm">เหตุผลสัญญาณ</h2>
      <div className="reasons">{sig.reasons.map((r) => <div className="card reason" key={r}>{r}</div>)}</div>

      <details className="card about"><summary>ℹ️ สัญญาณและโซนซื้อ/ขายคำนวณอย่างไร</summary>
        <p><b>คะแนนสัญญาณ</b> รวมจาก 4 กฎ (ใช้ราคาปิดของช่วงเวลาที่เลือก)</p>
        <div className="tbl-wrap"><table><thead><tr><th>กฎ</th><th>เงื่อนไข</th><th>คะแนน</th></tr></thead><tbody>
          <tr><td>แนวโน้มระยะยาว</td><td>ราคา &gt; SMA50 &gt; SMA200 / ราคา &lt; SMA50 &lt; SMA200</td><td>+2 / −2</td></tr>
          <tr><td>แนวโน้มระยะสั้น</td><td>ราคาเหนือ SMA20 / ต่ำกว่า SMA20</td><td>+1 / −1</td></tr>
          <tr><td>RSI (14)</td><td>ต่ำกว่า 30 (oversold) / สูงกว่า 70 (overbought)</td><td>+2 / −2</td></tr>
          <tr><td>MACD</td><td>เพิ่งตัดขึ้น / เพิ่งตัดลงเทียบเส้น Signal</td><td>+2 / −2</td></tr>
          <tr><td>MACD</td><td>อยู่เหนือ / ใต้ Signal (ไม่ได้เพิ่งตัด)</td><td>+1 / −1</td></tr></tbody></table></div>
        <p>รวมคะแนน ≥ 3 = โน้มเอียงฝั่งซื้อ · ≤ −3 = โน้มเอียงฝั่งขาย · อื่นๆ = ไม่ชัดเจน</p>
        <p><b>โซนซื้อ/ขาย</b> หาจากจุดสูงสุด/ต่ำสุดย่อยในกราฟ (pivot) แล้วรวมจุดที่ห่างกันไม่ถึง 1.5% เป็นระดับเดียว แนวรับ = ระดับใต้ราคาปัจจุบัน แนวต้าน = ระดับเหนือราคาปัจจุบัน จุดตัดขาดทุนตั้งไว้ที่ 2% ใต้แนวรับ 2 (หรือ 4% ใต้แนวรับ 1 ถ้ามีแนวรับเดียว) ผลจึงเปลี่ยนตามช่วงเวลาและแท่งที่เลือก</p>
        <p><b>ข้อจำกัด</b> เป็นกฎมาตรฐานของการวิเคราะห์เทคนิค ยังไม่ได้ทดสอบย้อนหลังว่าให้ผลตอบแทนจริง และไม่รวมข่าว งบการเงิน หรือภาวะตลาดโดยรวม</p>
      </details>

      <p className="warn-box">⚠️ นี่เป็นสัญญาณ rule-based จาก indicator เท่านั้น ไม่ใช่คำแนะนำการลงทุน ควรพิจารณาปัจจัยพื้นฐาน ข่าว และการจัดการความเสี่ยงของตนเองประกอบ</p>
    </>
  );
}
