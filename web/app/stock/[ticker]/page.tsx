import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import { WatchButton } from "@/components/Watchlist";
import MacroStrip from "@/components/MacroStrip";
import Controls from "@/components/stock/Controls";
import { ChartTab, NewsTab, SignalTab, StatsTab } from "@/components/stock/tabs";
import { getHistory, getProfile, getQuote } from "@/lib/yahoo";
import {
  DEFAULT_INTERVAL, DEFAULT_PERIOD, INTERVALS, isInterval, isIntraday, isPeriod, type IntervalKey, type PeriodKey,
} from "@/lib/ranges";
import { direction, fmtBig, fmtDateTimeTh, fmtPct, fmtPrice, fmtSigned, marketStateTh } from "@/lib/format";

type Props = {
  params: Promise<{ ticker: string }>;
  searchParams: Promise<{ tab?: string; period?: string; interval?: string; tr?: string }>;
};

const TABS = [
  { key: "chart", label: "📊 กราฟเทคนิค" },
  { key: "stats", label: "📋 สถิติ" },
  { key: "news", label: "📰 ข่าว" },
  { key: "signal", label: "🎯 สัญญาณสรุป" },
] as const;
type TabKey = (typeof TABS)[number]["key"];

const cleanSymbol = (raw: string) => decodeURIComponent(raw).trim().toUpperCase();

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { ticker } = await params;
  return { title: `${cleanSymbol(ticker)} · Stockscope` };
}

export default function StockPage(props: Props) {
  return (
    <Suspense fallback={<StockSkeleton />}>
      <StockView {...props} />
    </Suspense>
  );
}

function StockSkeleton() {
  return (
    <section className="stock-hero" aria-busy>
      <div className="rings" aria-hidden />
      <div className="stock-id">
        <div className="sym">LOADING</div>
        <h1 className="glow-text" style={{ opacity: 0.35 }}>กำลังโหลดข้อมูล…</h1>
      </div>
    </section>
  );
}

async function StockView({ params, searchParams }: Props) {
  const symbol = cleanSymbol((await params).ticker);
  const sp = await searchParams;
  const tab: TabKey = TABS.some((t) => t.key === sp.tab) ? (sp.tab as TabKey) : "chart";
  const period: PeriodKey = isPeriod(sp.period) ? sp.period : DEFAULT_PERIOD;
  const interval: IntervalKey = isInterval(sp.interval) ? sp.interval : DEFAULT_INTERVAL;
  const translate = sp.tr !== "0";

  const quote = await getQuote(symbol);
  if (!quote) {
    return (
      <div className="empty">
        <div className="kicker">ไม่พบข้อมูล</div>
        <h1 className="glow-text" style={{ margin: "16px 0" }}>{symbol}</h1>
        <p>เช็กรหัสหุ้นอีกครั้ง เช่น หุ้นไทยต้องลงท้ายด้วย .BK (PTT.BK) และคริปโตใช้ -USD (BTC-USD)</p>
        <p><Link href="/" style={{ textDecoration: "underline" }}>กลับหน้าแรก</Link></p>
      </div>
    );
  }

  const [profile, thb] = await Promise.all([getProfile(symbol), quote.currency === "USD" ? getQuote("THB=X") : null]);
  const dir = direction(quote.change);
  const live = quote.marketState === "REGULAR";
  const thbValue = thb?.price && quote.price !== null ? quote.price * thb.price : null;
  const ext =
    quote.marketState.startsWith("POST") && quote.postPrice !== null
      ? { label: "After Hours", price: quote.postPrice, chg: quote.postChange, pct: quote.postChangePct }
      : quote.marketState.startsWith("PRE") && quote.prePrice !== null
        ? { label: "Pre-Market", price: quote.prePrice, chg: quote.preChange, pct: quote.preChangePct }
        : quote.marketState === "CLOSED" && quote.postPrice !== null
          ? { label: "Last After Hours", price: quote.postPrice, chg: quote.postChange, pct: quote.postChangePct }
          : null;

  const href = (patch: Record<string, string>) => {
    const q = new URLSearchParams({ tab, period, interval, ...(translate ? {} : { tr: "0" }), ...patch });
    return `?${q.toString()}`;
  };

  const needsBars = tab === "chart" || tab === "stats" || tab === "signal";
  const bars = needsBars ? await getHistory(symbol, period, interval) : [];
  const rangeLow = bars.length ? Math.min(...bars.map((b) => b.low)) : null;
  const rangeHigh = bars.length ? Math.max(...bars.map((b) => b.high)) : null;
  const lastBar = bars.at(-1);
  const intervalMeta = INTERVALS.find((i) => i.key === interval)!;

  return (
    <>
      <section className="stock-hero">
        <div className="rings" aria-hidden />
        <div className="stock-id">
          <div className="sym">{quote.symbol}</div>
          <h1 className="glow-text">{quote.name}</h1>
          <div className="meta">{[quote.exchange, profile?.sector, profile?.industry].filter(Boolean).join(" · ") || "—"}</div>
          <div style={{ marginTop: 14 }}><WatchButton symbol={quote.symbol} /></div>
        </div>
        <div className="price-block">
          <div>
            <span className="price num glow-text">{fmtPrice(quote.price)}</span>
            <span className="cur">{quote.currency}</span>
          </div>
          {thbValue !== null && <div className="thb num">≈ ฿{fmtPrice(thbValue)} บาท</div>}
          <div className="line">
            <span className={`chip num ${dir}`}>
              {dir === "up" ? "▲" : dir === "down" ? "▼" : "•"} {fmtSigned(quote.change)} ({fmtPct(quote.changePct)})
            </span>
            <span className={`state${live ? " live" : ""}`}>{marketStateTh(quote.marketState)}</span>
          </div>
          {ext && (
            <div className="ext num">
              {ext.label} {fmtPrice(ext.price)} <span className={direction(ext.pct)}>({fmtSigned(ext.chg)} · {fmtPct(ext.pct)})</span>
            </div>
          )}
        </div>
      </section>

      <MacroStrip />

      <p className="source-note top">
        แหล่งข้อมูล: Yahoo Finance · ราคา ณ {fmtDateTimeTh(quote.marketTime)} (เวลาไทย) · สกุลเงิน {quote.currency} · อาจหน่วงตามเงื่อนไขของแต่ละตลาด (บางตลาดหน่วงได้ถึง 15 นาที) และเก็บแคชไว้ประมาณ 1 นาที
        {thbValue !== null && " · ราคาบาทคำนวณจากอัตรา USD/THB ล่าสุด เป็นค่าประมาณเท่านั้น"}
      </p>

      <section className="stats">
        <div className="card stat"><div className="label">ปริมาณซื้อขาย</div><div className="value num">{lastBar && interval !== "1m" ? lastBar.volume.toLocaleString("en-US") : fmtBig(quote.volume)}</div></div>
        <div className="card stat"><div className="label">ช่วงราคา (ย้อนหลัง)</div><div className="value num">{rangeLow !== null ? `${fmtPrice(rangeLow)} – ${fmtPrice(rangeHigh)}` : `${fmtPrice(quote.dayLow)} – ${fmtPrice(quote.dayHigh)}`}</div></div>
        <div className="card stat"><div className="label">มูลค่าตลาด</div><div className="value num">{fmtBig(quote.marketCap)}</div></div>
        <div className="card stat"><div className="label">ช่วงราคา 52 สัปดาห์</div><div className="value num">{quote.low52 !== null ? `${fmtPrice(quote.low52)} – ${fmtPrice(quote.high52)}` : "—"}</div></div>
      </section>

      <div className="tabbar">
        <nav className="tabs" aria-label="หัวข้อ">
          {TABS.map((t) => (
            <Link key={t.key} href={href({ tab: t.key })} scroll={false} className={t.key === tab ? "on" : ""}>{t.label}</Link>
          ))}
        </nav>
        {needsBars && <Suspense><Controls period={period} interval={interval} /></Suspense>}
      </div>
      {needsBars && Number.isFinite(intervalMeta.maxDays) && (
        <p className="muted-note" style={{ margin: "8px 0 0" }}>{intervalMeta.label}ใช้ได้กับข้อมูลย้อนหลังไม่เกิน {intervalMeta.maxDays} วัน ระบบจึงตัดช่วงย้อนหลังให้อัตโนมัติ</p>
      )}

      <section className="section" style={{ marginTop: 20 }}>
        {needsBars && !bars.length ? (
          <p className="empty-note">ไม่พบข้อมูลกราฟสำหรับ {symbol} ช่วงนี้ ลองเลือกช่วงเวลาหรือแท่งอื่น</p>
        ) : tab === "chart" ? (
          <ChartTab symbol={symbol} bars={bars} intraday={isIntraday(interval)} />
        ) : tab === "stats" ? (
          <Suspense fallback={<p className="empty-note">กำลังโหลดสถิติ…</p>}>
            <StatsTab symbol={symbol} quote={quote} bars={bars} />
          </Suspense>
        ) : tab === "news" ? (
          <Suspense fallback={<p className="empty-note">กำลังโหลดข่าว…</p>}>
            <NewsTab symbol={symbol} translate={translate} hrefToggle={href({ tab: "news", tr: translate ? "0" : "1" })} />
          </Suspense>
        ) : (
          <SignalTab bars={bars} />
        )}
      </section>

      <p className="source-note">เพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน</p>
    </>
  );
}
