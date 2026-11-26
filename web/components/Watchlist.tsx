"use client";

import Link from "next/link";
import { useCallback, useEffect, useState, useSyncExternalStore } from "react";

const KEY = "stockscope.watchlist";
const EVENT = "stockscope:watchlist";

// Stored in this browser only (localStorage), so it survives reloads — unlike the old session-only list.
function read(): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(v) ? v.filter((x) => typeof x === "string") : [];
  } catch {
    return [];
  }
}
function write(list: string[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list));
  } catch {
    /* private mode: ignore */
  }
  window.dispatchEvent(new Event(EVENT));
}
const subscribe = (cb: () => void) => {
  window.addEventListener(EVENT, cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener(EVENT, cb);
    window.removeEventListener("storage", cb);
  };
};
// useSyncExternalStore needs a stable snapshot reference, so cache by the raw string.
let lastRaw = "";
let lastList: string[] = [];
function snapshot(): string[] {
  let raw = "[]";
  try { raw = localStorage.getItem(KEY) ?? "[]"; } catch { /* ignore */ }
  if (raw !== lastRaw) { lastRaw = raw; lastList = read(); }
  return lastList;
}
const EMPTY: string[] = [];

export function useWatchlist() {
  return useSyncExternalStore(subscribe, snapshot, () => EMPTY);
}

export function WatchButton({ symbol }: { symbol: string }) {
  const list = useWatchlist();
  const on = list.includes(symbol);
  return (
    <button
      type="button"
      className={`watch-btn${on ? " on" : ""}`}
      onClick={() => write(on ? list.filter((s) => s !== symbol) : [...list, symbol])}
      aria-pressed={on}
    >
      {on ? "★ อยู่ใน Watchlist" : "☆ เพิ่มใน Watchlist"}
    </button>
  );
}

type Row = { symbol: string; name: string; price: number | null; changePct: number | null; currency: string };

export function WatchlistSummary() {
  const list = useWatchlist();
  const [rows, setRows] = useState<Row[] | null>(null);

  useEffect(() => {
    if (!list.length) return;
    let alive = true;
    fetch(`/api/quotes?s=${encodeURIComponent(list.join(","))}`)
      .then((r) => r.json())
      .then((d) => alive && setRows(d.quotes as Row[]))
      .catch(() => alive && setRows([]));
    return () => { alive = false; };
  }, [list]);

  const clear = useCallback(() => write([]), []);

  if (!list.length) {
    return <p className="empty-note">⭐ ยังไม่มี Watchlist · เปิดหน้าหุ้นแล้วกด “เพิ่มใน Watchlist” จะเห็นสรุปที่นี่</p>;
  }
  return (
    <>
      <div className="ticker-grid" style={{ ["--cols" as string]: 4 }}>
        {(rows ?? list.map((s) => ({ symbol: s, name: "", price: null, changePct: null, currency: "" }))).map((q) => {
          const dir = q.changePct === null || q.changePct === 0 ? "flat" : q.changePct > 0 ? "up" : "down";
          return (
            <Link key={q.symbol} href={`/stock/${encodeURIComponent(q.symbol)}`} className="card ticker-card">
              <div className="row">
                <span className="sym">{q.symbol.replace(".BK", "").replace("-USD", "")}</span>
                <span className={`num ${dir}`}>{q.changePct === null ? "…" : `${q.changePct >= 0 ? "+" : ""}${q.changePct.toFixed(2)}%`}</span>
              </div>
              <span className="name">{q.name || " "}</span>
              <span className="num glow-text" style={{ fontSize: 20 }}>
                {q.price === null ? "—" : q.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: q.price < 1 ? 4 : 2 })}{" "}
                <span className="unit">{q.currency}</span>
              </span>
            </Link>
          );
        })}
      </div>
      <button type="button" className="link-btn" onClick={clear}>ล้าง Watchlist</button>
    </>
  );
}
