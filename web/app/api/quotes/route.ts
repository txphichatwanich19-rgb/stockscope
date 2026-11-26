import { NextResponse } from "next/server";
import { getQuotes } from "@/lib/yahoo";

// Used by the Watchlist (its symbols live in the browser, so the server cannot render them).
export async function GET(request: Request) {
  const raw = new URL(request.url).searchParams.get("s") ?? "";
  const symbols = [...new Set(raw.split(",").map((s) => s.trim().toUpperCase()).filter((s) => /^[A-Z0-9.^=-]{1,15}$/.test(s)))].slice(0, 30);
  const quotes = (await getQuotes(symbols)).map((q) => ({
    symbol: q.symbol, name: q.name, price: q.price, changePct: q.changePct, currency: q.currency,
  }));
  return NextResponse.json({ quotes });
}
