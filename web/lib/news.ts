import "server-only";
import YahooFinance from "yahoo-finance2";
import { cacheLife } from "next/cache";
// eslint-disable-next-line @typescript-eslint/no-require-imports
const vader = require("vader-sentiment") as {
  SentimentIntensityAnalyzer: { polarity_scores: (t: string) => { compound: number } };
};

const yf = new YahooFinance({ suppressNotices: ["yahooSurvey"] });

export type NewsItem = { title: string; publisher: string; link: string | null; time: string | null; summary: string };

const UA = { "User-Agent": "Mozilla/5.0" };

/* eslint-disable @typescript-eslint/no-explicit-any */
async function fromSearch(symbol: string): Promise<NewsItem[]> {
  try {
    const r: any = await (yf.search as any)(symbol, { newsCount: 15, quotesCount: 0 }, { validateResult: false });
    return (r.news ?? []).map((n: any) => ({
      title: n.title ?? "",
      publisher: n.publisher ?? "",
      link: n.link ?? null,
      time: n.providerPublishTime instanceof Date ? n.providerPublishTime.toISOString() : null,
      summary: "",
    }));
  } catch {
    return [];
  }
}
/* eslint-enable @typescript-eslint/no-explicit-any */

const decode = (s: string) =>
  s.replace(/<!\[CDATA\[|\]\]>/g, "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#x27;|&#39;|&apos;/g, "'").trim();

async function fromRss(symbol: string): Promise<NewsItem[]> {
  try {
    const url = `https://feeds.finance.yahoo.com/rss/2.0/headline?s=${encodeURIComponent(symbol)}&region=US&lang=en-US`;
    const res = await fetch(url, { headers: UA, signal: AbortSignal.timeout(10000) });
    if (!res.ok) return [];
    const xml = await res.text();
    const pick = (block: string, tag: string) => {
      const m = block.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`));
      return m ? decode(m[1]) : "";
    };
    return [...xml.matchAll(/<item>([\s\S]*?)<\/item>/g)].map((m) => {
      const pub = pick(m[1], "pubDate");
      const t = pub ? new Date(pub) : null;
      return {
        title: pick(m[1], "title"),
        publisher: "Yahoo Finance",
        link: pick(m[1], "link") || null,
        time: t && !Number.isNaN(t.getTime()) ? t.toISOString() : null,
        summary: pick(m[1], "description"),
      };
    });
  } catch {
    return [];
  }
}

async function fetchNews(symbol: string): Promise<NewsItem[]> {
  // Yahoo retired the per-ticker news endpoint, so use the search API and top up from RSS when thin
  // (Thai stocks have very few articles).
  let items = await fromSearch(symbol);
  if (items.length < 5) {
    const seen = new Set(items.map((i) => i.title.trim().toLowerCase()));
    items = items.concat((await fromRss(symbol)).filter((i) => !seen.has(i.title.trim().toLowerCase())));
  }
  return items
    .filter((i) => i.title)
    .sort((a, b) => (b.time ?? "").localeCompare(a.time ?? ""));
}

export async function getNews(symbol: string) {
  "use cache";
  cacheLife({ stale: 60, revalidate: 300, expire: 3600 });
  return fetchNews(symbol);
}

// ---------- Sentiment (VADER, English text) ----------
export function sentimentScore(text: string): number {
  return vader.SentimentIntensityAnalyzer.polarity_scores(text).compound;
}

export function sentimentLabel(score: number): { label: string; cls: "up" | "down" | "flat" } {
  if (score >= 0.4) return { label: "บวกมาก", cls: "up" };
  if (score >= 0.1) return { label: "บวก", cls: "up" };
  if (score > -0.1) return { label: "กลาง", cls: "flat" };
  if (score > -0.4) return { label: "ลบ", cls: "down" };
  return { label: "ลบมาก", cls: "down" };
}

// ---------- Translation (free endpoints; both rate-limit, so use one then the other) ----------
let googleDownUntil = 0;

async function googleTranslate(text: string): Promise<string> {
  const url = `https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=th&dt=t&q=${encodeURIComponent(text)}`;
  const res = await fetch(url, { headers: UA, signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error(`google ${res.status}`);
  const data = (await res.json()) as [string, string][][];
  return data[0].map((seg) => seg[0]).join("");
}

async function myMemoryTranslate(text: string): Promise<string> {
  const url = `https://api.mymemory.translated.net/get?q=${encodeURIComponent(text.slice(0, 450))}&langpair=en|th`;
  const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
  const data = (await res.json()) as { responseStatus: number | string; responseData?: { translatedText?: string } };
  const out = data.responseData?.translatedText;
  if (Number(data.responseStatus) !== 200 || !out || /MYMEMORY WARNING/i.test(out)) throw new Error("mymemory");
  return out;
}

async function translateOne(text: string): Promise<string> {
  if (Date.now() >= googleDownUntil) {
    try {
      return await googleTranslate(text);
    } catch {
      googleDownUntil = Date.now() + 10 * 60 * 1000; // skip Google for 10 minutes
    }
  }
  return myMemoryTranslate(text);
}

async function fetchTranslations(texts: string[]): Promise<string[]> {
  const clean = texts.map((t) => (t || "").replace(/\s+/g, " ").trim());
  // One request for all headlines when Google is available.
  if (Date.now() >= googleDownUntil && clean.every(Boolean) && clean.join("\n").length <= 4500) {
    try {
      const out = (await googleTranslate(clean.join("\n"))).split("\n");
      if (out.length === clean.length) return out;
    } catch {
      googleDownUntil = Date.now() + 10 * 60 * 1000;
    }
  }
  const out = await Promise.all(clean.map((t) => (t ? translateOne(t).catch(() => "") : Promise.resolve(""))));
  if (clean.some(Boolean) && out.every((o) => !o)) throw new Error("translation unavailable"); // not cached
  return out;
}

export async function translateTexts(texts: string[]) {
  "use cache";
  cacheLife("hours");
  return fetchTranslations(texts);
}
