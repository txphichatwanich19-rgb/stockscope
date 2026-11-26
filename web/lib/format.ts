export function fmtPrice(v: number | null, digits?: number): string {
  if (v === null) return "—";
  // Sub-dollar prices (e.g. DOGE ≈ 0.06) need more decimals to mean anything.
  digits ??= Math.abs(v) > 0 && Math.abs(v) < 1 ? 4 : 2;
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtPct(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
}

export function fmtSigned(v: number | null): string {
  if (v === null) return "—";
  return `${v >= 0 ? "+" : ""}${fmtPrice(v)}`;
}

export function fmtBig(v: number | null): string {
  if (v === null) return "—";
  const units: [number, string][] = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];
  for (const [div, u] of units) if (Math.abs(v) >= div) return `${(v / div).toFixed(2)}${u}`;
  return v.toLocaleString("en-US");
}

export function direction(v: number | null): "up" | "down" | "flat" {
  if (v === null || v === 0) return "flat";
  return v > 0 ? "up" : "down";
}

const STATE_TH: Record<string, string> = {
  REGULAR: "ตลาดเปิด",
  PRE: "ก่อนตลาดเปิด",
  PREPRE: "ก่อนตลาดเปิด",
  POST: "หลังตลาดปิด",
  POSTPOST: "หลังตลาดปิด",
  CLOSED: "ตลาดปิด",
};
export const marketStateTh = (s: string) => STATE_TH[s] ?? s ?? "—";

export function fmtThaiTime(isoStr: string | null): string {
  if (!isoStr) return "—";
  return new Date(isoStr).toLocaleString("th-TH", {
    timeZone: "Asia/Bangkok",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function displaySymbol(sym: string): string {
  return sym.replace(".BK", "").replace("-USD", "");
}

/** Plain (unsigned) percent, e.g. 12.3% */
export function fmtPlainPct(v: number | null, digits = 1): string {
  return v === null ? "—" : `${v.toFixed(digits)}%`;
}

export function fmtMoneyUsd(v: number): string {
  const a = Math.abs(v);
  const s = a >= 1e9 ? `${(a / 1e9).toFixed(2)}B` : a >= 1e6 ? `${(a / 1e6).toFixed(2)}M` : a >= 1e3 ? `${(a / 1e3).toFixed(1)}K` : a.toFixed(0);
  return `${v < 0 ? "-" : ""}$${s}`;
}

export function fmtDateTh(isoStr: string | null): string {
  if (!isoStr) return "—";
  return new Date(isoStr).toLocaleDateString("th-TH", { timeZone: "Asia/Bangkok", day: "numeric", month: "short", year: "numeric" });
}

export function fmtDateTimeTh(isoStr: string | null): string {
  if (!isoStr) return "—";
  return new Date(isoStr).toLocaleString("th-TH", {
    timeZone: "Asia/Bangkok",
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function recLabel(mean: number | null): string {
  if (mean === null) return "—";
  if (mean <= 1.5) return "Strong Buy";
  if (mean <= 2.5) return "Buy";
  if (mean <= 3.5) return "Hold";
  if (mean <= 4.5) return "Sell";
  return "Strong Sell";
}

/** Whole days from now until an ISO date (kept out of components, which must stay pure). */
export function daysUntil(isoStr: string): number {
  return Math.ceil((new Date(isoStr).getTime() - Date.now()) / 864e5);
}
