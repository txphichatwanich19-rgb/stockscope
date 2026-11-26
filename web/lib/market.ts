// US market session helpers. Uses the New York time zone, so daylight saving is handled correctly
// (the Streamlit version hard-coded UTC-4 and took the weekday from UTC).

export type UsSession = "weekend" | "pre" | "regular" | "after" | "closed";

export function usSession(now = new Date()): UsSession {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
    hour: "numeric",
    minute: "numeric",
    hour12: false,
  }).formatToParts(now);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  const wd = get("weekday");
  const mins = (Number(get("hour")) % 24) * 60 + Number(get("minute"));
  if (wd === "Sat" || wd === "Sun") return "weekend";
  if (mins >= 4 * 60 && mins < 9 * 60 + 30) return "pre";
  if (mins >= 9 * 60 + 30 && mins < 16 * 60) return "regular";
  if (mins >= 16 * 60 && mins < 20 * 60) return "after";
  return "closed";
}

export const SESSION_HINT: Record<UsSession, string> = {
  weekend: "🌙 ตลาดสหรัฐฯ ปิดวันเสาร์-อาทิตย์ · แสดงราคา after-hours ล่าสุดของวันศุกร์",
  pre: "🌅 กำลังเป็นช่วง Pre-Market ของสหรัฐฯ",
  regular: "🟢 ตลาดสหรัฐฯ เปิดอยู่ตอนนี้ · แสดงราคา pre-market ของเช้านี้ (ถ้ามี)",
  after: "🌙 กำลังเป็นช่วง After-Hours ของสหรัฐฯ",
  closed: "⚪ ตลาดสหรัฐฯ ปิดอยู่ · แสดงราคา after-hours ล่าสุด",
};

/** Pre/regular/after windows in Thai time (UTC+7) for whichever offset New York is on right now. */
export function sessionWindowsThai(now = new Date()): string {
  const fmt = (hhmm: string) => hhmm;
  const nyHour = Number(
    new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", hour: "numeric", hour12: false }).format(now),
  );
  const thaiHour = Number(
    new Intl.DateTimeFormat("en-US", { timeZone: "Asia/Bangkok", hour: "numeric", hour12: false }).format(now),
  );
  const diff = (((thaiHour - nyHour) % 24) + 24) % 24; // 11 in US daylight time, 12 in standard time
  const at = (nyH: number, nyM = 0) => {
    const t = (((nyH + diff) % 24) * 60 + nyM) % 1440;
    return fmt(`${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`);
  };
  return `pre-market ${at(4)}–${at(9, 30)} ไทย · regular ${at(9, 30)}–${at(16)} · after-hours ${at(16)}–${at(20)}`;
}

export const STATE_LABEL: Record<string, string> = {
  REGULAR: "🟢 ตลาดปกติเปิดอยู่",
  PRE: "🌅 อยู่ในช่วง Pre-Market",
  PREPRE: "🌅 Pre-Market (เช้ามาก)",
  POST: "🌙 อยู่ในช่วง After-Hours",
  POSTPOST: "🌙 After-Hours (ดึก)",
  CLOSED: "⚪ ตลาดปิดอยู่",
};
