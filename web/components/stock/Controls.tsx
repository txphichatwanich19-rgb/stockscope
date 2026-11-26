"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { INTERVALS, PERIODS, type IntervalKey, type PeriodKey } from "@/lib/ranges";

export default function Controls({ period, interval }: { period: PeriodKey; interval: IntervalKey }) {
  const router = useRouter();
  const path = usePathname();
  const sp = useSearchParams();

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(sp.toString());
    next.set(key, value);
    router.push(`${path}?${next.toString()}`, { scroll: false });
  };

  return (
    <div className="controls">
      <label>
        <span>ช่วงย้อนหลัง</span>
        <select value={period} onChange={(e) => set("period", e.target.value)}>
          {PERIODS.map((p) => (
            <option key={p.key} value={p.key}>{p.label}</option>
          ))}
        </select>
      </label>
      <label>
        <span>ช่วงเวลาต่อแท่ง</span>
        <select value={interval} onChange={(e) => set("interval", e.target.value)}>
          {INTERVALS.map((i) => (
            <option key={i.key} value={i.key}>{i.label}</option>
          ))}
        </select>
      </label>
    </div>
  );
}
