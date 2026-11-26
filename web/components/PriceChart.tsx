"use client";

import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  createChart,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  type UTCTimestamp,
} from "lightweight-charts";
import type { Bar } from "@/lib/yahoo";
import { sma } from "@/lib/indicators";

const GREEN = "#5cff7a";
const RED = "#d06a6a";

// Daily-or-longer bars carry a plain date; intraday bars carry a full ISO timestamp.
const toTime = (d: string) => (Date.parse(d.length === 10 ? `${d}T00:00:00Z` : d) / 1000) as UTCTimestamp;

export default function PriceChart({ bars }: { bars: Bar[] }) {
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = box.current;
    if (!el || !bars.length) return;

    const chart = createChart(el, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#76767c",
        fontFamily: "var(--font-geist-mono), monospace",
        fontSize: 11,
        panes: { separatorColor: "rgba(255,255,255,0.06)" },
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.03)" },
        horzLines: { color: "rgba(255,255,255,0.04)" },
      },
      rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
      timeScale: { borderColor: "rgba(255,255,255,0.08)", timeVisible: bars[0]?.time.length !== 10 },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(255,255,255,0.25)", labelBackgroundColor: "#1a1a1c" },
        horzLine: { color: "rgba(255,255,255,0.25)", labelBackgroundColor: "#1a1a1c" },
      },
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "rgba(92,255,122,0.18)",
      borderUpColor: GREEN,
      wickUpColor: GREEN,
      downColor: "rgba(208,106,106,0.25)",
      borderDownColor: RED,
      wickDownColor: RED,
    });
    candles.setData(bars.map((b) => ({ time: toTime(b.time), open: b.open, high: b.high, low: b.low, close: b.close })));

    const closes = bars.map((b) => b.close);
    const addSma = (window: number, color: string) => {
      const line = chart.addSeries(LineSeries, {
        color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      const v = sma(closes, window);
      line.setData(bars.flatMap((b, i) => (v[i] === null ? [] : [{ time: toTime(b.time), value: v[i] as number }])));
    };
    addSma(20, "rgba(255,255,255,0.85)");
    addSma(50, "rgba(255,255,255,0.3)");

    const vol = chart.addSeries(
      HistogramSeries,
      { priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false },
      1,
    );
    vol.setData(
      bars.map((b, i) => ({
        time: toTime(b.time),
        value: b.volume,
        color: i > 0 && b.close < bars[i - 1].close ? "rgba(208,106,106,0.35)" : "rgba(255,255,255,0.22)",
      })),
    );
    chart.panes()[1]?.setHeight(80);
    // autoSize measures the box after the first layout, so fit the bars once the real width is known.
    chart.timeScale().fitContent();
    // Wait a frame so the chart's own autoSize has applied the new width before fitting.
    const ro = new ResizeObserver(() => requestAnimationFrame(() => chart.timeScale().fitContent()));
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
    };
  }, [bars]);

  if (!bars.length) return <div className="empty">ไม่มีข้อมูลกราฟสำหรับช่วงนี้</div>;
  return <div ref={box} className="chart-box" />;
}
