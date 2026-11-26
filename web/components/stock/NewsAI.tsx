"use client";

import { useState } from "react";

type Item = { date: string; title: string; summary: string };
const KEY = "stockscope.anthropicKey";

export default function NewsAI({ ticker, items }: { ticker: string; items: Item[] }) {
  // Read the saved key lazily (client only) instead of setting state inside an effect.
  const [key, setKey] = useState(() => {
    try { return typeof window === "undefined" ? "" : (localStorage.getItem(KEY) ?? ""); } catch { return ""; }
  });
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState("");

  async function run() {
    setBusy(true); setError(""); setText("");
    try { localStorage.setItem(KEY, key); } catch { /* ignore */ }
    try {
      const res = await fetch("/api/summarize", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ key, ticker, items }),
      });
      const data = (await res.json()) as { text?: string; error?: string };
      if (!res.ok) setError(data.error ?? "เกิดข้อผิดพลาด");
      else setText(data.text ?? "");
    } catch {
      setError("เชื่อมต่อไม่สำเร็จ");
    } finally {
      setBusy(false);
    }
  }

  return (
    <details className="card ai-box">
      <summary>🤖 AI สรุปข่าว (Claude) · ไม่บังคับ</summary>
      <p className="muted-note">
        ใส่ Anthropic API key ของคุณเองเพื่อให้ AI สรุปข่าวเป็นภาษาไทย (ขอได้ที่ console.anthropic.com ใช้ประมาณ 1 บาทต่อการสรุป)
        key ถูกส่งไปเรียก Anthropic ครั้งเดียวต่อการกด และเก็บไว้ในเบราว์เซอร์ของคุณเท่านั้น
      </p>
      <div className="ai-row">
        <input type="password" suppressHydrationWarning value={key} onChange={(e) => setKey(e.target.value)} placeholder="sk-ant-…" aria-label="Anthropic API key" autoComplete="off" />
        <button type="button" className="btn" disabled={busy || !key} onClick={run}>
          {busy ? "AI กำลังอ่านและสรุป…" : "🤖 สรุปด้วย AI"}
        </button>
      </div>
      {error && <p className="down" role="alert">{error}</p>}
      {text && <div className="ai-out">{text}</div>}
    </details>
  );
}
