import { NextResponse } from "next/server";

// The visitor's own Anthropic key is sent per request, used once, and never stored or logged.
type Body = { key?: string; ticker?: string; items?: { date: string; title: string; summary: string }[] };

export async function POST(request: Request) {
  let body: Body;
  try {
    body = (await request.json()) as Body;
  } catch {
    return NextResponse.json({ error: "คำขอไม่ถูกต้อง" }, { status: 400 });
  }
  const key = (body.key ?? "").trim();
  const ticker = (body.ticker ?? "").slice(0, 20);
  const items = (body.items ?? []).slice(0, 15);
  if (!key.startsWith("sk-ant-")) return NextResponse.json({ error: "ใส่ Anthropic API key ให้ถูกต้อง (ขึ้นต้นด้วย sk-ant-)" }, { status: 400 });
  if (!items.length) return NextResponse.json({ error: "ไม่มีข่าวให้สรุป" }, { status: 400 });

  const payload = items.map((n) => `- [${n.date}] ${n.title}: ${(n.summary ?? "").slice(0, 300)}`).join("\n");
  const prompt =
    `ฉันมีข่าวเกี่ยวกับหุ้น ${ticker} ดังนี้:\n\n${payload}\n\n` +
    "ช่วยวิเคราะห์เป็นภาษาไทยให้ครบทุกข้อ:\n" +
    "1. **สรุปภาพรวม** (2-3 ประโยค)\n" +
    "2. **ประเด็นสำคัญ 3-5 ข้อ** (bullet)\n" +
    "3. **Sentiment โดยรวม** (บวก/ลบ/กลาง) พร้อมเหตุผลสั้นๆ\n" +
    "4. **สิ่งที่ควรจับตา** (บอกว่าควรตามดูอะไรต่อ)\n\n" +
    "ใช้ภาษาไทยที่อ่านง่าย ตรงไปตรงมา ไม่ต้องเกริ่นนำ";

  try {
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      body: JSON.stringify({ model: "claude-haiku-4-5", max_tokens: 1024, messages: [{ role: "user", content: prompt }] }),
      signal: AbortSignal.timeout(45000),
    });
    const data = (await res.json()) as { content?: { type: string; text?: string }[]; error?: { message?: string } };
    if (!res.ok) return NextResponse.json({ error: data.error?.message?.slice(0, 200) ?? `Anthropic ตอบกลับ ${res.status}` }, { status: 502 });
    const text = (data.content ?? []).filter((c) => c.type === "text").map((c) => c.text).join("\n");
    return NextResponse.json({ text });
  } catch {
    return NextResponse.json({ error: "เรียก Anthropic ไม่สำเร็จ ลองใหม่อีกครั้ง" }, { status: 502 });
  }
}
