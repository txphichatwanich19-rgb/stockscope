"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function SearchBox({ big = false, autoFocus = false }: { big?: boolean; autoFocus?: boolean }) {
  const router = useRouter();
  const [value, setValue] = useState("");

  return (
    <form
      className={`search${big ? " big" : ""}`}
      role="search"
      onSubmit={(e) => {
        e.preventDefault();
        const sym = value.trim().toUpperCase();
        if (sym) {
          router.push(`/stock/${encodeURIComponent(sym)}`);
          setValue("");
        }
      }}
    >
      <svg width={big ? 18 : 15} height={big ? 18 : 15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </svg>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder={big ? "พิมพ์รหัสหุ้น เช่น NVDA, PTT.BK, BTC-USD" : "ค้นหารหัสหุ้น"}
        aria-label="รหัสหุ้น"
        autoFocus={autoFocus}
        autoCapitalize="characters"
        spellCheck={false}
      />
    </form>
  );
}
