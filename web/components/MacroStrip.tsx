import { Suspense } from "react";
import MacroBar from "@/components/MacroBar";

export default function MacroStrip() {
  return (
    <Suspense fallback={<div className="macro" aria-busy />}>
      <MacroBar />
    </Suspense>
  );
}
