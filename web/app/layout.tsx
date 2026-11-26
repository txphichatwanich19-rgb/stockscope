import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono, IBM_Plex_Sans_Thai } from "next/font/google";
import SearchBox from "@/components/SearchBox";
import "./globals.css";
import "./panels.css";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });
const plexThai = IBM_Plex_Sans_Thai({
  variable: "--font-thai",
  subsets: ["thai", "latin"],
  weight: ["300", "400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Stockscope",
  description: "ส่องหุ้นสหรัฐ หุ้นไทย และคริปโต ในที่เดียว",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="th" className={`${geist.variable} ${geistMono.variable} ${plexThai.variable}`}>
      <body>
        <header className="topbar">
          <div className="shell">
            <Link href="/" className="brand" aria-label="Stockscope หน้าแรก">
              <div className="brand-mark" />
              <span>STOCKSCOPE</span>
            </Link>
            <nav className="nav">
              <Link href="/">ภาพรวมตลาด</Link>
              <Link href="/browse">หมวดหมู่และไอเดีย</Link>
              <Link href="/scan">สแกนหุ้น</Link>
            </nav>
            <div className="spacer" />
            <SearchBox />
          </div>
        </header>
        <main className="shell">{children}</main>
      </body>
    </html>
  );
}
