import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import SmoothScroll from "@/src/components/SmoothScroll";
import Navbar from "@/src/components/Navbar";

// Use Google Fonts instead of local files
const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Maths AI Tutor",
  description: "Advanced AI Tutor for CS109",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} antialiased`}>
        <SmoothScroll>
          <div className="flex flex-col min-h-screen">
             <Navbar /> 
             <main className="flex-grow pt-16">{children}</main>
          </div>
        </SmoothScroll>
      </body>
    </html>
  );
}