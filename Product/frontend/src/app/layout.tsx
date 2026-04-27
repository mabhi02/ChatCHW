import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CHW Navigator",
  description:
    "Clinical decision logic extractor for Community Health Worker manuals",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
