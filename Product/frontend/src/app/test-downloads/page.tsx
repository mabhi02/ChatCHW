"use client";

import { useState } from "react";
import { parseContentDispositionFilename } from "@/lib/downloads";

const BACKEND = "http://localhost:8000";
const ENDPOINTS = [
  { fmt: "json", expected: "test-artifact.json" },
  { fmt: "csv", expected: "test-artifact.csv" },
  { fmt: "zip", expected: "test-bundle.zip" },
  { fmt: "txt", expected: "hello-world.txt" },
];

type MechanismResult = {
  method: string;
  status: "idle" | "fired" | "error";
  error?: string;
  notes?: string;
};

/**
 * Smoke test page for download mechanisms. Tries 5 download methods against
 * the backend's /api/test-download endpoint (which sends Content-Disposition:
 * attachment; filename="..."). Check your Downloads folder after clicking
 * each button to see which method preserved the correct filename.
 */
export default function TestDownloadsPage() {
  const [results, setResults] = useState<Record<string, MechanismResult>>({});
  const [selectedFmt, setSelectedFmt] = useState<string>("json");

  const expected = ENDPOINTS.find((e) => e.fmt === selectedFmt)?.expected ?? "?";
  const url = `${BACKEND}/api/test-download/${selectedFmt}`;

  const updateResult = (key: string, r: Partial<MechanismResult>) => {
    setResults((prev) => {
      const existing = prev[key] || { method: key, status: "idle" as const };
      return { ...prev, [key]: { ...existing, ...r } };
    });
  };

  // --- Method 1: anchor with download attribute (cross-origin; Chrome may ignore download attr) ---
  const m1_anchorWithDownload = () => {
    const a = document.createElement("a");
    a.href = url;
    a.download = expected;
    a.target = "_blank";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => a.parentNode?.removeChild(a), 100);
    updateResult("1_anchor_download", { status: "fired", notes: "Cross-origin anchor with download attr" });
  };

  // --- Method 2: plain anchor (no download attr, just rely on Content-Disposition) ---
  const m2_anchorPlain = () => {
    const a = document.createElement("a");
    a.href = url;
    a.target = "_blank";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => a.parentNode?.removeChild(a), 100);
    updateResult("2_anchor_plain", { status: "fired", notes: "Plain anchor, Content-Disposition only" });
  };

  // --- Method 3: hidden iframe navigation (the v3 approach in downloads.ts) ---
  const m3_iframe = () => {
    const iframe = document.createElement("iframe");
    iframe.style.display = "none";
    iframe.style.width = "0";
    iframe.style.height = "0";
    iframe.style.border = "0";
    iframe.src = url;
    document.body.appendChild(iframe);
    setTimeout(() => iframe.parentNode?.removeChild(iframe), 10_000);
    updateResult("3_iframe", { status: "fired", notes: "Hidden iframe navigation" });
  };

  // --- Method 4: fetch as blob + anchor.click synchronously ---
  const m4_blobSync = async () => {
    try {
      const r = await fetch(url, { credentials: "omit" });
      const blob = await r.blob();
      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objUrl;
      a.download = expected;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        URL.revokeObjectURL(objUrl);
        a.parentNode?.removeChild(a);
      }, 1000);
      updateResult("4_blob_sync", { status: "fired", notes: "fetch -> blob URL -> anchor.download" });
    } catch (e) {
      updateResult("4_blob_sync", { status: "error", error: String(e) });
    }
  };

  // --- Method 5: fetch blob + parse Content-Disposition header manually ---
  const m5_blobWithCDParse = async () => {
    try {
      const r = await fetch(url, { credentials: "omit" });
      const cd = r.headers.get("Content-Disposition") || "";
      const m = cd.match(/filename\s*=\s*"?([^";]+)"?/i);
      const filename = m ? m[1] : expected;
      const blob = await r.blob();
      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        URL.revokeObjectURL(objUrl);
        a.parentNode?.removeChild(a);
      }, 1000);
      updateResult("5_blob_cd_parse", {
        status: "fired",
        notes: `blob + parsed C-D filename: ${filename}`,
      });
    } catch (e) {
      updateResult("5_blob_cd_parse", { status: "error", error: String(e) });
    }
  };

  // --- Method 6: direct window navigation ---
  const m6_windowLocation = () => {
    // Download attribute doesn't apply; rely 100% on server's Content-Disposition.
    window.location.assign(url);
    updateResult("6_window_location", { status: "fired", notes: "window.location.assign(url) - pure browser nav" });
  };

  // --- Method 8: v5 production pattern - bare anchor, no download attr, no target ---
  const m8_v5Production = () => {
    const a = document.createElement("a");
    a.href = url;
    a.rel = "noopener";
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    setTimeout(() => a.parentNode?.removeChild(a), 100);
    updateResult("8_v5_production", {
      status: "fired",
      notes: "v5 production pattern: bare anchor click, no target, no download attr",
    });
  };

  // --- Method 7: diagnostic fetch - show what headers we receive ---
  const m7_diagnostic = async () => {
    try {
      const resp = await fetch(url, { credentials: "omit" });
      const headerPairs: string[] = [];
      resp.headers.forEach((v, k) => headerPairs.push(`${k}: ${v}`));
      const cd = resp.headers.get("Content-Disposition");
      const parsedName = parseContentDispositionFilename(cd);
      updateResult("7_diagnostic", {
        status: "fired",
        notes:
          `Status: ${resp.status}\n` +
          `Content-Disposition: ${cd || "MISSING"}\n` +
          `Parsed filename: ${parsedName || "(parse failed)"}\n` +
          `All headers:\n${headerPairs.join("\n")}`,
      });
    } catch (e) {
      updateResult("7_diagnostic", { status: "error", error: String(e) });
    }
  };

  const methods = [
    { key: "1_anchor_download", label: "1. Anchor with download attr + target=_blank", fn: m1_anchorWithDownload },
    { key: "2_anchor_plain", label: "2. Plain anchor (C-D only)", fn: m2_anchorPlain },
    { key: "3_iframe", label: "3. Hidden iframe (current v3 method)", fn: m3_iframe },
    { key: "4_blob_sync", label: "4. fetch -> blob + anchor.download", fn: m4_blobSync },
    { key: "5_blob_cd_parse", label: "5. fetch -> parse C-D + blob", fn: m5_blobWithCDParse },
    { key: "6_window_location", label: "6. window.location.assign (pure nav)", fn: m6_windowLocation },
    { key: "7_diagnostic", label: "7. DIAGNOSTIC: show received headers (no download)", fn: m7_diagnostic },
    { key: "8_v5_production", label: "8. v5 PRODUCTION PATTERN (bare anchor, no target, no download attr)", fn: m8_v5Production },
  ];

  return (
    <main className="min-h-screen bg-white text-slate-900 p-8 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-2">Download Mechanism Smoke Test</h1>
      <p className="text-sm text-slate-600 mb-6">
        Each button triggers a download using a different mechanism. Backend sends
        <code className="mx-1 rounded bg-slate-100 px-1">{`Content-Disposition: attachment; filename="${expected}"`}</code>.
        Check your <strong>Downloads folder</strong> after each click — the
        mechanism wins if the saved file has the expected filename (not a UUID).
      </p>

      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Test file format:</label>
        <div className="flex gap-2">
          {ENDPOINTS.map((e) => (
            <button
              key={e.fmt}
              onClick={() => setSelectedFmt(e.fmt)}
              className={`rounded px-3 py-1 text-sm border ${
                selectedFmt === e.fmt
                  ? "bg-amber-500 text-white border-amber-600"
                  : "bg-white text-slate-700 border-slate-300 hover:bg-slate-50"
              }`}
            >
              {e.fmt}
            </button>
          ))}
        </div>
        <p className="mt-2 text-xs text-slate-500">
          URL: <code>{url}</code>
          <br />
          Expected filename: <code>{expected}</code>
        </p>
      </div>

      <div className="space-y-3">
        {methods.map((m) => {
          const r = results[m.key];
          return (
            <div
              key={m.key}
              className="rounded-lg border border-slate-200 p-4 flex items-start justify-between gap-4"
            >
              <div className="flex-1">
                <h3 className="font-semibold text-slate-900">{m.label}</h3>
                {r?.notes && (
                  <pre className="text-xs text-slate-700 mt-1 whitespace-pre-wrap font-mono bg-slate-50 p-2 rounded border border-slate-200">
                    {r.notes}
                  </pre>
                )}
                {r?.error && (
                  <p className="text-xs text-red-600 mt-1 font-mono">{r.error}</p>
                )}
              </div>
              <div className="flex flex-col gap-2 items-end">
                <button
                  onClick={m.fn}
                  className="rounded bg-slate-900 text-white px-4 py-2 text-sm hover:bg-slate-700"
                >
                  Try
                </button>
                {r && (
                  <span className={`text-xs ${r.status === "error" ? "text-red-600" : "text-green-700"}`}>
                    {r.status}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-8 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm">
        <h3 className="font-semibold mb-2">How to read the results:</h3>
        <ul className="list-disc list-inside space-y-1 text-slate-700">
          <li>
            <strong>Correct filename saved</strong> (e.g. <code>test-artifact.json</code>):
            that method preserves Content-Disposition.
          </li>
          <li>
            <strong>UUID filename saved</strong> (e.g. <code>5193384e-...</code>):
            blob/anchor activation expired; mechanism is broken.
          </li>
          <li>
            <strong>URL path as filename</strong> (e.g. <code>json</code> or <code>test-download</code>):
            download attribute was ignored for cross-origin.
          </li>
        </ul>
      </div>
    </main>
  );
}
