"use client";

import { use } from "react";
import { useSession } from "@/hooks/useSession";
import { useSSE } from "@/hooks/useSSE";
import { ReplStream } from "@/components/ReplStream";
import { ExtractionJournal } from "@/components/ExtractionJournal";
import { ValidationPanel } from "@/components/ValidationPanel";
import { Z3Panel } from "@/components/Z3Panel";
import { ArtifactDownloads } from "@/components/ArtifactDownloads";
import { CostTracker } from "@/components/CostTracker";
import { MermaidDiagram } from "@/components/MermaidDiagram";
import { createArtifactDownloadUrl } from "@/lib/api";

export default function SessionPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id: sessionId } = use(params);
  const { apiKey, status, cancel } = useSession();
  const { events, isConnected } = useSSE(sessionId, apiKey);

  const isComplete = status?.status === "passed";
  const isFailed = status?.status === "failed" || status?.status === "halted";
  const isRunning = status?.status === "running";
  const errorMessage = status?.errorMessage ?? null;

  return (
    <main className="mx-auto max-w-7xl px-6 py-8 bg-white text-slate-900 min-h-screen">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Extraction Session</h1>
          <p className="text-sm text-slate-500 font-mono">
            {sessionId}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`rounded-full px-3 py-1 text-xs font-bold ${
              isComplete
                ? "bg-green-100 text-green-800"
                : isFailed
                  ? "bg-red-100 text-red-800"
                  : "bg-yellow-100 text-yellow-800"
            }`}
          >
            {status?.status?.toUpperCase() || "CONNECTING"}
          </span>
          {isRunning && (
            <button
              onClick={cancel}
              className="rounded-lg border border-red-500 px-3 py-1.5 text-xs text-red-600 hover:bg-red-50"
            >
              Cancel
            </button>
          )}
        </div>
      </div>

      {/* Failure banner: shows error message + partial-artifacts hint.
          Error message is in a scrollable pre with max-height so long
          stack traces or API errors don't push the page layout around. */}
      {isFailed && (
        <div className="mb-6 rounded-lg border border-red-300 bg-red-50 p-3">
          <div className="flex items-start gap-3">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-red-200 text-xs font-bold text-red-800">
              !
            </span>
            <div className="flex-1 min-w-0 space-y-1.5">
              <p className="text-sm font-semibold text-red-900">
                Extraction {status?.status === "halted" ? "halted" : "failed"}
              </p>
              {errorMessage && (
                <details className="group" open>
                  <summary className="text-[11px] text-red-700 cursor-pointer hover:text-red-900 select-none">
                    <span className="group-open:hidden">Show error details</span>
                    <span className="hidden group-open:inline">Hide error details</span>
                  </summary>
                  <pre className="mt-1 max-h-24 overflow-auto rounded bg-red-100 p-2 text-[11px] font-mono text-red-800 whitespace-pre-wrap break-words">
                    {errorMessage}
                  </pre>
                </details>
              )}
              <p className="text-[11px] text-slate-600">
                Intermediate artifacts from completed phases are still available
                for download in the Artifacts panel.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-6">
          <ExtractionJournal events={events} />
          {/* Mermaid flowchart -- only shown after successful extraction */}
          {isComplete && apiKey && (
            <MermaidDiagram
              sourceUrl={createArtifactDownloadUrl(sessionId, "mermaid", apiKey)}
              title="Clinical Logic Flowchart"
            />
          )}
          <ReplStream events={events} isConnected={isConnected} />
        </div>

        <div className="space-y-4">
          <CostTracker events={events} status={status} />
          <ValidationPanel events={events} />
          <Z3Panel events={events} />
          <ArtifactDownloads
            sessionId={sessionId}
            apiKey={apiKey}
            isComplete={isComplete}
            isFailed={isFailed}
          />
        </div>
      </div>
    </main>
  );
}
