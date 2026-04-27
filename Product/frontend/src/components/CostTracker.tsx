"use client";

import { useEffect, useState } from "react";
import type { ReplEvent } from "@/hooks/useSSE";
import type { SessionStatus } from "@/lib/api";

interface CostTrackerProps {
  events: ReplEvent[];
  status: SessionStatus | null;
}

function formatDuration(ms: number): string {
  if (ms < 0) return "0s";
  const totalSeconds = Math.floor(ms / 1000);
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  if (mins >= 60) {
    const hrs = Math.floor(mins / 60);
    const remainingMins = mins % 60;
    return `${hrs}h ${remainingMins}m ${secs}s`;
  }
  return `${mins}m ${secs}s`;
}

export function CostTracker({ events, status }: CostTrackerProps) {
  // Live elapsed timer -- updates every second while running
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    if (!status?.startedAt) return;
    const startedAt = new Date(status.startedAt).getTime();
    const isRunning = status.status === "running";

    const tick = () => setElapsedMs(Date.now() - startedAt);
    tick(); // initial
    if (!isRunning) return; // freeze on terminal state

    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [status?.startedAt, status?.status]);

  // Prefer live stats from status endpoint (populated by backend every step).
  // Fall back to computing from SSE events if status doesn't have live data.
  const iterations = status?.totalIterations || events.filter((e) => e.type === "exec").length;
  const callsOpus = status?.callsOpus ?? 0;
  const callsSonnet = status?.callsSonnet ?? 0;
  const callsHaiku = status?.callsHaiku ?? 0;

  // Sub-calls: the RLM emits one Opus/Sonnet call per iteration plus extra
  // calls via `llm_query_batched`. The backend's `total_subcalls` counter
  // only fires for recursive sub-RLMs, not batched sub-queries, so we
  // derive it from the routed call counts instead. Haiku calls are
  // catchers, displayed separately. Clamp to >=0 so rounding jitter on
  // a still-warming-up run doesn't go negative.
  const derivedSubcalls = Math.max(0, callsOpus + callsSonnet - iterations);
  const subcalls = status?.totalSubcalls
    ? Math.max(status.totalSubcalls, derivedSubcalls)
    : derivedSubcalls;

  // Token counts: prefer status endpoint (live from backend accumulator)
  const inputTokens = status?.inputTokensTotal ?? 0;
  const outputTokens = status?.outputTokensTotal ?? 0;
  const cachedTokens = status?.cachedTokensTotal ?? 0;
  const cacheWriteTokens = status?.cacheWriteTokensTotal ?? 0;

  const cost = status?.costEstimateUsd ?? 0;
  const isRunning = status?.status === "running";

  return (
    <div className="rounded-lg border border-[var(--border)] p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Cost Tracker</h3>
        <div className="flex items-center gap-1.5">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              isRunning ? "bg-green-400 animate-pulse" : "bg-gray-400"
            }`}
          />
          <span className="font-mono text-[11px] text-[var(--muted-foreground)]">
            {formatDuration(elapsedMs)}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
        <div>
          <p className="text-[var(--muted-foreground)]">Iterations</p>
          <p className="font-mono font-bold">{iterations}</p>
        </div>
        <div>
          <p className="text-[var(--muted-foreground)]">Sub-calls</p>
          <p className="font-mono font-bold">{subcalls}</p>
        </div>
        <div>
          <p className="text-[var(--muted-foreground)]">Input tokens</p>
          <p className="font-mono font-bold">{inputTokens.toLocaleString()}</p>
        </div>
        <div>
          <p className="text-[var(--muted-foreground)]">Output tokens</p>
          <p className="font-mono font-bold">{outputTokens.toLocaleString()}</p>
        </div>
        {cachedTokens > 0 && (
          <div>
            <p className="text-[var(--muted-foreground)]">Cache read</p>
            <p className="font-mono font-bold text-[var(--accent)]">
              {cachedTokens.toLocaleString()}
            </p>
          </div>
        )}
        {cacheWriteTokens > 0 && (
          <div>
            <p className="text-[var(--muted-foreground)]">Cache write</p>
            <p className="font-mono font-bold text-yellow-400">
              {cacheWriteTokens.toLocaleString()}
            </p>
          </div>
        )}
      </div>

      {(callsOpus > 0 || callsSonnet > 0 || callsHaiku > 0) && (
        <div className="border-t border-[var(--border)] pt-2 space-y-1">
          <p className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
            Model routing
          </p>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
            <span>
              <span className="text-[var(--muted-foreground)]">Opus:</span>{" "}
              <span className="font-mono font-bold">{callsOpus}</span>
            </span>
            <span>
              <span className="text-[var(--muted-foreground)]">Sonnet:</span>{" "}
              <span className="font-mono font-bold">{callsSonnet}</span>
            </span>
            <span>
              <span className="text-[var(--muted-foreground)]">Haiku:</span>{" "}
              <span className="font-mono font-bold">{callsHaiku}</span>
              <span className="ml-1 text-[9px] text-[var(--muted-foreground)]">
                (catchers)
              </span>
            </span>
          </div>
        </div>
      )}

      <div className="border-t border-[var(--border)] pt-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-[var(--muted-foreground)]">
            Estimated cost
          </span>
          <span className="text-sm font-mono font-bold text-[var(--accent)]">
            ${cost.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}
