"use client";

import { useRef, useEffect } from "react";
import type { ReplEvent } from "@/hooks/useSSE";

interface ReplStreamProps {
  events: ReplEvent[];
  isConnected: boolean;
}

function EventCard({ event }: { event: ReplEvent }) {
  // Light-mode base classes + dark: overrides so the labels stay legible on
  // both white and dark page backgrounds. Dark text on light, pale text on dark.
  const typeColors: Record<string, string> = {
    exec: "text-blue-700",
    subcall: "text-purple-700",
    validate: "text-amber-700",
    z3: "text-cyan-700",
    final: "text-green-700",
    error: "text-red-700",
    status: "text-slate-600",
  };

  const typeLabels: Record<string, string> = {
    exec: "EXEC",
    subcall: "SUB-CALL",
    validate: "VALIDATE",
    z3: "Z3 CHECK",
    final: "FINAL",
    error: "ERROR",
    status: "STATUS",
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 space-y-2 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className={`text-xs font-mono font-bold ${typeColors[event.type] || "text-slate-600"}`}
          >
            [{typeLabels[event.type] || event.type.toUpperCase()}]
          </span>
          <span className="text-xs text-slate-500">
            Step {event.stepNumber}
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          {event.executionMs !== undefined && (
            <span>{event.executionMs}ms</span>
          )}
          {event.tokens && (
            <span>
              {event.tokens.input}+{event.tokens.output} tokens
            </span>
          )}
        </div>
      </div>

      {event.code && (
        <pre className="overflow-x-auto rounded border border-slate-200 bg-slate-50 p-3 text-xs font-mono text-slate-900">
          {event.code}
        </pre>
      )}

      {event.stdout && (
        <pre className="overflow-x-auto rounded border border-slate-200 bg-slate-50 p-3 text-xs font-mono text-slate-700">
          {event.stdout}
        </pre>
      )}

      {event.stderr && (
        <pre className="overflow-x-auto rounded border border-red-200 bg-red-50 p-3 text-xs font-mono text-red-900">
          {event.stderr}
        </pre>
      )}

      {event.prompt && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-purple-700">Sub-call prompt:</p>
          <pre className="overflow-x-auto rounded border border-purple-200 bg-purple-50 p-3 text-xs font-mono text-purple-900">
            {event.prompt.length > 500
              ? event.prompt.slice(0, 500) + "..."
              : event.prompt}
          </pre>
        </div>
      )}

      {event.validationResult && (
        <div
          className={`rounded border p-3 text-xs ${
            event.validationResult.passed
              ? "border-green-200 bg-green-50 text-green-900"
              : "border-red-200 bg-red-50 text-red-900"
          }`}
        >
          <p className="font-bold">
            Validation: {event.validationResult.passed ? "PASSED" : "FAILED"} (
            {event.validationResult.errorCount} errors)
          </p>
          {event.validationResult.errors.slice(0, 10).map((err, i) => (
            <p key={i} className="mt-1">
              [{err.validator}] {err.message}
            </p>
          ))}
        </div>
      )}

      {event.z3Result && (
        <div
          className={`rounded border p-3 text-xs ${
            event.z3Result.allPassed
              ? "border-cyan-200 bg-cyan-50 text-cyan-900"
              : "border-red-200 bg-red-50 text-red-900"
          }`}
        >
          <p className="font-bold">
            Z3: {event.z3Result.allPassed ? "ALL PASSED" : "FAILURES FOUND"}
          </p>
          {event.z3Result.checks
            .filter((c) => !c.passed)
            .map((c, i) => (
              <p key={i} className="mt-1">
                [{c.testId}] {c.tableId ? `table=${c.tableId}: ` : ""}
                {c.message}
              </p>
            ))}
        </div>
      )}
    </div>
  );
}

export function ReplStream({ events, isConnected }: ReplStreamProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events.length]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">REPL Trajectory</h2>
        <div className="flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full ${
              isConnected ? "bg-green-600 animate-pulse" : "bg-slate-400"
            }`}
          />
          <span className="text-xs text-slate-500">
            {isConnected ? "Live" : "Disconnected"}
          </span>
        </div>
      </div>

      <div className="max-h-[600px] space-y-2 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-4">
        {events.length === 0 ? (
          isConnected ? (
            <div className="flex flex-col items-center gap-3 py-10">
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-amber-500" />
              <p className="text-sm font-medium text-slate-900">
                Initializing extraction model
              </p>
              <p className="text-xs text-slate-500 max-w-sm text-center">
                Caching the 28K-token system prompt and building the REPL environment.
                First call to Opus takes 2-3 minutes; subsequent iterations are faster.
              </p>
            </div>
          ) : (
            <p className="text-center text-sm text-slate-500 py-8">
              Waiting for REPL events...
            </p>
          )
        ) : (
          events.map((event, i) => <EventCard key={i} event={event} />)
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
