"use client";

import type { ReplEvent } from "@/hooks/useSSE";

interface Z3PanelProps {
  events: ReplEvent[];
}

export function Z3Panel({ events }: Z3PanelProps) {
  const z3Events = events.filter((e) => e.type === "z3" && e.z3Result);

  if (z3Events.length === 0) {
    return (
      <div className="rounded-lg border border-[var(--border)] p-4">
        <h3 className="text-sm font-semibold">Z3 Verification</h3>
        <p className="mt-2 text-xs text-[var(--muted-foreground)]">
          No Z3 checks run yet
        </p>
      </div>
    );
  }

  const latest = z3Events[z3Events.length - 1];
  const result = latest.z3Result!;
  const passed = result.checks.filter((c) => c.passed);
  const failed = result.checks.filter((c) => !c.passed);

  return (
    <div className="rounded-lg border border-[var(--border)] p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Z3 Verification</h3>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-bold ${
            result.allPassed
              ? "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/30 dark:text-cyan-400"
              : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
          }`}
        >
          {result.allPassed
            ? "ALL PROVED"
            : `${failed.length} failures`}
        </span>
      </div>

      <div className="space-y-1 text-xs">
        {result.checks.map((check, i) => (
          <div key={i} className="flex items-start gap-2">
            <span
              className={
                check.passed
                  ? "text-cyan-700 dark:text-cyan-400"
                  : "text-red-700 dark:text-red-400"
              }
            >
              {check.passed ? "PASS" : "FAIL"}
            </span>
            <span className="text-[var(--muted-foreground)]">
              {check.testId}
              {check.tableId && ` (${check.tableId})`}
            </span>
          </div>
        ))}
      </div>

      <div className="text-xs text-[var(--muted-foreground)]">
        {passed.length} passed, {failed.length} failed
      </div>
    </div>
  );
}
