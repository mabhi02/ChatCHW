"use client";

import type { ReplEvent } from "@/hooks/useSSE";

interface ValidationPanelProps {
  events: ReplEvent[];
}

export function ValidationPanel({ events }: ValidationPanelProps) {
  const validationEvents = events.filter(
    (e) => e.type === "validate" && e.validationResult
  );

  if (validationEvents.length === 0) {
    return (
      <div className="rounded-lg border border-[var(--border)] p-4">
        <h3 className="text-sm font-semibold">Validation</h3>
        <p className="mt-2 text-xs text-[var(--muted-foreground)]">
          No validation runs yet
        </p>
      </div>
    );
  }

  const latest = validationEvents[validationEvents.length - 1];
  const result = latest.validationResult!;

  const validatorGroups = new Map<string, { passed: number; failed: number }>();
  for (const err of result.errors) {
    const existing = validatorGroups.get(err.validator) || {
      passed: 0,
      failed: 0,
    };
    if (err.severity === "error") {
      existing.failed++;
    } else {
      existing.passed++;
    }
    validatorGroups.set(err.validator, existing);
  }

  // Add validators with no errors as passed
  for (const name of ["architecture", "completeness", "clinical", "naming"]) {
    if (!validatorGroups.has(name)) {
      validatorGroups.set(name, { passed: 1, failed: 0 });
    }
  }

  return (
    <div className="rounded-lg border border-[var(--border)] p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Validation</h3>
        <span
          className={`rounded-full px-2 py-0.5 text-xs font-bold ${
            result.passed
              ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400"
              : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
          }`}
        >
          {result.passed ? "PASSED" : `${result.errorCount} errors`}
        </span>
      </div>

      <div className="space-y-1">
        {Array.from(validatorGroups.entries()).map(([name, counts]) => (
          <div
            key={name}
            className="flex items-center justify-between text-xs"
          >
            <span className="capitalize">{name}</span>
            <span
              className={
                counts.failed > 0
                  ? "text-red-700 dark:text-red-400"
                  : "text-green-700 dark:text-green-400"
              }
            >
              {counts.failed > 0 ? `${counts.failed} errors` : "Pass"}
            </span>
          </div>
        ))}
      </div>

      <p className="text-xs text-[var(--muted-foreground)]">
        Run {validationEvents.length} of session
      </p>
    </div>
  );
}
