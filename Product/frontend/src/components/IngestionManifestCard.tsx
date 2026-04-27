"use client";

import { useState } from "react";
import type { GuideMetadataResponse, IngestionFlaggedItem } from "@/lib/api";

interface IngestionManifestCardProps {
  metadata: GuideMetadataResponse;
  forceOverride: boolean;
  onForceOverrideChange: (value: boolean) => void;
}

const SEVERITY_STYLES: Record<string, string> = {
  critical:
    "border-red-300 bg-red-50 text-red-900 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-400",
  warning:
    "border-yellow-300 bg-yellow-50 text-yellow-900 dark:border-yellow-500/40 dark:bg-yellow-500/10 dark:text-yellow-400",
  info:
    "border-blue-300 bg-blue-50 text-blue-900 dark:border-blue-500/40 dark:bg-blue-500/10 dark:text-blue-400",
};

const HIERARCHY_LABEL: Record<string, string> = {
  good: "Good — clean section hierarchy detected",
  sparse: "Sparse — few sections detected, may have gaps",
  noisy: "Noisy — too many sections, likely false-positive titles",
  fallback: "Fallback — no sections detected, using per-page navigation",
  unknown: "Unknown",
};

export function IngestionManifestCard({
  metadata,
  forceOverride,
  onForceOverrideChange,
}: IngestionManifestCardProps) {
  const [showItems, setShowItems] = useState(false);
  const manifest = metadata.manifest ?? {};
  const items = (manifest.flagged_items as IngestionFlaggedItem[] | undefined) ?? [];
  const critical = metadata.criticalCount ?? 0;
  const warnings = metadata.warningCount ?? 0;
  const info = metadata.infoCount ?? 0;
  const hasIssues = critical + warnings > 0;
  const blocked = critical > 0 && !forceOverride;

  return (
    <div className="space-y-3 rounded-lg border border-[var(--border)] p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">
            Guide ready: {metadata.id.slice(0, 12)}...
          </p>
          <p className="text-xs text-[var(--muted-foreground)]">
            {metadata.pageCount} pages · {metadata.sectionCount} sections ·{" "}
            {metadata.filename}
          </p>
        </div>
        <div className="flex gap-2">
          {critical > 0 && (
            <span className="rounded-md border border-red-300 bg-red-50 px-2 py-0.5 text-xs font-medium text-red-900 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-400">
              {critical} critical
            </span>
          )}
          {warnings > 0 && (
            <span className="rounded-md border border-yellow-300 bg-yellow-50 px-2 py-0.5 text-xs font-medium text-yellow-900 dark:border-yellow-500/40 dark:bg-yellow-500/10 dark:text-yellow-400">
              {warnings} warning{warnings !== 1 ? "s" : ""}
            </span>
          )}
          {info > 0 && (
            <span className="rounded-md border border-blue-300 bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-900 dark:border-blue-500/40 dark:bg-blue-500/10 dark:text-blue-400">
              {info} info
            </span>
          )}
          {!hasIssues && (
            <span className="rounded-md border border-green-300 bg-green-50 px-2 py-0.5 text-xs font-medium text-green-900 dark:border-green-500/40 dark:bg-green-500/10 dark:text-green-400">
              clean
            </span>
          )}
        </div>
      </div>

      <div className="text-xs text-[var(--muted-foreground)]">
        <span className="font-medium">Hierarchy:</span>{" "}
        {HIERARCHY_LABEL[metadata.hierarchyQuality] ?? metadata.hierarchyQuality}
      </div>

      {hasIssues && (
        <button
          type="button"
          onClick={() => setShowItems((s) => !s)}
          className="text-xs text-[var(--accent)] underline"
        >
          {showItems ? "Hide" : "Show"} {items.length} flagged item
          {items.length !== 1 ? "s" : ""}
        </button>
      )}

      {showItems && items.length > 0 && (
        <div className="max-h-64 space-y-2 overflow-y-auto">
          {items.map((item, idx) => (
            <div
              key={idx}
              className={`rounded border px-2 py-1.5 text-xs ${
                SEVERITY_STYLES[item.severity] ?? SEVERITY_STYLES.info
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="font-medium uppercase">{item.severity}</span>
                <span className="text-[var(--muted-foreground)]">
                  {item.issue_type}
                </span>
                {item.page_number && (
                  <span className="text-[var(--muted-foreground)]">
                    p.{item.page_number}
                  </span>
                )}
              </div>
              <p className="mt-1">{item.message}</p>
            </div>
          ))}
        </div>
      )}

      {blocked && (
        <div className="rounded-md border border-red-300 bg-red-50 p-3 dark:border-red-500/50 dark:bg-red-500/10">
          <p className="text-xs font-semibold text-red-900 dark:text-red-400">
            Cannot start extraction: {critical} critical ingestion issue
            {critical !== 1 ? "s" : ""}
          </p>
          <p className="mt-1 text-xs text-[var(--muted-foreground)]">
            Re-ingest the PDF, or check the box below to override and proceed
            anyway.
          </p>
          <label className="mt-2 flex items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={forceOverride}
              onChange={(e) => onForceOverrideChange(e.target.checked)}
              className="h-3 w-3"
            />
            <span>Override quality gate and proceed anyway</span>
          </label>
        </div>
      )}
    </div>
  );
}
