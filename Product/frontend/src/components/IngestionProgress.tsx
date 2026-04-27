"use client";

import { useEffect, useState } from "react";
import { pollIngestion, type IngestStatus } from "@/lib/api";

interface IngestionProgressProps {
  apiKey: string;
  jobId: string;
  onComplete: (guideId: string) => void;
  onError: (error: string) => void;
}

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued",
  cache_check: "Checking cache",
  cache_hit: "Cache hit! Loading existing guide",
  acquiring_lock: "Acquiring ingestion lock",
  warming_pool: "Warming parser pool",
  parsing_with_unstructured: "Parsing PDF with Unstructured.io",
  rendering_crops: "Rendering page regions",
  vision_table_refinement: "Refining tables with vision AI",
  vision_image_description: "Describing images",
  vision_flowchart_parsing: "Parsing flowcharts",
  assembling: "Assembling guide",
  writing_to_neon: "Saving to database",
  done: "Complete",
  failed: "Failed",
};

export function IngestionProgress({
  apiKey,
  jobId,
  onComplete,
  onError,
}: IngestionProgressProps) {
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    pollIngestion(apiKey, jobId, (s) => {
      if (!cancelled) setStatus(s);
    })
      .then((final) => {
        if (cancelled) return;
        if (final.guide_id) {
          onComplete(final.guide_id);
        } else {
          onError("Ingestion completed but no guide_id was returned");
        }
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        onError(msg);
      });

    return () => {
      cancelled = true;
    };
  }, [apiKey, jobId, onComplete, onError]);

  const progress = status?.progress ?? 0;
  const stageLabel = status?.stage
    ? STAGE_LABELS[status.stage] ?? status.stage
    : "Starting";
  const note = status?.note ?? "";

  return (
    <div className="space-y-3 p-4 rounded-lg border border-[var(--border)]">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">{stageLabel}</span>
        <span className="text-sm text-[var(--muted-foreground)]">
          {Math.round(progress * 100)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-[var(--border)] overflow-hidden">
        <div
          className="h-full bg-[var(--accent)] transition-all duration-300 ease-out"
          style={{ width: `${progress * 100}%` }}
        />
      </div>
      {note && (
        <p className="text-xs text-[var(--muted-foreground)]">{note}</p>
      )}
      {error && (
        <p className="text-xs text-[var(--destructive)]">{error}</p>
      )}
    </div>
  );
}
