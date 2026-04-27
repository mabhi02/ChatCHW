"use client";

import { useState, useEffect } from "react";
import { getArtifacts, createArtifactDownloadUrl } from "@/lib/api";
import type { Artifact } from "@/lib/api";
import { downloadBlobAs } from "@/lib/downloads";

interface ArtifactDownloadsProps {
  sessionId: string;
  apiKey: string;
  isComplete: boolean;
  isFailed?: boolean;
}

const ARTIFACT_LABELS: Record<string, { label: string; icon: string }> = {
  journal: { label: "Extraction Journal", icon: "MD" },
  json: { label: "Clinical Logic (JSON)", icon: "{ }" },
  system_prompt: { label: "System Prompt", icon: "MD" },
  dmn: { label: "DMN XML", icon: "XML" },
  xlsx: { label: "XLSForm (XLSX)", icon: "XLS" },
  mermaid: { label: "Flowchart (source)", icon: "MD" },
  mermaid_png: { label: "Flowchart (PNG)", icon: "PNG" },
  predicates_csv: { label: "Predicates (CSV)", icon: "CSV" },
  phrases_csv: { label: "Phrase Bank (CSV)", icon: "CSV" },
  final_dmn_validator: { label: "Final DMN Validator", icon: "OK" },
  artifact_supply_list: { label: "Supply List", icon: "{ }" },
  artifact_variables: { label: "Variables", icon: "{ }" },
  artifact_predicates: { label: "Predicates", icon: "{ }" },
  artifact_modules: { label: "Modules", icon: "{ }" },
  artifact_router: { label: "Router", icon: "{ }" },
  artifact_integrative: { label: "Integrative", icon: "{ }" },
  artifact_phrase_bank: { label: "Phrase Bank", icon: "{ }" },
};

export function ArtifactDownloads({
  sessionId,
  apiKey,
  isComplete,
  isFailed,
}: ArtifactDownloadsProps) {
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch whenever the run reaches a terminal state (passed OR failed).
  // Failed runs still have intermediate artifacts on disk (phase checkpoints)
  // that are useful for debugging.
  const shouldFetch = isComplete || isFailed;

  useEffect(() => {
    if (!shouldFetch || !apiKey) return;

    setLoading(true);
    getArtifacts(apiKey, sessionId)
      .then(setArtifacts)
      .catch(() => setArtifacts([]))
      .finally(() => setLoading(false));
  }, [shouldFetch, sessionId, apiKey]);

  if (!shouldFetch) {
    return (
      <div className="rounded-lg border border-[var(--border)] p-4">
        <h3 className="text-sm font-semibold">Artifacts</h3>
        <p className="mt-2 text-xs text-[var(--muted-foreground)]">
          Available after extraction completes
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="rounded-lg border border-[var(--border)] p-4">
        <h3 className="text-sm font-semibold">Artifacts</h3>
        <p className="mt-2 text-xs text-[var(--muted-foreground)]">
          Loading...
        </p>
      </div>
    );
  }

  if (artifacts.length === 0) {
    return (
      <div className="rounded-lg border border-[var(--border)] p-4">
        <h3 className="text-sm font-semibold">Artifacts</h3>
        <p className="mt-2 text-xs text-[var(--muted-foreground)]">
          No artifacts available for this session
        </p>
      </div>
    );
  }

  const zipUrl = createArtifactDownloadUrl(sessionId, "artifacts.zip", apiKey);

  return (
    <div className="rounded-lg border border-[var(--border)] p-4 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Download Artifacts</h3>
        {isFailed && (
          <span className="text-[10px] rounded bg-yellow-900/30 px-2 py-0.5 text-yellow-400">
            partial
          </span>
        )}
      </div>

      {/* ZIP bundle button -- downloads everything at once.
          Uses fetch-blob-download to work around Chrome ignoring the
          `download` attribute on cross-origin anchors. */}
      <button
        type="button"
        onClick={() =>
          downloadBlobAs(
            zipUrl,
            `${sessionId.slice(0, 8)}-artifacts.zip`,
          ).catch((e) => {
            console.error("ZIP download failed:", e);
            alert(`Could not download artifacts.zip.\n\n${e?.message ?? e}`);
          })
        }
        className="flex items-center justify-center gap-2 w-full rounded-lg border border-[var(--accent)] bg-[var(--accent)]/10 px-3 py-2.5 text-sm font-medium text-[var(--accent)] hover:bg-[var(--accent)]/20 transition-colors cursor-pointer"
      >
        <span className="font-mono text-xs">ZIP</span>
        <span>Download all ({artifacts.length} files)</span>
      </button>

      <div className="border-t border-[var(--border)] pt-3">
        <p className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)] mb-2">
          Individual files
        </p>
        <div className="grid grid-cols-1 gap-1.5 max-h-[320px] overflow-y-auto">
          {artifacts.map((artifact) => {
            const meta = ARTIFACT_LABELS[artifact.type] || {
              label: artifact.type,
              icon: "?",
            };
            const url = createArtifactDownloadUrl(
              sessionId,
              artifact.type,
              apiKey,
            );
            return (
              <button
                key={artifact.type}
                type="button"
                onClick={() =>
                  downloadBlobAs(url, artifact.filename).catch((e) => {
                    console.error(
                      `Download failed for ${artifact.filename}:`,
                      e,
                    );
                    alert(
                      `Could not download ${artifact.filename}.\n\n${e?.message ?? e}`,
                    );
                  })
                }
                className="flex items-center gap-2 rounded border border-[var(--border)] p-2 text-xs hover:bg-[var(--muted)] transition-colors cursor-pointer text-left"
              >
                <span className="flex h-6 w-8 shrink-0 items-center justify-center rounded bg-[var(--accent)]/10 text-[10px] font-bold text-[var(--accent)]">
                  {meta.icon}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="font-medium truncate">{meta.label}</p>
                  <p className="text-[10px] text-[var(--muted-foreground)] truncate font-mono">
                    {artifact.filename}
                  </p>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
