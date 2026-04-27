"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { ApiKeyInput } from "@/components/ApiKeyInput";
import { GuideUpload } from "@/components/GuideUpload";
import { IngestionProgress } from "@/components/IngestionProgress";
import { IngestionManifestCard } from "@/components/IngestionManifestCard";
import { TierBadge } from "@/components/TierBadge";
import { useSession } from "@/hooks/useSession";
import { ingestPdf, fetchGuideMetadata, type GuideMetadataResponse } from "@/lib/api";

export default function Home() {
  const router = useRouter();
  const { apiKey, setApiKey, sessionId, start, isLoading, error } =
    useSession();
  const [guideJson, setGuideJson] = useState<object | null>(null);
  const [manualName, setManualName] = useState<string>("");
  const [ingestionJobId, setIngestionJobId] = useState<string | null>(null);
  const [sourceGuideId, setSourceGuideId] = useState<string | null>(null);
  const [guideMetadata, setGuideMetadata] = useState<GuideMetadataResponse | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [forceOverride, setForceOverride] = useState(false);

  // Navigate to session page when session starts
  useEffect(() => {
    if (sessionId) {
      router.push(`/session/${sessionId}`);
    }
  }, [sessionId, router]);

  const handleUploadPdf = useCallback(
    async (file: File, checkDupes: boolean) => {
      if (!apiKey) {
        setIngestError("API key is required");
        return;
      }
      // Reset any prior guide/ingestion state
      setGuideJson(null);
      setSourceGuideId(null);
      setIngestionJobId(null);
      setIngestError(null);
      setIsIngesting(true);
      setManualName(file.name.replace(/\.pdf$/i, ""));
      try {
        const res = await ingestPdf(apiKey, file, {
          checkDupes,
          manualName: file.name.replace(/\.pdf$/i, ""),
        });
        setIngestionJobId(res.jobId);
      } catch (err) {
        setIsIngesting(false);
        setIngestError(
          err instanceof Error ? err.message : "Failed to start ingestion"
        );
      }
    },
    [apiKey]
  );

  const handleUploadJson = useCallback(
    (json: object, filename: string) => {
      // Clear any PDF ingestion state
      setIngestionJobId(null);
      setSourceGuideId(null);
      setIsIngesting(false);
      setIngestError(null);
      setGuideJson(json);
      setManualName(filename.replace(/\.json$/i, ""));
    },
    []
  );

  const handleIngestComplete = useCallback(
    async (guideId: string) => {
      setSourceGuideId(guideId);
      setIsIngesting(false);
      // Fetch the guide metadata so we can show the manifest summary inline.
      // Failures are non-fatal; the user can still try to start a session and
      // the backend's quality gate will catch critical issues.
      try {
        const meta = await fetchGuideMetadata(apiKey, guideId);
        setGuideMetadata(meta);
      } catch (err) {
        // Non-fatal; just don't show the manifest preview.
        console.warn("Failed to fetch guide metadata", err);
      }
    },
    [apiKey]
  );

  const handleIngestError = useCallback((msg: string) => {
    setIngestError(msg);
    setIsIngesting(false);
    setIngestionJobId(null);
  }, []);

  const handleStart = async () => {
    // Read pipeline override from `?pipeline=gen8|gen8.5|gen7|legacy` so we can
    // smoke-test new pipelines without changing the default.
    const pipelineParam = (typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("pipeline")
      : null) as ("gen7" | "gen8" | "gen8.5" | "legacy" | null);

    if (sourceGuideId) {
      await start({
        sourceGuideId,
        manualName: manualName || undefined,
        forceIngestionOverride: forceOverride,
        ...(pipelineParam ? { pipeline: pipelineParam } : {}),
      });
    } else if (guideJson) {
      await start({
        guideJson,
        manualName: manualName || undefined,
        ...(pipelineParam ? { pipeline: pipelineParam } : {}),
      });
    }
  };

  const canStart =
    !!apiKey &&
    (sourceGuideId !== null || guideJson !== null) &&
    !isLoading &&
    !isIngesting;

  return (
    <main className="mx-auto max-w-2xl px-6 py-16">
      <div className="space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">
            CHW Navigator
          </h1>
          <p className="text-[var(--muted-foreground)]">
            Extract clinical decision logic from CHW manuals using Recursive
            Language Models
          </p>
        </div>

        <div className="space-y-6 rounded-xl border border-[var(--border)] bg-[var(--background)] p-6 shadow-sm">
          <ApiKeyInput
            apiKey={apiKey}
            onApiKeyChange={setApiKey}
            disabled={isLoading}
          />

          {/* Tier badge: probes the user's Anthropic key and shows a green
              "Tier X — ready" banner for Tier 2+ or a red "upgrade required"
              banner for Tier 1. Debounced so it only fires after the user
              stops typing. Costs ~$0.0001 per probe on the user's bill. */}
          {apiKey && <TierBadge apiKey={apiKey} />}

          <GuideUpload
            onUploadPdf={handleUploadPdf}
            onUploadJson={handleUploadJson}
            disabled={isLoading || !apiKey || isIngesting}
          />

          {ingestionJobId && (
            <IngestionProgress
              apiKey={apiKey}
              jobId={ingestionJobId}
              onComplete={handleIngestComplete}
              onError={handleIngestError}
            />
          )}

          {sourceGuideId && guideMetadata && (
            <IngestionManifestCard
              metadata={guideMetadata}
              forceOverride={forceOverride}
              onForceOverrideChange={setForceOverride}
            />
          )}
          {sourceGuideId && !guideMetadata && (
            <div className="rounded-md border border-[var(--success)]/40 bg-[var(--success)]/10 px-3 py-2 text-xs text-[var(--success)]">
              Guide ready: {sourceGuideId.slice(0, 12)}...
            </div>
          )}

          {ingestError && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-transparent dark:bg-red-900/20 dark:text-red-400">
              {ingestError}
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-transparent dark:bg-red-900/20 dark:text-red-400">
              {error}
            </div>
          )}

          <button
            onClick={handleStart}
            disabled={!canStart}
            className="w-full rounded-lg bg-[var(--primary)] px-4 py-3 text-sm font-medium text-[var(--primary-foreground)] transition-colors hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isLoading
              ? "Starting extraction..."
              : isIngesting
                ? "Ingesting PDF..."
                : "Start Extraction"}
          </button>
        </div>

        <div className="rounded-xl border border-blue-300 bg-blue-50 p-4 text-center space-y-2 dark:border-blue-800/50 dark:bg-blue-950/30">
          <p className="text-sm font-semibold text-blue-900 dark:text-blue-300">
            Arena Mode: A/B Pipeline Comparison
          </p>
          <p className="text-xs text-blue-800/80 dark:text-zinc-400">
            Run REPL vs Sequential pipelines side-by-side on the same guide.
            Requires both Anthropic and OpenAI API keys.
          </p>
          <a
            href="/arena"
            className="inline-block mt-2 px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium transition-colors"
          >
            Open Arena
          </a>
        </div>

        <div className="text-center text-xs text-[var(--muted-foreground)] space-y-1">
          <p>
            Your API keys are held in browser memory only. Never stored.
          </p>
          <p>
            Both Anthropic and OpenAI keys are BYOK for arena mode.
            Unstructured.io is server-side.
          </p>
        </div>
      </div>
    </main>
  );
}
