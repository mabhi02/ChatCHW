"use client";

import { useEffect, useState, useCallback } from "react";
import { checkAnthropicTier, type TierCheckResult } from "@/lib/api";

interface TierBadgeProps {
  apiKey: string;
  /**
   * Debounce delay in ms before firing the tier probe after the key changes.
   * Prevents spamming the probe endpoint (and the user's Anthropic bill)
   * on every keystroke. Default 800ms is long enough that most pastes
   * settle before we probe.
   */
  debounceMs?: number;
}

/**
 * Renders a status badge for the user's Anthropic API key:
 *   - grey "Checking..." while the probe is in flight
 *   - green "Tier X — ready to run" for Tier 2+
 *   - red "Upgrade required" for Tier 1
 *   - red "Invalid key" on auth errors
 *
 * Debounces the probe so typing a key one character at a time doesn't
 * fire one probe per character. The probe costs ~$0.0001 on the user's
 * Anthropic bill — cheap, but we're respectful.
 *
 * Called once on mount (if the key is already present from sessionStorage)
 * and once per key change after the debounce. Re-renders don't re-probe.
 */
export function TierBadge({ apiKey, debounceMs = 800 }: TierBadgeProps) {
  const [state, setState] = useState<
    | { kind: "idle" }
    | { kind: "checking" }
    | { kind: "ok"; result: TierCheckResult }
    | { kind: "warn"; result: TierCheckResult }
    | { kind: "error"; message: string }
  >({ kind: "idle" });

  const runCheck = useCallback(async (key: string) => {
    if (!key) {
      setState({ kind: "idle" });
      return;
    }
    setState({ kind: "checking" });
    try {
      const result = await checkAnthropicTier(key);
      if (result.meetsMinimum) {
        setState({ kind: "ok", result });
      } else {
        setState({ kind: "warn", result });
      }
    } catch (err) {
      setState({
        kind: "error",
        message:
          err instanceof Error
            ? err.message
            : "Tier check failed — the backend may be unreachable.",
      });
    }
  }, []);

  // Debounced probe on key change
  useEffect(() => {
    if (!apiKey) {
      setState({ kind: "idle" });
      return;
    }
    const t = setTimeout(() => {
      runCheck(apiKey);
    }, debounceMs);
    return () => clearTimeout(t);
  }, [apiKey, debounceMs, runCheck]);

  if (state.kind === "idle") {
    return null;
  }

  if (state.kind === "checking") {
    return (
      <div className="rounded-md border border-zinc-300 bg-zinc-100 px-3 py-2 text-xs text-zinc-700 dark:border-zinc-700 dark:bg-zinc-900/40 dark:text-zinc-400">
        <span className="inline-flex items-center gap-2">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-zinc-500 dark:bg-zinc-400" />
          Verifying Anthropic key tier...
        </span>
      </div>
    );
  }

  if (state.kind === "error") {
    return (
      <div className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-900 dark:border-red-800/50 dark:bg-red-950/30 dark:text-red-400">
        <div className="flex items-start gap-2">
          <span className="mt-0.5 font-mono text-red-600 dark:text-red-500">!</span>
          <div className="flex-1">
            <p className="font-medium">Tier check failed</p>
            <p className="mt-0.5 text-red-700 dark:text-red-400/80">{state.message}</p>
          </div>
        </div>
      </div>
    );
  }

  if (state.kind === "warn") {
    const r = state.result;
    const tierLabel = r.tier ? `Tier ${r.tier}` : "Invalid / unreachable";
    return (
      <div className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800/50 dark:bg-amber-950/30 dark:text-amber-300">
        <div className="flex items-start gap-2">
          <span className="mt-0.5 font-mono text-amber-600 dark:text-amber-500">!</span>
          <div className="flex-1 space-y-1">
            <p className="font-medium">
              {tierLabel} — not sufficient for full-guide extractions
            </p>
            <p className="text-amber-800 dark:text-amber-400/80 whitespace-pre-line">{r.message}</p>
            <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-amber-700 dark:text-amber-400/70">
              <span>RPM: {r.requestsLimit.toLocaleString()}</span>
              <span>Input TPM: {r.inputTpm.toLocaleString()}</span>
              <span>Output TPM: {r.outputTpm.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // state.kind === "ok"
  const r = state.result;
  return (
    <div className="rounded-md border border-green-300 bg-green-50 px-3 py-2 text-xs text-green-900 dark:border-green-800/50 dark:bg-green-950/20 dark:text-green-300">
      <div className="flex items-start gap-2">
        <span className="mt-0.5 font-mono text-green-700 dark:text-green-400">OK</span>
        <div className="flex-1 space-y-1">
          <p className="font-medium">
            Tier {r.tier} — ready to run full-guide extractions
          </p>
          <p className="text-green-800 dark:text-green-400/80">{r.message}</p>
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[10px] text-green-700 dark:text-green-400/70">
            <span>RPM: {r.requestsLimit.toLocaleString()}</span>
            <span>Input TPM: {r.inputTpm.toLocaleString()}</span>
            <span>Output TPM: {r.outputTpm.toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
