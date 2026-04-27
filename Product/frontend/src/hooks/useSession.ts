"use client";

import { useState, useCallback, useEffect } from "react";
import {
  startSession,
  getSessionStatus,
  cancelSession,
  StartSessionError,
} from "@/lib/api";
import type { SessionStatus, StartSessionOptions } from "@/lib/api";

interface UseSessionReturn {
  apiKey: string;
  setApiKey: (key: string) => void;
  sessionId: string | null;
  status: SessionStatus | null;
  isLoading: boolean;
  error: string | null;
  start: (options: StartSessionOptions) => Promise<void>;
  cancel: () => Promise<void>;
  clearSession: () => void;
  clearApiKey: () => void;
}

export function useSession(): UseSessionReturn {
  // Seed from sessionStorage so the key survives Next.js route navigation
  // but is wiped on tab close (sessionStorage is per-tab, auto-cleared).
  const [apiKey, setApiKeyRaw] = useState(() => {
    if (typeof window !== "undefined") {
      return sessionStorage.getItem("_chw_ak") ?? "";
    }
    return "";
  });
  const [sessionId, setSessionId] = useState<string | null>(() => {
    if (typeof window !== "undefined") {
      return sessionStorage.getItem("_chw_sid") ?? null;
    }
    return null;
  });
  const [status, setStatus] = useState<SessionStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Wrap setApiKey to mirror into sessionStorage
  const setApiKey = useCallback((key: string) => {
    setApiKeyRaw(key);
    if (typeof window !== "undefined") {
      if (key) {
        sessionStorage.setItem("_chw_ak", key);
      } else {
        sessionStorage.removeItem("_chw_ak");
      }
    }
  }, []);

  // Mirror sessionId into sessionStorage
  useEffect(() => {
    if (typeof window !== "undefined") {
      if (sessionId) {
        sessionStorage.setItem("_chw_sid", sessionId);
      } else {
        sessionStorage.removeItem("_chw_sid");
      }
    }
  }, [sessionId]);

  // sessionStorage auto-clears when the tab closes -- no beforeunload
  // handler needed. Adding one would break full-page navigations (e.g.,
  // browser refresh, Playwright goto) by clearing the key mid-session.

  // Poll session status while running
  useEffect(() => {
    if (!sessionId || !apiKey) return;
    if (status?.status === "passed" || status?.status === "failed" || status?.status === "halted") {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const s = await getSessionStatus(apiKey, sessionId);
        setStatus(s);
      } catch {
        // Silently retry on transient errors
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [sessionId, apiKey, status?.status]);

  const start = useCallback(
    async (options: StartSessionOptions) => {
      if (!apiKey) {
        setError("API key is required");
        return;
      }
      setIsLoading(true);
      setError(null);
      try {
        const res = await startSession(apiKey, options);
        setSessionId(res.sessionId);
        setStatus({
          sessionId: res.sessionId,
          status: "running",
          totalIterations: 0,
          totalSubcalls: 0,
          validationErrors: 0,
          costEstimateUsd: 0,
        });
      } catch (err) {
        // StartSessionError carries structured detail from the backend.
        // We surface the `message` field (which the backend writes to be
        // user-friendly) instead of the generic "API error 400: ..." text.
        // For generic Errors, fall through to the plain message.
        if (err instanceof StartSessionError) {
          const detail = err.detail;
          if (typeof detail === "object" && detail !== null && "message" in detail && detail.message) {
            setError(detail.message);
          } else {
            setError(err.message);
          }
        } else {
          setError(err instanceof Error ? err.message : "Failed to start session");
        }
      } finally {
        setIsLoading(false);
      }
    },
    [apiKey]
  );

  const cancel = useCallback(async () => {
    if (!sessionId || !apiKey) return;
    try {
      await cancelSession(apiKey, sessionId);
      setStatus((prev) => (prev ? { ...prev, status: "halted" } : null));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel");
    }
  }, [sessionId, apiKey]);

  const clearSession = useCallback(() => {
    setSessionId(null);
    setStatus(null);
    setError(null);
  }, []);

  const clearApiKey = useCallback(() => {
    setApiKey("");
    setSessionId(null);
    setStatus(null);
    setError(null);
  }, []);

  return {
    apiKey,
    setApiKey,
    sessionId,
    status,
    isLoading,
    error,
    start,
    cancel,
    clearSession,
    clearApiKey,
  };
}
