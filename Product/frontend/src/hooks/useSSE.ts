"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { createSSEUrl } from "@/lib/api";

export interface ReplEvent {
  id: string;
  type: "exec" | "subcall" | "validate" | "z3" | "final" | "error" | "status";
  stepNumber: number;
  code?: string;
  stdout?: string;
  stderr?: string;
  prompt?: string;
  response?: string;
  validationResult?: {
    passed: boolean;
    errorCount: number;
    errors: Array<{ validator: string; message: string; severity: string }>;
  };
  z3Result?: {
    allPassed: boolean;
    checks: Array<{ testId: string; tableId?: string; message: string; passed: boolean }>;
  };
  tokens?: { input: number; output: number };
  executionMs?: number;
  timestamp: string;
}

interface UseSSEReturn {
  events: ReplEvent[];
  isConnected: boolean;
  error: string | null;
  connect: () => void;
  disconnect: () => void;
}

export function useSSE(
  sessionId: string | null,
  apiKey: string
): UseSSEReturn {
  const [events, setEvents] = useState<ReplEvent[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
    }
  }, []);

  const connect = useCallback(() => {
    if (!sessionId || !apiKey) return;

    disconnect();

    // SSE with auth: pass API key as query param (SSE doesn't support headers).
    // The backend validates this and strips it from logs.
    const url = `${createSSEUrl(sessionId)}?token=${encodeURIComponent(apiKey)}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    es.onmessage = (event) => {
      try {
        const data: ReplEvent = JSON.parse(event.data);
        setEvents((prev) => [...prev, data]);
      } catch {
        // Skip malformed events
      }
    };

    es.addEventListener("done", () => {
      disconnect();
    });

    es.onerror = () => {
      setError("SSE connection lost. Reconnecting...");
      setIsConnected(false);
      // EventSource auto-reconnects
    };
  }, [sessionId, apiKey, disconnect]);

  // Auto-connect when sessionId is set
  useEffect(() => {
    if (sessionId && apiKey) {
      connect();
    }
    return () => disconnect();
  }, [sessionId, apiKey, connect, disconnect]);

  return { events, isConnected, error, connect, disconnect };
}
