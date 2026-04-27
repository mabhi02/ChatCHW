/**
 * Backend API client for CHW Navigator.
 * API key flows via Authorization header, never stored.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface StartSessionResponse {
  sessionId: string;
  status: string;
}

interface SessionStatus {
  sessionId: string;
  status: "running" | "passed" | "failed" | "halted";
  totalIterations: number;
  totalSubcalls: number;
  validationErrors: number;
  costEstimateUsd: number;
  inputTokensTotal?: number;
  outputTokensTotal?: number;
  cachedTokensTotal?: number;
  cacheWriteTokensTotal?: number;
  callsOpus?: number;
  callsSonnet?: number;
  callsHaiku?: number;
  startedAt?: string | null;
  errorMessage?: string | null;
}

interface Artifact {
  type: string;
  filename: string;
  downloadUrl: string;
}

export interface IngestStartResponse {
  jobId: string;
  status: string;
}

export interface IngestStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  stage: string;
  progress: number; // 0.0 to 1.0
  note: string;
  guide_id?: string | null;
  content_hash?: string | null;
  error?: string | null;
  updated_at: number;
}

async function apiFetch<T>(
  path: string,
  apiKey: string,
  options: RequestInit = {}
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      ...options.headers,
    },
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }

  return res.json();
}

// ---------------------------------------------------------------------------
// Anthropic tier pre-flight check
// ---------------------------------------------------------------------------

export interface TierCheckResult {
  tier: number | null;           // 1..4, or null if probe failed
  requestsLimit: number;         // RPM ceiling reported by Anthropic
  inputTpm: number;              // Input tokens per minute ceiling
  outputTpm: number;             // Output tokens per minute ceiling
  meetsMinimum: boolean;         // True iff tier >= 2 (chunked catchers work)
  message: string;               // Canonical user-facing status
  error?: string | null;         // Probe error code if any
}

/**
 * Ask the backend to probe the user's Anthropic API key and classify its
 * tier. Cheap (~$0.0001 per call on the user's Anthropic bill, since the
 * probe goes through the user's BYOK key).
 *
 * Never throws on tier/auth/network errors — always returns a structured
 * result the UI can display. Only throws if the backend itself is unreachable.
 */
export async function checkAnthropicTier(
  apiKey: string,
): Promise<TierCheckResult> {
  const res = await fetch(`${API_BASE}/api/anthropic/tier-check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ apiKey }),
  });
  if (!res.ok) {
    // The backend is supposed to never 4xx this endpoint — it wraps auth
    // errors in the structured response. If we DO see a non-2xx, it's a
    // backend bug or a deployment issue, not a user problem.
    const body = await res.text();
    throw new Error(`Tier check backend error ${res.status}: ${body}`);
  }
  return res.json();
}

/**
 * Structured error payload from /api/session/start when the user's tier
 * is insufficient. Returned as the `detail` field of a 400 response.
 */
export interface InsufficientTierError {
  error: "insufficient_tier";
  tier: number | null;
  requestsLimit: number;
  inputTpm: number;
  outputTpm: number;
  message: string;
  probeError?: string | null;
}

/**
 * Structured error payload when the server is already running
 * MAX_CONCURRENT_SESSIONS extractions. Returned as the `detail` field of
 * a 429 response on /api/session/start.
 */
export interface TooManyConcurrentSessionsError {
  error: "too_many_concurrent_sessions";
  running: number;
  limit: number;
  message: string;
}

export type StartSessionOptions = (
  | { guideJson: object; manualName?: string }
  | {
      sourceGuideId: string;
      manualName?: string;
      forceIngestionOverride?: boolean;
    }
) & {
  // Pipeline override. When omitted, the backend defaults to "gen7".
  // Drive via `?pipeline=gen8` on the frontend URL for ad-hoc smoke tests.
  pipeline?: "gen7" | "gen8" | "gen8.5" | "legacy";
};

/**
 * Thrown by `startSession` when /api/session/start returns a structured
 * error. Carries the parsed `detail` payload so callers can distinguish
 * tier errors from concurrency errors from generic failures, and show
 * the appropriate UI for each.
 */
export class StartSessionError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail:
      | InsufficientTierError
      | TooManyConcurrentSessionsError
      | { error?: string; message?: string }
      | string,
  ) {
    const message =
      typeof detail === "string"
        ? detail
        : "message" in detail && detail.message
          ? detail.message
          : `Session start failed (${status})`;
    super(message);
    this.name = "StartSessionError";
  }
}

export async function startSession(
  apiKey: string,
  options: StartSessionOptions
): Promise<StartSessionResponse> {
  const body: Record<string, unknown> = {};
  if ("guideJson" in options) {
    body.guide_json = options.guideJson;
  } else {
    body.sourceGuideId = options.sourceGuideId;
    if (options.forceIngestionOverride) {
      body.forceIngestionOverride = true;
    }
  }
  if (options.manualName) body.manual_name = options.manualName;
  if (options.pipeline) body.pipeline = options.pipeline;

  const res = await fetch(`${API_BASE}/api/session/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    // The backend returns structured `detail` objects for the two errors
    // we know about (insufficient_tier at 400, too_many_concurrent_sessions
    // at 429). Try to parse JSON; fall back to text on anything else.
    let detail: StartSessionError["detail"];
    try {
      const parsed = await res.json();
      detail = parsed?.detail ?? parsed;
    } catch {
      detail = await res.text();
    }
    throw new StartSessionError(res.status, detail);
  }

  return res.json();
}

export async function getSessionStatus(
  apiKey: string,
  sessionId: string
): Promise<SessionStatus> {
  return apiFetch<SessionStatus>(
    `/api/session/${sessionId}/status`,
    apiKey
  );
}

export async function getArtifacts(
  apiKey: string,
  sessionId: string
): Promise<Artifact[]> {
  return apiFetch<Artifact[]>(
    `/api/session/${sessionId}/artifacts`,
    apiKey
  );
}

export async function cancelSession(
  apiKey: string,
  sessionId: string
): Promise<void> {
  await apiFetch(`/api/session/${sessionId}/cancel`, apiKey, {
    method: "POST",
  });
}

export function createSSEUrl(sessionId: string): string {
  return `${API_BASE}/api/session/${sessionId}/stream`;
}

/**
 * Build a download URL with the API key as a query-param token.
 * Used by <a href> tags which can't send custom Authorization headers.
 * `artifactType` can be an individual type id (e.g. "journal") or
 * "artifacts.zip" for the full bundle endpoint.
 */
export function createArtifactDownloadUrl(
  sessionId: string,
  artifactType: string,
  apiKey: string,
): string {
  const token = encodeURIComponent(apiKey);
  if (artifactType === "artifacts.zip") {
    return `${API_BASE}/api/session/${sessionId}/artifacts.zip?token=${token}`;
  }
  return `${API_BASE}/api/session/${sessionId}/artifacts/${artifactType}?token=${token}`;
}

export async function ingestPdf(
  apiKey: string,
  file: File,
  options: { checkDupes: boolean; manualName?: string } = { checkDupes: true }
): Promise<IngestStartResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (options.manualName) formData.append("manual_name", options.manualName);
  const url = `${API_BASE}/api/ingest?check_dupes=${options.checkDupes}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Ingestion start failed ${res.status}: ${body}`);
  }
  return res.json();
}

export async function getIngestStatus(
  apiKey: string,
  jobId: string
): Promise<IngestStatus> {
  return apiFetch<IngestStatus>(`/api/ingest/${jobId}`, apiKey);
}

export interface IngestionFlaggedItem {
  page_number?: number | null;
  element_id?: string | null;
  issue_type: string;
  severity: "critical" | "warning" | "info";
  message: string;
  context?: Record<string, unknown>;
}

export interface GuideMetadataResponse {
  id: string;
  filename: string;
  manualName?: string | null;
  pageCount: number;
  ingestedAt: string;
  sectionCount: number;
  hierarchyQuality: string;
  manifest?: Record<string, unknown>;
  criticalCount: number;
  warningCount: number;
  infoCount: number;
}

export async function fetchGuideMetadata(
  apiKey: string,
  guideId: string
): Promise<GuideMetadataResponse> {
  return apiFetch<GuideMetadataResponse>(
    `/api/ingest/guides/${guideId}`,
    apiKey
  );
}

export async function pollIngestion(
  apiKey: string,
  jobId: string,
  onProgress?: (status: IngestStatus) => void,
  intervalMs: number = 1000,
  timeoutMs: number = 1_800_000 // 30 min — covers cold Unstructured + vision for 500-page manuals
): Promise<IngestStatus> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const status = await getIngestStatus(apiKey, jobId);
    if (onProgress) onProgress(status);
    if (status.status === "done") return status;
    if (status.status === "failed") {
      throw new Error(`Ingestion failed: ${status.error || "unknown"}`);
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error(
    `Ingestion timed out after ${Math.round(timeoutMs / 60000)} minutes`
  );
}

// ---------------------------------------------------------------------------
// Arena API
// ---------------------------------------------------------------------------

export interface ArenaStartResponse {
  arenaId: string;
  status: string;
}

export interface GuideListItem {
  id: string;
  filename: string;
  manualName: string | null;
  pageCount: number;
  pdfBytes: number;
  ingestedAt: string | null;
}

export async function startArena(
  anthropicKey: string,
  openaiKey: string,
  sourceGuideId: string,
  nRuns: number = 1,
  manualName?: string
): Promise<ArenaStartResponse> {
  const res = await fetch(`${API_BASE}/api/arena/start`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${anthropicKey}`,
      "X-OpenAI-Key": openaiKey,
    },
    body: JSON.stringify({ sourceGuideId, nRuns, manualName }),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Arena start failed ${res.status}: ${body}`);
  }
  return res.json();
}

export function createArenaSSEUrl(arenaId: string): string {
  return `${API_BASE}/api/arena/${arenaId}/stream`;
}

export async function getArenaStatus(
  anthropicKey: string,
  arenaId: string
): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>(
    `/api/arena/${arenaId}/status`,
    anthropicKey
  );
}

export async function listGuides(
  apiKey: string
): Promise<GuideListItem[]> {
  return apiFetch<GuideListItem[]>("/api/guides", apiKey);
}

export function getArenaArtifactUrl(
  arenaId: string,
  pipeline: string,
  runNumber: number,
  artifactName: string
): string {
  return `${API_BASE}/api/arena/${arenaId}/artifacts/${pipeline}/${runNumber}/${artifactName}`;
}

export type { StartSessionResponse, SessionStatus, Artifact };
