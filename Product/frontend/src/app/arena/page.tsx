"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  startArena,
  createArenaSSEUrl,
  listGuides,
  getArenaArtifactUrl,
  type GuideListItem,
} from "@/lib/api";
import { downloadBlobAs } from "@/lib/downloads";

interface ArenaEvent {
  type: string;
  data: Record<string, unknown>;
}

interface RunState {
  status: "pending" | "running" | "completed" | "failed";
  steps: ArenaEvent[];
  artifacts: string[];
  cost_usd: number;
  wall_clock_ms: number;
  error?: string;
}

export default function ArenaPage() {
  // API Keys
  const [anthropicKey, setAnthropicKey] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [keysValid, setKeysValid] = useState(false);

  // Guide selection
  const [guides, setGuides] = useState<GuideListItem[]>([]);
  const [selectedGuide, setSelectedGuide] = useState<string>("");
  const [nRuns, setNRuns] = useState(1);

  // Arena state
  const [arenaId, setArenaId] = useState<string | null>(null);
  const [arenaStatus, setArenaStatus] = useState<string>("idle");
  const [replRuns, setReplRuns] = useState<Record<number, RunState>>({});
  const [seqRuns, setSeqRuns] = useState<Record<number, RunState>>({});
  const [error, setError] = useState<string | null>(null);

  // Refs for auto-scroll
  const replLogRef = useRef<HTMLDivElement>(null);
  const seqLogRef = useRef<HTMLDivElement>(null);

  // Validate keys
  useEffect(() => {
    const anthValid =
      anthropicKey.startsWith("sk-ant-") && anthropicKey.length > 20;
    const oaiValid =
      openaiKey.startsWith("sk-proj-") && openaiKey.length > 20;
    setKeysValid(anthValid && oaiValid);
  }, [anthropicKey, openaiKey]);

  // Load guides when keys are valid
  useEffect(() => {
    if (!keysValid) return;
    listGuides(anthropicKey)
      .then(setGuides)
      .catch((e) => setError(`Failed to load guides: ${e.message}`));
  }, [keysValid, anthropicKey]);

  // Start arena
  const handleStart = useCallback(async () => {
    if (!selectedGuide || !keysValid) return;
    setError(null);
    setArenaStatus("starting");
    setReplRuns({});
    setSeqRuns({});

    try {
      const res = await startArena(
        anthropicKey,
        openaiKey,
        selectedGuide,
        nRuns
      );
      setArenaId(res.arenaId);
      setArenaStatus("running");

      // Connect to SSE
      const eventSource = new EventSource(createArenaSSEUrl(res.arenaId));
      eventSource.onmessage = (e) => {
        try {
          const event: ArenaEvent = JSON.parse(e.data);
          handleArenaEvent(event);
        } catch {
          // ignore parse errors on heartbeats
        }
      };
      eventSource.addEventListener("done", () => {
        eventSource.close();
        setArenaStatus("completed");
      });
      eventSource.onerror = () => {
        eventSource.close();
        setArenaStatus((prev) =>
          prev === "running" || prev === "starting" ? "completed" : prev
        );
      };
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start arena");
      setArenaStatus("idle");
    }
  }, [selectedGuide, keysValid, anthropicKey, openaiKey, nRuns]);

  const handleArenaEvent = useCallback((event: ArenaEvent) => {
    const { type, data } = event;

    if (type === "run_start") {
      const pipeline = data.pipeline as string;
      const runNum = data.run_number as number;
      const newState: RunState = {
        status: "running",
        steps: [],
        artifacts: [],
        cost_usd: 0,
        wall_clock_ms: 0,
      };
      if (pipeline === "repl") {
        setReplRuns((prev) => ({ ...prev, [runNum]: newState }));
      } else {
        setSeqRuns((prev) => ({ ...prev, [runNum]: newState }));
      }
    } else if (type === "seq_step" || type === "repl_step") {
      const runNum = data.run_number as number;
      const setter = type === "seq_step" ? setSeqRuns : setReplRuns;
      setter((prev) => {
        const run = prev[runNum] || {
          status: "running",
          steps: [],
          artifacts: [],
          cost_usd: 0,
          wall_clock_ms: 0,
        };
        return {
          ...prev,
          [runNum]: { ...run, steps: [...run.steps, event] },
        };
      });
    } else if (type === "seq_artifact" || type === "repl_artifact") {
      const runNum = data.run_number as number;
      const artifactName = data.artifact_name as string;
      const setter = type === "seq_artifact" ? setSeqRuns : setReplRuns;
      setter((prev) => {
        const run = prev[runNum] || {
          status: "running",
          steps: [],
          artifacts: [],
          cost_usd: 0,
          wall_clock_ms: 0,
        };
        return {
          ...prev,
          [runNum]: {
            ...run,
            artifacts: [...run.artifacts, artifactName],
            steps: [...run.steps, event],
          },
        };
      });
    } else if (type === "run_complete" || type === "run_error") {
      const pipeline = data.pipeline_type as string;
      const runNum = data.run_number as number;
      const setter =
        pipeline === "REPL" ? setReplRuns : setSeqRuns;
      setter((prev) => {
        const run = prev[runNum] || {
          status: "running",
          steps: [],
          artifacts: [],
          cost_usd: 0,
          wall_clock_ms: 0,
        };
        return {
          ...prev,
          [runNum]: {
            ...run,
            status:
              type === "run_complete" ? "completed" : "failed",
            cost_usd: (data.cost_usd as number) || 0,
            wall_clock_ms: (data.wall_clock_ms as number) || 0,
            error: data.error as string | undefined,
            artifacts: (data.artifacts as string[]) || run.artifacts,
          },
        };
      });
    } else if (type === "arena_done") {
      setArenaStatus("completed");
    }
  }, []);

  // Auto-scroll log panels
  useEffect(() => {
    replLogRef.current?.scrollTo(0, replLogRef.current.scrollHeight);
  }, [replRuns]);
  useEffect(() => {
    seqLogRef.current?.scrollTo(0, seqLogRef.current.scrollHeight);
  }, [seqRuns]);

  const selectedGuideInfo = guides.find((g) => g.id === selectedGuide);

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">
              CHW Navigator Arena
            </h1>
            <p className="text-sm text-zinc-400">
              A/B test: REPL vs Sequential pipeline
            </p>
          </div>
          {arenaId && (
            <div className="text-xs text-zinc-500 font-mono">
              {arenaId}
            </div>
          )}
        </div>
      </header>

      {/* Config Panel */}
      {arenaStatus === "idle" && (
        <div className="max-w-2xl mx-auto p-6 space-y-6">
          {/* API Keys */}
          <div className="space-y-4">
            <h2 className="text-lg font-medium">API Keys</h2>
            <div>
              <label className="block text-sm text-zinc-400 mb-1">
                Anthropic API Key
              </label>
              <input
                type="password"
                value={anthropicKey}
                onChange={(e) => setAnthropicKey(e.target.value)}
                placeholder="sk-ant-..."
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none"
              />
              {anthropicKey && !anthropicKey.startsWith("sk-ant-") && (
                <p className="text-xs text-red-400 mt-1">
                  Must start with sk-ant-
                </p>
              )}
            </div>
            <div>
              <label className="block text-sm text-zinc-400 mb-1">
                OpenAI API Key
              </label>
              <input
                type="password"
                value={openaiKey}
                onChange={(e) => setOpenaiKey(e.target.value)}
                placeholder="sk-proj-..."
                className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm font-mono focus:border-blue-500 focus:outline-none"
              />
              {openaiKey && !openaiKey.startsWith("sk-proj-") && (
                <p className="text-xs text-red-400 mt-1">
                  Must start with sk-proj-
                </p>
              )}
            </div>
          </div>

          {/* Guide Selection */}
          {keysValid && (
            <div className="space-y-4">
              <h2 className="text-lg font-medium">Select Guide</h2>
              {guides.length === 0 ? (
                <p className="text-sm text-zinc-500">
                  No ingested guides found. Upload a PDF first on the{" "}
                  <a href="/" className="text-blue-400 underline">
                    home page
                  </a>
                  .
                </p>
              ) : (
                <select
                  value={selectedGuide}
                  onChange={(e) => setSelectedGuide(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-700 rounded px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
                >
                  <option value="">Choose a guide...</option>
                  {guides.map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.manualName || g.filename} ({g.pageCount} pages)
                    </option>
                  ))}
                </select>
              )}

              {selectedGuideInfo && (
                <div className="text-xs text-zinc-500 bg-zinc-900 rounded p-3">
                  <div>
                    Pages: {selectedGuideInfo.pageCount} | Size:{" "}
                    {Math.round(selectedGuideInfo.pdfBytes / 1024)} KB
                  </div>
                  <div>
                    Ingested:{" "}
                    {selectedGuideInfo.ingestedAt
                      ? new Date(
                          selectedGuideInfo.ingestedAt
                        ).toLocaleString()
                      : "unknown"}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Run Config */}
          {keysValid && selectedGuide && (
            <div className="space-y-4">
              <h2 className="text-lg font-medium">Run Configuration</h2>
              <div className="flex gap-4">
                <button
                  onClick={() => setNRuns(1)}
                  className={`px-4 py-2 rounded text-sm ${
                    nRuns === 1
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-800 text-zinc-400"
                  }`}
                >
                  N=1 (single run)
                </button>
                <button
                  onClick={() => setNRuns(3)}
                  className={`px-4 py-2 rounded text-sm ${
                    nRuns === 3
                      ? "bg-blue-600 text-white"
                      : "bg-zinc-800 text-zinc-400"
                  }`}
                >
                  N=3 (test-retest)
                </button>
              </div>
              <p className="text-xs text-zinc-500">
                N=1: One REPL + one Sequential run ($10-20). N=3: Three of
                each for reliability measurement ($30-60).
              </p>

              <button
                onClick={handleStart}
                className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-3 rounded text-sm transition-colors"
              >
                Start Arena (N={nRuns})
              </button>
            </div>
          )}

          {error && (
            <div className="bg-red-900/30 border border-red-700 rounded p-3 text-sm text-red-300">
              {error}
            </div>
          )}
        </div>
      )}

      {/* Split Screen: Arena Running/Completed */}
      {(arenaStatus === "running" ||
        arenaStatus === "starting" ||
        arenaStatus === "completed") && (
        <div className="flex h-[calc(100vh-73px)]">
          {/* Left: REPL Pipeline */}
          <div className="w-1/2 border-r border-zinc-800 flex flex-col">
            <div className="px-4 py-3 border-b border-zinc-800 bg-zinc-900 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
                <span className="text-sm font-medium">
                  Pipeline B: REPL
                </span>
              </div>
              <span className="text-xs text-zinc-500">
                Single session, full context
              </span>
            </div>
            <div
              ref={replLogRef}
              className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-xs"
            >
              {Object.entries(replRuns).map(([num, run]) => (
                <RunPanel
                  key={`repl-${num}`}
                  runNumber={parseInt(num)}
                  pipeline="repl"
                  run={run}
                  arenaId={arenaId}
                />
              ))}
              {Object.keys(replRuns).length === 0 && (
                <div className="text-zinc-600 text-center py-8">
                  Waiting for REPL runs to start...
                </div>
              )}
            </div>
          </div>

          {/* Right: Sequential Pipeline */}
          <div className="w-1/2 flex flex-col">
            <div className="px-4 py-3 border-b border-zinc-800 bg-zinc-900 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-amber-500" />
                <span className="text-sm font-medium">
                  Pipeline A: Sequential
                </span>
              </div>
              <span className="text-xs text-zinc-500">
                Separate calls, maker+redteam+repair
              </span>
            </div>
            <div
              ref={seqLogRef}
              className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-xs"
            >
              {Object.entries(seqRuns).map(([num, run]) => (
                <RunPanel
                  key={`seq-${num}`}
                  runNumber={parseInt(num)}
                  pipeline="seq"
                  run={run}
                  arenaId={arenaId}
                />
              ))}
              {Object.keys(seqRuns).length === 0 && (
                <div className="text-zinc-600 text-center py-8">
                  Waiting for Sequential runs to start...
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Status Bar */}
      {arenaStatus !== "idle" && (
        <div className="fixed bottom-0 left-0 right-0 bg-zinc-900 border-t border-zinc-800 px-6 py-2 flex items-center justify-between text-xs">
          <div className="flex items-center gap-4">
            <span
              className={`px-2 py-0.5 rounded ${
                arenaStatus === "running"
                  ? "bg-green-900 text-green-300"
                  : arenaStatus === "completed"
                  ? "bg-blue-900 text-blue-300"
                  : "bg-zinc-800 text-zinc-400"
              }`}
            >
              {arenaStatus.toUpperCase()}
            </span>
            <span className="text-zinc-500">
              REPL:{" "}
              {
                Object.values(replRuns).filter(
                  (r) => r.status === "completed"
                ).length
              }
              /{nRuns} | Sequential:{" "}
              {
                Object.values(seqRuns).filter(
                  (r) => r.status === "completed"
                ).length
              }
              /{nRuns}
            </span>
          </div>
          <div className="flex items-center gap-4 text-zinc-500">
            <span>
              REPL cost: $
              {Object.values(replRuns)
                .reduce((sum, r) => sum + r.cost_usd, 0)
                .toFixed(2)}
            </span>
            <span>
              Seq cost: $
              {Object.values(seqRuns)
                .reduce((sum, r) => sum + r.cost_usd, 0)
                .toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function RunPanel({
  runNumber,
  pipeline,
  run,
  arenaId,
}: {
  runNumber: number;
  pipeline: string;
  run: RunState;
  arenaId: string | null;
}) {
  const statusColors: Record<string, string> = {
    pending: "text-zinc-500",
    running: "text-green-400",
    completed: "text-blue-400",
    failed: "text-red-400",
  };

  return (
    <div className="border border-zinc-800 rounded p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-medium text-sm">
          Run #{runNumber}
        </span>
        <span className={statusColors[run.status] || "text-zinc-500"}>
          {run.status}
          {run.wall_clock_ms > 0 &&
            ` (${(run.wall_clock_ms / 1000).toFixed(1)}s)`}
        </span>
      </div>

      {/* Steps log */}
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {run.steps.map((step, i) => (
          <StepLine key={i} event={step} />
        ))}
      </div>

      {/* Artifacts */}
      {run.artifacts.length > 0 && (
        <div className="border-t border-zinc-800 pt-2">
          <div className="text-zinc-500 mb-1">
            Artifacts ({run.artifacts.length})
          </div>
          <div className="flex flex-wrap gap-1">
            {run.artifacts.map((name) => {
              const filename = `${name}.json`;
              return (
                <button
                  key={name}
                  type="button"
                  disabled={!arenaId}
                  onClick={() => {
                    if (!arenaId) return;
                    const url = getArenaArtifactUrl(
                      arenaId,
                      pipeline,
                      runNumber,
                      name,
                    );
                    downloadBlobAs(url, filename).catch((e) => {
                      console.error(`Arena download failed for ${filename}:`, e);
                      alert(
                        `Could not download ${filename}.\n\n${e?.message ?? e}`,
                      );
                    });
                  }}
                  className="px-2 py-0.5 bg-zinc-800 hover:bg-zinc-700 rounded text-zinc-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                >
                  {filename}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {run.error && (
        <div className="text-red-400 text-xs">Error: {run.error}</div>
      )}
    </div>
  );
}

function StepLine({ event }: { event: ArenaEvent }) {
  const { type, data } = event;

  if (type === "seq_step") {
    const stageType = data.stage_type as string;
    const stageId = data.stage_id as string;
    const passed = data.passed;
    const durationMs = data.duration_ms as number;
    const cost = data.cost_usd as number;

    const icon =
      stageType === "maker"
        ? "\u25B6"
        : stageType === "redteam"
        ? "\u26A0"
        : "\u2692";
    const passStr =
      passed === true
        ? " PASS"
        : passed === false
        ? " FAIL"
        : "";

    return (
      <div className="text-zinc-400">
        <span className="text-zinc-600">{icon}</span>{" "}
        {stageId}.{stageType}
        {passStr && (
          <span
            className={
              passed ? "text-green-500" : "text-red-500"
            }
          >
            {passStr}
          </span>
        )}
        {durationMs && (
          <span className="text-zinc-600">
            {" "}
            {(durationMs / 1000).toFixed(1)}s
          </span>
        )}
        {cost !== undefined && cost > 0 && (
          <span className="text-zinc-600">
            {" "}
            ${cost.toFixed(3)}
          </span>
        )}
      </div>
    );
  }

  if (type === "seq_artifact" || type === "repl_artifact") {
    const name = data.artifact_name as string;
    return (
      <div className="text-emerald-500">
        \u2713 emit_artifact({name})
      </div>
    );
  }

  if (type === "run_complete") {
    return (
      <div className="text-blue-400 font-medium">
        Completed | $
        {((data.cost_usd as number) || 0).toFixed(2)} |{" "}
        {(((data.wall_clock_ms as number) || 0) / 1000).toFixed(1)}s
      </div>
    );
  }

  if (type === "repl_step") {
    const stepType = data.step_type as string;
    const icon =
      stepType === "exec"
        ? "\u25B6"
        : stepType === "subcall"
        ? "\u2192"
        : stepType === "validate"
        ? "\u2713"
        : "\u2022";
    const code = data.code as string;
    return (
      <div className="text-zinc-400">
        <span className="text-zinc-600">{icon}</span>{" "}
        {stepType}
        {code && (
          <span className="text-zinc-500">
            {" "}
            {code.slice(0, 80)}
            {code.length > 80 ? "..." : ""}
          </span>
        )}
      </div>
    );
  }

  // Generic step
  return (
    <div className="text-zinc-500">
      {type}: {JSON.stringify(data).slice(0, 120)}
    </div>
  );
}
