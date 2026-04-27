"use client";

import { useRef, useEffect, useState } from "react";
import type { ReplEvent } from "@/hooks/useSSE";

interface JournalEntry {
  stepNumber: number;
  phase: string;
  summary: string;
  markdown: string;
  stats: {
    modules_found: number;
    modules: string[];
    predicates_found: number;
    subcalls: number;
    validation_runs: number;
    z3_runs: number;
    total_steps: number;
    current_phase: string;
  };
}

interface ExtractionJournalProps {
  events: ReplEvent[];
}

export function ExtractionJournal({ events }: ExtractionJournalProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [expanded, setExpanded] = useState(true);

  // Extract journal entries from SSE events
  const journalEntries: JournalEntry[] = events
    .filter((e) => (e as unknown as { type: string }).type === "journal")
    .map((e) => e as unknown as JournalEntry);

  const latestStats =
    journalEntries.length > 0
      ? journalEntries[journalEntries.length - 1].stats
      : null;

  useEffect(() => {
    if (expanded) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [journalEntries.length, expanded]);

  return (
    <div className="rounded-lg border border-slate-200 overflow-hidden bg-white">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-100 hover:bg-slate-200 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-900">Extraction Journal</span>
          {latestStats && (
            <span className="text-xs text-amber-600 font-mono">
              {latestStats.current_phase}
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {latestStats && (
            <div className="flex gap-3 text-xs text-slate-500">
              <span>{latestStats.modules_found} modules</span>
              <span>{latestStats.predicates_found} predicates</span>
              <span>{latestStats.total_steps} steps</span>
            </div>
          )}
          <span className="text-xs text-slate-600">{expanded ? "Collapse" : "Expand"}</span>
        </div>
      </button>

      {latestStats && latestStats.modules.length > 0 && (
        <div className="px-4 py-2 border-b border-slate-200 bg-slate-50">
          <div className="flex flex-wrap gap-1.5">
            {latestStats.modules.map((mod) => (
              <span
                key={mod}
                className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-mono text-blue-800"
              >
                {mod}
              </span>
            ))}
          </div>
        </div>
      )}

      {expanded && (
        <div className="max-h-[500px] overflow-y-auto p-4 space-y-4 bg-white">
          {journalEntries.length === 0 ? (
            <p className="text-center text-sm text-slate-500 py-8">
              Journal entries will appear as the extraction progresses...
            </p>
          ) : (
            journalEntries.map((entry, i) => (
              <JournalLine key={i} entry={entry} />
            ))
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}

function formatMarkdownSegment(text: string): React.ReactNode[] {
  // Split on bold markers and code backticks, return safe React elements
  const parts: React.ReactNode[] = [];
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  tokens.forEach((token, i) => {
    if (token.startsWith("**") && token.endsWith("**")) {
      parts.push(
        <strong key={i} className="text-[var(--foreground)]">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith("`") && token.endsWith("`")) {
      parts.push(
        <code
          key={i}
          className="bg-slate-100 border border-slate-200 px-1 rounded text-slate-800 text-[0.7rem] font-mono"
        >
          {token.slice(1, -1)}
        </code>
      );
    } else if (token) {
      parts.push(<span key={i}>{token}</span>);
    }
  });
  return parts;
}

function JournalLine({ entry }: { entry: JournalEntry }) {
  const rawLines = entry.markdown.split("\n");
  // Preserve fenced code blocks verbatim; strip empty lines outside them
  const lines: string[] = [];
  let inFence = false;
  let fenceBuf: string[] = [];
  for (const l of rawLines) {
    if (l.trim().startsWith("```")) {
      if (inFence) {
        lines.push("__CODEBLOCK__" + fenceBuf.join("\n"));
        fenceBuf = [];
        inFence = false;
      } else {
        inFence = true;
      }
      continue;
    }
    if (inFence) {
      fenceBuf.push(l);
    } else if (l.trim()) {
      lines.push(l);
    }
  }
  if (inFence && fenceBuf.length) {
    lines.push("__CODEBLOCK__" + fenceBuf.join("\n"));
  }

  return (
    <div className="space-y-1.5 border-l-2 border-slate-200 pl-3 py-1.5">
      {lines.map((line, i) => {
        if (line.startsWith("__CODEBLOCK__")) {
          return (
            <pre
              key={i}
              className="overflow-x-auto rounded border border-slate-200 bg-slate-50 p-2 text-[0.7rem] font-mono text-slate-800 my-1.5"
            >
              {line.replace("__CODEBLOCK__", "")}
            </pre>
          );
        }

        if (line.startsWith("## ")) {
          // Parse "## Label - starting" / "## Label - complete" style headings
          // Render as moderate semibold text (not h1/h2) with a compact status chip.
          const heading = line.replace("## ", "");
          const statusMatch = heading.match(/^(.+?)\s+-\s+(starting|complete|done|failed|running)\s*$/i);
          if (statusMatch) {
            const label = statusMatch[1];
            const status = statusMatch[2].toLowerCase();
            const statusColor =
              status === "complete" || status === "done"
                ? "text-green-700 bg-green-50 border-green-200"
                : status === "failed"
                  ? "text-red-700 bg-red-50 border-red-200"
                  : "text-amber-700 bg-amber-50 border-amber-200";
            return (
              <div key={i} className="flex items-center gap-2 mt-2 mb-1">
                <span className="text-[0.8rem] font-semibold text-slate-900">
                  {label}
                </span>
                <span
                  className={`text-[0.6rem] font-mono uppercase tracking-wide px-1.5 py-0.5 rounded border ${statusColor}`}
                >
                  {status}
                </span>
              </div>
            );
          }
          return (
            <p
              key={i}
              className="text-[0.8rem] font-semibold text-slate-900 mt-2 mb-1"
            >
              {heading}
            </p>
          );
        }

        if (line.startsWith("**[Step")) {
          const match = line.match(/\*\*\[Step (\d+)\]\*\* (.+)/);
          if (match) {
            return (
              <p key={i} className="text-xs flex items-start gap-2 leading-relaxed">
                <span className="font-mono text-[0.6rem] text-slate-400 mt-[0.2rem] shrink-0 tabular-nums">
                  {match[1].padStart(2, "0")}
                </span>
                <span className="text-slate-800">
                  {formatMarkdownSegment(match[2])}
                </span>
              </p>
            );
          }
        }

        if (line.trim().startsWith("- ")) {
          const content = line.trim().slice(2);
          const isError =
            content.startsWith("[X]") || content.startsWith("[!]");
          return (
            <p
              key={i}
              className={`text-xs pl-6 leading-relaxed ${isError ? "text-red-700" : "text-slate-600"}`}
            >
              {formatMarkdownSegment(content)}
            </p>
          );
        }

        return (
          <p key={i} className="text-xs text-slate-600 leading-relaxed">
            {formatMarkdownSegment(line)}
          </p>
        );
      })}
    </div>
  );
}
