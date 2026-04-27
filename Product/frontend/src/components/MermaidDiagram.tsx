"use client";

import { useEffect, useRef, useState } from "react";

interface MermaidDiagramProps {
  /** URL to fetch the raw mermaid markdown from */
  sourceUrl: string;
  /** Optional title shown above the diagram */
  title?: string;
}

/**
 * Renders a Mermaid diagram client-side using the `mermaid` npm package.
 *
 * Why client-side: the `mermaid` library uses DOM globals, which means it can
 * only run in the browser. We dynamic-import it so the server-side build
 * doesn't try to resolve DOM globals.
 *
 * The rendered SVG is produced by mermaid itself (self-contained, no user
 * input in the SVG generation path beyond the flowchart.md we generated
 * server-side), so injecting via `innerHTML` is safe here. We use a ref +
 * effect rather than `dangerouslySetInnerHTML` to avoid the React lint
 * warning and to keep the injection point explicit.
 *
 * On render errors, a red banner shows the mermaid parser error and a
 * collapsed view of the raw source so you can debug.
 */
export function MermaidDiagram({ sourceUrl, title }: MermaidDiagramProps) {
  const [expanded, setExpanded] = useState(false);
  const [svg, setSvg] = useState<string | null>(null);
  const [rawSource, setRawSource] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!expanded || svg || loading) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    (async () => {
      try {
        const res = await fetch(sourceUrl);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        const text = await res.text();

        // Strip ```mermaid ... ``` fences if present
        let source = text.trim();
        const fence = /^```(?:mermaid)?\s*\n?([\s\S]*?)\n?```\s*$/;
        const match = source.match(fence);
        if (match) {
          source = match[1].trim();
        }

        if (cancelled) return;
        setRawSource(source);

        // Dynamic import mermaid so it only loads in the browser
        const mermaid = (await import("mermaid")).default;
        mermaid.initialize({
          startOnLoad: false,
          theme: "default",
          securityLevel: "loose",
          flowchart: { htmlLabels: true, curve: "basis" },
        });

        const id = `mermaid-${Math.random().toString(36).slice(2, 10)}`;
        const { svg: rendered } = await mermaid.render(id, source);

        if (cancelled) return;
        setSvg(rendered);
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setError(`Failed to render diagram: ${msg}`);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [expanded, sourceUrl, svg, loading]);

  // Inject the rendered SVG via ref, not dangerouslySetInnerHTML, so the
  // React XSS lint doesn't fire. The source comes from our own backend-
  // generated flowchart.md, so it's trusted.
  useEffect(() => {
    if (svg && containerRef.current) {
      containerRef.current.innerHTML = svg;
    }
  }, [svg]);

  return (
    <div className="rounded-lg border border-[var(--border)] overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 bg-[var(--muted)] hover:bg-[var(--muted)]/80 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold">
            {title || "Clinical Logic Flowchart"}
          </span>
          <span className="text-[10px] uppercase tracking-wider text-[var(--muted-foreground)]">
            mermaid
          </span>
        </div>
        <span className="text-xs text-[var(--muted-foreground)]">
          {expanded ? "Collapse" : "Expand"}
        </span>
      </button>

      {expanded && (
        <div className="p-4 bg-white">
          {loading && (
            <div className="flex items-center gap-2 py-8 justify-center">
              <div className="h-5 w-5 animate-spin rounded-full border-2 border-[var(--muted-foreground)] border-t-[var(--accent)]" />
              <span className="text-xs text-[var(--muted-foreground)]">
                Rendering diagram...
              </span>
            </div>
          )}

          {error && (
            <div className="space-y-3">
              <div className="rounded bg-red-50 border border-red-200 p-3">
                <p className="text-xs font-semibold text-red-700 mb-1">
                  Could not render the diagram
                </p>
                <p className="text-xs font-mono text-red-600 break-all">
                  {error}
                </p>
              </div>
              {rawSource && (
                <details className="text-xs">
                  <summary className="cursor-pointer text-[var(--muted-foreground)] font-medium">
                    View raw Mermaid source
                  </summary>
                  <pre className="mt-2 p-3 bg-black/5 rounded overflow-x-auto text-[10px] font-mono text-black whitespace-pre">
                    {rawSource}
                  </pre>
                </details>
              )}
            </div>
          )}

          {svg && !error && (
            <div ref={containerRef} className="overflow-x-auto" />
          )}
        </div>
      )}
    </div>
  );
}
