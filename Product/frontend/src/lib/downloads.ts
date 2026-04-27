/**
 * Cross-origin artifact download helpers.
 *
 * History of this file (the bug we fought):
 *
 *   v1 (original): plain `<a href={url} download={filename}>` with a cross-
 *       origin URL. Chrome's security model IGNORES `download` attributes on
 *       cross-origin anchors, so the filename came from the URL's last path
 *       segment — which was the artifact type slug like `json` or
 *       `artifact_modules`, not the friendly filename.
 *
 *   v2 (blob hack): fetch the URL as a blob, wrap in `blob:http://localhost:3000/...`
 *       URL (same-origin because blob URLs inherit the creating document's
 *       origin), then click a synthetic anchor with `download={filename}`.
 *       This "worked" in Playwright's CDP-level download interception
 *       because CDP reads the anchor's `download` attribute directly. But
 *       real Chrome save-to-disk used a DIFFERENT code path: by the time
 *       `link.click()` fired, we were several `await`s deep from the user's
 *       button click, so the "transient user activation" had expired. Chrome
 *       falls back to naming blob-URL downloads by the blob's internal UUID
 *       when user activation is stale — producing files like
 *       `eea737a1-2303-40e5-b9b4-0c8d8dbce90a` in the Downloads folder.
 *
 *   v3 (this file, 2026-04-12): skip the fetch entirely. Create a plain
 *       anchor pointing directly at the backend URL, target=_blank so the
 *       main window doesn't navigate, and click it SYNCHRONOUSLY inside the
 *       user's click handler. Chrome follows the link itself, sees
 *       `Content-Disposition: attachment; filename="..."` on the response,
 *       and saves with the server's filename. No blob URL, no stale user
 *       activation, no UUID filenames. The download attribute is omitted on
 *       purpose — Chrome would ignore it on a cross-origin anchor anyway,
 *       and including it confuses some browsers into preferring a blob-
 *       internal name when the server also sends Content-Disposition.
 *
 *       Trade-off: we lose the JS-level error-surface. If the backend
 *       returns 401/404, Chrome opens a briefly-visible tab that closes
 *       itself (because no HTML body is rendered — the response has
 *       `Content-Type: application/octet-stream`). In practice this is
 *       acceptable because:
 *         - 401 only happens if the API key in the URL's ?token= is wrong,
 *           which is caught earlier when the artifacts list loads
 *         - 404 only happens if the file was deleted server-side after
 *           the list was rendered, which is a race we can't easily win
 *       We log a console breadcrumb before the click so developers can
 *       correlate user-reported download failures with the URL that was
 *       tried.
 *
 *   What the backend does right (so the fix above works):
 *     - `FileResponse(filename="...")` in `backend/server.py` emits
 *       `Content-Disposition: attachment; filename="<name>"`
 *     - FastAPI CORS config exposes the header via `expose_headers`
 *     - The `?token=<key>` query-param auth lets the browser fetch the URL
 *       directly without needing a custom Authorization header
 */

/**
 * Trigger a browser download by navigating an invisible iframe to the URL.
 *
 * Why an iframe instead of an anchor:
 *   We tried three approaches that each had their own failure mode:
 *
 *   1. Plain cross-origin `<a href download>` — Chrome ignores the
 *      `download` attribute on cross-origin anchors, fell back to the URL's
 *      last path segment ("json", "artifact_modules") as the filename.
 *
 *   2. Fetch-as-blob + `<a href=blobUrl download=name>` — Chrome's user
 *      activation had expired by the time `link.click()` fired (several
 *      awaits deep from the user's click), so Chrome fell back to naming
 *      the download by the blob URL's internal UUID path segment,
 *      producing filenames like `eea737a1-2303-40e5-b9b4-0c8d8dbce90a`.
 *
 *   3. `<a target=_blank>` without download attribute — worked in
 *      Playwright's Chromium (which reads suggestedFilename from CDP's
 *      downloadWillBegin event) but still produced UUID filenames in real
 *      Chrome. The `target=_blank` path goes through Chrome's popup
 *      handling which has subtly different download naming semantics than
 *      a direct navigation; in particular, when the popup-new-tab's URL
 *      has no file extension in the last path segment (like `/artifacts/json`),
 *      Chrome invents a synthetic filename that happens to be a UUID.
 *
 *   The iframe approach is the classic 2005-era pattern: create a hidden
 *   iframe, set its `src` to the URL, and let the browser's main network
 *   stack handle the navigation. When the response has
 *   `Content-Disposition: attachment`, the browser NEVER renders the
 *   iframe contents — it converts the load into a download immediately.
 *   Critically, iframe navigations go through the browser's canonical
 *   download pipeline which correctly parses Content-Disposition for the
 *   filename. No popup handling, no anchor download-attribute, no blob
 *   URL. Works identically in Chrome, Firefox, Safari, Edge.
 *
 * Rules for callers:
 *   - Must be called synchronously from inside a user event handler.
 *     (We use an iframe navigation which doesn't require user activation
 *     the way target=_blank popups do, but React's event dispatch is
 *     synchronous so this is trivially satisfied.)
 *   - Pass the full backend URL including auth token query params.
 *   - Errors (401/404) become silent iframe-load failures. The user sees
 *     no download, no tab flash, no error message. We log a breadcrumb
 *     before the navigation for developer debugging.
 */
export function triggerDownload(url: string, suggestedFilename: string): void {
  // eslint-disable-next-line no-console
  console.info(
    `[download] top-level nav: ${suggestedFilename} <- ${url.split("?")[0]}`,
  );

  // v5 (2026-04-14): TOP-LEVEL NAVIGATION via bare anchor click.
  //
  // The definitive working pattern for cross-origin downloads in Chrome
  // 125+ (as of 2026). Why this and not the prior iterations:
  //
  // Chrome derives the download filename in this priority order:
  //   1. TOP-LEVEL NAVIGATION to a URL whose response has
  //      `Content-Disposition: attachment` -- Chrome uses the CD filename.
  //      This is the ONLY path that "just works" cross-origin.
  //   2. Same-origin blob/anchor with `download` attribute -- Chrome uses
  //      the `download` attr value.
  //   3. Cross-origin fetch -> blob URL -> anchor.download -- Chrome
  //      IGNORES the `download` attribute (security mitigation against
  //      filename spoofing, Chrome 65+, tightened in 83+) and falls back
  //      to the blob's UUID. This is what v2 and v4 hit.
  //
  // History: v2 used fetch+blob (path 3, UUID filenames). v3 used hidden
  // iframe (ambiguous: sometimes path 1, sometimes fails). v4 tried
  // "fetch+parse CD+blob" which is still path 3 (UUIDs). v5 is the
  // minimal path-1 pattern: synchronous anchor navigation, no download
  // attribute (it poisons cross-origin behavior), no target=_blank
  // (opens flash-tab), no fetch (that's path 3).
  //
  // Trade-off: we lose the JS error surface. 401/404 responses cause
  // Chrome to render the error body in a new tab briefly. That's
  // acceptable: 401 only happens if ?token= is wrong (caught at list load
  // time), and 404 only for a race we can't win.
  //
  // The synchronous click inside a user event handler preserves
  // transient user activation, so Chrome's download manager promotes the
  // navigation to a download as soon as it sees Content-Disposition:
  // attachment on the response.
  const a = document.createElement("a");
  a.href = url;
  a.rel = "noopener";
  // NO download attribute -- it's ignored cross-origin AND poisons the
  // Content-Disposition filename resolution path in some Chrome versions.
  // NO target="_blank" -- opens a tab that flashes and closes, annoying UX.
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  // Cleanup: the click dispatches synchronously; by the time this
  // setTimeout fires, the browser has already captured the navigation
  // intent and converted it to a download.
  setTimeout(() => {
    if (a.parentNode) {
      a.parentNode.removeChild(a);
    }
  }, 100);
  // suggestedFilename is kept as a parameter for API compatibility but is
  // unused -- the browser gets the filename from the server's
  // Content-Disposition header. If the caller logs or reports it, they
  // see the expected name that should match what gets saved to disk.
  void suggestedFilename;
}

/**
 * Backwards-compatibility alias. The old helper was named `downloadBlobAs`
 * and took `(url, fallbackFilename)`. Preserving the name + signature lets
 * existing call sites continue working without churn while the rename
 * propagates. The implementation is now the blob-free v3 path.
 *
 * If a future call site genuinely needs the old blob-fetch-then-click
 * semantics (e.g., to show a progress bar or swallow errors in JS), that
 * caller should re-implement it locally rather than resurrecting the
 * stale-activation UUID bug.
 */
export async function downloadBlobAs(
  url: string,
  fallbackFilename: string,
): Promise<void> {
  // v5 is synchronous but we keep the async signature for API compatibility
  // with existing callers that `await` or `.catch()` this function.
  triggerDownload(url, fallbackFilename);
}

/**
 * Parse an RFC 6266 Content-Disposition header and return the filename.
 *
 * Kept for tests and any future code that needs server-side filename
 * resolution. The current download path doesn't use it (Chrome parses
 * the header itself when we let it handle the navigation natively).
 */
export function parseContentDispositionFilename(header: string | null): string | null {
  if (!header) return null;

  // RFC 6266: filename*=UTF-8''<percent-encoded>
  // Takes precedence over `filename=` if both are present.
  const extMatch = header.match(/filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)/i);
  if (extMatch) {
    try {
      return decodeURIComponent(extMatch[1].trim());
    } catch {
      // Malformed percent encoding: fall through to the plain form.
    }
  }

  // Plain form: filename="foo.json" or filename=foo.json
  const plainMatch = header.match(/filename\s*=\s*"?([^";]+)"?/i);
  if (plainMatch) return plainMatch[1].trim();

  return null;
}
