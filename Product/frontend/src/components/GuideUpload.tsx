"use client";

import { useCallback, useState } from "react";

interface GuideUploadProps {
  onUploadPdf: (file: File, checkDupes: boolean) => void;
  onUploadJson: (json: object, filename: string) => void;
  disabled?: boolean;
}

const SUPPORTED_EXTENSIONS = [".pdf", ".json"] as const;
const MAX_PDF_BYTES = 50 * 1024 * 1024; // 50 MB

export function GuideUpload({
  onUploadPdf,
  onUploadJson,
  disabled,
}: GuideUploadProps) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [checkDupes, setCheckDupes] = useState(true);

  const showError = useCallback((msg: string) => {
    setError(msg);
    setTimeout(() => setError(null), 5000);
  }, []);

  const handleFiles = useCallback(
    (files: File[]) => {
      setError(null);
      if (files.length > 1) {
        showError("Only 1 document can be uploaded at a time.");
        return;
      }
      if (files.length === 0) return;
      const file = files[0];
      const ext = "." + (file.name.split(".").pop()?.toLowerCase() ?? "");
      if (!SUPPORTED_EXTENSIONS.includes(ext as (typeof SUPPORTED_EXTENSIONS)[number])) {
        showError(
          `Unsupported file type. Use ${SUPPORTED_EXTENSIONS.join(" or ")}`
        );
        return;
      }
      if (ext === ".pdf") {
        if (file.size > MAX_PDF_BYTES) {
          showError("PDF too large (max 50 MB)");
          return;
        }
        setFilename(file.name);
        onUploadPdf(file, checkDupes);
      } else {
        // .json path
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const json = JSON.parse(e.target?.result as string);
            setFilename(file.name);
            onUploadJson(json, file.name);
          } catch {
            showError("Invalid JSON file");
          }
        };
        reader.readAsText(file);
      }
    },
    [checkDupes, onUploadPdf, onUploadJson, showError]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const dropped = Array.from(e.dataTransfer.files ?? []);
      handleFiles(dropped);
    },
    [handleFiles]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = Array.from(e.target.files ?? []);
      handleFiles(selected);
      // Reset so selecting the same file again still fires onChange
      e.target.value = "";
    },
    [handleFiles]
  );

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium">
        Clinical Manual (PDF or JSON)
      </label>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={`relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
          dragActive
            ? "border-[var(--accent)] bg-[var(--accent)]/5"
            : "border-[var(--border)]"
        } ${
          disabled
            ? "opacity-50 pointer-events-none"
            : "cursor-pointer hover:border-[var(--accent)]"
        }`}
      >
        <input
          type="file"
          multiple
          accept=".pdf,.json"
          onChange={handleInputChange}
          disabled={disabled}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
        {filename ? (
          <div className="text-center">
            <p className="text-sm font-medium text-[var(--success)]">
              Uploaded: {filename}
            </p>
            <p className="mt-1 text-xs text-[var(--muted-foreground)]">
              Drop another file to replace
            </p>
          </div>
        ) : (
          <div className="text-center">
            <p className="text-sm">
              Drop your PDF or JSON guide here, or click to browse
            </p>
            <p className="mt-1 text-xs text-[var(--muted-foreground)]">
              Accepts PDF (clinical manual) or JSON (pre-parsed guide)
            </p>
            <p className="mt-1 text-[10px] text-[var(--muted-foreground)]">
              Only 1 file at a time
            </p>
          </div>
        )}
      </div>

      <label className="flex items-center gap-2 text-xs text-[var(--muted-foreground)] cursor-pointer select-none">
        <input
          type="checkbox"
          checked={checkDupes}
          onChange={(e) => setCheckDupes(e.target.checked)}
          disabled={disabled}
          className="h-3.5 w-3.5 rounded border-[var(--border)] accent-[var(--accent)]"
        />
        Check for duplicates (cache hit on re-upload)
      </label>

      {error && (
        <div className="rounded-md border border-[var(--destructive)]/40 bg-[var(--destructive)]/10 px-3 py-2 text-sm text-[var(--destructive)]">
          {error}
        </div>
      )}
    </div>
  );
}
