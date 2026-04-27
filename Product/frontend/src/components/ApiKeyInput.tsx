"use client";

import { useState } from "react";

interface ApiKeyInputProps {
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  disabled?: boolean;
}

export function ApiKeyInput({
  apiKey,
  onApiKeyChange,
  disabled,
}: ApiKeyInputProps) {
  const [visible, setVisible] = useState(false);

  return (
    <div className="space-y-2">
      <label
        htmlFor="api-key"
        className="block text-sm font-medium"
      >
        Anthropic API Key
      </label>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            id="api-key"
            type={visible ? "text" : "password"}
            value={apiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            placeholder="sk-ant-..."
            disabled={disabled}
            className="w-full rounded-lg border border-[var(--border)] bg-[var(--muted)] px-4 py-2.5 text-sm font-mono placeholder:text-[var(--muted-foreground)] focus:outline-none focus:ring-2 focus:ring-[var(--accent)] disabled:opacity-50"
          />
          <button
            type="button"
            onClick={() => setVisible(!visible)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
          >
            {visible ? "Hide" : "Show"}
          </button>
        </div>
        {apiKey && (
          <button
            onClick={() => onApiKeyChange("")}
            className="rounded-lg border border-[var(--border)] px-3 py-2 text-sm hover:bg-[var(--muted)]"
          >
            Clear
          </button>
        )}
      </div>
      <p className="text-xs text-[var(--muted-foreground)]">
        Your key is held in memory only and never sent to our servers or stored.
        It is passed directly to the Anthropic API for each LLM call.
      </p>
    </div>
  );
}
