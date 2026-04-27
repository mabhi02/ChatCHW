"""Gen 7: Opus-only pipeline. Label + compile with a single model and provider.

Phase 0: Micro-chunk to ~1-1.5K tokens (reuses Gen 6 chunker)
Phase 1: 1x Opus per chunk (temp=0, strict JSON, prompt-cached system)
Phase 2: Opus REPL compiles labeled chunks into 7 artifacts in DAG order
"""
