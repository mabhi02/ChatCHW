"""Gen 6: Codebook-labeled micro-chunks + REPL navigation.

Phase 0: Re-chunk guide to ~1-1.5K tokens at clean JSON boundaries
Phase 1: 3x GPT-4o-mini (temp=0, strict JSON) labels each chunk inline
Phase 2: Disagreement resolution via Sonnet, extreme chunks auto-Sonnet
Phase 3: RLM REPL loads labeled chunks, queries structured data, builds artifacts
"""
