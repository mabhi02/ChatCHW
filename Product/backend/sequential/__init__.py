"""Sequential pipeline for CHW clinical logic extraction.

Architecture: Many separate LLM calls chained in order. Each stage has a
MAKER prompt, a RED TEAM prompt, and a REPAIR prompt. Each call is a fresh
context window. Raw Anthropic + OpenAI API calls (no rlm library).

This is Pipeline A in the arena A/B test. Pipeline B is the REPL pipeline
in backend/rlm_runner.py.
"""
