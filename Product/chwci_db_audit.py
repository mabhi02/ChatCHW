"""Pull reliability-relevant numbers from the CHW-Chat-Interface Neon database.

Uses the corrected column names from schema.prisma. Read-only SELECTs only.
"""
import psycopg
import sys
from pathlib import Path

env_path = Path(r"C:\Users\athar\Documents\GitHub\CHW-Chat-Interface\.env")
db_url = None
for line in env_path.read_text().splitlines():
    if line.startswith("NEON_DB_URL="):
        db_url = line.split("=", 1)[1].strip().strip('"')
        break

QUERIES = [
    ("01. Table row counts", """
        SELECT
            (SELECT COUNT(*) FROM guides) AS guides,
            (SELECT COUNT(*) FROM chats) AS chats,
            (SELECT COUNT(*) FROM messages) AS messages,
            (SELECT COUNT(*) FROM dmn_versions) AS dmn_versions,
            (SELECT COUNT(*) FROM catcher_runs) AS catcher_runs,
            (SELECT COUNT(*) FROM cache_state) AS cache_state,
            (SELECT COUNT(*) FROM llm_call_log) AS llm_call_log
    """),
    ("02. Chat count per guide (non-stationary baseline evidence)", """
        SELECT g.id AS guide_id, g.filename, COUNT(DISTINCT c.id) AS chat_count
        FROM guides g
        LEFT JOIN chats c ON c.guide_id = g.id
        GROUP BY g.id, g.filename
        ORDER BY chat_count DESC
    """),
    ("03. Messages with catcher_results (= user decision points)", """
        SELECT
            COUNT(*) AS total_messages,
            COUNT(*) FILTER (WHERE catcher_results IS NOT NULL) AS with_catcher_results,
            COUNT(*) FILTER (WHERE catcher_results IS NOT NULL AND fix_applied = true) AS catcher_fix_applied_true,
            COUNT(*) FILTER (WHERE catcher_results IS NOT NULL AND fix_applied = false) AS catcher_fix_applied_false
        FROM messages
    """),
    ("04. Catcher runs by catcher name and severity", """
        SELECT catcher_name, severity, COUNT(*) AS n
        FROM catcher_runs
        GROUP BY catcher_name, severity
        ORDER BY catcher_name, severity
    """),
    ("05. Distinct catcher names observed", """
        SELECT catcher_name, model_used, COUNT(*) AS runs, AVG(execution_ms)::int AS avg_ms
        FROM catcher_runs
        GROUP BY catcher_name, model_used
        ORDER BY catcher_name
    """),
    ("06. CacheState fields: how many NULL in the observability fields?", """
        SELECT
            COUNT(*) AS total_rows,
            COUNT(*) FILTER (WHERE last_turn_at IS NULL) AS null_last_turn_at,
            COUNT(*) FILTER (WHERE expires_at IS NULL) AS null_expires_at,
            COUNT(*) FILTER (WHERE cached_at IS NULL) AS null_cached_at,
            COUNT(*) FILTER (WHERE is_cached = true) AS currently_cached,
            COUNT(*) FILTER (WHERE gemini_cache_name IS NOT NULL) AS has_gemini_cache,
            COUNT(*) FILTER (WHERE claude_cache_hash IS NOT NULL) AS has_claude_hash
        FROM cache_state
    """),
    ("07. LLM call log: operation + model breakdown", """
        SELECT operation, provider, model, COUNT(*) AS calls,
               SUM(input_tokens) AS total_input,
               SUM(output_tokens) AS total_output,
               SUM(cached_tokens) AS total_cached,
               ROUND(SUM(cost_usd)::numeric, 4) AS total_cost_usd
        FROM llm_call_log
        GROUP BY operation, provider, model
        ORDER BY calls DESC
    """),
    ("08. LLM call log: cache_hit rate", """
        SELECT
            COUNT(*) AS total_calls,
            COUNT(*) FILTER (WHERE cached_tokens > 0) AS calls_with_cache_hit,
            ROUND(100.0 * COUNT(*) FILTER (WHERE cached_tokens > 0) / NULLIF(COUNT(*),0), 1) AS pct_cache_hit,
            SUM(cached_tokens) AS total_cached_tokens,
            SUM(input_tokens) AS total_input_tokens
        FROM llm_call_log
    """),
    ("09. DMN version diff hint: chats with >1 version (re-runs within a chat)", """
        SELECT chat_id, COUNT(*) AS versions, MIN(created_at) AS first, MAX(created_at) AS last,
               EXTRACT(EPOCH FROM (MAX(created_at) - MIN(created_at)))::int AS span_seconds
        FROM dmn_versions
        GROUP BY chat_id
        HAVING COUNT(*) > 1
        ORDER BY versions DESC
    """),
    ("10. For the biggest guide: how many chats, messages, dmn_versions?", """
        WITH big_guide AS (
            SELECT g.id
            FROM guides g
            LEFT JOIN chats c ON c.guide_id = g.id
            GROUP BY g.id
            ORDER BY COUNT(DISTINCT c.id) DESC
            LIMIT 1
        )
        SELECT
            (SELECT COUNT(DISTINCT c.id) FROM chats c WHERE c.guide_id = (SELECT id FROM big_guide)) AS chats_using_this_guide,
            (SELECT COUNT(*) FROM messages m JOIN chats c ON m.chat_id = c.id WHERE c.guide_id = (SELECT id FROM big_guide)) AS messages,
            (SELECT COUNT(*) FROM dmn_versions dv JOIN chats c ON dv.chat_id = c.id WHERE c.guide_id = (SELECT id FROM big_guide)) AS dmn_versions,
            (SELECT COUNT(*) FROM cache_state cs JOIN chats c ON cs.chat_id = c.id WHERE c.guide_id = (SELECT id FROM big_guide)) AS cache_state_rows,
            (SELECT COUNT(*) FROM catcher_runs cr JOIN chats c ON cr.chat_id = c.id WHERE c.guide_id = (SELECT id FROM big_guide)) AS catcher_runs
    """),
    ("11. Timestamps: earliest to latest activity", """
        SELECT
            (SELECT MIN(created_at) FROM chats) AS first_chat,
            (SELECT MAX(created_at) FROM chats) AS last_chat,
            (SELECT MIN(created_at) FROM dmn_versions) AS first_dmn,
            (SELECT MAX(created_at) FROM dmn_versions) AS last_dmn,
            (SELECT MIN(created_at) FROM catcher_runs) AS first_catcher,
            (SELECT MAX(created_at) FROM catcher_runs) AS last_catcher
    """),
    ("12. Sample catcher_results JSON structure from one message (inspect the mediator trace)", """
        SELECT m.id, m.fix_applied,
               jsonb_array_length(m.catcher_results) AS num_catchers_in_report
        FROM messages m
        WHERE m.catcher_results IS NOT NULL
          AND jsonb_typeof(m.catcher_results) = 'array'
        LIMIT 10
    """),
]

try:
    with psycopg.connect(db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            for label, sql in QUERIES:
                print(f"=== {label} ===")
                try:
                    cur.execute(sql)
                    cols = [d.name for d in cur.description] if cur.description else []
                    rows = cur.fetchall()
                    if not rows:
                        print("  (no rows)")
                    else:
                        if cols:
                            print(f"  {' | '.join(cols)}")
                        for r in rows:
                            print(f"  {r}")
                except Exception as e:
                    print(f"  QUERY FAILED: {e}")
                print()
except Exception as e:
    print(f"CONNECT FAILED: {e}")
    sys.exit(1)
