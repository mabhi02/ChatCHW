"""CHW Navigator Backend -- FastAPI entry point.

Run from repo root:  python -m backend.main
Run from backend/:   python main.py
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `backend.*` imports work
# whether we're invoked from repo root or from backend/
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv

# Load .env from repo root (where the user's .env lives)
load_dotenv(repo_root / ".env")

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
