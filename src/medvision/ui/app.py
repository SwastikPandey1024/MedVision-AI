"""Streamlit Web Application dashboard entrypoint (Phase 10)."""

import sys
from pathlib import Path

# Ensure root and app directories are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
APP_DIR = str(ROOT_DIR / "app")
SRC_DIR = str(ROOT_DIR / "src")
for p in [APP_DIR, SRC_DIR, str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app.streamlit_app import main

if __name__ == "__main__":
    main()
