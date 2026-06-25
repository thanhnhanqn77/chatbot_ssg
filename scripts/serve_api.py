from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn

from rag_app.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "rag_app.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
