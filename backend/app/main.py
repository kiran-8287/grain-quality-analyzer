"""FastAPI application entry point for the active rice-analysis method."""

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.services.event_logger import structured_logger

log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])

logs_dir = Path(os.environ.get("GRAIN_LOG_DIR", PROJECT_ROOT / "logs"))
logs_dir.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5)
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
if not getattr(structured_logger.logger, "_rice_analysis_handlers_configured", False):
    structured_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    structured_logger.logger.addHandler(file_handler)
    structured_logger.logger.setLevel(log_level)
    structured_logger.logger._rice_analysis_handlers_configured = True


class RequestLoggingMiddleware:
    """Emit one structured start and finish event per HTTP request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import uuid
        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        structured_logger.info("request_started", request_id, method=method, path=path)
        started = time.monotonic()
        status_code = 500

        async def wrapped_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, wrapped_send)
        except Exception as exc:
            structured_logger.error("request_failed", request_id, method=method, path=path,
                                    status_code=status_code, duration_ms=int((time.monotonic() - started) * 1000),
                                    error_type=type(exc).__name__, error_message=str(exc))
            raise
        else:
            structured_logger.info("request_finished", request_id, method=method, path=path,
                                   status_code=status_code, ok=status_code < 400,
                                   duration_ms=int((time.monotonic() - started) * 1000))


app = FastAPI(
    title="Rice Quality Analysis API",
    description="Image-based rice grain and foreign-matter analysis using the active classical CV method.",
    version="1.0.0",
)
app.add_middleware(RequestLoggingMiddleware)

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173,http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in frontend_origin.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(router)

frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"message": "Rice Quality Analysis API is running.", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), reload=True)
