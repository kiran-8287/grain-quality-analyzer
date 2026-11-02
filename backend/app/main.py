"""FastAPI application entry point for the active rice-analysis method."""

import logging
import json
import os
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from backend.app.services.event_logger import structured_logger

log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
logs_dir = Path(os.environ.get("GRAIN_LOG_DIR", PROJECT_ROOT / "logs"))
logs_dir.mkdir(parents=True, exist_ok=True)
log_file = logs_dir / "app.log"
if log_file.exists() and log_file.stat().st_size:
    with log_file.open("rb") as existing_log:
        first_line = existing_log.readline().lstrip()
    if first_line and not first_line.startswith(b"{"):
        legacy_log = logs_dir / f"app.legacy.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.log"
        os.replace(log_file, legacy_log)
class _JsonLineFormatter(logging.Formatter):
    """Keep application and Uvicorn records in the same JSONL log file."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        try:
            payload = json.loads(message)
            if isinstance(payload, dict) and payload.get("timestamp") and payload.get("event"):
                return message
        except (TypeError, json.JSONDecodeError):
            pass

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc)
        payload = {
            "timestamp": timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "event": "application_log",
            "logger": record.name,
            "message": message,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)


def _configure_file_logging() -> RotatingFileHandler:
    root_logger = logging.getLogger()
    file_handler = next(
        (handler for handler in root_logger.handlers if getattr(handler, "_rice_app_log_handler", False)),
        None,
    )
    if file_handler is None:
        # Replace preconfigured console handlers too (for example, Uvicorn's
        # default stderr handler when the app is started with the Uvicorn CLI).
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            if not getattr(handler, "_rice_app_log_handler", False):
                handler.close()
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler._rice_app_log_handler = True

    file_handler.setFormatter(_JsonLineFormatter())
    root_logger.addHandler(file_handler)
    root_logger.setLevel(log_level)

    # Uvicorn's server/error messages go to the same JSONL file. Access logs
    # are disabled because the request middleware already records every HTTP
    # request without duplicating client IP addresses.
    uvicorn_logger = logging.getLogger("uvicorn")
    for handler in uvicorn_logger.handlers[:]:
        uvicorn_logger.removeHandler(handler)
        if handler is not file_handler:
            handler.close()
    uvicorn_logger.addHandler(file_handler)
    uvicorn_logger.setLevel(log_level)
    uvicorn_logger.propagate = False

    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    for handler in uvicorn_error_logger.handlers[:]:
        uvicorn_error_logger.removeHandler(handler)
        if handler is not file_handler:
            handler.close()
    uvicorn_error_logger.setLevel(log_level)
    uvicorn_error_logger.propagate = True

    access_logger = logging.getLogger("uvicorn.access")
    for handler in access_logger.handlers[:]:
        access_logger.removeHandler(handler)
        handler.close()
    access_logger.disabled = True
    access_logger.propagate = False

    event_logger = structured_logger.logger
    for handler in event_logger.handlers[:]:
        event_logger.removeHandler(handler)
        if handler is not file_handler:
            handler.close()
    event_logger.addHandler(file_handler)
    # Request/lifecycle events remain enabled even when general logging is
    # configured above INFO; diagnostics must not silently stop recording.
    event_logger.setLevel(min(log_level, logging.INFO))
    event_logger.propagate = False
    return file_handler


_application_log_handler = _configure_file_logging()

# Configure file-only logging before importing routes so dependency/model
# startup warnings are also captured instead of being printed by Uvicorn.
from backend.app.api.routes import router

_in_flight_requests = 0
_process = psutil.Process(os.getpid())
MAX_ANALYSIS_REQUEST_BYTES = 26 * 1024 * 1024


def _memory_usage_mb() -> float:
    return round(_process.memory_info().rss / (1024 * 1024), 2)


def _process_cpu_percent() -> float:
    return round(_process.cpu_percent(interval=None), 2)


class RequestLoggingMiddleware:
    """Write start and completion JSON events for every HTTP request."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        global _in_flight_requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_content_length = headers.get(b"content-length")
        try:
            content_length = int(raw_content_length) if raw_content_length is not None else None
        except ValueError:
            content_length = None

        started_at = time.time()
        started = time.monotonic()
        status_code = 500
        failure_type = None
        _in_flight_requests += 1
        structured_logger.info(
            "request_started", request_id, method=method, path=path,
            content_length=content_length, in_flight=_in_flight_requests,
            rss_mb=_memory_usage_mb(), process_cpu_percent=_process_cpu_percent(),
        )

        async def wrapped_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
            await send(message)

        is_analysis_upload = method == "POST" and path.endswith("/rice-quality/analyze")

        received_body_bytes = content_length

        try:
            if is_analysis_upload and content_length is not None and content_length > MAX_ANALYSIS_REQUEST_BYTES:
                status_code = 413
                await JSONResponse(
                    {"detail": "Upload request exceeds the 26 MB request limit."}, status_code=413
                )(scope, receive, wrapped_send)
            elif is_analysis_upload and content_length is None:
                # Chunked bodies have no declared size. Buffer them with a
                # disk-backed spool so we can reject the request before
                # Starlette's multipart parser creates an UploadFile.
                with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as body_spool:
                    body_size = 0
                    body_complete = False
                    while not body_complete:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            status_code = 400
                            body_complete = True
                            break
                        if message["type"] != "http.request":
                            continue
                        body = message.get("body", b"")
                        body_size += len(body)
                        if body_size > MAX_ANALYSIS_REQUEST_BYTES:
                            status_code = 413
                            await JSONResponse(
                                {"detail": "Upload request exceeds the 26 MB request limit."}, status_code=413
                            )(scope, receive, wrapped_send)
                            break
                        body_spool.write(body)
                        body_complete = not message.get("more_body", False)
                    else:
                        body_complete = True

                    received_body_bytes = body_size
                    if status_code != 413 and body_complete:
                        body_spool.seek(0)

                        async def replay_body():
                            chunk = body_spool.read(64 * 1024)
                            more_body = body_spool.tell() < body_size
                            return {"type": "http.request", "body": chunk, "more_body": more_body}

                        await self.app(scope, replay_body, wrapped_send)
            else:
                await self.app(scope, receive, wrapped_send)
        except BaseException as exc:
            failure_type = type(exc).__name__
            raise
        finally:
            _in_flight_requests = max(0, _in_flight_requests - 1)
            finished_at = time.time()
            duration_ms = int((time.monotonic() - started) * 1000)
            fields = {
                "method": method,
                "path": path,
                "status": status_code,
                "ok": status_code < 400,
                "content_length": content_length,
                "received_body_bytes": received_body_bytes,
                "duration_ms": duration_ms,
                "duration_s": round(duration_ms / 1000, 3),
                "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started_at)) + f".{int(started_at * 1000) % 1000:03d}Z",
                "finishedAt": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(finished_at)) + f".{int(finished_at * 1000) % 1000:03d}Z",
                "in_flight": _in_flight_requests,
                "rss_mb": _memory_usage_mb(),
                "process_cpu_percent": _process_cpu_percent(),
            }
            if failure_type:
                fields["error_type"] = failure_type
                structured_logger.error("request_failed", request_id, **fields)
            else:
                structured_logger.info("request_finished", request_id, **fields)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    structured_logger.info(
        "server_started", pid=os.getpid(), host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)), log_file=str(log_file),
        rss_mb=_memory_usage_mb(), process_cpu_percent=_process_cpu_percent(),
    )
    try:
        yield
    finally:
        structured_logger.info(
            "server_stopped", pid=os.getpid(), in_flight=_in_flight_requests,
            rss_mb=_memory_usage_mb(), process_cpu_percent=_process_cpu_percent(),
        )
        for handler in structured_logger.logger.handlers:
            handler.flush()


app = FastAPI(
    title="Grain Quality Analyzer API",
    description="Image-based rice grain and foreign-matter analysis using the active classical CV method.",
    version="1.0.0",
    lifespan=lifespan,
)

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173,http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in frontend_origin.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

app.include_router(router, prefix="/api")
app.include_router(router)

frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {"message": "Grain Quality Analyzer API is running.", "docs": "/docs", "health": "/health"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        reload=True,
        access_log=False,
        log_config=None,
    )
