"""
FastAPI Main Application for GRAIN QUALITY ANALYZER.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add project root to sys.path so ml and backend can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.app.api.routes import router
from backend.app.services.run_logger import structured_logger

log_level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logs_dir = Path(os.environ.get("GRAIN_LOG_DIR", PROJECT_ROOT / "monitoring" / "logs"))
logs_dir.mkdir(parents=True, exist_ok=True)
file_handler = RotatingFileHandler(
    logs_dir / "app.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
)
file_handler.setLevel(log_level)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

structured_logger_handler_key = "_grain_structured_handlers_configured"
if not getattr(structured_logger.logger, structured_logger_handler_key, False):
    structured_logger.logger.addHandler(logging.StreamHandler(sys.stdout))
    structured_logger.logger.addHandler(file_handler)
    structured_logger.logger.setLevel(log_level)
    setattr(structured_logger.logger, structured_logger_handler_key, True)

logger = logging.getLogger("rice_quality_ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load experimental models lazily; keep the web app available without them."""
    app.state.grain_ml_runtime = None
    app.state.grain_ml_config = None
    app.state.grain_ml_error = "Experimental grain models are not configured or available."
    try:
        # This module defers Torch/torchvision/Ultralytics imports until runtime construction.
        from backend.models.grain_infer import GrainInferenceConfig, GrainInferenceRuntime

        config = GrainInferenceConfig.from_env(PROJECT_ROOT)
        app.state.grain_ml_config = config
        app.state.grain_ml_runtime = await run_in_threadpool(GrainInferenceRuntime, config)
        app.state.grain_ml_error = None
        structured_logger.info(
            "grain_ml_models_loaded",
            device=str(app.state.grain_ml_runtime.device),
            mask_checkpoint=config.mask_checkpoint.name,
            yolo_checkpoint=config.yolo_weights.name,
            experimental=True,
        )
    except ImportError as exc:
        app.state.grain_ml_error = f"Experimental ML dependencies are unavailable: {exc}"
        structured_logger.warning("grain_ml_startup_unavailable", reason=app.state.grain_ml_error)
    except FileNotFoundError as exc:
        app.state.grain_ml_error = str(exc)
        structured_logger.warning("grain_ml_startup_unavailable", reason=app.state.grain_ml_error)
    except Exception as exc:
        app.state.grain_ml_error = f"Experimental model loading failed: {type(exc).__name__}: {exc}"
        logger.exception("Experimental grain models could not be loaded")
        structured_logger.error("grain_ml_startup_unavailable", reason=app.state.grain_ml_error)
    yield


app = FastAPI(
    title="GRAIN QUALITY ANALYZER API",
    description="Automated Image-Based Raw Milled Rice Grain Quality Analysis System",
    version="1.0.0",
    lifespan=lifespan,
)


def _is_grain_ml_path(path: str) -> bool:
    return path.rstrip("/") in {"/analyze-ml", "/api/analyze-ml"}


@app.exception_handler(HTTPException)
async def experimental_http_exception_handler(request: Request, exc: HTTPException):
    if _is_grain_ml_path(request.url.path):
        return JSONResponse(
            status_code=exc.status_code,
            content={"experimental": True, "detail": exc.detail},
            headers=exc.headers,
        )
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def experimental_validation_exception_handler(request: Request, exc: RequestValidationError):
    if _is_grain_ml_path(request.url.path):
        issues = [
            {"location": error.get("loc", []), "message": error.get("msg", "Invalid request")}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder({"experimental": True, "detail": "Invalid request", "issues": issues}),
        )
    return await request_validation_exception_handler(request, exc)


class RequestLoggingMiddleware:
    """Middleware to emit structured request lifecycle events."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        route = scope.get("route", None)
        route_path = getattr(route, "path", path) if route else path

        import uuid as _uuid
        request_id = str(_uuid.uuid4())
        if "state" not in scope:
            scope["state"] = {}
        scope["state"]["request_id"] = request_id
        structured_logger.info(
            "request_started",
            request_id=request_id,
            method=method,
            path=path,
            route=route_path,
        )

        t_start = time.monotonic()
        error_type = None
        error_message = None
        status_code = 500
        original_send = send

        async def wrapped_send(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message.get("status", 500))
            await original_send(message)

        try:
            await self.app(scope, receive, wrapped_send)
            return
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
            raise
        finally:
            duration_ms = int((time.monotonic() - t_start) * 1000)
            if error_type is not None:
                structured_logger.error(
                    "request_failed",
                    request_id=request_id,
                    method=method,
                    path=path,
                    route=route_path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    error_type=error_type,
                    error_message=error_message,
                )
            else:
                structured_logger.info(
                    "request_finished",
                    request_id=request_id,
                    method=method,
                    path=path,
                    route=route_path,
                    status_code=status_code,
                    ok=True,
                    duration_ms=duration_ms,
                )


app.add_middleware(RequestLoggingMiddleware)

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173,http://localhost:3000")
ALLOWED_ORIGINS = [origin.strip() for origin in FRONTEND_ORIGIN.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")
# Mount directly at root for convenience (e.g. /health, /models, /standards)
app.include_router(router)

# Mount frontend dist static files if built
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
else:
    @app.get("/")
    def read_root():
        return {
            "message": "GRAIN QUALITY ANALYZER API is running.",
            "docs": "/docs",
            "health": "/health",
        }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=port, reload=True)
