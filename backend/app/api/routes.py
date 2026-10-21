"""
API Routes for GRAIN QUALITY ANALYZER.
"""

import base64
import asyncio
import logging
import math
import time
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.models import (
    AnalysisResultResponse,
    ExperimentalGrainAnalysisResponse,
    HealthResponse,
    JobStatusResponse,
    ModelsResponse,
    StandardsResponse,
)
from backend.app.services.export import export_result_csv, export_result_json
from backend.app.services.job_manager import job_manager
from backend.app.services.run_logger import structured_logger
from ml.config import load_model_registry, load_standards

logger = logging.getLogger(__name__)

router = APIRouter()


def _sanitize_numpy(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_numpy(v) for v in obj)
    return obj


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="GRAIN QUALITY ANALYZER Backend",
        gpu_available=False,
        device="cpu",
    )


@router.get("/models")
def get_models():
    """Return model registry status and metadata."""
    registry = load_model_registry()
    return {"models": registry.get("models", {})}


@router.get("/standards")
def get_standards():
    """Return current standards definition."""
    try:
        standards_data = load_standards()
        return standards_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Standards file not found.")


@router.post("/analyze")
async def analyze_rice_image(
    request: Request,
    file: UploadFile = File(...),
    grade: str = Form("grade_a"),
    reference_pixels: Optional[float] = Form(None),
    reference_mm: Optional[float] = Form(None),
    profile: Optional[str] = Form(None),
):
    """Analyze uploaded rice-grain image."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    manual_scale = None
    if reference_pixels is not None and reference_mm is not None and reference_pixels > 0:
        manual_scale = {
            "reference_pixels": reference_pixels,
            "reference_mm": reference_mm,
        }

    result = job_manager.process_sync(
        image_bytes=contents,
        filename=file.filename or "uploaded.jpg",
        manual_scale=manual_scale,
        grade=grade,
        profile=profile,
        request_id=getattr(request.state, "request_id", None),
    )

    if not result.get("success", False) and "error" in result:
        raise HTTPException(status_code=400, detail=result.get("error"))

    return _sanitize_numpy(result)


@router.post("/analyze-ml", response_model=ExperimentalGrainAnalysisResponse)
async def analyze_grain_ml(
    request: Request,
    file: UploadFile = File(...),
    px_per_mm: Optional[float] = Form(None),
):
    """Run the isolated experimental Mask R-CNN + YOLO analysis endpoint."""
    runtime = getattr(request.app.state, "grain_ml_runtime", None)
    if runtime is None:
        message = getattr(
            request.app.state,
            "grain_ml_error",
            "Experimental grain models are not configured or available.",
        )
        raise HTTPException(status_code=503, detail=message)

    supported_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in supported_types:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, or BMP image.")

    max_bytes = runtime.config.max_upload_bytes
    contents = await file.read(max_bytes + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {max_bytes}-byte upload limit.")
    if px_per_mm is not None and (not math.isfinite(px_per_mm) or px_per_mm <= 0):
        raise HTTPException(status_code=422, detail="px_per_mm must be a finite positive number.")

    request_id = getattr(request.state, "request_id", None)
    started = time.perf_counter()
    structured_logger.info(
        "grain_ml_analysis_started",
        job_id=request_id,
        content_type=content_type,
        bytes_received=len(contents),
    )
    try:
        result = await asyncio.wait_for(
            run_in_threadpool(runtime.analyze_image, contents, px_per_mm),
            timeout=runtime.config.timeout_seconds,
        )
    except asyncio.TimeoutError:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        structured_logger.error(
            "grain_ml_analysis_failed",
            job_id=request_id,
            error_type="TimeoutError",
            duration_ms=elapsed_ms,
        )
        raise HTTPException(status_code=504, detail="Experimental grain analysis timed out.")
    except ValueError as exc:
        structured_logger.error(
            "grain_ml_analysis_failed",
            job_id=request_id,
            error_type=type(exc).__name__,
            message=str(exc),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Experimental grain inference failed")
        structured_logger.error(
            "grain_ml_analysis_failed",
            job_id=request_id,
            error_type=type(exc).__name__,
            message=str(exc),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise HTTPException(status_code=500, detail="Experimental grain inference failed.") from exc

    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    structured_logger.info(
        "grain_ml_analysis_completed",
        job_id=request_id,
        duration_ms=duration_ms,
        device=result.get("device"),
        experimental=True,
    )
    return result


@router.get("/analysis/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Check the status of an analysis job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress_stage=job["progress_stage"],
        progress_percent=job["progress_percent"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
        error=job["error"],
    )


@router.get("/analysis/{job_id}/result")
def get_job_result(job_id: str):
    """Retrieve full analysis result for a completed job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet (current status: {job['status']}).",
        )
    return job["result"]


@router.get("/analysis/{job_id}/image")
def get_annotated_image(job_id: str):
    """Retrieve the annotated JPEG image binary."""
    job = job_manager.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job result not found.")

    b64_data = job["result"].get("annotated_image_base64")
    if not b64_data or not b64_data.startswith("data:image/jpeg;base64,"):
        raise HTTPException(status_code=404, detail="Annotated image not available.")

    raw_b64 = b64_data.split(",", 1)[1]
    image_bytes = base64.b64decode(raw_b64)
    return Response(content=image_bytes, media_type="image/jpeg")


@router.get("/analysis/{job_id}/export/json")
def export_job_json(job_id: str):
    """Download analysis result as a JSON file."""
    job = job_manager.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job result not found.")

    json_str = export_result_json(job["result"])
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=rice_analysis_{job_id[:8]}.json"},
    )


@router.get("/analysis/{job_id}/export/csv")
def export_job_csv(job_id: str):
    """Download analysis result as a CSV file."""
    job = job_manager.get_job(job_id)
    if not job or not job.get("result"):
        raise HTTPException(status_code=404, detail="Job result not found.")

    csv_str = export_result_csv(job["result"])
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=rice_analysis_{job_id[:8]}.csv"},
    )
