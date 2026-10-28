"""
API Routes for GRAIN QUALITY ANALYZER.
"""

import base64
import asyncio
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import cv2
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
from autolabel_proxy_fresh.rice_labeler import label_image

logger = logging.getLogger(__name__)

router = APIRouter()


def _run_fresh_proxy(contents: bytes, filename: str) -> dict[str, Any]:
    """Run the fresh classical proxy and return its mask overlay and counts."""
    suffix = Path(filename).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}:
        suffix = ".jpg"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        image, instances, _reference_area, whole_length_reference = label_image(
            temp_path, return_details=True
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    rice_count = sum(category == "rice" for _, category in instances)
    foreign_count = sum(category == "foreign" for _, category in instances)

    overlay = image.copy()
    color_layer = image.copy()
    # OpenCV uses BGR channel order: rice blue, foreign matter red.
    objects = []
    category_ids = {"rice": 0, "foreign": 0}
    grain_lengths = []
    grain_breadths = []
    grain_lb_ratios = []
    whole_grain_count = 0
    whole_threshold_length = 0.75 * float(whole_length_reference)
    for mask, category in instances:
        color = (255, 0, 0) if category == "rice" else (0, 0, 255)
        color_layer[mask] = color
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue

        category_ids[category] += 1
        object_id = f"R{category_ids[category]}" if category == "rice" else f"F{category_ids[category]}"
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop_mask = mask[y1:y2, x1:x2]
        rgba_mask = np.zeros((*crop_mask.shape, 4), dtype=np.uint8)
        rgba_mask[crop_mask] = (*color, 255)
        mask_ok, mask_encoded = cv2.imencode(".png", rgba_mask)
        if not mask_ok:
            raise ValueError(f"Could not encode the isolated mask for {object_id}.")

        object_data = {
            "id": object_id,
            "category": category,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "centroid": [float(xs.mean()), float(ys.mean())],
            "pixel_count": int(crop_mask.sum()),
            "mask_image": f"data:image/png;base64,{base64.b64encode(mask_encoded.tobytes()).decode('ascii')}",
        }
        if category == "rice":
            points = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
            rect_width, rect_height = cv2.minAreaRect(points)[1]
            length = float(max(rect_width, rect_height))
            breadth = float(min(rect_width, rect_height))
            lb_ratio = length / breadth if breadth > 0 else 0.0
            is_whole = length >= whole_threshold_length
            whole_grain_count += int(is_whole)
            grain_lengths.append(length)
            grain_breadths.append(breadth)
            grain_lb_ratios.append(lb_ratio)
            object_data.update({
                "length": round(length, 2),
                "breadth": round(breadth, 2),
                "lb_ratio": round(lb_ratio, 2),
                "whole_label": "whole" if is_whole else "broken",
            })
        objects.append(object_data)
    overlay = cv2.addWeighted(image, 0.58, color_layer, 0.42, 0)

    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Could not encode the masked output image.")
    original_ok, original_encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not original_ok:
        raise ValueError("Could not encode the original output image.")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    rice_denominator = len(grain_lengths)
    summary = {
        "whole_grain_count": whole_grain_count,
        "whole_percent": round(100 * whole_grain_count / rice_denominator, 2) if rice_denominator else 0.0,
        "whole_threshold_length": round(whole_threshold_length, 2),
        "reference_length": round(float(whole_length_reference), 2),
        "average_length": round(float(np.mean(grain_lengths)), 2) if grain_lengths else None,
        "median_length": round(float(np.median(grain_lengths)), 2) if grain_lengths else None,
        "average_breadth": round(float(np.mean(grain_breadths)), 2) if grain_breadths else None,
        "median_breadth": round(float(np.median(grain_breadths)), 2) if grain_breadths else None,
        "average_lb_ratio": round(float(np.mean(grain_lb_ratios)), 2) if grain_lb_ratios else None,
        "median_lb_ratio": round(float(np.median(grain_lb_ratios)), 2) if grain_lb_ratios else None,
    }
    return {
        "masked_image": f"data:image/jpeg;base64,{image_base64}",
        "original_image": f"data:image/jpeg;base64,{base64.b64encode(original_encoded.tobytes()).decode('ascii')}",
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "rice_count": int(rice_count),
        "foreign_matter_count": int(foreign_count),
        "objects": objects,
        "summary": summary,
    }


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


@router.post("/analyze-proxy")
async def analyze_with_fresh_proxy(file: UploadFile = File(...)):
    """Run the standalone fresh auto-label proxy for the simplified website flow."""
    supported_types = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in supported_types:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, TIFF, or BMP image.")

    max_bytes = 25 * 1024 * 1024
    contents = await file.read(max_bytes + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the 25 MB upload limit.")

    try:
        return await run_in_threadpool(_run_fresh_proxy, contents, file.filename or "upload.jpg")
    except (ValueError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail=f"Could not analyze this image: {exc}") from exc


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
