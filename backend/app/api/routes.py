"""HTTP endpoints for the Grain Quality Analyzer application."""

import asyncio
import base64
import io
import math
import time
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from backend.app.services.event_logger import structured_logger
from rice_analysis_method.rice_analysis import label_image

router = APIRouter()
_analysis_semaphore = asyncio.Semaphore(1)
_pending_analyses = 0
MAX_PENDING_ANALYSES = 2
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "TIFF": "image/tiff",
    "BMP": "image/bmp",
}


def _decode_upload(contents: bytes) -> tuple[np.ndarray, str]:
    """Validate image metadata before OpenCV allocates a decoded pixel buffer."""
    try:
        with Image.open(io.BytesIO(contents)) as image_header:
            image_format = image_header.format
            width, height = image_header.size
            if width <= 0 or height <= 0:
                raise HTTPException(status_code=400, detail="Image dimensions are invalid.")
            if min(width, height) < 8:
                raise HTTPException(status_code=400, detail="Image is too small for analysis; each side must be at least 8 pixels.")
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Image dimensions exceed the {MAX_IMAGE_PIXELS // 1_000_000} megapixel processing limit.",
                )
            if image_format not in SUPPORTED_FORMATS:
                raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, TIFF, or BMP image.")
            image_header.verify()
        with Image.open(io.BytesIO(contents)) as image_metadata:
            try:
                exif_orientation = int(image_metadata.getexif().get(274, 1) or 1)
            except (AttributeError, TypeError, ValueError):
                exif_orientation = 1
    except HTTPException:
        raise
    except Image.DecompressionBombError as exc:
        raise HTTPException(
            status_code=413,
            detail=f"Image dimensions exceed the {MAX_IMAGE_PIXELS // 1_000_000} megapixel processing limit.",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid supported image.") from exc

    decoded = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise HTTPException(status_code=400, detail="The uploaded image could not be decoded.")

    if decoded.dtype == np.uint16:
        decoded = cv2.convertScaleAbs(decoded, alpha=255.0 / 65535.0)
    elif decoded.dtype != np.uint8:
        raise HTTPException(status_code=415, detail="Use an 8-bit or 16-bit image for analysis.")

    alpha = None
    if decoded.ndim == 2:
        source_image = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 1:
        source_image = cv2.cvtColor(decoded[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 2:
        gray, alpha = decoded[:, :, 0], decoded[:, :, 1]
        source_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif decoded.ndim == 3 and decoded.shape[2] == 3:
        source_image = decoded
    elif decoded.ndim == 3 and decoded.shape[2] == 4:
        source_image, alpha = decoded[:, :, :3], decoded[:, :, 3]
    else:
        raise HTTPException(status_code=415, detail="This image color format is not supported.")

    if alpha is not None:
        alpha_float = alpha[:, :, None].astype(np.float32) / 255.0
        source_image = np.rint(source_image.astype(np.float32) * alpha_float + 255.0 * (1.0 - alpha_float)).astype(np.uint8)

    orientation_transforms = {
        2: lambda image: cv2.flip(image, 1),
        3: lambda image: cv2.rotate(image, cv2.ROTATE_180),
        4: lambda image: cv2.flip(image, 0),
        5: lambda image: cv2.transpose(image),
        6: lambda image: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        7: lambda image: cv2.flip(cv2.transpose(image), -1),
        8: lambda image: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    if exif_orientation in orientation_transforms:
        source_image = orientation_transforms[exif_orientation](source_image)
    return source_image, SUPPORTED_FORMATS[image_format]


def _run_fresh_proxy(contents: bytes) -> dict[str, Any]:
    """Analyze at the method's working scale, then render masks at source resolution."""
    source_image, source_mime_type = _decode_upload(contents)
    source_height, source_width = source_image.shape[:2]

    # Keep only each object's bounding-box mask in working-resolution memory.
    instances = []

    def collect_instance(mask: np.ndarray, category: str) -> None:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        instances.append((mask[y1:y2, x1:x2].copy(), category, x1, y1))

    working_image, _unused_instances, _reference_area, proxy_details = label_image(
        source_image, instance_callback=collect_instance, return_details=True
    )
    working_height, working_width = working_image.shape[:2]
    scale_x = source_width / working_width
    scale_y = source_height / working_height
    reference_scale = math.sqrt(scale_x * scale_y)

    overlay = source_image.copy()
    objects = []
    category_ids = {"rice": 0, "foreign": 0}
    grain_lengths = []
    grain_breadths = []
    grain_lb_ratios = []
    whole_grain_count = 0
    broken_grain_count = 0
    whole_length_reference = float(proxy_details.get("reference_length", 0.0)) * reference_scale
    if not np.isfinite(whole_length_reference) or whole_length_reference <= 0:
        whole_length_reference = 0.0
    whole_threshold_length = 0.75 * whole_length_reference

    for working_mask, category, wx1, wy1 in instances:
        working_x2 = wx1 + working_mask.shape[1]
        working_y2 = wy1 + working_mask.shape[0]
        x1 = max(0, int(math.floor(wx1 * scale_x)))
        y1 = max(0, int(math.floor(wy1 * scale_y)))
        x2 = min(source_width, int(math.ceil(working_x2 * scale_x)))
        y2 = min(source_height, int(math.ceil(working_y2 * scale_y)))
        if x2 <= x1 or y2 <= y1:
            continue
        crop_mask = cv2.resize(
            working_mask.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        ys, xs = np.nonzero(crop_mask)
        xs += x1
        ys += y1

        category_ids[category] += 1
        object_id = f"R{category_ids[category]}" if category == "rice" else f"F{category_ids[category]}"
        grain_metrics = {}

        if category == "rice":
            local_points = np.column_stack((xs - x1, ys - y1)).astype(np.float32).reshape(-1, 1, 2)
            points = np.column_stack((xs, ys)).astype(np.float32).reshape(-1, 1, 2)
            rect_width, rect_height = cv2.minAreaRect(local_points)[1]
            length = float(max(rect_width, rect_height))
            breadth = float(min(rect_width, rect_height))
            lb_ratio = length / breadth if breadth > 0 else 0.0
            is_whole = whole_length_reference > 0 and length >= whole_threshold_length
            whole_grain_count += int(is_whole)
            broken_grain_count += int(whole_length_reference > 0 and not is_whole)
            grain_lengths.append(length)
            grain_breadths.append(breadth)
            grain_lb_ratios.append(lb_ratio)

            mask_contours, _ = cv2.findContours(crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull_area = cv2.contourArea(cv2.convexHull(max(mask_contours, key=cv2.contourArea))) if mask_contours else 0.0
            solidity = min(1.0, float(crop_mask.sum()) / max(hull_area, 1.0))
            ellipse_length = length
            if len(points) >= 5:
                try:
                    ellipse_axes = cv2.fitEllipse(points)[1]
                    ellipse_length = float(max(ellipse_axes))
                except cv2.error:
                    pass

            whole_label = "undetermined" if whole_length_reference <= 0 else ("whole" if is_whole else "broken")
            color = (255, 0, 0) if whole_label == "whole" else (
                (0, 255, 255) if whole_label == "broken" else (128, 128, 128)
            )
            grain_metrics = {
                "length": round(length, 2),
                "effective_length": round(length, 2),
                "ellipse_length": round(ellipse_length, 2),
                "breadth": round(breadth, 2),
                "lb_ratio": round(lb_ratio, 2),
                "length_ratio": round(length / whole_length_reference, 4) if whole_length_reference else None,
                "mask_area": int(crop_mask.sum()),
                "solidity": round(solidity, 4),
                "whole_label": whole_label,
            }
        else:
            color = (0, 0, 255)

        overlay_region = overlay[y1:y2, x1:x2]
        source_region = source_image[y1:y2, x1:x2]
        blended_region = cv2.addWeighted(source_region, 0.58, np.full_like(source_region, color), 0.42, 0)
        overlay_region[crop_mask] = blended_region[crop_mask]

        object_contours, _ = cv2.findContours(crop_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_polygon = []
        if object_contours:
            contour = max(object_contours, key=cv2.contourArea)
            epsilon = max(1.0, 0.005 * cv2.arcLength(contour, True))
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            mask_polygon = [[int(px + x1), int(py + y1)] for px, py in simplified]

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
            "mask_polygon": mask_polygon,
            "pixel_count": int(crop_mask.sum()),
            "mask_image": f"data:image/png;base64,{base64.b64encode(mask_encoded.tobytes()).decode('ascii')}",
        }
        if category == "rice":
            object_data.update(grain_metrics)
        objects.append(object_data)

    rice_count = category_ids["rice"]
    foreign_count = category_ids["foreign"]
    ok, masked_encoded = cv2.imencode(".png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("Could not encode the lossless masked output image.")

    # Keep the original upload bytes for browser-supported formats. TIFF is
    # converted to PNG because browsers do not consistently render TIFF data URLs.
    if source_mime_type == "image/tiff":
        original_ok, original_encoded = cv2.imencode(".png", source_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not original_ok:
            raise ValueError("Could not encode the original image for display.")
        original_bytes = original_encoded.tobytes()
        original_mime_type = "image/png"
    else:
        original_bytes = contents
        original_mime_type = source_mime_type

    rice_denominator = len(grain_lengths)
    summary = {
        "whole_grain_count": whole_grain_count,
        "broken_grain_count": broken_grain_count,
        "whole_percent": round(100 * whole_grain_count / rice_denominator, 2) if rice_denominator else 0.0,
        "whole_threshold_length": round(whole_threshold_length, 2),
        "reference_length": round(whole_length_reference, 2) if whole_length_reference else None,
        "touching_cluster_count": int(proxy_details.get("split_cluster_count", 0)),
        "average_length": round(float(np.mean(grain_lengths)), 2) if grain_lengths else None,
        "median_length": round(float(np.median(grain_lengths)), 2) if grain_lengths else None,
        "average_breadth": round(float(np.mean(grain_breadths)), 2) if grain_breadths else None,
        "median_breadth": round(float(np.median(grain_breadths)), 2) if grain_breadths else None,
        "average_lb_ratio": round(float(np.mean(grain_lb_ratios)), 2) if grain_lb_ratios else None,
        "median_lb_ratio": round(float(np.median(grain_lb_ratios)), 2) if grain_lb_ratios else None,
    }
    return {
        "masked_image": f"data:image/png;base64,{base64.b64encode(masked_encoded.tobytes()).decode('ascii')}",
        "original_image": f"data:{original_mime_type};base64,{base64.b64encode(original_bytes).decode('ascii')}",
        "image_width": int(source_width),
        "image_height": int(source_height),
        "rice_count": int(rice_count),
        "foreign_matter_count": int(foreign_count),
        "objects": objects,
        "summary": summary,
    }


@router.get("/health")
def get_health() -> dict[str, str]:
    """Report that the API process is ready to accept requests."""
    return {"status": "healthy", "version": "1.0.0", "service": "Grain Quality Analyzer API"}


@router.post("/rice-quality/analyze")
async def analyze_with_rice_method(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze one uploaded image using the active classical CV method."""
    global _pending_analyses
    request_id = getattr(request.state, "request_id", None)
    queue_started = time.monotonic()
    queued = False
    if _analysis_semaphore.locked():
        if _pending_analyses >= MAX_PENDING_ANALYSES:
            structured_logger.warning("analysis_queue_full", request_id, pending=_pending_analyses)
            raise HTTPException(status_code=503, detail="Analysis is busy. Please retry in a moment.")
        _pending_analyses += 1
        queued = True

    try:
        async with _analysis_semaphore:
            if queued:
                _pending_analyses -= 1
                queued = False
            queue_wait_ms = int((time.monotonic() - queue_started) * 1000)
            contents = await file.read(MAX_UPLOAD_BYTES + 1)
            if not contents:
                structured_logger.warning("analysis_rejected", request_id, reason="empty_upload")
                raise HTTPException(status_code=400, detail="Uploaded image is empty.")
            if len(contents) > MAX_UPLOAD_BYTES:
                structured_logger.warning(
                    "analysis_rejected", request_id, reason="file_size_limit", file_bytes=len(contents)
                )
                raise HTTPException(status_code=413, detail="Image exceeds the 25 MB upload limit.")

            started = time.monotonic()
            structured_logger.info(
                "analysis_started", request_id, file_bytes=len(contents), queue_wait_ms=queue_wait_ms
            )
            try:
                result = await run_in_threadpool(_run_fresh_proxy, contents)
            except HTTPException as exc:
                structured_logger.warning(
                    "analysis_rejected", request_id, status_code=exc.status_code,
                    reason=exc.detail, processing_ms=int((time.monotonic() - started) * 1000),
                )
                raise
            except (ValueError, cv2.error) as exc:
                structured_logger.warning(
                    "analysis_rejected", request_id, error_type=type(exc).__name__,
                    processing_ms=int((time.monotonic() - started) * 1000),
                )
                raise HTTPException(status_code=400, detail=f"Could not analyze this image: {exc}") from exc
            except Exception as exc:
                structured_logger.error(
                    "analysis_failed", request_id, error_type=type(exc).__name__,
                    processing_ms=int((time.monotonic() - started) * 1000),
                )
                raise

            structured_logger.info(
                "analysis_completed", request_id,
                processing_ms=int((time.monotonic() - started) * 1000),
                image_width=result["image_width"], image_height=result["image_height"],
                rice_count=result["rice_count"], foreign_matter_count=result["foreign_matter_count"],
                whole_grain_count=result["summary"]["whole_grain_count"],
                broken_grain_count=result["summary"]["broken_grain_count"],
                reference_length=result["summary"]["reference_length"],
                response_objects=len(result["objects"]),
            )
            return result
    finally:
        if queued:
            _pending_analyses = max(0, _pending_analyses - 1)
