"""HTTP endpoints used by the live rice quality analysis application."""

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from rice_analysis_method.rice_analysis import label_image

router = APIRouter()
_fresh_proxy_semaphore = asyncio.Semaphore(1)


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

        # Keep only each object's bounding-box mask. The labeler works at a
        # reduced resolution, but retaining one full-frame mask per grain can
        # still consume hundreds of MB on a busy sample.
        instances = []

        def collect_instance(mask: np.ndarray, category: str) -> None:
            ys, xs = np.nonzero(mask)
            if not len(xs):
                return
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            instances.append((mask[y1:y2, x1:x2].copy(), category, x1, y1))

        image, _unused_instances, _reference_area, proxy_details = label_image(
            temp_path, instance_callback=collect_instance, return_details=True
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    overlay = image.copy()
    # OpenCV uses BGR channel order: rice blue, foreign matter red.
    objects = []
    category_ids = {"rice": 0, "foreign": 0}
    grain_lengths = []
    grain_breadths = []
    grain_lb_ratios = []
    whole_grain_count = 0
    broken_grain_count = 0
    whole_length_reference = float(proxy_details.get("reference_length", 0.0))
    if not np.isfinite(whole_length_reference) or whole_length_reference <= 0:
        whole_length_reference = 0.0
    whole_threshold_length = 0.75 * whole_length_reference
    for mask, category, x1, y1 in instances:
        ys, xs = np.nonzero(mask)
        xs += x1
        ys += y1

        category_ids[category] += 1
        object_id = f"R{category_ids[category]}" if category == "rice" else f"F{category_ids[category]}"
        x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
        crop_mask = mask

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
            solidity = min(1.0, float(mask.sum()) / max(hull_area, 1.0))
            ellipse_length = length
            if len(points) >= 5:
                try:
                    ellipse_axes = cv2.fitEllipse(points)[1]
                    ellipse_length = float(max(ellipse_axes))
                except cv2.error:
                    pass
            whole_label = (
                "undetermined"
                if whole_length_reference <= 0
                else ("whole" if is_whole else "broken")
            )
            color = (255, 0, 0) if whole_label == "whole" else ((0, 255, 255) if whole_label == "broken" else (128, 128, 128))
            grain_metrics = {
                "length": round(length, 2),
                "effective_length": round(length, 2),
                "ellipse_length": round(ellipse_length, 2),
                "breadth": round(breadth, 2),
                "lb_ratio": round(lb_ratio, 2),
                "length_ratio": round(length / whole_length_reference, 4) if whole_length_reference else None,
                "mask_area": int(mask.sum()),
                "solidity": round(solidity, 4),
                "whole_label": whole_label,
            }
        else:
            color = (0, 0, 255)

        overlay_region = overlay[y1:y2, x1:x2]
        source_region = image[y1:y2, x1:x2]
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
        "masked_image": f"data:image/jpeg;base64,{image_base64}",
        "original_image": f"data:image/jpeg;base64,{base64.b64encode(original_encoded.tobytes()).decode('ascii')}",
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "rice_count": int(rice_count),
        "foreign_matter_count": int(foreign_count),
        "objects": objects,
        "summary": summary,
    }


@router.get("/health")
def get_health() -> dict[str, str]:
    """Report that the API process is ready to accept requests."""
    return {"status": "healthy", "version": "1.0.0", "service": "Rice Quality Analysis API"}


@router.post("/rice-quality/analyze")
async def analyze_with_rice_method(file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze one uploaded image using the active classical CV method."""
    supported_types = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type not in supported_types:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, WebP, TIFF, or BMP image.")

    max_bytes = 25 * 1024 * 1024
    async with _fresh_proxy_semaphore:
        contents = await file.read(max_bytes + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        if len(contents) > max_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds the 25 MB upload limit.")

        try:
            return await run_in_threadpool(_run_fresh_proxy, contents, file.filename or "upload.jpg")
        except (ValueError, cv2.error) as exc:
            raise HTTPException(status_code=400, detail=f"Could not analyze this image: {exc}") from exc
