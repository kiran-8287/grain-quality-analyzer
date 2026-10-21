"""Experimental, isolated inference for the grain Mask R-CNN and YOLO checkpoints.

This module does not import Torch at module import time. The web app can therefore
report a clean 503 when ML dependencies or experiment weights are unavailable.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrainInferenceConfig:
    project_root: Path
    mask_checkpoint: Path
    yolo_weights: Path
    overlap: int = 128
    mask_score_threshold: float = 0.5
    yolo_imgsz: int = 1280
    yolo_conf: float = 0.25
    device: str = "auto"
    timeout_seconds: float = 180.0
    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    mask_nms_iou: float = 0.5
    suppress_foreign_overlap: bool = False
    foreign_mask_iou: float = 0.5

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "GrainInferenceConfig":
        root = (project_root or Path(__file__).resolve().parents[2]).resolve()

        def configured_path(name: str, default: str) -> Path:
            value = Path(os.environ.get(name, default))
            return value if value.is_absolute() else root / value

        return cls(
            project_root=root,
            mask_checkpoint=configured_path(
                "GRAIN_MASKRCNN_CHECKPOINT", "runs/grain/maskrcnn/best.pth"
            ),
            yolo_weights=configured_path(
                "GRAIN_YOLO_WEIGHTS", "runs/grain/yolo/exp/weights/best.pt"
            ),
            overlap=int(os.environ.get("GRAIN_TILE_OVERLAP", "128")),
            mask_score_threshold=float(os.environ.get("GRAIN_MASK_SCORE_THRESHOLD", "0.5")),
            yolo_imgsz=int(os.environ.get("GRAIN_YOLO_IMGSZ", "1280")),
            yolo_conf=float(os.environ.get("GRAIN_YOLO_CONF", "0.25")),
            device=os.environ.get("GRAIN_DEVICE", "auto"),
            timeout_seconds=float(os.environ.get("GRAIN_INFER_TIMEOUT_SECONDS", "180")),
            max_upload_bytes=int(os.environ.get("GRAIN_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))),
            max_image_pixels=int(os.environ.get("GRAIN_MAX_IMAGE_PIXELS", "40000000")),
            mask_nms_iou=float(os.environ.get("GRAIN_MASK_NMS_IOU", "0.5")),
            suppress_foreign_overlap=os.environ.get(
                "GRAIN_SUPPRESS_FOREIGN_OVERLAP", "false"
            ).strip().lower() in {"1", "true", "yes", "on"},
            foreign_mask_iou=float(os.environ.get("GRAIN_FOREIGN_MASK_IOU", "0.5")),
        )

    def __post_init__(self) -> None:
        if self.overlap < 128:
            raise ValueError("GRAIN_TILE_OVERLAP must be at least 128 pixels")
        if not 0.0 <= self.mask_score_threshold <= 1.0:
            raise ValueError("Mask score threshold must be between 0 and 1")
        if not 0.0 <= self.yolo_conf <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")
        if self.yolo_imgsz < 32:
            raise ValueError("YOLO image size must be at least 32 pixels")
        if self.timeout_seconds <= 0 or self.max_upload_bytes <= 0 or self.max_image_pixels <= 0:
            raise ValueError("Timeout, upload, and pixel limits must be positive")


@dataclass
class InstanceMask:
    """A binary mask cropped to its tight bounding box in original-image coordinates."""

    class_name: str
    score: float
    bbox: tuple[int, int, int, int]  # x, y, width, height
    mask: Any  # numpy bool array, kept untyped to avoid importing numpy at module load


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= 0 or tile_size <= 0:
        raise ValueError("Image and tile dimensions must be positive")
    if overlap < 128:
        raise ValueError("Tile overlap must be at least 128 pixels")
    if tile_size <= overlap:
        raise ValueError("Tile size must be larger than overlap")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = [0]
    while starts[-1] + tile_size < length:
        starts.append(min(starts[-1] + step, length - tile_size))
    return starts


def image_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    """Return (x, y, width, height) windows that cover the image with required overlap."""
    xs = tile_starts(width, tile_size, overlap)
    ys = tile_starts(height, tile_size, overlap)
    return [
        (x, y, min(tile_size, width - x), min(tile_size, height - y))
        for y in ys
        for x in xs
    ]


def mask_iou(first: InstanceMask, second: InstanceMask) -> float:
    if first.class_name != second.class_name:
        return 0.0
    ax, ay, aw, ah = first.bbox
    bx, by, bw, bh = second.bbox
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    area_a = int(first.mask.sum())
    area_b = int(second.mask.sum())
    if x0 >= x1 or y0 >= y1:
        return 0.0
    a = first.mask[y0 - ay:y1 - ay, x0 - ax:x1 - ax]
    b = second.mask[y0 - by:y1 - by, x0 - bx:x1 - bx]
    intersection = int((a & b).sum())
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def mask_iou_nms(detections: Sequence[InstanceMask], iou_threshold: float) -> list[InstanceMask]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("Mask NMS IoU threshold must be between 0 and 1")
    kept: list[InstanceMask] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if all(mask_iou(detection, previous) <= iou_threshold for previous in kept):
            kept.append(detection)
    return kept


def _mask_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def _mask_touches_inner_tile(mask: Any, x: int, y: int, image_width: int, image_height: int) -> bool:
    if mask.size == 0:
        return False
    h, w = mask.shape
    return bool(
        (x > 0 and mask[:, 0].any())
        or (x + w < image_width and mask[:, -1].any())
        or (y > 0 and mask[0, :].any())
        or (y + h < image_height and mask[-1, :].any())
    )


def _expanded_mask_intersects(detection: InstanceMask, other: InstanceMask, image_width: int, image_height: int) -> bool:
    import cv2
    import numpy as np

    x, y, width, height = detection.bbox
    expanded_x0, expanded_y0 = max(0, x - 2), max(0, y - 2)
    expanded_x1, expanded_y1 = min(image_width, x + width + 2), min(image_height, y + height + 2)
    expanded = np.zeros((expanded_y1 - expanded_y0, expanded_x1 - expanded_x0), dtype=np.uint8)
    ox, oy = x - expanded_x0, y - expanded_y0
    expanded[oy:oy + height, ox:ox + width] = detection.mask.astype(np.uint8)
    expanded = cv2.dilate(expanded, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)

    bx, by, bw, bh = other.bbox
    ix0, iy0 = max(expanded_x0, bx), max(expanded_y0, by)
    ix1, iy1 = min(expanded_x1, bx + bw), min(expanded_y1, by + bh)
    if ix0 >= ix1 or iy0 >= iy1:
        return False
    a = expanded[iy0 - expanded_y0:iy1 - expanded_y0, ix0 - expanded_x0:ix1 - expanded_x0]
    b = other.mask[iy0 - by:iy1 - by, ix0 - bx:ix1 - bx]
    return bool((a & b).any())


def measure_mask(
    detection: InstanceMask,
    image_size: tuple[int, int],
    all_masks: Sequence[InstanceMask],
    px_per_mm: float | None = None,
) -> dict[str, Any]:
    """Measure the largest external contour; mark image-border/neighbor contact as not measurable."""
    import cv2
    import numpy as np

    if px_per_mm is not None and (not math.isfinite(px_per_mm) or px_per_mm <= 0):
        raise ValueError("px_per_mm must be a finite positive number")
    image_width, image_height = image_size
    x, y, width, height = detection.bbox
    mask_u8 = detection.mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Cannot measure an empty mask")
    contour = max(contours, key=cv2.contourArea)
    (_, _), (side_a, side_b), angle = cv2.minAreaRect(contour)
    length_px, breadth_px = sorted((float(side_a), float(side_b)), reverse=True)
    ratio = length_px / breadth_px if breadth_px > 0 else None
    touches_border = bool(
        (x == 0 and detection.mask[:, 0].any())
        or (x + width >= image_width and detection.mask[:, -1].any())
        or (y == 0 and detection.mask[0, :].any())
        or (y + height >= image_height and detection.mask[-1, :].any())
    )
    touches_neighbor = any(
        other is not detection
        and _expanded_mask_intersects(detection, other, image_width, image_height)
        for other in all_masks
    )
    flags = []
    if touches_border:
        flags.append("touches_image_border")
    if touches_neighbor:
        flags.append("touches_neighbor")
    if px_per_mm is None:
        flags.append("calibration_missing")
    measurable = not touches_border and not touches_neighbor
    return {
        "length_px": length_px,
        "breadth_px": breadth_px,
        "lb_ratio": ratio,
        "length_mm": length_px / px_per_mm if px_per_mm else None,
        "breadth_mm": breadth_px / px_per_mm if px_per_mm else None,
        "bbox": [x, y, width, height],
        "area_px": float(detection.mask.sum()),
        "orientation_deg": float(angle),
        "measurable": measurable,
        "flags": flags,
    }


def _build_model_from_checkpoint(num_classes: int, crop_size: int, anchor_sizes: Sequence[int]):
    """Minimal copy of train_maskrcnn.build_model, preserving its checkpoint architecture.

    The training module calls cv2.setNumThreads(0) at import. Keeping inference's builder
    local avoids changing the web application's OpenCV thread setting.
    """
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if len(anchor_sizes) != 5:
        raise ValueError("Checkpoint anchor_sizes must contain five values")
    model = maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        trainable_backbone_layers=5,
        min_size=crop_size,
        max_size=crop_size,
        box_detections_per_img=500,
        box_nms_thresh=0.6,
        rpn_pre_nms_top_n_train=2000,
        rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=2000,
        rpn_post_nms_top_n_test=1500,
        box_batch_size_per_image=512,
    )
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((int(size),) for size in anchor_sizes),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


class GrainInferenceRuntime:
    """Loaded-once Mask R-CNN + YOLO runtime, with serialized device access."""

    def __init__(self, config: GrainInferenceConfig):
        self.config = config
        missing = [str(path) for path in (config.mask_checkpoint, config.yolo_weights) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing experimental model weights: " + ", ".join(missing))
        self._lock = threading.Lock()
        self._torch = importlib.import_module("torch")
        self._np = importlib.import_module("numpy")
        self._cv2 = importlib.import_module("cv2")
        self._pil_image = importlib.import_module("PIL.Image")
        self._pil_image_ops = importlib.import_module("PIL.ImageOps")
        self._torchvision = importlib.import_module("torchvision")
        self._ultralytics = importlib.import_module("ultralytics")
        self._checkpoint = self._torch.load(
            config.mask_checkpoint, map_location="cpu", weights_only=False
        )
        self.class_names = list(self._checkpoint["class_names"])
        self.crop_size = int(self._checkpoint["crop_size"])
        self.anchor_sizes = [int(v) for v in self._checkpoint["anchor_sizes"]]
        if config.overlap >= self.crop_size:
            raise ValueError("Tile overlap must be smaller than checkpoint crop_size")
        self.checkpoint_epoch = int(self._checkpoint.get("epoch", -1))
        self.mask_version = f"{config.mask_checkpoint.name}:epoch-{self.checkpoint_epoch + 1}"
        self.yolo_version = config.yolo_weights.name
        self.device = self._select_device(config.device)
        self.mask_model = None
        self.yolo_model = None
        try:
            self._load_models(self.device)
        except Exception as exc:
            if self.device.type != "cuda":
                raise
            logger.warning(
                "grain_ml_cuda_model_load_failed",
                extra={"error_type": type(exc).__name__, "message": str(exc)},
            )
            self._empty_cuda_cache()
            self.device = self._torch.device("cpu")
            self._load_models(self.device)
            logger.warning("grain_ml_using_cpu_fallback", extra={"reason": "CUDA model load failed"})

    def _select_device(self, requested: str):
        torch = self._torch
        requested = requested.strip()
        normalized = requested.lower()
        if normalized == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
            normalized = requested.lower()
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("grain_ml_using_cpu_fallback", extra={"reason": "CUDA is unavailable"})
            requested = "cpu"
            normalized = "cpu"
        return torch.device("cuda:0" if normalized == "0" else normalized)

    def _load_models(self, device: Any) -> None:
        torch = self._torch
        model = _build_model_from_checkpoint(
            len(self.class_names) + 1, self.crop_size, self.anchor_sizes
        )
        model.load_state_dict(self._checkpoint["model"], strict=True)
        model.eval().to(device)
        from ultralytics import YOLO

        yolo = YOLO(str(self.config.yolo_weights))
        yolo.to(str(device))
        self.mask_model, self.yolo_model = model, yolo

    def _empty_cuda_cache(self) -> None:
        if self._torch.cuda.is_available():
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                logger.debug("grain_ml_cuda_cache_clear_failed", exc_info=True)

    def _mask_detections(self, image: Any, tiles: Sequence[tuple[int, int, int, int]]) -> list[InstanceMask]:
        torch, np = self._torch, self._np
        detections: list[InstanceMask] = []
        for x, y, width, height in tiles:
            rgb = image[y:y + height, x:x + width]
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    output = self.mask_model([tensor.to(self.device)])[0]
            scores = output["scores"].detach().float().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            masks = output["masks"]
            for index, score in enumerate(scores):
                if float(score) < self.config.mask_score_threshold:
                    continue
                label = int(labels[index])
                if label < 1 or label > len(self.class_names):
                    continue
                mask = (masks[index, 0] > 0.5).detach().cpu().numpy().astype(bool)
                if _mask_touches_inner_tile(mask, x, y, image.shape[1], image.shape[0]):
                    continue
                local_bbox = _mask_bbox(mask)
                if local_bbox is None:
                    continue
                bx, by, bw, bh = local_bbox
                tight_mask = mask[by:by + bh, bx:bx + bw].copy()
                detections.append(InstanceMask(
                    self.class_names[label - 1], float(score), (x + bx, y + by, bw, bh), tight_mask
                ))
            del output, masks
        return detections

    def _foreign_detections(self, image: Any, tiles: Sequence[tuple[int, int, int, int]]) -> list[dict[str, Any]]:
        detections = []
        for x, y, width, height in tiles:
            crop = image[y:y + height, x:x + width]
            prediction = self.yolo_model.predict(
                source=crop,
                imgsz=self.config.yolo_imgsz,
                conf=self.config.yolo_conf,
                device=str(self.device),
                half=self.device.type == "cuda",
                verbose=False,
            )[0]
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            class_names = getattr(self.yolo_model, "names", {})
            for box, score, class_id in zip(xyxy, scores, classes):
                x1, y1, x2, y2 = [float(v) for v in box]
                detections.append({
                    "class": class_names.get(int(class_id), "foreign_matter")
                    if isinstance(class_names, dict)
                    else class_names[int(class_id)],
                    "score": float(score),
                    "bbox": [round(x + x1, 2), round(y + y1, 2), round(x + x2, 2), round(y + y2, 2)],
                })
        return detections

    def _suppress_foreign(self, foreign: list[dict[str, Any]], rice_masks: Sequence[InstanceMask]) -> list[dict[str, Any]]:
        if not self.config.suppress_foreign_overlap:
            return foreign
        kept = []
        for item in foreign:
            x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            suppressed = False
            for grain in rice_masks:
                gx, gy, gw, gh = grain.bbox
                ix0, iy0, ix1, iy1 = max(x1, gx), max(y1, gy), min(x2, gx + gw), min(y2, gy + gh)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                overlap = int(grain.mask[iy0 - gy:iy1 - gy, ix0 - gx:ix1 - gx].sum())
                union = box_area + int(grain.mask.sum()) - overlap
                if union and overlap / union >= self.config.foreign_mask_iou:
                    suppressed = True
                    break
            if not suppressed:
                kept.append(item)
        return kept

    def _analyze_rgb(self, image: Any, px_per_mm: float | None) -> dict[str, Any]:
        height, width = image.shape[:2]
        tiles = image_tiles(width, height, self.crop_size, self.config.overlap)
        t0 = time.perf_counter()
        mask_detections = self._mask_detections(image, tiles)
        t_mask = time.perf_counter()
        rice_masks = mask_iou_nms(mask_detections, self.config.mask_nms_iou)
        foreign = self._foreign_detections(image, tiles)
        t_yolo = time.perf_counter()
        foreign = self._suppress_foreign(foreign, rice_masks)

        grains = []
        for idx, detection in enumerate(rice_masks, start=1):
            measured = measure_mask(detection, (width, height), rice_masks, px_per_mm)
            grains.append({
                "id": idx,
                "class": detection.class_name,
                "score": detection.score,
                **measured,
            })
        whole = [item for item in grains if item["class"] == "whole_rice" and item["measurable"]]
        broken_count = sum(item["class"] == "broken_rice" for item in grains)
        whole_count = sum(item["class"] == "whole_rice" for item in grains)
        rice_count = whole_count + broken_count
        total_count = rice_count + len(foreign)
        mean_length_px = statistics.fmean(item["length_px"] for item in whole) if whole else None
        mean_breadth_px = statistics.fmean(item["breadth_px"] for item in whole) if whole else None
        mean_ratio = statistics.fmean(item["lb_ratio"] for item in whole) if whole else None
        aggregates = {
            "measurable_whole_count": len(whole),
            "mean_length_px": mean_length_px,
            "mean_breadth_px": mean_breadth_px,
            "mean_lb_ratio": mean_ratio,
            "mean_length_mm": mean_length_px / px_per_mm if mean_length_px is not None and px_per_mm else None,
            "mean_breadth_mm": mean_breadth_px / px_per_mm if mean_breadth_px is not None and px_per_mm else None,
            "measurement_basis": "px_per_mm calibration" if px_per_mm else "pixels; calibration unavailable",
        }
        return {
            "experimental": True,
            "image_size": {"width": width, "height": height},
            "counts": {
                "whole_rice": whole_count,
                "broken_rice": broken_count,
                "foreign_matter": len(foreign),
            },
            "percentages": {
                "broken_of_rice": broken_count / rice_count * 100 if rice_count else None,
                "foreign_of_all_detections": len(foreign) / total_count * 100 if total_count else None,
            },
            "grains": grains,
            "foreign_matter": foreign,
            "aggregates": aggregates,
            "model_versions": {
                "maskrcnn": {
                    "checkpoint": self.config.mask_checkpoint.name,
                    "version": self.mask_version,
                    "torch": str(self._torch.__version__),
                    "torchvision": str(self._torchvision.__version__),
                },
                "yolo": {
                    "checkpoint": self.config.yolo_weights.name,
                    "version": self.yolo_version,
                    "ultralytics": str(self._ultralytics.__version__),
                },
            },
            "device": str(self.device),
            "timings_ms": {
                "maskrcnn": round((t_mask - t0) * 1000, 2),
                "yolo": round((t_yolo - t_mask) * 1000, 2),
                "total": round((time.perf_counter() - t0) * 1000, 2),
            },
        }

    def analyze_image(self, image_bytes: bytes, px_per_mm: float | None = None) -> dict[str, Any]:
        if px_per_mm is not None and (not math.isfinite(px_per_mm) or px_per_mm <= 0):
            raise ValueError("px_per_mm must be a finite positive number")
        image_module = self._pil_image
        try:
            with image_module.open(importlib.import_module("io").BytesIO(image_bytes)) as source:
                oriented = self._pil_image_ops.exif_transpose(source).convert("RGB")
                if oriented.width * oriented.height > self.config.max_image_pixels:
                    raise ValueError(
                        f"Image exceeds the {self.config.max_image_pixels}-pixel processing limit"
                    )
                image = self._np.asarray(oriented)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Uploaded file is not a readable image") from exc

        with self._lock:
            started = time.perf_counter()
            try:
                result = self._analyze_rgb(image, px_per_mm)
            except Exception as exc:
                if self.device.type != "cuda" or not self._looks_like_cuda_memory_error(exc):
                    raise
                logger.warning(
                    "grain_ml_cuda_inference_failed",
                    extra={"error_type": type(exc).__name__, "message": str(exc)},
                )
                self._empty_cuda_cache()
                self.device = self._torch.device("cpu")
                self.mask_model.to(self.device)
                self.yolo_model.to(str(self.device))
                logger.warning("grain_ml_using_cpu_fallback", extra={"reason": "CUDA inference allocation failed"})
                result = self._analyze_rgb(image, px_per_mm)
            finally:
                if self._torch.cuda.is_available():
                    self._empty_cuda_cache()
            result["timings_ms"]["total"] = round((time.perf_counter() - started) * 1000, 2)
            return result

    @staticmethod
    def _looks_like_cuda_memory_error(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return "cuda" in text or "out of memory" in text or "bad allocation" in text
