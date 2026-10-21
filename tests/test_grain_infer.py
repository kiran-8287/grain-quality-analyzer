"""Synthetic, weight-free unit and endpoint tests for experimental grain inference."""

from types import SimpleNamespace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.models.grain_infer import (
    GrainInferenceConfig,
    InstanceMask,
    _mask_bbox,
    image_tiles,
    mask_iou_nms,
    measure_mask,
)


def _instance(mask: np.ndarray, class_name: str = "whole_rice", score: float = 0.9) -> InstanceMask:
    bbox = _mask_bbox(mask)
    assert bbox is not None
    x, y, width, height = bbox
    return InstanceMask(class_name, score, bbox, mask[y:y + height, x:x + width].astype(bool))


def _synthetic_image() -> bytes:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:] = (45, 105, 38)
    cv2.ellipse(image, (128, 128), (50, 15), 0, 0, 360, (220, 228, 224), -1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_tiles_cover_image_with_configured_overlap_and_small_images_use_one_tile():
    tiles = image_tiles(2048, 1600, 512, 128)
    assert len(tiles) > 1
    covered = np.zeros((1600, 2048), dtype=bool)
    for x, y, width, height in tiles:
        covered[y:y + height, x:x + width] = True
    assert covered.all()
    xs = sorted({x for x, _, _, _ in tiles})
    assert all((left + 512) - right >= 128 for left, right in zip(xs, xs[1:]))
    assert image_tiles(300, 200, 512, 128) == [(0, 0, 300, 200)]


def test_mask_iou_nms_keeps_highest_score_duplicate_and_different_class():
    first = np.zeros((20, 20), dtype=bool)
    first[5:15, 5:15] = True
    duplicate = np.zeros_like(first)
    duplicate[5:15, 6:16] = True
    higher = _instance(first, score=0.8)
    lower_duplicate = _instance(duplicate, score=0.7)
    other_class = _instance(duplicate, class_name="broken_rice", score=0.6)
    kept = mask_iou_nms([lower_duplicate, other_class, higher], iou_threshold=0.5)
    assert kept == [higher, other_class]


def test_known_ellipse_measurement_and_calibration_conversion():
    mask = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(mask, (128, 128), (50, 15), 0, 0, 360, 255, -1)
    detection = _instance(mask > 0)
    measured = measure_mask(detection, (256, 256), [detection], px_per_mm=10.0)
    assert abs(measured["length_px"] - 100) / 100 <= 0.02
    assert abs(measured["breadth_px"] - 30) / 30 <= 0.02
    assert measured["length_mm"] == measured["length_px"] / 10.0
    assert measured["breadth_mm"] == measured["breadth_px"] / 10.0
    assert measured["measurable"] is True


def test_measurement_flags_image_border_and_neighbor_after_two_pixel_dilation():
    border = np.zeros((40, 40), dtype=bool)
    border[10:20, 0:10] = True
    border_detection = _instance(border)
    measured_border = measure_mask(border_detection, (40, 40), [border_detection])
    assert measured_border["measurable"] is False
    assert "touches_image_border" in measured_border["flags"]
    assert "calibration_missing" in measured_border["flags"]

    left = np.zeros((40, 40), dtype=bool)
    right = np.zeros((40, 40), dtype=bool)
    left[10:20, 10:20] = True
    right[10:20, 21:31] = True
    first, second = _instance(left), _instance(right)
    measured_neighbor = measure_mask(first, (40, 40), [first, second])
    assert measured_neighbor["measurable"] is False
    assert "touches_neighbor" in measured_neighbor["flags"]


def test_config_defaults_and_minimum_overlap(monkeypatch, tmp_path):
    monkeypatch.delenv("GRAIN_MASKRCNN_CHECKPOINT", raising=False)
    monkeypatch.delenv("GRAIN_YOLO_WEIGHTS", raising=False)
    config = GrainInferenceConfig.from_env(tmp_path)
    assert config.mask_checkpoint == tmp_path / "runs/grain/maskrcnn/best.pth"
    assert config.yolo_weights == tmp_path / "runs/grain/yolo/exp/weights/best.pt"
    monkeypatch.setenv("GRAIN_TILE_OVERLAP", "64")
    try:
        GrainInferenceConfig.from_env(tmp_path)
    except ValueError as exc:
        assert "at least 128" in str(exc)
    else:
        raise AssertionError("overlap below 128 must be rejected")


def test_analyze_ml_returns_503_when_experiment_weights_are_unavailable():
    with TestClient(app) as client:
        response = client.post(
            "/analyze-ml",
            files={"file": ("synthetic.png", _synthetic_image(), "image/png")},
        )
    assert response.status_code == 503
    assert response.json()["experimental"] is True
    assert "weights" in response.json()["detail"].lower()


def test_analyze_ml_endpoint_accepts_synthetic_image_and_marks_result_experimental():
    fake_response = {
        "experimental": True,
        "image_size": {"width": 256, "height": 256},
        "counts": {"whole_rice": 1, "broken_rice": 0, "foreign_matter": 0},
        "percentages": {"broken_of_rice": 0.0, "foreign_of_all_detections": 0.0},
        "grains": [{
            "id": 1,
            "class": "whole_rice",
            "score": 0.9,
            "length_px": 100.0,
            "breadth_px": 30.0,
            "lb_ratio": 10 / 3,
            "length_mm": None,
            "breadth_mm": None,
            "bbox": [78, 113, 101, 31],
            "area_px": 2000.0,
            "orientation_deg": 0.0,
            "measurable": True,
            "flags": ["calibration_missing"],
        }],
        "foreign_matter": [],
        "aggregates": {"measurable_whole_count": 1, "mean_length_px": 100.0},
        "model_versions": {"maskrcnn": {"checkpoint": "smoke.pth"}, "yolo": {"checkpoint": "smoke.pt"}},
        "device": "cpu",
        "timings_ms": {"maskrcnn": 1.0, "yolo": 1.0, "total": 2.0},
    }

    class FakeRuntime:
        config = SimpleNamespace(max_upload_bytes=1024 * 1024, timeout_seconds=5)

        def analyze_image(self, image_bytes, px_per_mm=None):
            assert image_bytes
            assert px_per_mm == 10.0
            return fake_response

    try:
        with TestClient(app) as client:
            app.state.grain_ml_runtime = FakeRuntime()
            response = client.post(
                "/analyze-ml",
                data={"px_per_mm": "10"},
                files={"file": ("synthetic.png", _synthetic_image(), "image/png")},
            )
    finally:
        app.state.grain_ml_runtime = None
    assert response.status_code == 200
    assert response.json()["experimental"] is True
    assert response.json()["grains"][0]["class"] == "whole_rice"
    assert response.json()["grains"][0]["length_mm"] is None


def test_analyze_ml_validation_errors_are_marked_experimental():
    with TestClient(app) as client:
        response = client.post("/analyze-ml")
    assert response.status_code == 422
    assert response.json()["experimental"] is True

