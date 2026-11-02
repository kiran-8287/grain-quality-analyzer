"""
Case 1 — rice-presence gate tests ("objects detected" != "rice detected").

Scenarios
    TEST 1  foreign matter only (18 stones, 0 rice)  -> NOT RICE, analysis skipped
    TEST 2  single valid rice grain                  -> RICE, analysis continues
    TEST 3  rice + foreign matter                    -> RICE, FM reported separately
    TEST 4  empty / background-only image            -> NOT RICE, analysis skipped
"""

import io
import sys
from pathlib import Path

import PIL.Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from ml.config import get_threshold
from ml.quality.foreign_matter import _detect_foreign_heuristic
from ml.segmentation.pipeline import RiceQualityPipeline
from ml.segmentation.rice_gate import load_project_class_mappings


@pytest.fixture
def pipeline():
    return RiceQualityPipeline(small_sample_threshold=30)


def _encode(img: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".png", img)
    assert ok
    return enc.tobytes()


def create_stone_image(img_size: int = 900, num_stones: int = 18, seed: int = 7) -> bytes:
    """
    Roundish grey / tan / dark pebbles with a speckled stone surface on a white
    background — the Case 1 situation (many objects, zero rice).
    """
    rng = np.random.default_rng(seed)
    img = np.full((img_size, img_size, 3), 250, dtype=np.uint8)
    cols, rows = 6, 3
    step_x, step_y = img_size // (cols + 1), img_size // (rows + 1)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= num_stones:
                break
            cx = (c + 1) * step_x + int(rng.integers(-12, 12))
            cy = (r + 1) * step_y + int(rng.integers(-12, 12))
            base = int(rng.integers(120, 180))
            tone = int(rng.integers(0, 3))
            if tone == 1:  # tan stone
                color = (min(255, base + 18), base, max(0, base - 25))
            elif tone == 2:  # dark stone
                color = (max(0, base - 45), max(0, base - 45), max(0, base - 40))
            else:  # grey stone
                color = (base, base, base)

            axes = (int(rng.integers(34, 47)), int(rng.integers(26, 39)))  # roundish
            angle = int(rng.integers(0, 180))
            cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)

            # Speckled stone surface (per-pixel texture inside the pebble)
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
            noise = rng.integers(-22, 22, size=(img_size, img_size, 1))
            speckled = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img[mask > 0] = speckled[mask > 0]
            idx += 1

    return _encode(img)


def create_rice_grain_image(img_size: int = 600) -> bytes:
    """Single white elongated grain on a black background."""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cv2.ellipse(
        img, (img_size // 2, img_size // 2), (70, 24), 20, 0, 360, (235, 235, 235), -1
    )
    return _encode(img)


def test_foreign_heuristic_ignores_rice_mask_edge_fringe():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    grain_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(image, (150, 120), (70, 24), 27, 0, 360, (235, 235, 235), -1)
    cv2.ellipse(grain_mask, (150, 120), (70, 24), 27, 0, 360, 255, -1)
    slightly_eroded_mask = cv2.erode(
        grain_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    result = _detect_foreign_heuristic(image, [slightly_eroded_mask])

    assert result.objects == []


def create_rice_plus_stone_image(img_size: int = 700) -> bytes:
    """One rice grain plus two stones on a black background (CASE B)."""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    cv2.ellipse(img, (200, 250), (70, 24), 25, 0, 360, (235, 235, 235), -1)
    for cx, cy, radius, value in ((480, 200, 45, 150), (470, 480, 40, 135)):
        cv2.circle(img, (cx, cy), radius, (value, value, value), -1)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        rng = np.random.default_rng(11)
        noise = rng.integers(-22, 22, size=(img_size, img_size, 1))
        speckled = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img[mask > 0] = speckled[mask > 0]
    return _encode(img)


def create_many_rice_image(img_size: int = 900, rows: int = 7, cols: int = 8) -> bytes:
    """Many rice grains on a dark tray background to reproduce the false NOT_RICE bug."""
    img = np.full((img_size, img_size, 3), 12, dtype=np.uint8)
    rng = np.random.default_rng(42)
    start_x = 80
    start_y = 80
    step_x = (img_size - 160) // cols
    step_y = (img_size - 160) // rows

    for r in range(rows):
        for c in range(cols):
            cx = start_x + c * step_x + int(rng.integers(-10, 10))
            cy = start_y + r * step_y + int(rng.integers(-10, 10))
            angle = int(rng.integers(-25, 25))
            cv2.ellipse(img, (cx, cy), (32 + int(rng.integers(-4, 4)), 10 + int(rng.integers(-2, 2))), angle, 0, 360, (235, 235, 235), -1)
    return _encode(img)


def test_class_mapping_comes_from_model_configuration():
    """The gate must use the project's real class names / IDs, not invented ones."""
    mappings = load_project_class_mappings(force_reload=True)
    assert mappings["rice_class_name"] == "rice_grain"
    assert mappings["rice_class_id"] == 1
    assert mappings["segmentation"] == {"background": 0, "rice_grain": 1}
    for name in ("stone", "inorganic", "organic", "other_foreign_matter"):
        assert name in mappings["foreign_matter_class_names"]
    assert mappings["rice_class_name"] not in mappings["foreign_matter_class_names"]


# TEST 1 — the failing image: foreign matter only (18 objects, 0 rice)
def test_case1_foreign_matter_only_is_not_rice(pipeline):
    res = pipeline.analyze(create_stone_image())

    gate = res["rice_gate"]
    print("\n[TEST 1] " + gate["debug"])

    assert res["success"] is True
    assert res["rice_detected"] is False
    assert gate["status"] in ("NOT_RICE", "NOT_RICE_SPARSE", "NO_ANALYSABLE_RICE")
    assert gate["has_rice"] is False
    assert gate["analysis_stopped"] is True
    assert gate["rice_detections"] == 0
    assert gate["total_detections"] >= 10, "the stone image must produce many detections"
    assert gate["foreign_matter_detections"] == gate["total_detections"]
    assert gate["rice_confidence_threshold"] == pytest.approx(
        get_threshold("rice_gate", "rice_confidence_threshold", 0.65)
    )

    # Every detection keeps a foreign-matter class — never a rice class
    for detection in gate["detections"]:
        assert detection["is_rice"] is False
        assert detection["class_name"] != gate["rice_class_name"]
        assert detection["class_name"] in gate["foreign_matter_classes"]
        assert detection["confidence"] < gate["rice_confidence_threshold"]

    # Rice analysis must NOT have run
    assert res["grains"] == []
    assert res["summary"] == {}
    assert res["sample"]["analysed"] == 0
    assert res["sample"]["rice_detections"] == 0
    assert res["admixture"]["admixture_status"] == "not_applicable"
    assert res["calibration"]["mode"] == "not_evaluated"
    assert res["standards"]["status"].startswith("Not evaluated")
    assert res["message"] == (
        "No rice grains detected. Please upload an image containing rice grains."
    )

    # Foreign matter is still reported separately (foreign matter is NOT rice)
    assert res["sample"]["foreign_matter_detections"] >= 1
    assert res["foreign_matter"], "foreign matter must remain available"
    assert res["foreign_matter_summary"]["foreign_object_count"] == len(
        res["foreign_matter"]
    )


def test_case1_explicit_gate_status_stops_pipeline_before_segmentation(pipeline):
    """The no-rice gate must terminate the rice-analysis path before segmentation."""
    res = pipeline.analyze(create_stone_image())
    gate = res["rice_gate"]

    assert gate["status"] in ("NOT_RICE", "NOT_RICE_SPARSE", "NO_ANALYSABLE_RICE")
    assert gate["analysis_stopped"] is True
    assert gate["has_rice"] is False
    assert res["sample"]["analysed"] == 0
    assert res["grains"] == []
    assert res["summary"] == {}


def test_learned_gate_is_primary_when_model_is_available(monkeypatch):
    """When a trained rice gate model is present, it should be the primary rice/no-rice signal."""
    from ml.segmentation import rice_gate as rice_gate_module

    class DummyGateModel:
        def predict(self, crop):
            return 0.99

    monkeypatch.setattr(rice_gate_module, "load_rice_gate_model", lambda: DummyGateModel())

    encoded = create_rice_grain_image()
    img = np.array(PIL.Image.open(io.BytesIO(encoded)))
    result = rice_gate_module.evaluate_rice_presence(img)

    assert result["has_rice"] is True
    assert result["method"] == "learned_rice_gate"
    assert result["model_status"].startswith("trained")
    assert result["model_data_provenance"] == "synthetic_hand_sampled_feature_vectors_only"
    assert result["model_confidence_calibrated"] is False
    assert any("synthetic feature vectors" in warning for warning in result["warnings"])


# TEST 2 — one valid rice grain
def test_case2_single_rice_grain_is_rice(pipeline):
    res = pipeline.analyze(create_rice_grain_image())

    gate = res["rice_gate"]
    print("\n[TEST 2] " + gate["debug"])

    assert res["success"] is True
    assert res["rice_detected"] is True
    assert gate["status"] == "RICE"
    assert gate["has_rice"] is True
    assert gate["analysis_stopped"] is False
    assert gate["rice_detections"] >= 1
    assert len(res["grains"]) == 1
    assert res["sample"]["analysed"] == 1
    assert res["summary"]


def test_case2a_only_rice_boxes_are_kept_after_gate_filter(pipeline):
    """A mixed image must keep only the rice-gate detections in the grain list."""
    res = pipeline.analyze(create_rice_plus_stone_image())
    gate = res["rice_gate"]

    assert gate["has_rice"] is True
    assert gate["rice_detections"] >= 1
    assert len(res["grains"]) == gate["rice_detections"]
    assert res["summary"]["total_rice_grains"] == gate["rice_detections"]
    assert len(res["foreign_matter"]) >= 1


# TEST 3 — rice + foreign matter (rice continues, FM reported separately)
def test_case3_rice_plus_foreign_matter(pipeline):
    res = pipeline.analyze(create_rice_plus_stone_image())

    gate = res["rice_gate"]
    print("\n[TEST 3] " + gate["debug"])

    assert res["success"] is True
    assert res["rice_detected"] is True
    assert gate["has_rice"] is True
    assert gate["rice_detections"] >= 1
    assert gate["foreign_matter_detections"] >= 1
    assert gate["total_detections"] == (
        gate["rice_detections"] + gate["foreign_matter_detections"]
    )

    # Rice analysis continues on the rice grains
    assert res["sample"]["analysed"] >= 1
    assert res["summary"]["total_rice_grains"] >= 1

    # Foreign matter remains separately identified (foreign matter is NOT rice)
    assert res["summary"]["foreign_matter_count"] >= 1
    assert len(res["foreign_matter"]) >= 1
    assert all(
        obj["class"] in gate["foreign_matter_classes"] for obj in res["foreign_matter"]
    )


# TEST 4 — empty / background-only image
def test_case4_background_only_is_not_rice(pipeline):
    background = np.full((400, 400, 3), 243, dtype=np.uint8)

    res = pipeline.analyze(_encode(background))

    gate = res["rice_gate"]
    print("\n[TEST 4] " + gate["debug"])

    assert res["success"] is True
    assert res["rice_detected"] is False
    assert gate["rice_detections"] == 0
    assert gate["has_rice"] is False
    assert gate["analysis_stopped"] is True
    assert res["grains"] == []
    assert res["summary"] == {}


def test_many_rice_scene_on_dark_background_is_rice(pipeline):
    """A many-rice scene must not be rejected merely because a detector misses many grains."""
    res = pipeline.analyze(create_many_rice_image())
    gate = res["rice_gate"]

    assert res["success"] is True
    assert res["rice_detected"] is True
    assert gate["status"] in ("RICE",)
    assert gate["has_rice"] is True
    assert gate["analysis_stopped"] is False
    assert gate["rice_detections"] >= 1
    assert res["summary"], "many-rice scene must continue to segmentation and analysis"
