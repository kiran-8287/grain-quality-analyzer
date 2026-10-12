"""
Rice + foreign-object (stones/wheat/mixed) validation harness — results1.

Runs every image in test_images/2 rice and stones/images through the
current production RiceQualityPipeline (classical CV gate + classical CV
segmentation cascade) and writes:

    test_images/2 rice and stones/results/results1/<NN>_annotated.jpg
    test_images/2 rice and stones/results/results1/validation_report.json
    test_images/2 rice and stones/results/results1/validation_summary.md

Validation focus:
    1. Rice-gate correctness on rice-present vs foreign-only scenes.
    2. Rice grain counts vs filename ground truth.
    3. Foreign-matter separation (stones/wheat must NOT become rice grains).
    4. Annotated output images for visual review.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.segmentation.pipeline import RiceQualityPipeline

DATASET_ROOT = PROJECT_ROOT / "test_images" / "2 rice and stones"
IMAGES_DIR = DATASET_ROOT / "images"
OUT_DIR = DATASET_ROOT / "results" / "results1"

# Ground truth derived from the filenames of the controlled test scenes.
GROUND_TRUTH: Dict[int, Dict[str, Any]] = {
    1:  {"expected_rice": 1,  "expected_fm": 0,  "desc": "One rice grain"},
    2:  {"expected_rice": 2,  "expected_fm": 0,  "desc": "Two rice grains"},
    3:  {"expected_rice": 5,  "expected_fm": 0,  "desc": "Five rice grains"},
    4:  {"expected_rice": 10, "expected_fm": 0,  "desc": "Ten rice grains"},
    5:  {"expected_rice": 25, "expected_fm": 0,  "desc": "Twenty five rice grains"},
    6:  {"expected_rice": 50, "expected_fm": 0,  "desc": "Fifty rice grains"},
    7:  {"expected_rice": 0,  "expected_fm": None, "desc": "Foreign objects only stones",
         "expect_rice_absent": True},
    8:  {"expected_rice": 0,  "expected_fm": None, "desc": "Foreign objects only wheat/other seeds",
         "expect_rice_absent": True},
    9:  {"expected_rice": 0,  "expected_fm": None, "desc": "Foreign objects only mixed",
         "expect_rice_absent": True},
    10: {"expected_rice": 10, "expected_fm": 1,  "desc": "10 rice + 1 stone"},
    11: {"expected_rice": 10, "expected_fm": 5,  "desc": "10 rice + 5 stones"},
    12: {"expected_rice": 25, "expected_fm": 5,  "desc": "25 rice + 5 wheat grains"},
    13: {"expected_rice": 25, "expected_fm": 10, "desc": "25 rice + 5 stones + 5 wheat"},
    14: {"expected_rice": 50, "expected_fm": 1,  "desc": "50 rice + 1 stone"},
    15: {"expected_rice": 50, "expected_fm": 10, "desc": "50 rice + 10 mixed foreign objects"},
}


def image_key(name: str) -> Optional[int]:
    try:
        return int(name.split(".")[0].strip())
    except (ValueError, IndexError):
        return None


def decode_data_uri(data_uri: str | None) -> bytes | None:
    if not data_uri or "," not in data_uri:
        return None
    try:
        return base64.b64decode(data_uri.split(",", 1)[1])
    except Exception:
        return None


def fm_class_counts(records: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for obj in records or []:
        cls = str(obj.get("class_name") or obj.get("class") or "unknown")
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def run_images(pipeline: RiceQualityPipeline, image_paths: List[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in image_paths:
        key = image_key(path.name)
        gt = GROUND_TRUTH.get(key, {}) if key is not None else {}
        try:
            result = pipeline.analyze(str(path), filename=path.name)
        except Exception as exc:
            records.append({"image": path.name, "image_key": key, "error": repr(exc)})
            print(f"  {key}: ERROR {exc!r}")
            continue

        annotated = decode_data_uri(result.get("annotated_image_base64"))
        if annotated and key is not None:
            (OUT_DIR / f"{key:02d}_annotated.jpg").write_bytes(annotated)

        gate = result.get("rice_gate") or {}
        sample = result.get("sample") or {}
        summary = result.get("summary") or {}
        fm_objects = result.get("foreign_matter") or []
        fm_counts = fm_class_counts(fm_objects)

        rice_detected = bool(result.get("rice_detected"))
        analysed = sample.get("analysed") or summary.get("total_rice_grains") or 0
        fm_count = sample.get("foreign_matter_detections")
        if fm_count is None:
            fm_count = summary.get("foreign_matter_count", len(fm_objects))

        expected_rice = gt.get("expected_rice")
        rice_diff = (analysed - expected_rice) if expected_rice is not None else None
        rice_status = "N/A"
        if rice_diff is not None:
            if rice_diff == 0:
                rice_status = "MATCH"
            elif rice_diff < 0:
                rice_status = "UNDER-DETECTED"
            else:
                rice_status = "OVER-DETECTED"

        if gt.get("expect_rice_absent"):
            if not rice_detected:
                rice_status = "CORRECTLY REJECTED (no rice)"
            else:
                rice_status = "FALSE RICE DETECTION"

        rec = {
            "image": path.name,
            "image_key": key,
            "description": gt.get("desc"),
            "success": result.get("success"),
            "rice_detected": rice_detected,
            "gate_status": gate.get("status"),
            "gate_rice_detections": gate.get("rice_detections"),
            "gate_total_detections": gate.get("total_detections"),
            "gate_foreign_matter_detections": gate.get("foreign_matter_detections"),
            "segmentation_method": (result.get("segmentation_info") or {}).get("segmentation_method_used"),
            "expected_rice": expected_rice,
            "detected_rice": analysed,
            "rice_difference": rice_diff,
            "rice_count_status": rice_status,
            "expected_foreign_matter": gt.get("expected_fm"),
            "detected_foreign_matter": fm_count,
            "foreign_matter_classes": fm_counts,
            "reference_source": summary.get("reference_source"),
            "reference_status": summary.get("reference_status"),
            "reference_data_status": summary.get("reference_data_status"),
            "reference_unit": summary.get("reference_unit"),
            "whole_count": summary.get("whole_count"),
            "broken_count": summary.get("broken_count"),
            "undetermined_count": summary.get("undetermined_count"),
            "broken_percent": summary.get("broken_percent"),
            "measurement_unit": summary.get("measurement_unit"),
            "calibration_mode": summary.get("calibration_mode"),
            "is_small_sample": summary.get("is_small_sample"),
            "grains": [
                {
                    "id": g.get("id"),
                    "bbox": g.get("bbox"),
                    "length_px": (g.get("geometry") or {}).get("length_pixels"),
                    "breadth_px": (g.get("geometry") or {}).get("breadth_pixels"),
                    "lb_ratio": (g.get("geometry") or {}).get("lb_ratio"),
                    "broken_label": (g.get("defects") or {}).get("broken", {}).get("broken_label"),
                }
                for g in (result.get("grains") or [])
            ],
            "annotated_saved": bool(annotated),
            "warnings": result.get("warnings"),
        }
        records.append(rec)
        print(
            f"  [{key:02d}] gate={gate.get('status')} rice={analysed} "
            f"(exp {expected_rice}) fm={fm_count} "
            f"fm_classes={fm_counts} ref={summary.get('reference_source')} "
            f"-> {rice_status}"
        )
    return records


def build_summary_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Rice + Foreign Objects — results1 validation summary")
    add("")
    add(f"Dataset: `{report['dataset']}`")
    add(f"Images: {report['image_count']}")
    add("")
    add("> **Method status.** Production pipeline is classical CV only:")
    add("> rice gate = classical heuristic, segmentation = classical CV cascade,")
    add("> foreign matter = heuristic fallback (no YOLO/Mask R-CNN weights present).")
    add("> Counts below are image-derived, not laboratory measurements.")
    add("")

    add("## 1. Rice gate — rice-present vs foreign-only scenes")
    add("")
    add("| # | Scene | Gate status | Rice detected | Expected rice | Result |")
    add("|---|-------|-------------|---------------|---------------|--------|")
    for rec in report["images"]:
        add(
            f"| {rec.get('image_key')} | {rec.get('description')} | "
            f"`{rec.get('gate_status')}` | {rec.get('detected_rice')} | "
            f"{rec.get('expected_rice')} | {rec.get('rice_count_status')} |"
        )
    add("")

    add("## 2. Foreign matter — separation from rice")
    add("")
    add("| # | Scene | FM detected | Expected FM | FM classes |")
    add("|---|-------|-------------|-------------|------------|")
    for rec in report["images"]:
        classes = rec.get("foreign_matter_classes") or {}
        cls_str = ", ".join(f"{k}={v}" for k, v in sorted(classes.items())) or "—"
        exp = rec.get("expected_foreign_matter")
        exp_str = str(exp) if exp is not None else "(not asserted)"
        add(
            f"| {rec.get('image_key')} | {rec.get('description')} | "
            f"{rec.get('detected_foreign_matter')} | {exp_str} | {cls_str} |"
        )
    add("")

    add("## 3. Whole/Broken (no explicit profile — sample-derived / undetermined)")
    add("")
    add("| # | Rice analysed | Reference source | Reference status | Whole | Broken | Undetermined | Broken % |")
    add("|---|-------------|----------------|------------------|-------|--------|------------|----------|")
    for rec in report["images"]:
        if not rec.get("rice_detected"):
            add(f"| {rec.get('image_key')} | 0 | — | — | — | — | — | — |")
            continue
        bp = rec.get("broken_percent")
        bp_str = f"{bp}" if bp is not None else "N/A"
        add(
            f"| {rec.get('image_key')} | {rec.get('detected_rice')} | "
            f"`{rec.get('reference_source')}` | `{rec.get('reference_status')}` | "
            f"{rec.get('whole_count')} | {rec.get('broken_count')} | "
            f"{rec.get('undetermined_count')} | {bp_str} |"
        )
    add("")

    add("## 4. Contract check")
    add("")
    add("- Rice-only scenes (1–6): rice gate must report RICE and count grains.")
    add("- Foreign-only scenes (7–9): rice gate must reject (NOT_RICE / NO_ANALYSABLE_RICE / NO_RICE_CLASS).")
    add("- Mixed scenes (10–15): stones/wheat must stay in the foreign-matter channel, never counted as rice.")
    add("- No reference profile supplied -> reference_source is sample_derived or undetermined (Tier 2/3), never a silently injected pixel profile.")
    add("")

    add("## 5. Limitations")
    add("")
    add("- No trained model weights are present; all outputs are heuristic/classical-CV.")
    add("- Whole/Broken without an explicit calibrated profile is sample-derived or undetermined.")
    add("- Foreign-matter classes are heuristic labels, not validated model predictions.")
    add("- Pixel measurements are scale-specific to each capture setup.")
    add("")
    return "\n".join(lines)


def main() -> int:
    if not IMAGES_DIR.is_dir():
        print(f"ERROR: image directory not found: {IMAGES_DIR}")
        return 1

    image_paths = sorted(
        (p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
        key=lambda p: image_key(p.name) or 0,
    )
    if not image_paths:
        print(f"ERROR: no images found in {IMAGES_DIR}")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing {len(image_paths)} images with RiceQualityPipeline ...")
    pipeline = RiceQualityPipeline()
    records = run_images(pipeline, image_paths)

    report: Dict[str, Any] = {
        "dataset": "test_images/2 rice and stones",
        "image_count": len(image_paths),
        "images": records,
        "notes": [
            "Classical CV production pipeline (no YOLO/Mask R-CNN weights present).",
            "Foreign-only scenes (7-9) are expected to be rejected by the rice gate.",
            "No reference profile supplied: Whole/Broken reference is sample-derived or undetermined.",
            "Image-derived counts are not laboratory measurements.",
        ],
    }

    report_path = OUT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = OUT_DIR / "validation_summary.md"
    summary_path.write_text(build_summary_markdown(report), encoding="utf-8")

    print()
    print(f"artifacts written to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
