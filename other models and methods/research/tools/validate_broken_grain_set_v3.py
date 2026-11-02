"""
Whole-vs-Broken validation harness for test_images/3 broken_grain — results_3.

Runs the production RiceQualityPipeline over the controlled broken-grain
dataset in two reference modes:

    no_profile    -> profile=None  (Tier 2 sample-derived / Tier 3 undetermined)
    with_profile  -> profile="default_rice" (Tier 1 explicit pixel profile)

and writes into test_images/3 broken_grain/results/results_3/:

    validation_report.json
    validation_summary.md
    no_profile/
    with_profile/

The report explicitly records:
    reference source
    reference data status
    calibration status
    reference unit
    reference value
    count / whole / broken / undetermined
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.segmentation.pipeline import RiceQualityPipeline

BROKEN_ROOT = PROJECT_ROOT / "test_images" / "3 broken_grain"
IMAGES_DIR = BROKEN_ROOT / "images"
OUT_DIR = BROKEN_ROOT / "results" / "results_3"

PROFILE_NAME = "default_rice"

GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "B01": {"expected_total": 1,  "expected_broken": 0,  "source": "filename"},
    "B02": {"expected_total": 1,  "expected_broken": 1,  "source": "filename"},
    "B03": {"expected_total": 5,  "expected_broken": 0,  "source": "filename"},
    "B04": {"expected_total": 5,  "expected_broken": 5,  "source": "filename"},
    "B05": {"expected_total": 11, "expected_broken": 1,  "source": "filename"},
    "B06": {"expected_total": 2,  "expected_broken": 2,  "source": "filename"},
    "B07": {"expected_total": 12, "expected_broken": 2,  "source": "filename"},
    "B08": {"expected_total": 5,  "expected_broken": 5,  "source": "filename"},
    "B09": {"expected_total": 15, "expected_broken": 5,  "source": "filename"},
    "B10": {"expected_total": 20, "expected_broken": 10, "source": "filename"},
    "B11": {"expected_total": 30, "expected_broken": 5,  "source": "filename"},
    "B12": {"expected_total": 35, "expected_broken": 10, "source": "filename"},
    "B13": {"expected_total": 12, "expected_broken": 12, "source": "visual_verification"},
}

MODES = (("no_profile", None), ("with_profile", PROFILE_NAME))


def image_key(name: str) -> str:
    return name.split(" ")[0].split(".")[0].strip()


def decode_data_uri(data_uri: str | None) -> bytes | None:
    if not data_uri or "," not in data_uri:
        return None
    try:
        import base64
        return base64.b64decode(data_uri.split(",", 1)[1])
    except Exception:
        return None


def per_grain_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for grain in result.get("grains") or []:
        defects = grain.get("defects") or {}
        broken = defects.get("broken") or {}
        geom = grain.get("geometry") or {}
        rows.append({
            "grain_id": grain.get("id"),
            "status": broken.get("broken_label"),
            "effective_length": broken.get("effective_length"),
            "length_ratio": broken.get("length_ratio"),
            "breadth_pixels": geom.get("breadth_pixels"),
            "lb_ratio": geom.get("lb_ratio"),
            "area_pixels": geom.get("area_pixels"),
            "measurement_quality": geom.get("measurement_quality"),
            "bbox": grain.get("bbox"),
        })
    return rows


def run_mode(pipeline: RiceQualityPipeline, image_paths: List[Path], mode: str, profile) -> List[Dict[str, Any]]:
    mode_dir = OUT_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for path in image_paths:
        key = image_key(path.name)
        gt = GROUND_TRUTH.get(key, {})
        try:
            result = pipeline.analyze(str(path), filename=path.name, profile=profile)
        except Exception as exc:
            records.append({"image": path.name, "image_key": key, "mode": mode, "error": repr(exc)})
            continue

        annotated = decode_data_uri(result.get("annotated_image_base64"))
        if annotated:
            (mode_dir / f"{key}_annotated.jpg").write_bytes(annotated)

        summary = result.get("summary") or {}
        sample = result.get("sample") or {}
        gate = result.get("rice_gate") or {}

        expected_total = gt.get("expected_total")
        detected = sample.get("analysed")
        records.append({
            "image": path.name,
            "image_key": key,
            "mode": mode,
            "success": result.get("success"),
            "rice_detected": result.get("rice_detected"),
            "gate_status": gate.get("status"),
            "gate_rice_detections": gate.get("rice_detections"),
            "gate_total_detections": gate.get("total_detections"),
            "segmentation_method": (result.get("segmentation_info") or {}).get("segmentation_method_used"),
            "expected_total": expected_total,
            "expected_broken": gt.get("expected_broken"),
            "expected_source": gt.get("source"),
            "detected_masks": sample.get("total_detected"),
            "analysed_grains": detected,
            "detected_total": detected,
            "uncertain_grains": sample.get("uncertain"),
            "rejected_grains": sample.get("rejected"),
            "count_difference": (detected - expected_total) if (detected is not None and expected_total is not None) else None,
            "reference_source": summary.get("reference_source"),
            "reference_status": summary.get("reference_status"),
            "reference_data_status": summary.get("reference_data_status"),
            "reference_profile_name": summary.get("reference_profile_name"),
            "reference_length": summary.get("whole_reference_length"),
            "reference_unit": summary.get("reference_unit"),
            "reference_production_eligible": summary.get("reference_production_eligible"),
            "reference_calibration_required": summary.get("reference_calibration_required"),
            "reference_calibration_status": summary.get("reference_calibration_status"),
            "reference_count": summary.get("reference_count"),
            "reference_explanation": summary.get("reference_explanation"),
            "measurement_unit": summary.get("measurement_unit"),
            "calibration_mode": summary.get("calibration_mode"),
            "whole_count": summary.get("whole_count"),
            "broken_count": summary.get("broken_count"),
            "undetermined_count": summary.get("undetermined_count"),
            "broken_percent": summary.get("broken_percent"),
            "whole_percent": summary.get("whole_percent"),
            "small_sample": summary.get("is_small_sample"),
            "grains": per_grain_records(result),
        })
        print(
            f"  [{mode}] {key}: detected={detected} expected={expected_total} "
            f"whole={summary.get('whole_count')} broken={summary.get('broken_count')} "
            f"undet={summary.get('undetermined_count')} "
            f"broken%={summary.get('broken_percent')} ref={summary.get('reference_source')} "
            f"data_status={summary.get('reference_data_status')}"
        )
    return records


def build_summary_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    add = lines.append

    add("# Whole vs Broken — results_3 validation summary")
    add("")
    add(f"Dataset: `{report['dataset']}`")
    add(f"Explicit profile used: `{report['profile_used']}`")
    add("")
    add(f"> **Profile scale caveat.** {report['profile_scale_note']}")
    add("")
    add("## 1. Segmentation table (count vs physical ground truth)")
    add("")
    add("| Image | Expected physical | Detected masks | Analysed grains | Difference | Status |")
    add("|------|--------------------|----------------|-----------------|------------|--------|")

    with_profile = report["modes"].get("with_profile", [])
    for rec in with_profile:
        exp = rec.get("expected_total")
        det = rec.get("detected_total")
        diff = rec.get("count_difference")
        if det is None:
            status = "ERROR"
        elif diff == 0:
            status = "MATCH"
        else:
            status = "UNDER-DETECTED" if diff < 0 else "OVER-DETECTED"
        add(
            f"| {rec.get('image_key')} | {exp} | {rec.get('detected_masks')} | "
            f"{det} | {diff} | {status} |"
        )
    add("")
    add("## 2. Primary benchmark — B13 (all broken)")
    add("")
    for mode in ("no_profile", "with_profile"):
        rec = report.get("primary_benchmark", {}).get(mode)
        add(f"### mode `{mode}`")
        add("")
        if not rec:
            add("(no record)")
            add("")
            continue
        add(f"- detected masks: **{rec.get('detected_total')}**")
        add(f"- reference_source: `{rec.get('reference_source')}`")
        add(f"- reference_status: `{rec.get('reference_status')}`")
        add(f"- reference_data_status: `{rec.get('reference_data_status')}`")
        add(f"- reference_production_eligible: `{rec.get('reference_production_eligible')}`")
        add(f"- whole_kernel_reference_length: {rec.get('reference_length')} {rec.get('reference_unit')}")
        add(f"- whole_count: **{rec.get('whole_count')}**")
        add(f"- broken_count: **{rec.get('broken_count')}**")
        add(f"- undetermined_count: **{rec.get('undetermined_count')}**")
        add(f"- broken_percent: **{rec.get('broken_percent')}**")
        add("")

    add("## 3. Per-image Whole/Broken detail (with_profile)")
    add("")
    add("| Image | Whole | Broken | Undetermined | Broken % | Reference | Data status |")
    add("|-------|-------|--------|--------------|----------|-----------|------------|")
    for rec in with_profile:
        add(
            f"| {rec.get('image_key')} | {rec.get('whole_count')} | "
            f"{rec.get('broken_count')} | {rec.get('undetermined_count')} | "
            f"{rec.get('broken_percent')} | {rec.get('reference_source')} | "
            f"{rec.get('reference_data_status')} |"
        )
    add("")
    add("## 4. Contract check")
    add("")
    add("- B13 + explicit legacy profile: 12 masks, 0 whole, 12 broken, 0 undetermined, 100% broken, data_status=Proxy")
    add("- B13 with no reference: 12 masks, 0 whole, 0 broken, 12 undetermined")
    add("- Pixel legacy profile is never automatically injected")
    add("")
    return "\n".join(lines)


def main() -> int:
    if not IMAGES_DIR.is_dir():
        print(f"ERROR: image directory not found: {IMAGES_DIR}")
        return 1

    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = RiceQualityPipeline()
    report: Dict[str, Any] = {
        "dataset": "test_images/3 broken_grain",
        "profile_used": PROFILE_NAME,
        "profile_scale_note": (
            "Pixel-unit profile, scale-specific to the controlled capture setup that "
            "produced grain_profiles/default_rice.json (154.1 px whole-kernel reference). "
            "Not a universal physical measurement."
        ),
        "images": [p.name for p in image_paths],
        "modes": {},
    }

    for mode, profile in MODES:
        print(f"=== mode: {mode} (profile={profile}) ===")
        report["modes"][mode] = run_mode(pipeline, image_paths, mode, profile)

    report["primary_benchmark"] = {}
    for mode in report["modes"]:
        for rec in report["modes"][mode]:
            if rec.get("image_key") == "B13":
                report["primary_benchmark"][mode] = rec

    report_path = OUT_DIR / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = OUT_DIR / "validation_summary.md"
    summary_path.write_text(build_summary_markdown(report), encoding="utf-8")

    print()
    print(f"artifacts written to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
