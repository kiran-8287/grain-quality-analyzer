"""
Export utilities for Rice Quality Analysis results (JSON and CSV).
"""

import csv
import io
import json
from typing import Any, Dict


def export_result_json(result: Dict[str, Any]) -> str:
    """Export limited Phase 1 result as formatted JSON string."""
    summary = result.get("summary", {})
    grains = result.get("grains", [])

    limited_grains = []
    for g in grains:
        geom = g.get("geometry", {})
        length_val = geom.get("length_mm") if geom.get("length_mm") is not None else geom.get("length_pixels")
        breadth_val = geom.get("breadth_mm") if geom.get("breadth_mm") is not None else geom.get("breadth_pixels")
        limited_grains.append({
            "grain_id": g.get("id"),
            "confidence": g.get("confidence"),
            "confidence_label": g.get("confidence_label"),
            "length": length_val,
            "breadth": breadth_val,
            "lb_ratio": geom.get("lb_ratio"),
        })

    clean_result = {
        "total_rice_grains": summary.get("total_rice_grains", len(grains)),
        "measurement_unit": summary.get("measurement_unit", "pixels"),
        "calibration_mode": summary.get("calibration_mode", result.get("calibration", {}).get("mode", "none")),
        "grains": limited_grains,
    }
    return json.dumps(clean_result, indent=2)


def export_result_csv(result: Dict[str, Any]) -> str:
    """
    Export limited grain-level data as CSV format.

    Current visible UI fields only:
    grain_id, confidence, length, breadth, L_over_B
    """
    output = io.StringIO()
    writer = csv.writer(output)

    summary = result.get("summary", {})
    writer.writerow(["# Rice Grain Segmentation Export"])
    writer.writerow(["# Sample Summary"])
    writer.writerow(["Parameter", "Value", "Unit / Notes"])
    writer.writerow(["Total Rice Grains", summary.get("total_rice_grains", 0), "count"])
    writer.writerow(["Measurement Unit", summary.get("measurement_unit", "pixels"), ""])
    writer.writerow(["Calibration Mode", summary.get("calibration_mode", result.get("calibration", {}).get("mode", "none")), ""])
    writer.writerow([])

    writer.writerow(["# Per-Grain Data"])
    grain_headers = [
        "grain_id",
        "confidence",
        "length",
        "breadth",
        "L_over_B",
    ]
    writer.writerow(grain_headers)

    grains = result.get("grains", [])
    for g in grains:
        geom = g.get("geometry", {})
        length_val = geom.get("length_mm") if geom.get("length_mm") is not None else geom.get("length_pixels", "")
        breadth_val = geom.get("breadth_mm") if geom.get("breadth_mm") is not None else geom.get("breadth_pixels", "")
        lb_val = geom.get("lb_ratio", "")

        writer.writerow([
            g.get("id"),
            g.get("confidence", ""),
            length_val,
            breadth_val,
            lb_val,
        ])

    return output.getvalue()
