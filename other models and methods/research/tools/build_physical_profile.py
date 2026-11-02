"""
Physical whole-kernel reference profile builder.

Accepts measured whole-kernel lengths (and optional breadth/L-B values)
and writes a production-style GrainProfile JSON.

The documented measurement protocol is:

    10 whole kernels
    x 3 sets
    = 30 measurements
    -> averaged

This script does NOT fabricate physical values.
It only packages user-supplied real measurements into a profile.

Usage:
    python scripts/build_physical_profile.py --output grain_profiles/my_rice.json \
        --lengths 6.7 6.8 6.9 6.6 7.0 6.5 6.9 7.1 6.8 6.7 \
        --breadths 2.0 2.1 1.9 2.0 2.2 1.8 2.0 2.1 1.9 2.0 \
        --source "operator-lab-01" \
        --variety "PR-106" \
        --notes "10 whole kernels x 3 sets = 30 measurements averaged."

    # Or from a JSON file:
    python scripts/build_physical_profile.py --input measurements.json --output profile.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.quality.profiles import GrainProfile


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float], mean_val: float) -> float:
    return math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))


def build_profile(
    lengths_mm: List[float],
    breadths_mm: Optional[List[float]] = None,
    source: str = "user_measured",
    variety: Optional[str] = None,
    measurement_method: str = "FSSAI-style 10x3 manual measurement",
    measurement_date: Optional[str] = None,
    operator: Optional[str] = None,
    sample_identifier: Optional[str] = None,
    notes: Optional[str] = None,
    production_eligible: bool = False,
    profile_name: str = "physical_rice_profile",
) -> Dict[str, Any]:
    if not lengths_mm:
        raise ValueError("At least one length measurement is required.")
    if any(v <= 0 or not math.isfinite(v) for v in lengths_mm):
        raise ValueError("All length values must be positive finite numbers.")

    if breadths_mm is not None:
        if len(breadths_mm) != len(lengths_mm):
            raise ValueError("breadths_mm must have the same count as lengths_mm.")
        if any(v <= 0 or not math.isfinite(v) for v in breadths_mm):
            raise ValueError("All breadth values must be positive finite numbers.")

    n = len(lengths_mm)
    mean_len = _mean(lengths_mm)
    std_len = _std(lengths_mm, mean_len)
    cv_len = std_len / mean_len if mean_len else 0.0

    lb_ratio = None
    if breadths_mm is not None and all(b > 0 for b in breadths_mm):
        mean_brd = _mean(breadths_mm)
        lb_ratio = mean_len / mean_brd if mean_brd > 0 else None

    profile = GrainProfile(
        profile_name=profile_name,
        reference_unit="mm",
        whole_kernel_length=round(mean_len, 3),
        whole_kernel_breadth=round(_mean(breadths_mm), 3) if breadths_mm else None,
        whole_kernel_lb_ratio=round(lb_ratio, 3) if lb_ratio is not None else None,
        source=source,
        reference_count=n,
        notes=notes,
        data_status="Measured",
        production_eligible=production_eligible,
        variety=variety,
        measurement_method=measurement_method,
        measurement_date=measurement_date,
        operator=operator,
        sample_identifier=sample_identifier,
    )

    is_valid, msg = profile.validate()
    if not is_valid:
        raise ValueError(f"Profile validation failed: {msg}")

    return {
        "profile": profile.to_dict(),
        "statistics": {
            "count": n,
            "mean_length_mm": round(mean_len, 3),
            "std_length_mm": round(std_len, 3),
            "cv_length": round(cv_len, 4),
            "mean_breadth_mm": round(_mean(breadths_mm), 3) if breadths_mm else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a measured physical whole-kernel reference profile from real measurements."
    )
    parser.add_argument("--output", "-o", required=True, help="Output profile JSON path")
    parser.add_argument("--input", "-i", help="Input JSON file with measurements")
    parser.add_argument("--lengths", nargs="+", type=float, help="Whole-kernel lengths in mm")
    parser.add_argument("--breadths", nargs="+", type=float, help="Whole-kernel breadths in mm")
    parser.add_argument("--source", default="user_measured", help="Measurement source/operator")
    parser.add_argument("--variety", help="Rice variety label")
    parser.add_argument("--measurement-method", default="FSSAI-style 10x3 manual measurement")
    parser.add_argument("--measurement-date", help="Measurement date")
    parser.add_argument("--operator", help="Operator name")
    parser.add_argument("--sample-identifier", help="Sample identifier")
    parser.add_argument("--notes", help="Free-text notes")
    parser.add_argument("--production-eligible", action="store_true", help="Mark as production eligible")

    args = parser.parse_args()

    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        lengths = [float(x) for x in data.get("lengths", [])]
        breadths = [float(x) for x in data.get("breadths", [])] if data.get("breadths") else None
    elif args.lengths:
        lengths = args.lengths
        breadths = args.breadths
    else:
        parser.error("Either --input or --lengths is required.")

    try:
        result = build_profile(
            lengths_mm=lengths,
            breadths_mm=breadths,
            source=args.source,
            variety=args.variety,
            measurement_method=args.measurement_method,
            measurement_date=args.measurement_date,
            operator=args.operator,
            sample_identifier=args.sample_identifier,
            notes=args.notes,
            production_eligible=args.production_eligible,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Profile written to: {out_path}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
