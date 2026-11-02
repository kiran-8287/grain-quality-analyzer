"""
Standards comparison engine.

Compares image-derived measurements against configured historical/reference
raw rice limits without asserting current-season verification.

Creates TWO separate outputs:
A. Image-based standard screening (parameter-by-parameter comparison)
B. Formal official grade status (honest about limitations)

Does NOT invent a formal Government of India grade from image features.
"""

import logging
from typing import Dict, List, Optional

from ml.config import load_standards

logger = logging.getLogger(__name__)


def compare_with_standards(
    sample_stats: Dict,
    total_count: int,
    calibration_mode: str = "none",
    grade: str = "grade_a",
    quality_tier: Optional[str] = None,
    minimum_sample_size: int = 30,
) -> Dict:
    """
    Compare image-derived measurements with official standards.
    
    Args:
        sample_stats: Dict with defect percentages and counts
        total_count: Total analysed grain count
        calibration_mode: 'aruco', 'manual', or 'none'
        grade: 'grade_a' or 'common'
        
    Returns:
        Standards comparison result
    """
    try:
        standards = load_standards()
    except FileNotFoundError:
        return {
            "screening": {},
            "official_grade": {
                "status": "Standards file not found",
                "message": "Cannot perform standards comparison.",
            },
            "warnings": ["Standards JSON file not found. Supply it manually."],
        }
    
    grade_limits = standards.get(grade, standards.get("grade_a", {}))
    assessment_map = standards.get("image_assessment_mapping", {})
    
    # A. Image-based standard screening
    screening = {}
    warnings = []
    
    param_mappings = [
        ("broken", "broken_percent", "broken", "broken_count", "broken_analyzed_count"),
        ("damaged_slightly_damaged", "damaged_percent", "damaged_slightly_damaged", "damaged_count", "damaged_analyzed_count"),
        ("discoloured", "discoloured_percent", "discoloured", "discoloured_count", "discoloured_analyzed_count"),
        ("chalky", "chalky_percent", "chalky", "chalky_count", "chalky_analyzed_count"),
        ("red", "red_percent", "red", "red_count", "red_analyzed_count"),
        ("dehusked", "dehusked_percent", "dehusked", "dehusked_count", "dehusked_analyzed_count"),
        ("foreign_matter", "foreign_matter_percent", "foreign_matter", "foreign_matter_count", "foreign_matter_analyzed_count"),
        ("admixture_of_lower_class", "admixture_percent", "admixture_of_lower_class", "admixture_count", "admixture_analyzed_count"),
    ]

    sample_too_small = total_count < minimum_sample_size
    quality_unreliable = quality_tier in ("POOR", "UNRELIABLE")

    for param_key, stat_key, assessment_key, count_key, analyzed_key in param_mappings:
        limit_info = grade_limits.get(param_key, {})
        max_percent = limit_info.get("max_percent")
        basis = limit_info.get("basis", "weight")
        
        observed = sample_stats.get(stat_key)
        detected_count = int(sample_stats.get(count_key, 0) or 0)
        analyzed_count = int(
            sample_stats.get(analyzed_key, total_count)
            or 0
        )
        observed_fraction = (
            round(detected_count / analyzed_count, 4) if analyzed_count else None
        )
        assess_info = assessment_map.get(assessment_key, {})
        image_assessable = assess_info.get("image_assessable", False)
        limitation = assess_info.get("limitation", "")
        
        if image_assessable is not True:
            status = "NOT ASSESSABLE"
            difference = None
        elif max_percent is None:
            status = "NOT APPLICABLE"
            difference = None
        elif sample_too_small or quality_unreliable:
            status = "NOT DETERMINABLE"
            difference = None
            if sample_too_small:
                limitation = (
                    f"Only {total_count} grains were analyzed; the project screening "
                    f"threshold is {minimum_sample_size} grains (engineering threshold, "
                    "not an official Government requirement)."
                )
            if quality_unreliable:
                limitation = (
                    f"Image quality is {quality_tier}; reliable compliance screening is suppressed."
                )
        elif observed is None:
            status = "NOT ASSESSABLE"
            difference = None
        elif observed <= max_percent:
            status = "WITHIN REFERENCE LIMIT"
            difference = round(max_percent - observed, 2)
        else:
            status = "EXCEEDS REFERENCE LIMIT"
            difference = round(observed - max_percent, 2)
        
        screening[param_key] = {
            "observed_percent": round(observed, 2) if observed is not None else None,
            "detected_count": detected_count,
            "analyzed_count": analyzed_count,
            "observed_fraction": observed_fraction,
            "observed_value": (
                f"{detected_count} / {analyzed_count} "
                f"({observed * 1.0:.2f}%)"
                if observed is not None and analyzed_count
                else f"{detected_count} / {analyzed_count} (not measurable)"
            ),
            "reference_limit_percent": max_percent,
            "reference_limit": f"{max_percent}% max" if max_percent is not None else "N/A",
            "official_basis": basis,
            "image_basis": assess_info.get("image_basis", "count"),
            "basis": assess_info.get("image_basis", "count"),
            "status": status,
            "difference": difference,
            "limitation": limitation,
            "note": "Image-based screening; not an official laboratory test.",
        }
    
    # B. Formal official grade status
    assessable_params = [s for s in screening.values() 
                        if s["status"] in ("WITHIN REFERENCE LIMIT", "EXCEEDS REFERENCE LIMIT")]
    exceeding = [k for k, s in screening.items() if s["status"] == "EXCEEDS REFERENCE LIMIT"]
    
    reasons_cannot_grade = [
        "Laboratory/weight-based requirements cannot be verified from image alone.",
        "Official dehusked test requires chemical staining — not reproduced by image analysis.",
    ]
    
    if sample_too_small:
        reasons_cannot_grade.append(
            f"Small sample size ({total_count} grains); project screening threshold: "
            f"{minimum_sample_size} grains (engineering threshold, not an official "
            "Government requirement)."
        )

    if quality_unreliable:
        reasons_cannot_grade.append(
            f"Image quality is {quality_tier}; reliable official compliance conclusions are suppressed."
        )
    
    if calibration_mode == "none":
        reasons_cannot_grade.append(
            "No metric calibration — length/breadth measurements are in pixels."
        )
    
    if sample_too_small or quality_unreliable:
        image_verdict = "Not determinable from this sample"
    elif not exceeding and assessable_params:
        image_verdict = "Meets image-assessable reference limits"
    elif exceeding:
        image_verdict = "Exceeds one or more image-assessable reference limits"
    else:
        image_verdict = "Insufficient data for screening"
    
    official_grade = {
        "formal_grade_determined": False,
        "status": (
            "Not determinable from this sample"
            if sample_too_small or quality_unreliable
            else "Formal grade not determined"
        ),
        "message": "Formal Grade A/Common determination is not established from this image alone.",
        "reason": " ".join(reasons_cannot_grade),
        "reasons": reasons_cannot_grade,
        "image_screening_verdict": image_verdict,
        "exceeding_parameters": exceeding,
        "disclaimer": "Observed image fractions are not official weight-based measurements.",
        "note": "This is NOT a government certification. "
                "Image-based analysis provides screening estimates only.",
    }
    
    # Footnotes from standard
    footnotes = standards.get("footnotes", [])
    
    return {
        "screening": screening,
        "official_grade": official_grade,
        "standard_reference": {
            "source": standards.get("_metadata", {}).get("source_title", ""),
            "season": standards.get("_metadata", {}).get("season", ""),
            "display_label": standards.get("_metadata", {}).get("display_label", "Reference standard"),
            "verification_status": standards.get("_metadata", {}).get("verification_status", "unverified"),
            "grade_profile": grade,
        },
        "footnotes": footnotes,
        "warnings": warnings,
    }
