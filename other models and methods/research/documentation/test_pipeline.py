#!/usr/bin/env python3
"""
Test script to run the RiceQualityPipeline on all testing images
and save annotated output images.
"""

import base64
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from ml.segmentation.pipeline import RiceQualityPipeline


def main():
    test_dir = Path("A:/grain/testing_images")
    output_dir = Path("A:/grain/test_image_results")
    output_dir.mkdir(exist_ok=True)

    pipeline = RiceQualityPipeline()

    image_files = sorted(test_dir.glob("*"))
    print(f"Found {len(image_files)} test images")

    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        try:
            result = pipeline.analyze(
                image_source=str(img_path),
                filename=img_path.name,
                grade="grade_a",
            )

            if not result.get("success", False):
                print(f"  ERROR: {result.get('error', 'Unknown error')}")
                continue

            # Save annotated image
            annotated_b64 = result.get("annotated_image_base64", "")
            if annotated_b64 and annotated_b64.startswith("data:image/jpeg;base64,"):
                raw_b64 = annotated_b64.split(",", 1)[1]
                img_bytes = base64.b64decode(raw_b64)
                
                output_name = img_path.stem + "_annotated.jpg"
                output_path = output_dir / output_name
                with open(output_path, "wb") as f:
                    f.write(img_bytes)
                
                print(f"  Saved: {output_path.name}")
                print(f"  Rice detected: {result.get('rice_detected')}")
                print(f"  Grains: {result.get('summary', {}).get('total_rice_grains', 0)}")
                print(f"  Method: {result.get('segmentation_info', {}).get('segmentation_method_used', 'unknown')}")
                print(f"  Foreign matter: {result.get('summary', {}).get('foreign_matter_count', 0)}")
            else:
                print(f"  WARNING: No annotated image in result")

        except Exception as e:
            print(f"  EXCEPTION: {e}")

    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()