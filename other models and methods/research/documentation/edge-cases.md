# Edge Case Handling and Robustness

This document details how the rice quality analysis system handles complex real-world edge cases.

---

## 1. Grain Sample Size Range ($N = 0$ to $N > 5000$)

| Sample Count | System Behavior | User Interface Indication |
|---|---|---|
| **$N = 0$** | Halts processing early. No fake grain measurements are generated. | Displays `"No rice grains detected. Please upload an image containing rice grains."` |
| **$N = 1$** | Fully segments and analyzes the single grain. Computes all geometric dimensions, classifications, and color profiles. Broken kernel distribution reference marked as `undetermined`. | Shows `"Sample size: 1 grain"` warning: *"Individual-grain analysis is possible, but sample-level quality percentages are not representative of a larger rice lot."* Percentage labeled as `"Observed sample fraction"`. |
| **$N = 2$ to $9$** | Segments each grain, assigns individual IDs `#1..#N`, computes geometry and defect labels. Broken status marked undetermined if whole reference cannot be established. | Shows small-sample advisory banner: *"Below the project screening threshold (30 grains; engineering threshold, not an official requirement)."* |
| **$N = 10$ to $29$** | Segments each grain. Geometry outlier diagnostic may run, but lower-class admixture remains unsupported. | Small-sample advisory displayed; official lot status is not determinable. |
| **$N = 30$ to $999$** | The project screening threshold permits image-based screening statuses; a geometry outlier remains diagnostic only and does not establish admixture or official lot quality. | Full dashboard with explicit model/proxy provenance and no claim of official compliance. |
| **$N \ge 1000$** | High throughput memory-safe processing, vectorized feature extraction, and cached model inference. | Full dashboard with processing time benchmark. |

---

## 2. No Fixed 12-Megapixel Gate

The system **does not reject images with less than 12 megapixels**.
- Total image resolution is uncoupled from grain resolution. A 2 MP macro close-up of 5 grains provides hundreds of pixels across each grain, whereas a 12 MP camera from 2 meters away may yield grains occupying only 10 pixels.
- Quality is evaluated **after segmentation** via:
   1. Median grain pixel area
   2. Bounding box width/height
   3. Mask-scoped Laplacian blur variance (engineering threshold; calibration against human review is still required)
   4. Illumination uniformity

---

## 3. Touching and Overlapping Grains
- **Detection**: Merged instances are detected using contour concavity, aspect ratio anomalies, and area ratio $> 2.5 \times \text{median area}$.
- **Splitting**: Watershed transform on Euclidean distance transform peaks splits touching grains along natural pinch points.
- **Uncertain Instances**: When a merged blob cannot be split with high confidence, it is tagged as `segmentation_quality: "uncertain"` and flagged in the UI rather than counted as a single giant grain.

---

## 4. Arbitrary Backgrounds
- Thresholding utilizes adaptive Gaussian thresholding and Otsu inverted background comparison.
- Evaluates foreground-to-background ratio to automatically invert polarity on black vs white vs textured backgrounds.

---

## 5. Calibration vs No Calibration
- **Mode A (Calibrated)**: ArUco marker detection or manual scale factor provided. Converts length, breadth, and area to metric millimeters ($mm$).
- **Mode B (Uncalibrated)**: No reference marker detected. Measurements reported strictly in pixels ($px$). L/B ratio is reported normally since it is dimensionless. The UI clearly displays: *"Metric calibration unavailable — measurements shown in pixels."*

---

## 6. Whole/Broken Reference and Measurement Semantics

- **Project binary threshold**: Length ratio $< 0.75$ against the whole-kernel reference is classified as broken; ratio $\ge 0.75$ is classified as NOT broken / WHOLE. The $0.75$ threshold is a project rule, not a formal standards category boundary.
- **Physical measured profile (preferred production route)**: Unit = $mm$, source = measured, `production_eligible = true`. Requires `calibration_validity == "valid"` before any metric comparison. Never compare $mm$ reference against $px$ measurements without conversion.
- **Same-image sample-derived reference**: Heuristic/proxy derived from intact-looking candidates in the same image using upper-population clustering. Does **not** use `mean(all grains)` or `max(all grains)` because broken grains can bias either estimate. Returns `data_status = "Sample-Derived"` and `production_eligible = false`.
- **Legacy pixel profile**: Demo-only proxy, unit = $px$, `data_status = "Proxy"`, `production_eligible = false`. Never automatically injected. Only used when explicitly requested.
- **All-broken / no-reference cases**: When no valid reference exists (all broken, no profile, no calibration), the result is `undetermined` with a clear explanation. The system does **not** fabricate a whole-kernel estimate from broken pieces.
- **Calibration validity states**: `valid`, `invalid`, or `unavailable`. A physical profile is usable only when validity is `valid`. Invalid/unavailable calibration yields `NOT_DETERMINABLE` with a clear reason.
- **Broken percentage**: Reported **by grain count**, not by mass. Regulatory specifications may use mass-based measurements; the image-based count should not be presented as mass-based regulatory compliance.
- **No fabrication**: No physical $mm$ reference values are invented. If a verified physical measurement is needed, it should be established from multiple whole kernels (e.g., 10 whole kernels x 3 sets = 30 measurements averaged) and recorded as a measured profile.
- **Physical profile builder**: `scripts/build_physical_profile.py` packages real user-supplied measurements into a measured mm profile. It does not invent values.
- **Metric measurement path**: When calibration validity is `valid`, geometry is computed in both pixels and mm (`length_mm`, `breadth_mm`, `effective_length_mm`). The pipeline reports both where available.
- **Unit safety**: Pixel and mm measurements are never mixed in the same ratio. Invalid unit comparisons fail safely with `NOT_DETERMINABLE`.
- **Homography status**: A homography is computed when >= 4 ArUco markers are detected, but it is not yet connected to metric grain measurement. Current metric measurement uses `pixels_per_mm` scaling.
