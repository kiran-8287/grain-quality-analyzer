==================================================
GRAIN QUALITY ANALYZER — PROJECT CONTEXT
==================================================

# 1. PROJECT PURPOSE

Grain Quality Analyzer is an image-based rice grain inspection application.

The long-term objective is to:

- detect rice grains
- segment individual rice grains
- measure grain geometry
- classify Whole vs Broken grains
- detect additional rice defects
- detect foreign matter
- provide image-based quality statistics
- eventually support standards/reference screening
- provide a user-facing web dashboard
- export analysis results

However, development is intentionally incremental.

The CURRENT active development priority is:

1. Rice grain detection
2. Accurate individual-grain segmentation/masking
3. Grain geometry
4. Whole vs Broken classification

Do NOT expand development into additional defect classifiers or standards work unless explicitly requested.

==================================================
# 2. CURRENT ARCHITECTURE

Frontend:
React + TypeScript + Vite + Tailwind

Backend:
FastAPI + Python

Runtime flow:

User
  ↓
Upload image OR camera capture
  ↓
FastAPI API
  ↓
Job manager
  ↓
RiceQualityPipeline
  ↓
Rice / non-rice gate
  ↓
Individual grain segmentation
  ↓
Geometry measurement
  ↓
Whole vs Broken classification
  ↓
Sample summary
  ↓
Annotated image
  ↓
Frontend dashboard
  ↓
Save / export

==================================================
# 3. CURRENT PRODUCTION METHODS

IMPORTANT:
The CURRENT production pipeline is NOT YOLO.

Rice gate:
Classical computer vision / heuristic image analysis

Segmentation:
Classical computer vision segmentation

Geometry:
Contour/mask-based geometric image processing

Whole vs Broken:
Rule-based geometric classification

YOLO:
NOT CURRENT PRODUCTION

Previous YOLO experiments/checkpoints were intentionally removed from the repository during cleanup.

Do not restore or activate YOLO automatically.

A future YOLO experiment must be treated as a NEW explicitly requested experiment and must not silently replace the current baseline.

==================================================
# 4. ACTIVE ML / COMPUTER VISION MODULES

Current important runtime modules:

ml/segmentation/
    inference.py
    pipeline.py
    postprocessing.py
    preprocessing.py
    rice_gate.py
    segmentation.py

ml/quality/
    geometry.py
    calibration.py
    profiles.py

ml/config.py

The current goal is to keep these modules stable while improving the segmentation and Whole/Broken capabilities carefully.

==================================================
# 5. RICE / NON-RICE GATE

The rice gate is currently classical CV.

Its responsibility is to answer:

"Does this image contain rice-like grain material?"

It should distinguish:

- rice present
- rice absent / non-rice scene

The gate must not be confused with the grain segmentation stage.

Foreign/non-rice material must not automatically become a rice grain.

Preserve the current gate behavior unless a regression test or controlled experiment demonstrates that a change is necessary.

==================================================
# 6. RICE GRAIN SEGMENTATION

The current production segmentation method is classical CV.

The segmentation output represents individual detected grain instances.

Each grain instance contains information such as:

- grain ID
- bounding box
- centroid
- mask polygon
- geometry
- classification information

The core requirement is:

ONE PHYSICAL RICE GRAIN
        ↓
ONE CORRESPONDING GRAIN INSTANCE / MASK

Avoid:

- false background objects
- foreign objects interpreted as rice
- one physical grain becoming multiple masks
- multiple physical grains becoming one mask

Important project assumption for the current controlled testing setup:

Rice grains used in the intended test scenes are placed so that grains should NOT touch or overlap.

Therefore, touching/overlapping separation is not currently the primary problem to solve.

The priority is accurate detection and clean masking of separated grains.

==================================================
# 7. GEOMETRY

Geometry is computed from the segmented grain mask.

Current measurements include:

- length
- breadth
- L/B ratio
- area
- perimeter
- centroid
- orientation
- bounding box
- solidity
- related geometry quality information

The system can represent measurements in pixels.

Physical millimetre measurement requires calibration.

Do not assume pixel measurements are physically universal across different camera distances.

==================================================
# 8. CALIBRATION

Calibration support exists.

The project supports calibration/reference handling including ArUco-based calibration.

Without calibration:
    measurements are generally reported in pixels.

Important:

Pixel size depends on imaging scale.

A whole rice grain can occupy very different pixel lengths when the camera distance changes.

Therefore:

A fixed pixel reference MUST NOT be treated as a universal physical rice-grain size.

Calibration validity states:
    valid        - marker detected, geometry valid, scale finite/positive, within sanity bounds
    invalid      - calibration claimed but markers invalid, scale non-finite/non-positive, or out of bounds
    unavailable  - no calibration reference detected

Physical mm profiles require calibration_validity == "valid" before metric comparison.

==================================================
# 9. WHOLE VS BROKEN

Whole-vs-Broken analysis is implemented.

Current mathematical rule:

length_ratio =
    effective_grain_length /
    whole_kernel_reference_length

Decision:

if length_ratio < 0.75:
    BROKEN

else:
    WHOLE

The 0.75 threshold represents the project's image-based three-fourths length criterion.

IMPORTANT:
This is a count/image-based classification for the project.

It must not be described as an official mass-based laboratory broken-percentage measurement.

==================================================
# 10. WHOLE-KERNEL REFERENCE HIERARCHY

Reference resolution follows:

Tier 1
Explicit physical measured profile (mm) + valid calibration
    -> production eligible

Tier 1b
Explicit legacy pixel profile
    -> Proxy / demo-only / not production eligible
    -> never automatically injected

Tier 2
Sample-derived intact population
    -> Sample-Derived / proxy / not production eligible

Tier 3
Undetermined

Meaning:

Tier 1:
A caller explicitly supplies a physical measured profile with valid calibration.

Tier 1b:
A caller explicitly supplies the legacy pixel profile (154.1 px). This is a scale-specific proxy.

Tier 2:
The system attempts to estimate a whole-kernel reference from an intact grain population inside the sample.

Tier 3:
If there is not enough evidence for a reliable reference, the system returns:

reference_source = unavailable
reference_status = undetermined

and Whole/Broken classification may remain undetermined.

Physical mm profile requirements:
    reference_unit == "mm"
    data_status indicates measured/reference-qualified profile
    calibration_validity == "valid"
    reference_count >= 3 for production-eligible profiles

==================================================
# 11. DEFAULT RICE PROFILE

Current file:

grain_profiles/default_rice.json

Current reference:

whole_kernel_length = 154.1 pixels
whole_kernel_breadth = 48.2 pixels
reference_unit = pixels

IMPORTANT:

This is a PIXEL-SCALE PROXY.

It is not a universal physical whole-rice reference.

It is only appropriate for an imaging setup with compatible scale.

Long-term robust solution:

physical calibration + millimetre-based reference.

Do not silently inject this profile as a universal default for arbitrary images.

==================================================
# 12. CURRENT BROKEN-GRAIN VALIDATION

Controlled Whole/Broken validation images are stored in:

test_images/3 broken_grain/

Current test structure:

images/
    B01 — 1 whole grain.jpeg
    B02 — 1 broken grain.jpeg
    B03 — 5 whole grains.jpeg
    B04 — 5 broken grains A.jpeg
    B05 — 10 whole + 1 broken.jpeg
    B06 — 2 broken grains.png
    B07 — 10 whole + 2 broken.jpeg
    B08 — 5 broken grains B.png
    B09 — 10 whole + 5 broken.jpeg
    B10 — 10 whole + 10 broken.jpeg
    B11 — 25 whole + 5 broken.jpeg
    B12 — 25 whole + 10 broken.jpeg
    B13 — All broken grains.png

Existing validation results are stored under:

test_images/3 broken_grain/results/

The controlled profile-based tests demonstrated that the Whole/Broken rule can correctly mark the intended clear broken samples as red and whole grains as green when the reference profile is appropriate to the imaging scale.

==================================================
# 13. IMPORTANT BROKEN-GRAIN FINDINGS

Previous validation established several important facts:

1. The mathematical Whole/Broken rule itself is deterministic.

2. Red annotated grains correspond to BROKEN.

3. Green annotated grains correspond to WHOLE.

4. Some earlier broken-image test failures were caused by segmentation over-splitting rather than the classifier itself.

5. Some generated "broken" test pieces were physically above the 0.75 threshold and were therefore not valid clear-broken examples.

6. Different test images were captured at different scales.

7. A pixel reference therefore cannot be universally reused across all camera distances.

8. An image containing only broken grains may not contain enough information to derive a whole-kernel reference internally.

9. With an explicit valid reference profile, an all-broken image can still be classified.

==================================================
# 14. CURRENT UI CAPABILITIES

Current frontend supports the current analysis flow including:

- image upload
- camera capture
- processing state
- annotated result viewer
- grain table
- per-grain detail
- Whole/Broken display
- Whole count
- Broken count
- Whole percentage
- Broken percentage
- annotated image saving/downloading
- CSV/JSON export
- reset flow

Whole and Broken grains are visually differentiated.

The UI should make the distinction understandable to the user.

==================================================
# 15. LOGGING

The project now maintains structured application logging.

Primary logging components include:

backend/app/services/run_logger.py
backend/app/main.py
backend/app/services/job_manager.py
ml/segmentation/pipeline.py
tests/test_logging.py

Current logging design:

- one logical lifecycle owner for analysis events
- consistent job_id throughout an analysis
- no duplicate lifecycle events
- individually measured stage durations
- total analysis duration
- success/failure terminal events
- structured logs
- rotating file logging
- stdout logging
- logging propagation protected against duplicate output

Typical event sequence:

analysis_started
image_decoded
rice_gate_started
rice_gate_completed
segmentation_started
segmentation_completed
geometry_started
geometry_completed
broken_classification_started
broken_classification_completed
annotation_started
annotation_completed
analysis_completed

Failure path:

analysis_started
...
analysis_failed

A failed analysis must not subsequently emit:
analysis_completed status=success

Log files are runtime artifacts.

monitoring/logs/
and *.log
must remain ignored by Git.

Do not log:
- uploaded image bytes
- base64 image data
- secrets
- API keys
- credentials

==================================================
# 16. TEST DATA ORGANIZATION

Current test images are organized as:

test_images/

1 all types/
    General rice / non-rice / defect scenarios

2 rice and stones/
    Rice + foreign-object scenarios

3 broken_grain/
    Whole/Broken controlled validation

Test images are validation/demo assets.

Do not assume they are training data.

==================================================
# 17. CURRENT TEST STATUS

Current verified backend test status:

42 passed
1 deselected

The deselected test was intentionally excluded during the logging validation task.

Whole/Broken tests pass.

Rice-gate tests pass.

Logging tests pass.

Frontend production build passes.

When future changes are made, regression testing must include the relevant existing suites.

Do not claim broader model accuracy based only on unit tests.

==================================================
# 18. FOREIGN MATTER STATUS

Foreign-matter functionality exists in the codebase but is NOT the current development priority.

Current immediate project priority remains:

1. rice detection
2. segmentation/masking
3. geometry
4. Whole/Broken

Do not begin foreign-matter model development or replace the current pipeline unless explicitly requested.

Foreign-matter predictions must be labeled according to their actual validation status.

==================================================
# 19. OTHER QUALITY PARAMETERS

Long-term intended parameters include:

1. Broken
2. Damaged
3. Discoloured
4. Chalky
5. Red
6. Dehusked
7. Immature/Shrunken
8. Sprouted/Weevilled
9. Foreign Matter
10. Admixture
11. Length
12. Breadth
13. L/B Ratio
14. Total Count

However:

The existence of an implementation file does NOT mean that a parameter is scientifically validated.

Use explicit status labels where appropriate:

Validated
Experimental
Proxy
Undetermined
Not yet implemented

==================================================
# 20. STANDARDS

Standards/reference-screening functionality exists in the project.

Do NOT claim:

- official lot compliance
- official laboratory certification
- official current-season grading
- official mass-based compliance

unless a verified authoritative source and equivalent measurement basis are explicitly available.

Image-derived values must remain clearly distinguished from laboratory measurements.

==================================================
# 21. DEVELOPMENT PRINCIPLES

Always:

1. Read PROJECT_CONTEXT.md before significant changes.
2. Inspect the actual code before changing it.
3. Treat current runtime behavior as authoritative.
4. Preserve working behavior unless evidence requires change.
5. Identify the earliest failing stage before modifying downstream logic.
6. Separate dataset problems from model problems.
7. Separate segmentation problems from classification problems.
8. Avoid speculative architecture changes.
9. Avoid unnecessary rewrites.
10. Make the smallest change necessary for the requested task.
11. Validate changes against controlled images and regression tests.
12. Never fabricate metrics or provenance.

==================================================
# 22. SCOPE CONTROL

The current active task is:

RICE GRAIN DETECTION + INDIVIDUAL GRAIN MASKING

followed by:

WHOLE VS BROKEN

Do not automatically move to:

- damaged classification
- chalky classification
- red/discoloured classification
- advanced foreign-matter detection
- standards grading
- authentication
- deployment redesign
- model replacement

unless explicitly instructed.

==================================================
# 23. MODEL POLICY

No trained YOLO/Mask R-CNN model is currently part of the production repository.

Previous experimental model work was removed intentionally.

Do not infer that a missing model should be downloaded or recreated.

Do not introduce model-based segmentation merely because it may appear more sophisticated.

Any new model experiment must:

- be explicitly approved
- be isolated from production
- have a clear dataset
- have independent evaluation
- have reproducible results
- not silently replace the current baseline

==================================================
# 24. SOURCE-OF-TRUTH ORDER

When documentation conflicts:

1. Current source code
2. Current configuration
3. Current tests
4. Current runtime behavior
5. PROJECT_CONTEXT.md
6. Older reports and historical documentation

Older experimental reports do not override current implementation.

==================================================
# 25. CURRENT STATUS SUMMARY

CURRENT PRODUCTION:

Rice gate:
    Classical CV

Segmentation:
    Classical CV

Geometry:
    Geometric image processing

Whole/Broken:
    Rule-based relative-length classification

Reference:
    Explicit profile / sample-derived / undetermined

Calibration:
    Supported

UI:
    Upload + camera + analysis + annotated visualization + Whole/Broken summary + per-grain details + save/export

Logging:
    Structured + job-aware + stage timing + rotating file logging
    Supports both pure JSONL and Python logging-prefixed JSON

Monitoring:
    Streamlit read-only dashboard
    Persistent structured JSON Lines logs (plain JSONL or logging-prefixed)
    API metrics + analysis stage metrics
    Errors/failures + memory monitoring
    Dashboard: C:\\Python314\\python.exe -m streamlit run monitoring/app.py
    Parser handles real production log format with Python logger prefix

YOLO:
    Not production

==================================================
# 26. CURRENT NEXT STEP

The immediate next development step is NOT to redesign the architecture.

The immediate next step is to continue carefully improving the existing grain-analysis foundation.

Priority:

1. Verify rice-grain detection
2. Verify one-mask-per-grain behavior
3. Reduce false masks
4. Verify mask boundaries
5. Verify grain counts
6. Verify geometry from masks
7. Verify Whole/Broken against controlled reference/profile
8. Expand validation only after the above is stable

Every change should be experimentally validated before being considered part of the production baseline. 
