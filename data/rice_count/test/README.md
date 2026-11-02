# Grain Quality Analyzer test images

This directory contains 20 JPEG sample images used for local API and image-output checks. The filenames describe the nominal sample composition (whole grains, broken grains, and stones/foreign objects), and repeated suffixes indicate additional captures of a composition.

The filename counts are nominal references only. This directory does not include reviewed per-grain masks or independently confirmed instance-level labels, so it is not by itself a ground-truth benchmark for precision, recall, or segmentation accuracy. See the project report and archived evaluation notes before interpreting model/count comparisons.

The images are 3072 × 4096 pixels. EXIF and JPEG comment metadata were removed before publication; the decoded pixel arrays were verified identical to the local originals. The larger 860-image raw set remains local and is not committed.
