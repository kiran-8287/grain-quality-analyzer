# Rice analysis method

This package is the single active analysis method used by the website. The implementation in `rice_analysis.py` uses classical computer vision and shape-aware watershed splitting to produce rice and foreign-matter masks and count estimates. It is not a trained model.

The backend imports `label_image` from this package. Method dependencies are installed with [`../backend/requirements.txt`](../backend/requirements.txt). Evaluation scripts, reports, and historical outputs are preserved under `other models and methods/methods/rice_analysis_method/`.
