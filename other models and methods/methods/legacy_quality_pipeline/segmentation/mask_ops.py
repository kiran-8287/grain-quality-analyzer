"""
Offset-aware helpers for bounding-box cropped binary masks.

A grain/detection mask is stored as a tight ``uint8`` crop of the image plus
the ``(x, y)`` offset of that crop in image coordinates.

Why crops
---------
The classical-CV pipeline runs on whatever the user uploads, including 12 MP
phone photos.  Keeping one full-frame mask per object makes the per-object
cost scale with the image area: 1000 objects x 12.6 MP = 12.6 GB of retained
masks, which exhausts RAM and kills the request.  Crops keep the per-object
cost proportional to the object instead of to the image, so memory and CPU
stay bounded no matter how large the upload is.

The crop of a mask is always its tight bounding box, so every pixel-wise
measurement (area, contour, moments) is identical to the full-frame mask.
Only the *position* of the mask changes, and that is carried by the offset.
"""

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Offset = Tuple[int, int]


def tight_bounds(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return the tight ``(x, y, w, h)`` bounding box of the non-zero mask."""
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def crop_to_bounds(mask: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop ``mask`` to ``(x, y, w, h)`` (a view; callers may copy if needed)."""
    x, y, w, h = bounds
    return mask[y : y + h, x : x + w]


def translate_points(
    points: Sequence[Sequence[float]], offset: Offset
) -> List[List[float]]:
    """Translate a list of ``[x, y]`` points by ``offset``."""
    ox, oy = offset
    return [[float(px) + ox, float(py) + oy] for px, py in points]


def translate_contour(contour: Optional[np.ndarray], offset: Offset) -> Optional[np.ndarray]:
    """Translate an OpenCV contour (Nx1x2 int32) by ``offset``."""
    if contour is None:
        return None
    ox, oy = offset
    out = contour.astype(np.int32, copy=True)
    out[..., 0] += int(ox)
    out[..., 1] += int(oy)
    return out


def draw_mask(canvas: np.ndarray, mask: np.ndarray, offset: Offset) -> np.ndarray:
    """Draw a cropped mask into a full-frame canvas at ``offset`` (in place)."""
    ox, oy = offset
    mh, mw = mask.shape[:2]
    ch, cw = canvas.shape[:2]
    x0 = max(0, ox)
    y0 = max(0, oy)
    x1 = min(cw, ox + mw)
    y1 = min(ch, oy + mh)
    if x1 <= x0 or y1 <= y0:
        return canvas
    src = mask[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
    region = canvas[y0:y1, x0:x1]
    np.maximum(region, src, out=region)
    return canvas


def union_masks(
    shape: Tuple[int, int],
    masks: Iterable[np.ndarray],
    offsets: Optional[Sequence[Offset]] = None,
) -> np.ndarray:
    """Build the union of cropped masks into a full-frame ``uint8`` canvas."""
    canvas = np.zeros(shape, dtype=np.uint8)
    mask_list = list(masks)
    if offsets is None:
        offsets = [(0, 0)] * len(mask_list)
    for mask, offset in zip(mask_list, offsets):
        if mask is None or mask.size == 0:
            continue
        draw_mask(canvas, mask, offset)
    return canvas


def extents_intersect(
    a_offset: Offset,
    a_shape: Tuple[int, int],
    b_offset: Offset,
    b_shape: Tuple[int, int],
    pad: int = 0,
) -> bool:
    """
    True when the image-space extents of two crops overlap.

    Used to skip mask comparisons that cannot possibly overlap, which keeps
    pairwise stages (NMS, touching detection) linear in practice instead of
    quadratic in the number of objects.
    """
    ax, ay = a_offset
    bx, by = b_offset
    ah, aw = a_shape[:2]
    bh, bw = b_shape[:2]
    x1 = max(ax - pad, bx)
    y1 = max(ay - pad, by)
    x2 = min(ax + aw + pad, bx + bw)
    y2 = min(ay + ah + pad, by + bh)
    return x2 > x1 and y2 > y1


def mask_iou(
    a: np.ndarray,
    a_offset: Offset,
    b: np.ndarray,
    b_offset: Offset,
) -> float:
    """
    IoU of two masks given as crops with their image-space offsets.

    Equivalent to comparing the two full-frame masks, but only the
    overlapping region is ever materialised.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    ax, ay = a_offset
    bx, by = b_offset
    ah, aw = a.shape[:2]
    bh, bw = b.shape[:2]

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    a_sub = a[y1 - ay : y2 - ay, x1 - ax : x2 - ax] > 0
    b_sub = b[y1 - by : y2 - by, x1 - bx : x2 - bx] > 0
    inter = int(np.count_nonzero(a_sub & b_sub))
    union = int(np.count_nonzero(a_sub | b_sub))
    if union == 0:
        return 0.0
    return inter / union


def dilated_overlap(
    a: np.ndarray,
    a_offset: Offset,
    b: np.ndarray,
    b_offset: Offset,
    kernel: Optional[np.ndarray] = None,
    iterations: int = 1,
) -> int:
    """
    Number of pixels shared by ``dilate(a)`` and ``b``.

    ``a`` is dilated inside a zero-padded frame so that the dilation can
    extend one pixel beyond the crop, exactly as it would on the full frame.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    pad = 1
    a_bin = (a > 0).astype(np.uint8)
    a_pad = cv2.copyMakeBorder(
        a_bin, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
    )
    a_dil = cv2.dilate(a_pad, kernel, iterations=iterations)

    ax, ay = a_offset
    bx, by = b_offset
    ah, aw = a.shape[:2]
    bh, bw = b.shape[:2]

    x1 = max(ax - pad, bx)
    y1 = max(ay - pad, by)
    x2 = min(ax + aw + pad, bx + bw)
    y2 = min(ay + ah + pad, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0

    a_sub = a_dil[y1 - (ay - pad) : y2 - (ay - pad), x1 - (ax - pad) : x2 - (ax - pad)]
    b_sub = b[y1 - by : y2 - by, x1 - bx : x2 - bx] > 0
    return int(np.count_nonzero((a_sub > 0) & b_sub))
