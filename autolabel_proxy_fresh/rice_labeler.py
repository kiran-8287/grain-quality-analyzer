"""
Classical auto-labeler for rice (pale) + foreign matter (brown/dark) on a green sheet.

v2 - touching / overlapping grains are separated with a *shape-aware* splitter
instead of KMeans on pixel coordinates.

Why KMeans failed
-----------------
KMeans uses round clusters, but a rice grain is a long thin ellipse, so on a clump
of elongated grains it cuts ACROSS grains.  And k = area / single-grain-area is
only a rough guess (overlapping grains hide area, broken grains are small).

What this version does
----------------------
1. Learn the grain size/shape (length, width, area) from the isolated grains of the
   SAME image, so there are no hand-tuned pixel sizes.
2. A connected component that already looks like one grain is kept as it is.
3. Otherwise fit k ELONGATED ellipses (fixed width, free position / angle / length)
   to the clump with EM, for k = 1, 2, 3 ...  The smallest k whose ellipses cover the
   clump without overshooting it wins; every pixel goes to its most likely ellipse.
4. The cut lines are snapped to the real seams / grain outlines with a
   marker-controlled watershed, and neighbouring pieces that together still look
   like ONE grain are merged back (cures over-splitting).

Same API as before:  im, inst, ref = label_image(path)   ->  inst = [(bool_mask, 'rice'|'foreign')]
Run `python rice_labeler.py photo1.jpg photo2.jpg` to get overlays (ov1.jpg, ...).
If the file name looks like "115_whole__15_broken___15_stones.jpg" the expected
counts are parsed and a warning is printed when the prediction differs
(handy for flagging bad proxy labels for manual review).
"""
import cv2
import numpy as np
from skimage.segmentation import watershed

# --------------------------------------------------------------------------- #
# 1. colour segmentation (unchanged logic from the original script)
# --------------------------------------------------------------------------- #

def segment_colour(im):
    """Return (stone_mask, rice_mask) as uint8 {0,1} arrays."""
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h, s, v = [x.astype(np.int32) for x in cv2.split(hsv)]

    green = (h > 55) & (h < 100) & (s > 90)
    fg = ~green
    # stone pixels: orange/tan hue (rice hue is greenish-white) OR very dark
    stone_px = fg & (((h <= 36) & (s > 70)) | (v < 110))
    stone = stone_px.astype(np.uint8)
    stone = cv2.morphologyEx(stone, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    stone = cv2.morphologyEx(stone, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    stone_d = cv2.dilate(stone, np.ones((7, 7), np.uint8))

    rice = (fg & (stone_d == 0))
    # Rice is BRIGHT and unsaturated.  The yellow-green fringe that surrounds a stone
    # (stone/sheet colour blend) is darker and more saturated -> reject it here.
    core = rice & (s < 80)
    if core.sum() > 1000:
        v_rice = np.median(v[core])
        rice &= (v > 0.78 * v_rice)
    rice = rice.astype(np.uint8)
    rice = cv2.morphologyEx(rice, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    rice = cv2.morphologyEx(rice, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return stone, rice


# --------------------------------------------------------------------------- #
# 2. grain model learned from isolated grains
# --------------------------------------------------------------------------- #

class GrainModel:
    """Typical single (whole) grain: area, length, width and PCA std-devs."""

    def __init__(self, area, length, width, sig_l, sig_w):
        self.area, self.length, self.width = area, length, width
        self.sig_l, self.sig_w = sig_l, sig_w


def _largest_contour(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def _shape_stats(mask_bool):
    """area, solidity, (long, short) side of min-area rect."""
    m8 = mask_bool.astype(np.uint8)
    c = _largest_contour(m8)
    if c is None or len(c) < 5:
        return 0, 0.0, 0.0, 0.0
    area = float(mask_bool.sum())
    hull = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
    solidity = cv2.contourArea(c) / hull
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    return area, solidity, max(rw, rh), min(rw, rh)


def learn_grain_model(comps, min_area):
    """comps: list of boolean masks (cropped) of connected components."""
    rows = []
    for m in comps:
        area, sol, L, W = _shape_stats(m)
        if area < min_area or W < 1:
            continue
        if sol > 0.95 and 2.2 < L / W < 6.5:
            ys, xs = np.nonzero(m)
            ev = np.linalg.eigvalsh(np.cov(np.c_[xs, ys].T))
            rows.append((area, L, W, np.sqrt(ev[1]), np.sqrt(ev[0])))
    if len(rows) >= 5:
        r = np.array(rows)
        # whole grains = the longer isolated ones (broken grains are shorter)
        keep = r[r[:, 1] >= 0.9 * np.percentile(r[:, 1], 75)]
        if len(keep) < 3:
            keep = r
        med = np.median(keep, axis=0)
        return GrainModel(*med)
    # fallback: no isolated grains found -> use smaller components
    areas = np.array(sorted(float(m.sum()) for m in comps if m.sum() >= min_area))
    a = float(np.median(areas[:max(1, int(len(areas) * 0.6))]))
    W = np.sqrt(a / (3.6 * 0.785))
    L = 3.6 * W
    return GrainModel(a, L, W, L / 3.6 * 0.9, W / 3.6 * 0.9)


# --------------------------------------------------------------------------- #
# 3. "does this piece look like one grain?"
# --------------------------------------------------------------------------- #

def single_grain_badness(mask_bool, g, scale=1.0):
    """0 = looks like one grain; grows when it is too big / wide / concave / tiny."""
    area, sol, L, W = _shape_stats(mask_bool)
    if area == 0:
        return 10.0
    a = area / (g.area * scale * scale)
    bad = 0.0
    bad += max(0.0, a - 1.30) * 3.0                       # too big for one grain
    bad += max(0.0, 0.90 - sol) * 12.0                    # concave -> glued grains
    bad += max(0.0, W / (g.width * scale) - 1.30) * 4.0   # too wide
    bad += max(0.0, L / (g.length * scale) - 1.30) * 4.0  # too long
    bad += max(0.0, 0.15 - a) * 8.0                       # tiny crumb
    return bad


SEAM_BETA = 1.0      # weight of luminance edges when snapping cuts (0 = colour seams only)
SINGLE_OK = 0.25     # badness below this == accepted as one grain


# --------------------------------------------------------------------------- #
# 4. EM with k elongated (fixed-width) ellipses
# --------------------------------------------------------------------------- #

def _kmeanspp(pts, k, rng):
    c = [pts[rng.integers(len(pts))]]
    d2 = ((pts - c[0]) ** 2).sum(1)
    for _ in range(1, k):
        p = d2 / d2.sum() if d2.sum() > 0 else None
        c.append(pts[rng.choice(len(pts), p=p)])
        d2 = np.minimum(d2, ((pts - c[-1]) ** 2).sum(1))
    return np.array(c, float)


def _logp(pts, mu, ang, sl, sw, pi):
    """log( pi_j * N(pts | ellipse j) )  ->  array (k, N). Fully vectorised."""
    d = pts[None, :, :] - mu[:, None, :]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    u = d[..., 0] * c + d[..., 1] * s            # along the grain
    v = -d[..., 0] * s + d[..., 1] * c           # across the grain
    return (np.log(pi + 1e-12)[:, None] - np.log(sl * sw)[:, None]
            - 0.5 * ((u / sl[:, None]) ** 2 + (v / sw[:, None]) ** 2))


def _em_ellipses(pts, k, sl0, sw0, rng, iters=30):
    """Mixture of k anisotropic Gaussians ("grains") with (almost) fixed width.
    Returns the fitted parameters."""
    mu = _kmeanspp(pts, k, rng)
    for _ in range(3):                                     # a few Lloyd steps
        lab = ((pts[None] - mu[:, None]) ** 2).sum(2).argmin(0)
        for j in range(k):
            if (lab == j).any():
                mu[j] = pts[lab == j].mean(0)
    ang = np.zeros(k)
    for j in range(k):
        q = pts[lab == j]
        if len(q) > 5:
            cv = np.cov(q.T)
            ang[j] = 0.5 * np.arctan2(2 * cv[0, 1], cv[0, 0] - cv[1, 1])
    sl, sw, pi = np.full(k, sl0), np.full(k, sw0), np.full(k, 1.0 / k)

    prev = -np.inf
    for _ in range(iters):
        lp = _logp(pts, mu, ang, sl, sw, pi)
        mx = lp.max(0)
        r = np.exp(lp - mx)
        tot = r.sum(0)
        ll = float((mx + np.log(tot)).sum())
        r /= tot
        nk = r.sum(1) + 1e-9
        pi = nk / nk.sum()
        mu = (r @ pts) / nk[:, None]
        d = pts[None] - mu[:, None]
        sxx = (r * d[..., 0] ** 2).sum(1) / nk + 1e-6
        syy = (r * d[..., 1] ** 2).sum(1) / nk + 1e-6
        sxy = (r * d[..., 0] * d[..., 1]).sum(1) / nk
        ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)         # major-axis direction
        root = np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
        l1, l2 = (sxx + syy) / 2 + root, (sxx + syy) / 2 - root
        sl = np.clip(np.sqrt(l1), 0.45 * sl0, 1.10 * sl0)
        sw = np.clip(np.sqrt(np.maximum(l2, 1e-6)), 0.80 * sw0, 1.20 * sw0)
        if ll - prev < 1e-4 * abs(ll):
            break
        prev = ll
    return mu, ang, sl, sw, pi


def _coverage_error(shape, mu, ang, sl, sw, clump, radius=2.0):
    """How well do k fixed-shape ellipses explain the clump?
    uncovered = clump pixels outside every ellipse   (k too small -> large)
    overshoot = ellipse area outside the clump       (ellipse on nothing -> large)"""
    Hs, Ws = shape
    yy, xx = np.mgrid[:Hs, :Ws]
    g = np.c_[xx.ravel(), yy.ravel()].astype(float)
    d = g[None] - mu[:, None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    u = d[..., 0] * c + d[..., 1] * s
    v = -d[..., 0] * s + d[..., 1] * c
    inside = ((u / sl[:, None]) ** 2 + (v / sw[:, None]) ** 2) <= radius ** 2   # (k, N)
    union = inside.any(0).reshape(shape)
    unc = (clump & ~union).sum() / max(clump.sum(), 1)
    ovs = (union & ~clump).sum() / max(union.sum(), 1)
    return unc, ovs


COVER_OK = 0.19      # uncovered + overshoot below this -> k ellipses explain the clump (isolated real grains score 0.10-0.17)


def split_clump(mask_bool, g, rng, k_max=14, restarts=6, work_len=45.0, max_pts=1400):
    """Return a label image (0 = background, 1..k) for one connected clump."""
    H, W = mask_bool.shape
    sc = min(1.0, work_len / g.length)          # a grain is ~work_len px long here
    small = cv2.resize(mask_bool.astype(np.uint8), None, fx=sc, fy=sc,
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(small)
    pts = np.c_[xs, ys].astype(float)
    fit_pts = pts if len(pts) <= max_pts else pts[rng.choice(len(pts), max_pts, replace=False)]
    sl0, sw0 = g.sig_l * sc, g.sig_w * sc
    area_est = len(pts) / (g.area * sc * sc)
    k_lo = max(1, int(np.floor(area_est / 1.5)))
    k_hi = max(k_lo, min(k_max, int(np.ceil(area_est / 0.45)) + 1))

    best = (1e9, None)
    for k in range(k_lo, k_hi + 1):
        k_best = (1e9, None)
        for _ in range(restarts if k > 1 else 1):
            mu, ang, sl, sw, pi = _em_ellipses(fit_pts, k, sl0, sw0, rng)
            unc, ovs = _coverage_error(small.shape, mu, ang, sl, sw, small)
            J = unc + ovs
            if J < k_best[0]:
                k_best = (J, (mu, ang, sl, sw, pi))
            if J < COVER_OK:
                break
        J = k_best[0] + 0.01 * k              # tiny preference for fewer grains
        if J < best[0]:
            best = (J, k_best[1])
        if k_best[0] < COVER_OK:              # smallest k that explains the clump
            best = (J, k_best[1])
            break
    mu, ang, sl, sw, pi = best[1]
    lab = _logp(pts, mu, ang, sl, sw, pi).argmax(0)
    L = np.zeros(small.shape, np.int32)
    L[ys, xs] = lab + 1
    L = _clean_labels(L)
    L_full = cv2.resize(L.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
    L_full[~mask_bool] = 0
    return L_full


def _clean_labels(L):
    """Keep the biggest connected piece of every label; give stray pixels to a neighbour."""
    out = np.zeros_like(L)
    for j in np.unique(L):
        if j == 0:
            continue
        n, cc, st, _ = cv2.connectedComponentsWithStats((L == j).astype(np.uint8))
        if n > 1:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            out[cc == big] = j
    stray = (L > 0) & (out == 0)
    if stray.any():
        out = watershed(np.zeros(L.shape, np.float32), out, mask=(L > 0))
    return out


# --------------------------------------------------------------------------- #
# 5. snap cuts to visible seams
# --------------------------------------------------------------------------- #

def refine_with_seams(labels, mask_bool, relief, g):
    ids = [j for j in np.unique(labels) if j]
    if len(ids) < 2:
        return labels
    r = max(1, int(round(0.18 * g.width)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    markers = np.zeros_like(labels)
    for j in ids:
        core = cv2.erode((labels == j).astype(np.uint8), ker) > 0
        if not core.any():                      # very thin piece -> keep its centre
            core = (labels == j)
        markers[core] = j
    out = watershed(relief, markers, mask=mask_bool, compactness=0.0)
    return out


def merge_pieces(L, g, ok=SINGLE_OK, crumb=0.30):
    """Greedily re-join neighbouring pieces when together they look like ONE grain
    (fixes over-splitting), and glue crumbs (< `crumb` x grain area) to a neighbour."""
    ker = np.ones((5, 5), np.uint8)
    while True:
        ids = [j for j in np.unique(L) if j]
        if len(ids) < 2:
            return L
        masks = {j: (L == j) for j in ids}
        dil = {j: cv2.dilate(masks[j].astype(np.uint8), ker) > 0 for j in ids}
        best = None
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                border = int((dil[i] & masks[j]).sum())
                if border < 6:
                    continue
                union = masks[i] | masks[j]
                bad = single_grain_badness(union, g)
                small = min(masks[i].sum(), masks[j].sum()) < crumb * g.area
                if small:
                    bad = min(bad, 0.2) if union.sum() < 1.5 * g.area else bad
                if bad < ok and (best is None or bad < best[0]):
                    best = (bad, i, j)
        if best is None:
            return L
        _, i, j = best
        L[L == j] = i


# --------------------------------------------------------------------------- #
# 6. public API (same as before)
# --------------------------------------------------------------------------- #

def label_image(path, min_area=400, instance_callback=None, max_split_clusters=None,
                seed=0, use_seams=True, merge=True, return_details=False):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f'Could not read image: {path}')
    processing_scale = 0.4
    im = cv2.resize(im, None, fx=processing_scale, fy=processing_scale, interpolation=cv2.INTER_AREA)
    min_area = max(1, int(round(min_area * processing_scale * processing_scale)))
    stone, rice = segment_colour(im)
    rng = np.random.default_rng(seed)
    inst = []

    def emit(m, cls):
        if instance_callback:
            instance_callback(m, cls)
        else:
            inst.append((m, cls))

    # foreign matter: one instance per connected stone blob
    n, cc, stats, _ = cv2.connectedComponentsWithStats(stone)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            emit(cc == i, 'foreign')

    # relief for seam snapping: dark seams between grains become ridges
    lab = cv2.cvtColor(cv2.GaussianBlur(im, (0, 0), 1.5), cv2.COLOR_BGR2LAB)
    a_ch = lab[..., 1].astype(np.float32)
    gl = cv2.GaussianBlur(lab[..., 0].astype(np.float32), (0, 0), 2.0)
    grad = cv2.magnitude(cv2.Sobel(gl, cv2.CV_32F, 1, 0), cv2.Sobel(gl, cv2.CV_32F, 0, 1))
    on = rice > 0
    if on.any():
        a_n = (a_ch - np.percentile(a_ch[on], 1)) / (np.ptp(a_ch[on]) + 1e-6)
        g_n = grad / (np.percentile(grad[on], 99) + 1e-6)
    else:
        a_n, g_n = a_ch * 0, grad * 0
    # valleys of greenness (dark seams) and luminance edges (grain outlines in a pile)
    relief_full = (-a_n + SEAM_BETA * g_n).astype(np.float32)
    pad = 4
    relief_pad = np.pad(relief_full, pad, mode='edge')

    n, cc, stats, _ = cv2.connectedComponentsWithStats(rice)
    boxes = []
    comps = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area:
            continue
        boxes.append((i, x, y, w, h))
        comps.append(cc[y:y + h, x:x + w] == i)
    g = learn_grain_model(comps, min_area)

    for (i, x, y, w, h), m in zip(boxes, comps):
        bad = single_grain_badness(m, g)
        if bad < SINGLE_OK:
            full = np.zeros(cc.shape, bool); full[y:y + h, x:x + w] = m
            emit(full, 'rice')
            continue
        k_est = m.sum() / g.area
        if max_split_clusters is not None and round(k_est) > max_split_clusters:
            continue
        mp = np.pad(m, pad)
        L = split_clump(mp, g, rng)
        if use_seams:
            L = refine_with_seams(L, mp, relief_pad[y:y + h + 2 * pad, x:x + w + 2 * pad], g)
        if merge:
            L = merge_pieces(L, g)
        L = L[pad:-pad, pad:-pad]
        for j in np.unique(L):
            if j == 0:
                continue
            piece = L == j
            if piece.sum() < 0.12 * g.area:      # crumbs
                continue
            full = np.zeros(cc.shape, bool); full[y:y + h, x:x + w] = piece
            emit(full, 'rice')
    if return_details:
        return im, inst, g.area, g.length
    return im, inst, g.area


def draw(im, inst, text):
    vis = im.copy()
    rng = np.random.default_rng(1)
    for m, c in inst:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = (0, 0, 255) if c == 'foreign' else (255, 0, 255)
        cv2.drawContours(vis, cnts, -1, col, max(2, im.shape[0] // 500))
    cv2.putText(vis, text, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, im.shape[0] / 900, (0, 0, 0), 6)
    return vis


def expected_from_name(path):
    """'115_whole__15_broken___15_stones.jpg' -> (rice=130, foreign=15) else None."""
    import re
    m = re.search(r'(\d+)\s*whole.*?(\d+)\s*broken.*?(\d+)\s*stones', str(path), re.I)
    return (int(m.group(1)) + int(m.group(2)), int(m.group(3))) if m else None


if __name__ == '__main__':
    import sys
    files = sys.argv[1:]
    if not files:
        sys.exit('usage: python rice_labeler.py photo1.jpg [photo2.jpg ...]')
    for i, f in enumerate(files, 1):
        im, inst, ref = label_image(f)
        nr = sum(c == 'rice' for _, c in inst)
        nf = sum(c == 'foreign' for _, c in inst)
        exp = expected_from_name(f)
        msg = f'photo {i}: rice={nr} foreign={nf} (single-grain ref area {ref:.0f})'
        if exp:
            msg += f'  expected rice={exp[0]} foreign={exp[1]}'
            if (nr, nf) != exp:
                msg += '   <-- count mismatch, check manually'
        print(msg)
        text = f'rice={nr} foreign={nf}' + (f' (expected {exp[0]}/{exp[1]})' if exp else '')
        out = draw(im, inst, text)
        cv2.imwrite(f'ov{i}.jpg', cv2.resize(out, None, fx=0.5, fy=0.5))


