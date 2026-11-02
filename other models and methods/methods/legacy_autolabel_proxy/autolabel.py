"""Classical auto-labeler for rice (pale) + foreign matter (brown/dark) on a green sheet."""
import cv2
import numpy as np
from sklearn.cluster import KMeans

K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _foreground_masks(im):
    """Separate objects from locally varying felt colour in Lab space."""
    h, w = im.shape[:2]
    small = cv2.resize(im, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    kernel = max(3, int(max(small.shape[:2]) * 0.12) | 1)
    paper = cv2.resize(cv2.medianBlur(small, kernel), (w, h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = cv2.cvtColor(paper, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_smooth = cv2.GaussianBlur(lab[..., 1], (0, 0), 1.5)
    foreground = ((a_smooth - bg[..., 1] > 7) | (lab[..., 0] - bg[..., 0] < -45))
    foreground = cv2.morphologyEx(foreground.astype(np.uint8) * 255, cv2.MORPH_OPEN, K5)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, K5)
    stone = ((foreground > 0) & ((a_smooth > 126) | (lab[..., 0] < 100))).astype(np.uint8) * 255
    stone = cv2.dilate(cv2.morphologyEx(stone, cv2.MORPH_OPEN, K5), K5)
    rice = cv2.bitwise_and(foreground, cv2.bitwise_not(stone))
    rice = cv2.morphologyEx(rice, cv2.MORPH_OPEN, K5)
    return rice, stone


def _emit_split(cc, component, shape, k, cls, callback, instances):
    ys, xs = np.where(cc == component)
    if k == 1:
        mask = np.zeros(shape, dtype=bool)
        mask[ys, xs] = True
        if callback:
            callback(mask, cls)
        else:
            instances.append((mask, cls))
        return
    labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(
        np.c_[xs, ys].astype(np.float32))
    for j in range(k):
        keep = labels == j
        mask = np.zeros(shape, dtype=bool)
        mask[ys[keep], xs[keep]] = True
        if callback:
            callback(mask, cls)
        else:
            instances.append((mask, cls))


def _assign_counts(components, ref, target, max_split_clusters):
    """Allocate a known image count across semantic connected components."""
    comps = sorted(components, key=lambda item: item[1], reverse=True)[:target]
    if not comps:
        raise ValueError("No rice components available for requested count")
    counts = [max(1, int(round(area / ref))) for _, area in comps]
    counts = [min(k, max_split_clusters) if max_split_clusters else k for k in counts]
    while sum(counts) < target:
        choices = [i for i, k in enumerate(counts)
                   if max_split_clusters is None or k < max_split_clusters]
        if not choices:
            raise ValueError("Cannot reach requested count within component split limit")
        i = max(choices, key=lambda j: comps[j][1] / (counts[j] + 1))
        counts[i] += 1
    while sum(counts) > target:
        choices = [i for i, k in enumerate(counts) if k > 1]
        if not choices:
            raise ValueError(f"At least {len(comps)} components remain for count {target}")
        i = min(choices, key=lambda j: comps[j][1] / counts[j])
        counts[i] -= 1
    return [(comps[i][0], counts[i]) for i in range(len(comps))]


def label_image(path, min_area=400, min_foreign_area=None, instance_callback=None,
                max_split_clusters=None, target_rice_count=None,
                target_foreign_count=None):
    im = cv2.imread(str(path))
    if im is None:
        raise ValueError(f"Could not read image: {path}")
    rice, stone = _foreground_masks(im)
    foreign_min_area = min_area if min_foreign_area is None else min_foreign_area

    inst = []
    # Foreign matter: connected stone blobs; known-count training masks split merged blobs.
    n, cc, stats, _ = cv2.connectedComponentsWithStats(stone)
    foreign_comps = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)
                     if stats[i, cv2.CC_STAT_AREA] >= foreign_min_area]
    if target_foreign_count is None:
        for component, _ in foreign_comps:
            _emit_split(cc, component, im.shape[:2], 1, 'foreign', instance_callback, inst)
    elif foreign_comps:
        selected = sorted(foreign_comps, key=lambda item: item[1], reverse=True)[:target_foreign_count]
        split_counts = [1] * len(selected)
        while sum(split_counts) < target_foreign_count:
            i = max(range(len(selected)), key=lambda j: selected[j][1] / (split_counts[j] + 1))
            split_counts[i] += 1
        for (component, _), count in zip(selected, split_counts):
            _emit_split(cc, component, im.shape[:2], count, 'foreign', instance_callback, inst)
    elif target_foreign_count:
        raise ValueError("No foreign components found for requested count")

    # rice: connected components, then split clumps by area / single-grain area
    n, cc, stats, _ = cv2.connectedComponentsWithStats(rice)
    comps = [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, n)]
    comps = [(i, area) for i, area in comps if area >= min_area]
    areas = sorted(area for _, area in comps)
    lone_areas = []
    for component, area in comps:
        m = (cc == component).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        aspect = max(rw, rh) / max(1.0, min(rw, rh))
        adjusted_area = cv2.contourArea(c) + 0.5 * cv2.arcLength(c, True) + 1
        if adjusted_area / max(hull, 1.0) >= 0.93 and 2.4 <= aspect <= 5.5:
            lone_areas.append(float(adjusted_area))
    ref = (float(np.median(lone_areas)) if len(lone_areas) >= 8 else
           float(np.median(areas[:max(1, int(len(areas) * 0.6))])))
    if target_rice_count is None:
        for component, area in comps:
            count = max(1, int(round(area / ref)))
            if max_split_clusters is not None and count > max_split_clusters:
                continue
            _emit_split(cc, component, im.shape[:2], count, 'rice', instance_callback, inst)
    elif target_rice_count:
        if max_split_clusters is None:
            max_split_clusters = target_rice_count
        assignments = _assign_counts(comps, ref, target_rice_count, max_split_clusters)
        for component, count in assignments:
            _emit_split(cc, component, im.shape[:2], count, 'rice', instance_callback, inst)
    return im, inst, ref

def draw(im, inst, text):
    vis = im.copy()
    for m, c in inst:
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255) if c == 'foreign' else (255, 0, 255), 4)
    cv2.putText(vis, text, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
    return vis

if __name__ == '__main__':
    files = [
     '/mnt/user-data/uploads/1792569661926_IMG_20261020_230916.jpg',
     '/mnt/user-data/uploads/1792569661926_IMG_20261020_230951.jpg',
     '/mnt/user-data/uploads/1792569661927_IMG_20261020_231051.jpg',
     '/mnt/user-data/uploads/1792569661927_IMG20261020230214.jpg']
    outs = []
    for i, f in enumerate(files, 1):
        im, inst, ref = label_image(f)
        nr = sum(c == 'rice' for _, c in inst); nf = sum(c == 'foreign' for _, c in inst)
        print(f'photo {i}: rice={nr} foreign={nf} (single-grain ref area {ref:.0f})')
        outs.append(draw(im, inst, f'rice={nr} foreign={nf}'))
        cv2.imwrite(f'ov{i}.jpg', cv2.resize(outs[-1], None, fx=0.5, fy=0.5))
