"""Classical auto-labeler for rice (pale) + foreign matter (brown/dark) on a green sheet."""
import cv2, numpy as np
from sklearn.cluster import KMeans

def label_image(path, min_area=400):
    im = cv2.imread(str(path))
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

    rice = (fg & (stone_d == 0)).astype(np.uint8)
    rice = cv2.morphologyEx(rice, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    rice = cv2.morphologyEx(rice, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    inst = []
    # foreign matter: one instance per connected stone blob
    n, cc = cv2.connectedComponents(stone)
    for i in range(1, n):
        m = cc == i
        if m.sum() >= min_area: inst.append((m, 'foreign'))

    # rice: connected components, then split clumps by area / single-grain area
    n, cc = cv2.connectedComponents(rice)
    comps = [(cc == i) for i in range(1, n) if (cc == i).sum() >= min_area]
    areas = sorted(m.sum() for m in comps)
    ref = float(np.median(areas[:max(1, int(len(areas) * 0.6))]))
    for m in comps:
        a = m.sum()
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        solidity = cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)
        k = max(1, int(round(a / ref)))
        if solidity > 0.93 and a < 1.9 * ref: k = 1       # clean convex blob = one grain
        if k == 1:
            inst.append((m, 'rice')); continue
        ys, xs = np.where(m)
        lab = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(np.c_[xs, ys].astype(np.float32))
        for j in range(k):
            mm = np.zeros(m.shape, bool); mm[ys[lab == j], xs[lab == j]] = True
            inst.append((mm, 'rice'))
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
