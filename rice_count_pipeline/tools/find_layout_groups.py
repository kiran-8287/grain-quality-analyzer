"""Group photos of the same grain layout using shifted mask correlation."""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_photos_v2 import EXTS, load_small, masks  # noqa: E402

FW, FH = 72, 96
PAD_W, PAD_H = 144, 192


def fingerprint(path, mask_args):
    img = load_small(path, 1536)
    if img is None:
        return None
    rice, stone = masks(img, mask_args)
    m = cv2.bitwise_or(rice, stone).astype(np.float32) / 255.0
    m = cv2.resize(m, (FW, FH), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    m -= m.mean()
    norm = float(np.linalg.norm(m))
    return m / norm if norm > 0 else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--out", default="groups_out")
    p.add_argument("--threshold", type=float, default=0.80)
    a = p.parse_args()
    if not 0 < a.threshold <= 1:
        p.error("--threshold must be in (0, 1]")

    root = Path(a.folder)
    paths = sorted(f for f in root.rglob("*") if f.is_file() and f.suffix.lower() in EXTS)
    if not paths:
        p.error(f"No photos found under {root.resolve()}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mask_args = SimpleNamespace(a_thr=7, stone_a=126, stone_l=100)
    fps, keep = [], []
    for i, f in enumerate(paths):
        fp = fingerprint(f, mask_args)
        if fp is not None:
            fps.append(fp)
            keep.append(f)
        elif i < 10:
            print(f"  unreadable or empty foreground: {f}")
        if (i + 1) % 50 == 0:
            print(f"  fingerprints {i + 1}/{len(paths)}")
    n = len(keep)
    if not n:
        raise ValueError("No readable photos had usable foreground masks")
    print(f"{n} photos fingerprinted; comparing all pairs ...")

    F = np.stack([np.fft.rfft2(f, s=(PAD_H, PAD_W)) for f in fps]).astype(np.complex64)
    best_sim = np.full(n, -1.0)
    best_idx = np.full(n, -1, dtype=int)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i in range(n - 1):
        cc = np.fft.irfft2(F[i] * np.conj(F[i + 1:]), s=(PAD_H, PAD_W))
        sims = cc.reshape(cc.shape[0], -1).max(axis=1)
        for k, score in enumerate(sims):
            j = i + 1 + k
            score = float(score)
            pairs.append((score, i, j))
            if score > best_sim[i]:
                best_sim[i], best_idx[i] = score, j
            if score > best_sim[j]:
                best_sim[j], best_idx[j] = score, i
            if score >= a.threshold:
                parent[find(i)] = find(j)
        if (i + 1) % 100 == 0:
            print(f"  compared {i + 1}/{n}")

    roots = [find(i) for i in range(n)]
    sizes = Counter(roots)
    gid = {r: k for k, r in enumerate(sorted(sizes, key=lambda r: (-sizes[r], r)))}
    with open(out / "groups.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "group_id", "group_size", "best_match", "best_sim"])
        for i in range(n):
            j = best_idx[i]
            writer.writerow([str(keep[i]), gid[roots[i]], sizes[roots[i]],
                             keep[j].name if j >= 0 else "", round(float(best_sim[i]), 3)])

    print(f"\nthreshold {a.threshold}:  {len(sizes)} layout groups from {n} photos")
    hist = Counter(min(size, 21) for size in sizes.values())
    print("group sizes (21 = 21 or more):", dict(sorted(hist.items())))
    print(f"photos that share a layout with another photo: {sum(size for size in sizes.values() if size > 1)}/{n}")
    if n > 1:
        print("similarity of each photo's best match -> p5/p25/p50/p75:",
              np.percentile(best_sim, [5, 25, 50, 75]).round(2))
    pairs.sort(key=lambda item: (abs(item[0] - a.threshold), -item[0]))
    print("\nBORDERLINE pairs (open a few to check the threshold):")
    for score, i, j in pairs[:6]:
        print(f"  sim {score:.2f}  {keep[i].name}  <->  {keep[j].name}")
    print(f"\nSaved {out / 'groups.csv'}")


if __name__ == "__main__":
    main()
