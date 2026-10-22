"""Convert HEIC images to JPEG in place, keeping the same numeric base name.

Usage:
    python scripts/convert_heic_to_jpg.py
    python scripts/convert_heic_to_jpg.py --raw-dir <path> --keep-heic
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

import pillow_heif

pillow_heif.register_heif_opener()

HEIC_EXTS = {".heic", ".heif"}


def convert_one(src: Path) -> Path:
    dst = src.with_suffix(".jpg")
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(dst, format="JPEG", quality=95, subsampling=0)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=r"A:\grain\data\rice_count\raw")
    ap.add_argument("--keep-heic", action="store_true", help="keep the original .heic files")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        print(f"Raw dir not found: {raw_dir}", file=sys.stderr)
        return 1

    sources = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in HEIC_EXTS)
    if not sources:
        print(f"No HEIC files under {raw_dir}")
        return 0

    print(f"Converting {len(sources)} HEIC file(s) to JPEG in {raw_dir}")
    failures = []
    for src in sources:
        dst = src.with_suffix(".jpg")
        try:
            convert_one(src)
        except Exception as exc:  # noqa: BLE001
            failures.append((src, exc))
            print(f"FAILED {src.name}: {type(exc).__name__}: {exc}")
            continue
        if not args.keep_heic:
            src.unlink()
        print(f"{src.name} -> {dst.name}")

    if failures:
        print(f"{len(failures)} file(s) failed", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
