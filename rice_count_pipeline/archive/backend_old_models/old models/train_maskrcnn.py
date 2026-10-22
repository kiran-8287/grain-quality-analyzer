#!/usr/bin/env python3
"""
train_maskrcnn.py -- fine-tune torchvision Mask R-CNN (COCO-pretrained) for rice-grain
instance segmentation (whole_rice / broken_rice).  Foreign matter is handled by YOLO.

Design decisions (all tuned for ~1000 small, touching, randomly rotated grains per photo)
-----------------------------------------------------------------------------------------
* CROP TRAINING.  A 12 MP phone photo cannot be shrunk to 800 px without turning a grain
  into a 10x4 px smudge, and it will not fit in GPU memory at full size.  We train on
  random 1024x1024 crops at NATIVE resolution (model min/max_size == crop size, so no
  resizing happens) and run inference on overlapping tiles.
* SMALL-OBJECT ANCHORS.  Default torchvision anchors start at 32 px.  We use 16..256 and
  print your real grain-size statistics at start-up so you can adjust --anchor-sizes.
* CROWDED SCENES.  torchvision caps detections at 100/image and uses NMS 0.5, both of
  which silently delete grains in a dense, overlapping pile.  We raise the caps and NMS.
* MASK TARGETS FROM POLYGONS, CROPPED ON THE FLY (no 12 MP mask is ever decoded).
* Honest validation = COCO mask AP on a tiled validation set.

Requirements: torch>=2.3, torchvision, pycocotools, opencv-python, numpy

Example
-------
python train_maskrcnn.py --images raw_images/ --train-json dataset/maskrcnn/train.json \
       --val-json dataset/maskrcnn/val.json --output-dir runs/maskrcnn \
       --epochs 30 --batch-size 2 --crop-size 1024

Inference reminder: slide a crop_size window with >=128 px overlap, discard detections that
touch an inner tile border, then merge with mask-IoU NMS. Measure length/breadth from the
final masks (cv2.minAreaRect on the largest contour).
"""
import argparse
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

cv2.setNumThreads(0)  # avoid CPU oversubscription inside DataLoader workers

# Relative weight of each Mask R-CNN loss term. Raise loss_mask (e.g. 2.0) if mask edges
# look sloppy -- edge accuracy directly limits your length/breadth measurement.
LOSS_WEIGHTS = {
    "loss_classifier": 1.0,
    "loss_box_reg": 1.0,
    "loss_mask": 1.0,
    "loss_objectness": 1.0,
    "loss_rpn_box_reg": 1.0,
}


# =========================================================================================
# Dataset
# =========================================================================================
def _poly_area(poly):
    """Shoelace area of a flat COCO polygon [x1,y1,x2,y2,...]."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _tile_grid(length, crop):
    """Start offsets that cover [0, length) with windows of size `crop` (last one is shifted back)."""
    if length <= crop:
        return [0]
    starts = list(range(0, length - crop + 1, crop))
    if starts[-1] + crop < length:
        starts.append(length - crop)
    return starts


class CocoGrainDataset(Dataset):
    """
    COCO-format instance segmentation dataset that yields CROPS.

    train=True : `crops_per_image` random crops per image per epoch + flips/rot90/colour jitter.
    train=False: deterministic grid of tiles covering every image (used for validation).

    Returns (image[3,H,W] float 0..1, target) where target has the keys torchvision wants:
        boxes[N,4] float32 xyxy | labels[N] int64 (1..K, 0 = background) | masks[N,H,W] uint8
    """

    def __init__(self, img_dir, ann_file, class_names, crop_size=1024, train=True,
                 crops_per_image=4, min_visible=0.5):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_file)
        self.class_names = list(class_names)
        self.crop = crop_size
        self.train = train
        self.crops_per_image = crops_per_image
        self.min_visible = min_visible  # drop instances cut by the crop edge to less than this fraction

        name_to_id = {c["name"]: c["id"] for c in self.coco.loadCats(self.coco.getCatIds())}
        missing = [n for n in self.class_names if n not in name_to_id]
        if missing:
            raise ValueError(f"Classes {missing} not in {ann_file}. Available: {sorted(name_to_id)}")
        self.cat_to_label = {name_to_id[n]: i + 1 for i, n in enumerate(self.class_names)}

        self.img_ids = sorted(self.coco.getImgIds())
        self.anns = {
            i: [a for a in self.coco.loadAnns(self.coco.getAnnIds(imgIds=i))
                if a["category_id"] in self.cat_to_label and not a.get("iscrowd", 0)]
            for i in self.img_ids
        }
        if not train:
            self.tiles = []
            for i in self.img_ids:
                info = self.coco.imgs[i]
                for y0 in _tile_grid(info["height"], crop_size):
                    for x0 in _tile_grid(info["width"], crop_size):
                        self.tiles.append((i, x0, y0))
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25,
                                                         saturation=0.2, hue=0.02)

    def __len__(self):
        return len(self.img_ids) * self.crops_per_image if self.train else len(self.tiles)

    # ---------------------------------------------------------------------------------
    def _rasterise(self, anns, x0, y0, cw, ch):
        """Draw polygons directly into a (cw x ch) canvas -> list of uint8 masks + labels."""
        masks, labels = [], []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            if bx + bw <= x0 or bx >= x0 + cw or by + bh <= y0 or by >= y0 + ch:
                continue  # bbox does not touch the crop
            seg = a["segmentation"]
            if isinstance(seg, list):  # polygons (CVAT / Roboflow default)
                # Rasterise with pycocotools (the same routine annToMask / COCOeval use) after
                # shifting the polygon into crop coordinates. cv2.fillPoly would also paint the
                # boundary pixels, i.e. inflate every grain by ~0.5 px per side -- a systematic
                # error that goes straight into your length/breadth measurements.
                polys = [(np.asarray(poly, np.float64).reshape(-1, 2) - [x0, y0]).reshape(-1).tolist()
                         for poly in seg if len(poly) >= 6]
                if not polys:
                    continue
                m = mask_util.decode(mask_util.merge(mask_util.frPyObjects(polys, ch, cw)))
                full_area = sum(_poly_area(poly) for poly in seg if len(poly) >= 6)
            else:  # RLE (e.g. CVAT brush tool) -> decode full mask once, then crop
                m = self.coco.annToMask(a)[y0:y0 + ch, x0:x0 + cw].astype(np.uint8)
                full_area = float(a.get("area", m.sum()))
            vis = int(m.sum())
            if vis < 4 or vis < self.min_visible * max(full_area, 1.0):
                continue
            masks.append(m)
            labels.append(self.cat_to_label[a["category_id"]])
        return masks, labels

    def __getitem__(self, idx):
        if self.train:
            img_id = self.img_ids[idx % len(self.img_ids)]
        else:
            img_id, tx0, ty0 = self.tiles[idx]
        info = self.coco.imgs[img_id]
        W, H = info["width"], info["height"]
        cw, ch = min(self.crop, W), min(self.crop, H)

        path = self.img_dir / info["file_name"]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # OpenCV applies EXIF rotation by default
        if img is None:
            raise FileNotFoundError(path)
        if img.shape[:2] != (H, W):
            raise RuntimeError(
                f"{path.name}: file is {img.shape[1]}x{img.shape[0]} but COCO says {W}x{H}. "
                "Usually an EXIF-rotation or resize mismatch between the annotation tool and the "
                "file on disk -- re-export or bake the orientation into the pixels.")

        anns = self.anns[img_id]
        if self.train:
            for _ in range(5):  # retry a few times to avoid empty crops (sheet margins)
                x0, y0 = random.randint(0, W - cw), random.randint(0, H - ch)
                masks, labels = self._rasterise(anns, x0, y0, cw, ch)
                if masks:
                    break
        else:
            x0, y0 = tx0, ty0
            masks, labels = self._rasterise(anns, x0, y0, cw, ch)

        img = cv2.cvtColor(img[y0:y0 + ch, x0:x0 + cw], cv2.COLOR_BGR2RGB)
        masks = np.stack(masks) if masks else np.zeros((0, ch, cw), np.uint8)

        if self.train:  # D4 symmetries: grains have no canonical orientation
            if random.random() < 0.5:
                img, masks = img[:, ::-1], masks[:, :, ::-1]
            if random.random() < 0.5:
                img, masks = img[::-1], masks[:, ::-1]
            k = random.randint(0, 3)
            img, masks = np.rot90(img, k), np.rot90(masks, k, axes=(1, 2))
        img = np.ascontiguousarray(img)
        masks = np.ascontiguousarray(masks)

        # Boxes are recomputed from the FINAL masks so they always agree with the augmentation.
        boxes = np.zeros((len(masks), 4), np.float32)
        for i, m in enumerate(masks):
            ys, xs = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            boxes[i] = (xs[0], ys[0], xs[-1] + 1, ys[-1] + 1)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        if self.train:
            img_t = self.jitter(img_t)  # hue kept tiny: the green sheet colour is a useful cue
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(masks),
        }
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def print_grain_stats(ds, anchor_sizes):
    sizes = [math.sqrt(a["bbox"][2] * a["bbox"][3]) for v in ds.anns.values() for a in v]
    if not sizes:
        raise SystemExit("No rice annotations found -- check --classes and your JSON.")
    p5, p50, p95 = np.percentile(sizes, [5, 50, 95])
    n = sum(len(v) for v in ds.anns.values())
    print(f"[data] {len(ds.img_ids)} images, {n} grain instances ({n / len(ds.img_ids):.0f}/image)")
    print(f"[data] grain bbox size sqrt(w*h): p5={p5:.0f}px  median={p50:.0f}px  p95={p95:.0f}px")
    print(f"[data] anchor sizes per FPN level: {anchor_sizes}  "
          f"(rule of thumb: median grain size should sit near one of them)")


# =========================================================================================
# Model
# =========================================================================================
def build_model(num_classes, crop_size, anchor_sizes, pretrained=True, trainable_layers=5,
                max_dets=500, box_nms=0.6):
    """COCO-pretrained Mask R-CNN R50-FPN with new heads + settings for dense small objects."""
    if len(anchor_sizes) != 5:
        raise ValueError("--anchor-sizes needs exactly 5 values (one per FPN level P2..P6)")
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None,
        weights_backbone=None,  # backbone weights come bundled with the COCO checkpoint
        trainable_backbone_layers=trainable_layers,  # 5 = fine-tune everything (lower if overfitting)
        min_size=crop_size, max_size=crop_size,      # == crop size  ->  no internal resizing
        # --- crowded-scene settings -------------------------------------------------------
        box_detections_per_img=max_dets,            # default 100 would drop most grains
        box_nms_thresh=box_nms,                     # default 0.5 suppresses touching, rotated grains
        rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1500,
        box_batch_size_per_image=512,
    )
    # Small-object anchors. One size per level x 3 aspect ratios keeps the RPN head shape
    # identical to the pretrained one, so pretrained RPN weights still load.
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((s,) for s in anchor_sizes), aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    # New class-specific heads (background + your classes)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes)
    return model


# =========================================================================================
# Training / evaluation
# =========================================================================================
def lr_at(it, total, base_lr, warmup, floor=0.01):
    """Linear warm-up, then cosine decay to base_lr*floor. Called every iteration."""
    if it < warmup:
        return base_lr * (0.001 + 0.999 * it / warmup)
    p = (it - warmup) / max(1, total - warmup)
    return base_lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model, lr, momentum, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)  # no weight decay on biases / norms
    return torch.optim.SGD(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, momentum=momentum)


def train_one_epoch(model, optimizer, loader, device, epoch, total_iters, args, scaler):
    model.train()
    sums, seen, t0 = defaultdict(float), 0, time.time()
    for it, (images, targets) in enumerate(loader):
        g_it = epoch * len(loader) + it
        lr = lr_at(g_it, total_iters, args.lr, args.warmup_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        images = [im.to(device, non_blocking=True) for im in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with torch.autocast(device_type=device.type, enabled=args.amp):
            loss_dict = model(images, targets)  # train mode -> dict of 5 losses
            loss = sum(LOSS_WEIGHTS.get(k, 1.0) * v for k, v in loss_dict.items())

        if not torch.isfinite(loss):
            print(f"  [warn] non-finite loss at iter {g_it}, skipping batch: "
                  f"{ {k: round(v.item(), 3) for k, v in loss_dict.items()} }")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        seen += 1
        sums["total"] += loss.item()
        for k, v in loss_dict.items():
            sums[k] += v.item()
        if it % args.print_freq == 0:
            print(f"  ep {epoch + 1} [{it}/{len(loader)}] lr {lr:.5f} loss {loss.item():.3f} "
                  + " ".join(f"{k.replace('loss_', '')} {v.item():.3f}" for k, v in loss_dict.items()))
    avg = {k: v / max(seen, 1) for k, v in sums.items()}
    print(f"epoch {epoch + 1} done in {time.time() - t0:.0f}s | " + " ".join(f"{k} {v:.3f}" for k, v in avg.items()))
    return avg


def _rle(binary_mask):
    r = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


@torch.no_grad()
def evaluate_coco(model, loader, device, class_names, max_dets=500):
    """COCO mask AP over the validation tiles. Returns dict with AP, AP50, AP75, per-class AP50."""
    model.eval()
    gt = {"images": [], "annotations": [],
          "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)]}
    dets, ann_id, tile_id = [], 1, 1
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for im, tgt, out in zip(images, targets, outputs):
            gt["images"].append({"id": tile_id, "height": im.shape[1], "width": im.shape[2]})
            for m, b, l in zip(tgt["masks"].numpy(), tgt["boxes"].numpy(), tgt["labels"].numpy()):
                gt["annotations"].append({
                    "id": ann_id, "image_id": tile_id, "category_id": int(l), "iscrowd": 0,
                    "segmentation": _rle(m), "area": float(m.sum()),
                    "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
                ann_id += 1
            pm = (out["masks"][:, 0] > 0.5).cpu().numpy()
            for m, s, l in zip(pm, out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
                dets.append({"image_id": tile_id, "category_id": int(l),
                             "segmentation": _rle(m), "score": float(s)})
            tile_id += 1
    if not dets or not gt["annotations"]:
        print("  [eval] no detections / no ground truth -> AP = 0")
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    E = COCOeval(coco_gt, coco_gt.loadRes(dets), "segm")
    E.params.maxDets = [1, 10, max_dets]  # default cap is 100 detections/image -> useless for dense tiles
    E.evaluate()
    E.accumulate()

    prec = E.eval["precision"]  # [IoU, recall, class, area, maxDets]

    def ap(iou=None, k=None):
        p = prec
        if iou is not None:
            p = p[np.isclose(E.params.iouThrs, iou)]
        if k is not None:
            p = p[:, :, k:k + 1]
        p = p[:, :, :, 0, -1]  # area 'all', largest maxDets
        p = p[p > -1]
        return float(p.mean()) if p.size else 0.0

    res = {"AP": ap(), "AP50": ap(0.5), "AP75": ap(0.75)}
    for k, n in enumerate(class_names):
        res[f"AP50_{n}"] = ap(0.5, k)
    print("  [eval] mask " + " ".join(f"{k} {v:.3f}" for k, v in res.items()))
    return res


def save_ckpt(path, model, optimizer, epoch, best_ap, args):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best_ap": best_ap, "class_names": args.classes, "crop_size": args.crop_size,
                "anchor_sizes": args.anchor_sizes}, path)


# =========================================================================================
# Main
# =========================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="folder with the images referenced by file_name")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--val-json", default=None)
    ap.add_argument("--val-images", default=None, help="defaults to --images")
    ap.add_argument("--output-dir", default="runs/maskrcnn")
    ap.add_argument("--classes", nargs="+", default=["whole_rice", "broken_rice"],
                    help="COCO category names to train on (order -> label ids 1..K). "
                         "Use a single name to treat every grain as one class.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--crop-size", type=int, default=1024, help="lower to 768/640 if you run out of GPU memory")
    ap.add_argument("--crops-per-image", type=int, default=4, help="random crops drawn per image per epoch")
    ap.add_argument("--anchor-sizes", type=int, nargs=5, default=[16, 32, 64, 128, 256])
    ap.add_argument("--lr", type=float, default=0.005, help="SGD lr for batch 2-4; scale linearly with batch")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--trainable-layers", type=int, default=5)
    ap.add_argument("--max-dets", type=int, default=500, help="max detections per crop/tile")
    ap.add_argument("--box-nms", type=float, default=0.6)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--print-freq", type=int, default=20)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--resume", default=None, help="path to last.pth")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = args.amp and device.type == "cuda"
    torch.backends.cudnn.benchmark = True  # fixed-size crops -> safe speed-up
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | amp: {args.amp} | classes: {args.classes}")

    # ---- data ------------------------------------------------------------------------------
    train_ds = CocoGrainDataset(args.images, args.train_json, args.classes, args.crop_size,
                                train=True, crops_per_image=args.crops_per_image)
    print_grain_stats(train_ds, args.anchor_sizes)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    val_loader = None
    if args.val_json:
        val_ds = CocoGrainDataset(args.val_images or args.images, args.val_json, args.classes,
                                  args.crop_size, train=False)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=args.workers,
                                collate_fn=collate_fn)
        print(f"[data] validation: {len(val_ds.img_ids)} images -> {len(val_ds)} tiles")

    # ---- model / optimiser ------------------------------------------------------------------
    model = build_model(len(args.classes) + 1, args.crop_size, args.anchor_sizes,
                        pretrained=not args.no_pretrained, trainable_layers=args.trainable_layers,
                        max_dets=args.max_dets, box_nms=args.box_nms).to(device)
    optimizer = make_optimizer(model, args.lr, args.momentum, args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    start_epoch, best_ap = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best_ap = ck["epoch"] + 1, ck.get("best_ap", -1.0)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    total_iters = args.epochs * len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, total_iters, args, scaler)

        is_last = epoch + 1 == args.epochs
        if val_loader is not None and ((epoch + 1) % args.eval_every == 0 or is_last):
            metrics = evaluate_coco(model, val_loader, device, args.classes, args.max_dets)
            if metrics["AP"] > best_ap:
                best_ap = metrics["AP"]
                save_ckpt(out_dir / "best.pth", model, optimizer, epoch, best_ap, args)
                print(f"  new best mask AP {best_ap:.3f} -> best.pth")
        save_ckpt(out_dir / "last.pth", model, optimizer, epoch, best_ap, args)

    if val_loader is None:  # no validation set: best == last
        save_ckpt(out_dir / "best.pth", model, optimizer, args.epochs - 1, best_ap, args)
    print(f"done. checkpoints in {out_dir}/ (best mask AP {best_ap:.3f})")


if __name__ == "__main__":
    main()
