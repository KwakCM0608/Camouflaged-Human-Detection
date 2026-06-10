# -*- coding: utf-8 -*-
"""Generate MoCH predictions for newly downloaded external models.

The output layout matches tools/evaluate_moch_baselines.py:
  <out>/<model>/MoCH/<split>/<sequence>/<frame_stem>.png

FastSAM/MobileSAM are promptable/proposal models, so this script uses an
automatic person-prior proposal selector without using GT masks.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from contextlib import contextmanager


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def sorted_images(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def list_moch_frames(root: Path, splits: Sequence[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for split in splits:
        split_root = root / "data" / split
        if not split_root.exists():
            continue
        for scene_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            img_dir = scene_dir / "images"
            gt_dir = scene_dir / "gts"
            if not img_dir.exists() or not gt_dir.exists():
                continue
            imgs = sorted_images(img_dir)
            gts = sorted_images(gt_dir)
            img_by_stem = {p.stem: p for p in imgs}
            valid = [g for g in gts if g.stem in img_by_stem]
            for idx, gt_path in enumerate(valid):
                rows.append(
                    {
                        "split": split,
                        "scene": scene_dir.name,
                        "frame_index": idx,
                        "frame_count": len(valid),
                        "frame_name": gt_path.stem,
                        "img_path": img_by_stem[gt_path.stem],
                    }
                )
    return rows


def ensure_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_prob(out_root: Path, model_name: str, row: Dict[str, object], prob: np.ndarray) -> Path:
    out = (
        out_root
        / model_name
        / "MoCH"
        / str(row["split"])
        / str(row["scene"])
        / (str(row["frame_name"]) + ".png")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob.squeeze()
    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    cv2.imwrite(str(out), (prob * 255).astype(np.uint8))
    return out


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def center_from_bbox(box: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[float, float]]:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def person_prior_score(mask: np.ndarray, image_shape: Tuple[int, int], prev_mask: Optional[np.ndarray]) -> float:
    h, w = image_shape
    area = float(mask.sum())
    if area <= 0:
        return -1e9
    area_ratio = area / max(1.0, h * w)
    if area_ratio < 0.001 or area_ratio > 0.75:
        return -1e6
    box = mask_bbox(mask)
    if box is None:
        return -1e9
    x1, y1, x2, y2 = box
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    cx, cy = center_from_bbox(box) or (w / 2.0, h / 2.0)

    # MoCH is human-centered; prefer compact person-like regions, but keep this
    # soft so crouched/lying humans are not automatically rejected.
    aspect = bh / bw
    aspect_score = math.exp(-abs(math.log(max(aspect, 1e-3) / 1.8)))
    area_score = math.exp(-abs(math.log(max(area_ratio, 1e-5) / 0.08)))
    center_score = 1.0 - min(1.0, abs(cx - w / 2.0) / (w / 2.0))
    vertical_score = 1.0 - min(1.0, abs(cy - h * 0.55) / (h * 0.55))

    score = 0.95 * area_score + 0.8 * aspect_score + 0.45 * center_score + 0.25 * vertical_score

    if prev_mask is not None and prev_mask.any():
        prev_box = mask_bbox(prev_mask)
        prev_center = center_from_bbox(prev_box)
        this_center = center_from_bbox(box)
        temporal_iou = mask_iou(mask, prev_mask)
        score += 1.25 * temporal_iou
        if prev_center and this_center:
            dist = math.hypot(this_center[0] - prev_center[0], this_center[1] - prev_center[1])
            diag = math.hypot(w, h)
            score += 0.45 * (1.0 - min(1.0, dist / max(1.0, diag * 0.5)))
    return float(score)


def select_proposal(
    masks: Iterable[np.ndarray],
    image_shape: Tuple[int, int],
    prev_mask: Optional[np.ndarray],
) -> np.ndarray:
    h, w = image_shape
    best_score = -1e18
    best_mask: Optional[np.ndarray] = None
    for raw in masks:
        mask = np.asarray(raw).astype(bool)
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        score = person_prior_score(mask, (h, w), prev_mask)
        if score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None:
        return np.zeros((h, w), dtype=np.float32)
    return best_mask.astype(np.float32)


def should_skip(out_root: Path, model_name: str, row: Dict[str, object], overwrite: bool) -> bool:
    if overwrite:
        return False
    out = out_root / model_name / "MoCH" / str(row["split"]) / str(row["scene"]) / (str(row["frame_name"]) + ".png")
    return out.exists()


def write_manifest(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = sorted({key for row in rows for key in row.keys()})
    else:
        fields = ["model", "status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_zoomnext(args: argparse.Namespace, rows: List[Dict[str, object]], device: str) -> List[Dict[str, object]]:
    sys.path.insert(0, str(Path("external_models/ZoomNeXt").resolve()))
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        @contextmanager
        def _sdp_kernel_compat(*_args, **_kwargs):
            yield

        torch.backends.cuda.sdp_kernel = _sdp_kernel_compat  # type: ignore[attr-defined]
    if not hasattr(F, "scaled_dot_product_attention"):
        def _scaled_dot_product_attention_compat(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
        ):
            scale_factor = (q.shape[-1] ** -0.5) if scale is None else scale
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
            if is_causal:
                q_len, k_len = q.shape[-2], k.shape[-2]
                causal_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril()
                attn = attn.masked_fill(~causal_mask, float("-inf"))
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask
            attn = torch.softmax(attn, dim=-1)
            if dropout_p:
                attn = torch.dropout(attn, dropout_p, train=False)
            return torch.matmul(attn, v)

        F.scaled_dot_product_attention = _scaled_dot_product_attention_compat  # type: ignore[attr-defined]
    from methods import PvtV2B5_ZoomNeXt

    model_name = "ZoomNeXt_PvtV2B5"
    weight = Path("external_models/ZoomNeXt/weights/pvtv2-b5-zoomnext.pth")
    model = PvtV2B5_ZoomNeXt(pretrained=False, num_frames=1, use_checkpoint=False)
    state = torch.load(str(weight), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    todo = [r for r in rows if not should_skip(args.out_root, model_name, r, args.overwrite)]
    done: List[Dict[str, object]] = []
    t0 = time.time()
    batch_size = max(1, args.zoomnext_batch_size)
    with torch.no_grad():
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            images_s, images_m, images_l = [], [], []
            orig_shapes: List[Tuple[int, int]] = []
            for row in batch:
                image = ensure_rgb(Path(row["img_path"]))
                oh, ow = image.shape[:2]
                orig_shapes.append((oh, ow))
                base_h = base_w = args.zoomnext_size
                s = cv2.resize(image, (base_w // 2, base_h // 2), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(image, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
                l = cv2.resize(image, (base_w * 3 // 2, base_h * 3 // 2), interpolation=cv2.INTER_LINEAR)
                images_s.append(torch.from_numpy(s).float().div(255).permute(2, 0, 1))
                images_m.append(torch.from_numpy(m).float().div(255).permute(2, 0, 1))
                images_l.append(torch.from_numpy(l).float().div(255).permute(2, 0, 1))
            data = {
                "image_s": torch.stack(images_s).to(device),
                "image_m": torch.stack(images_m).to(device),
                "image_l": torch.stack(images_l).to(device),
            }
            logits = model(data=data)
            probs = torch.sigmoid(logits).squeeze(1).detach().float().cpu().numpy()
            for row, prob, (oh, ow) in zip(batch, probs, orig_shapes):
                pmin, pmax = float(prob.min()), float(prob.max())
                if pmax > pmin:
                    prob = (prob - pmin) / (pmax - pmin)
                prob = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
                out = save_prob(args.out_root, model_name, row, prob)
                done.append({"model": model_name, "status": "ok", "pred_path": str(out), "img_path": str(row["img_path"])})
            if len(done) % max(batch_size, args.log_interval) == 0 or start + batch_size >= len(todo):
                elapsed = time.time() - t0
                rate = len(done) / max(elapsed, 1e-6)
                remain = (len(todo) - len(done)) / max(rate, 1e-6)
                print(f"[{model_name}] {len(done)}/{len(todo)} new frames, ETA {remain/60:.1f} min", flush=True)
    return done


def run_fastsam(args: argparse.Namespace, rows: List[Dict[str, object]], device: str) -> List[Dict[str, object]]:
    sys.path.insert(0, str(Path("external_models/FastSAM").resolve()))
    from fastsam import FastSAM

    model_name = "FastSAM_s_PersonPrior"
    model = FastSAM("external_models/FastSAM/weights/FastSAM-s.pt")
    done: List[Dict[str, object]] = []
    todo = [r for r in rows if not should_skip(args.out_root, model_name, r, args.overwrite)]
    prev_by_seq: Dict[Tuple[str, str], np.ndarray] = {}
    t0 = time.time()
    for idx, row in enumerate(todo, 1):
        img = Image.open(row["img_path"]).convert("RGB")
        results = model(
            img,
            device=device,
            retina_masks=True,
            imgsz=args.fastsam_imgsz,
            conf=args.fastsam_conf,
            iou=args.fastsam_iou,
            verbose=False,
        )
        h, w = np.asarray(img).shape[:2]
        masks = []
        if results is not None and len(results) and getattr(results[0], "masks", None) is not None:
            masks = [m.detach().cpu().numpy().astype(bool) for m in results[0].masks.data]
        key = (str(row["split"]), str(row["scene"]))
        pred = select_proposal(masks, (h, w), prev_by_seq.get(key))
        prev_by_seq[key] = pred.astype(bool)
        out = save_prob(args.out_root, model_name, row, pred)
        done.append({"model": model_name, "status": "ok", "pred_path": str(out), "img_path": str(row["img_path"])})
        if idx % args.log_interval == 0 or idx == len(todo):
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            remain = (len(todo) - idx) / max(rate, 1e-6)
            print(f"[{model_name}] {idx}/{len(todo)} new frames, ETA {remain/60:.1f} min", flush=True)
    return done


def run_mobilesam(args: argparse.Namespace, rows: List[Dict[str, object]], device: str) -> List[Dict[str, object]]:
    sys.path.insert(0, str(Path("external_models/MobileSAM").resolve()))
    from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

    model_name = f"MobileSAM_vit_t_P{args.mobilesam_points_per_side}_PersonPrior"
    sam = sam_model_registry["vit_t"](checkpoint="external_models/MobileSAM/weights/mobile_sam.pt")
    sam.to(device=device)
    sam.eval()
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.mobilesam_points_per_side,
        pred_iou_thresh=args.mobilesam_pred_iou_thresh,
        stability_score_thresh=args.mobilesam_stability_thresh,
        crop_n_layers=0,
        min_mask_region_area=args.mobilesam_min_region_area,
    )
    done: List[Dict[str, object]] = []
    todo = [r for r in rows if not should_skip(args.out_root, model_name, r, args.overwrite)]
    prev_by_seq: Dict[Tuple[str, str], np.ndarray] = {}
    t0 = time.time()
    for idx, row in enumerate(todo, 1):
        image = ensure_rgb(Path(row["img_path"]))
        h, w = image.shape[:2]
        anns = generator.generate(image)
        masks = [ann["segmentation"] for ann in anns]
        key = (str(row["split"]), str(row["scene"]))
        pred = select_proposal(masks, (h, w), prev_by_seq.get(key))
        prev_by_seq[key] = pred.astype(bool)
        out = save_prob(args.out_root, model_name, row, pred)
        done.append({"model": model_name, "status": "ok", "pred_path": str(out), "img_path": str(row["img_path"])})
        if idx % args.log_interval == 0 or idx == len(todo):
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            remain = (len(todo) - idx) / max(rate, 1e-6)
            print(f"[{model_name}] {idx}/{len(todo)} new frames, ETA {remain/60:.1f} min", flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moch-root", type=Path, default=Path("MoCH"))
    parser.add_argument("--out-root", type=Path, default=Path("MoCH_Test/predictions_new_downloaded"))
    parser.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    parser.add_argument("--models", nargs="+", default=["zoomnext", "fastsam", "mobilesam"])
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--zoomnext-size", type=int, default=384)
    parser.add_argument("--zoomnext-batch-size", type=int, default=4)
    parser.add_argument("--fastsam-imgsz", type=int, default=640)
    parser.add_argument("--fastsam-conf", type=float, default=0.25)
    parser.add_argument("--fastsam-iou", type=float, default=0.9)
    parser.add_argument("--mobilesam-points-per-side", type=int, default=16)
    parser.add_argument("--mobilesam-pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--mobilesam-stability-thresh", type=float, default=0.92)
    parser.add_argument("--mobilesam-min-region-area", type=int, default=64)
    args = parser.parse_args()

    rows = list_moch_frames(args.moch_root, args.splits)
    if args.max_frames > 0:
        rows = rows[: args.max_frames]
    print(f"MoCH frames: {len(rows)}; device={args.device}; models={args.models}", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    for model in args.models:
        started = time.time()
        try:
            if model == "zoomnext":
                manifest.extend(run_zoomnext(args, rows, args.device))
            elif model == "fastsam":
                manifest.extend(run_fastsam(args, rows, args.device))
            elif model == "mobilesam":
                manifest.extend(run_mobilesam(args, rows, args.device))
            else:
                manifest.append({"model": model, "status": "skipped", "reason": "unknown model key"})
            print(f"[{model}] finished in {(time.time()-started)/60:.1f} min", flush=True)
        except Exception as exc:
            print(f"[{model}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            manifest.append({"model": model, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})

    write_manifest(args.out_root / "manifest.csv", manifest)


if __name__ == "__main__":
    main()
