# -*- coding: utf-8 -*-
"""Generate ZoomNeXt predictions for MoCA_Video and CAD source training data."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def sorted_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def list_moca_frames(root: Path, splits: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            img_dir = seq_dir / "Imgs"
            gt_dir = seq_dir / "GT"
            if not img_dir.exists() or not gt_dir.exists():
                continue
            imgs = {p.stem: p for p in sorted_images(img_dir)}
            gts = [g for g in sorted_images(gt_dir) if g.stem in imgs]
            for idx, gt in enumerate(gts):
                rows.append(
                    {
                        "dataset": "MoCA_Video",
                        "split": split,
                        "scene": seq_dir.name,
                        "frame_index": idx,
                        "frame_count": len(gts),
                        "frame_name": gt.stem,
                        "img_path": imgs[gt.stem],
                    }
                )
    return rows


def list_cad_frames(root: Path) -> list[dict]:
    rows = []
    orig = root / "original_data"
    mask = root / "converted_mask"
    if not orig.exists() or not mask.exists():
        return rows
    for seq_dir in sorted(p for p in orig.iterdir() if p.is_dir()):
        frame_dir = seq_dir / "frames"
        gt_dir = mask / seq_dir.name / "groundtruth"
        if not frame_dir.exists() or not gt_dir.exists():
            continue
        imgs = {p.stem: p for p in sorted_images(frame_dir)}
        gts = sorted_images(gt_dir)
        valid = []
        for gt in gts:
            suffix = gt.stem
            prefix = seq_dir.name + "_"
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix) :]
            img = imgs.get(suffix)
            if img is not None:
                valid.append((gt, img))
        for idx, (gt, img) in enumerate(valid):
            rows.append(
                {
                    "dataset": "CAD",
                    "split": "All",
                    "scene": seq_dir.name,
                    "frame_index": idx,
                    "frame_count": len(valid),
                    "frame_name": gt.stem,
                    "img_path": img,
                }
            )
    return rows


def ensure_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def out_path(out_root: Path, row: dict) -> Path:
    return out_root / "ZoomNeXt_PvtV2B5" / row["dataset"] / row["split"] / row["scene"] / f"{row['frame_name']}.png"


def save_prob(path: Path, prob: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prob = np.nan_to_num(prob.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    cv2.imwrite(str(path), (prob * 255).astype(np.uint8))


def patch_torch_for_zoomnext() -> None:
    if not torch.cuda.is_available():
        class _FakeCudaProps:
            major = 0
            minor = 0

        torch.cuda.get_device_properties = lambda *_args, **_kwargs: _FakeCudaProps()  # type: ignore[assignment]
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        @contextmanager
        def _sdp_kernel_compat(*_args, **_kwargs):
            yield

        torch.backends.cuda.sdp_kernel = _sdp_kernel_compat  # type: ignore[attr-defined]
    if not hasattr(F, "scaled_dot_product_attention"):
        def _scaled_dot_product_attention_compat(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moca-root", default="MoCA_Video")
    ap.add_argument("--cad-root", default="CamouflagedAnimalDataset")
    ap.add_argument("--out-root", default="MoCH_Test/source_refiner_correct/source_predictions")
    ap.add_argument("--moca-splits", nargs="+", default=["TrainDataset_per_sq", "TestDataset_per_sq"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--log-interval", type=int, default=250)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    patch_torch_for_zoomnext()
    sys.path.insert(0, str(Path("external_models/ZoomNeXt").resolve()))
    from methods import PvtV2B5_ZoomNeXt

    rows = list_moca_frames(Path(args.moca_root), args.moca_splits) + list_cad_frames(Path(args.cad_root))
    if args.max_frames:
        rows = rows[: args.max_frames]
    out_root = Path(args.out_root)
    todo = [r for r in rows if args.overwrite or not out_path(out_root, r).exists()]
    print(f"source frames total={len(rows)} todo={len(todo)} device={args.device}", flush=True)

    model = PvtV2B5_ZoomNeXt(pretrained=False, num_frames=1, use_checkpoint=False)
    state = torch.load("external_models/ZoomNeXt/weights/pvtv2-b5-zoomnext.pth", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    manifest = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            images_s, images_m, images_l, shapes = [], [], [], []
            for row in batch:
                image = ensure_rgb(Path(row["img_path"]))
                oh, ow = image.shape[:2]
                shapes.append((oh, ow))
                s = cv2.resize(image, (args.size // 2, args.size // 2), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_LINEAR)
                l = cv2.resize(image, (args.size * 3 // 2, args.size * 3 // 2), interpolation=cv2.INTER_LINEAR)
                images_s.append(torch.from_numpy(s).float().div(255).permute(2, 0, 1))
                images_m.append(torch.from_numpy(m).float().div(255).permute(2, 0, 1))
                images_l.append(torch.from_numpy(l).float().div(255).permute(2, 0, 1))
            data = {
                "image_s": torch.stack(images_s).to(args.device),
                "image_m": torch.stack(images_m).to(args.device),
                "image_l": torch.stack(images_l).to(args.device),
            }
            logits = model(data=data)
            probs = torch.sigmoid(logits).squeeze(1).detach().float().cpu().numpy()
            for row, prob, (oh, ow) in zip(batch, probs, shapes):
                pmin, pmax = float(prob.min()), float(prob.max())
                if pmax > pmin:
                    prob = (prob - pmin) / (pmax - pmin)
                prob = cv2.resize(prob, (ow, oh), interpolation=cv2.INTER_LINEAR)
                pred = out_path(out_root, row)
                save_prob(pred, prob)
                manifest.append({**{k: str(v) for k, v in row.items() if k != "img_path"}, "img_path": str(row["img_path"]), "pred_path": str(pred)})
            done = min(start + len(batch), len(todo))
            if done % args.log_interval == 0 or done == len(todo):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-6)
                remain = (len(todo) - done) / max(rate, 1e-6)
                print(f"ZoomNeXt source {done}/{len(todo)} ETA {remain/60:.1f} min", flush=True)
    if manifest:
        manifest_path = out_root / "ZoomNeXt_PvtV2B5_source_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)
    print(f"done new={len(todo)}", flush=True)


if __name__ == "__main__":
    main()
