# -*- coding: utf-8 -*-
"""Source-only refiner training for MoCH evaluation.

This script is deliberately strict about data use:
  * training/validation frames come only from MoCA_Video and CAD
  * MoCH frames are used only in apply/evaluate steps

Prediction layouts:
  source base root:
    <root>/MoCA_Video/<split>/<sequence>/<frame_stem>.png
    <root>/CAD/All/<sequence>/<cad_frame_stem>.png

  MoCH base root:
    <root>/MoCH/<split>/<sequence>/<frame_stem>.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def sorted_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_bucket(text: str, modulo: int = 100) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def read_rgb(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as raw:
        img = raw.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1)
    return (ten - MEAN) / STD


def read_gray(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as raw:
        img = raw.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def save_prob(prob: np.ndarray, out_path: Path, ref_path: Path) -> None:
    with Image.open(ref_path) as ref:
        out_size = ref.size
    img = Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L")
    if img.size != out_size:
        img = img.resize(out_size, Image.BILINEAR)
    ensure_dir(out_path.parent)
    img.save(out_path)


def list_moca_sequences(root: Path, splits: list[str]) -> list[dict]:
    seqs = []
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
            frames = []
            for gt in sorted_images(gt_dir):
                img = imgs.get(gt.stem)
                if img is not None:
                    frames.append(
                        {
                            "dataset": "MoCA_Video",
                            "split": split,
                            "sequence": seq_dir.name,
                            "frame_name": gt.stem,
                            "img_path": img,
                            "gt_path": gt,
                        }
                    )
            if frames:
                seqs.append({"dataset": "MoCA_Video", "split": split, "sequence": seq_dir.name, "frames": frames})
    return seqs


def list_cad_sequences(root: Path) -> list[dict]:
    seqs = []
    orig = root / "original_data"
    mask = root / "converted_mask"
    if not orig.exists() or not mask.exists():
        return seqs
    for seq_dir in sorted(p for p in orig.iterdir() if p.is_dir()):
        frame_dir = seq_dir / "frames"
        gt_dir = mask / seq_dir.name / "groundtruth"
        if not frame_dir.exists() or not gt_dir.exists():
            continue
        imgs = {p.stem: p for p in sorted_images(frame_dir)}
        frames = []
        for gt in sorted_images(gt_dir):
            suffix = gt.stem
            prefix = seq_dir.name + "_"
            if suffix.startswith(prefix):
                suffix = suffix[len(prefix) :]
            img = imgs.get(suffix)
            if img is None:
                continue
            frames.append(
                {
                    "dataset": "CAD",
                    "split": "All",
                    "sequence": seq_dir.name,
                    "frame_name": gt.stem,
                    "img_path": img,
                    "gt_path": gt,
                }
            )
        if frames:
            seqs.append({"dataset": "CAD", "split": "All", "sequence": seq_dir.name, "frames": frames})
    return seqs


def list_source_sequences(moca_root: Path, cad_root: Path, mode: str) -> list[dict]:
    seqs: list[dict] = []
    if mode == "train":
        seqs.extend(list_moca_sequences(moca_root, ["TrainDataset_per_sq"]))
    elif mode == "val":
        seqs.extend(list_moca_sequences(moca_root, ["TestDataset_per_sq"]))
    else:
        raise ValueError(mode)

    for seq in list_cad_sequences(cad_root):
        bucket = stable_bucket("CAD/" + seq["sequence"], 100)
        if (mode == "val" and bucket >= 80) or (mode == "train" and bucket < 80):
            seqs.append(seq)
    return seqs


def list_moch_sequences(root: Path, splits: list[str]) -> list[dict]:
    seqs = []
    for split in splits:
        split_dir = root / "data" / split
        if not split_dir.exists():
            continue
        for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            img_dir = seq_dir / "images"
            gt_dir = seq_dir / "gts"
            if not img_dir.exists() or not gt_dir.exists():
                continue
            imgs = {p.stem: p for p in sorted_images(img_dir)}
            frames = []
            for gt in sorted_images(gt_dir):
                img = imgs.get(gt.stem)
                if img is not None:
                    frames.append((img, gt))
            if frames:
                seqs.append({"split": split, "sequence": seq_dir.name, "frames": frames})
    return seqs


def source_pred_path(root: Path, frame: dict) -> Path:
    return root / frame["dataset"] / frame["split"] / frame["sequence"] / f"{frame['frame_name']}.png"


def moch_pred_path(root: Path, split: str, sequence: str, stem: str) -> Path:
    return root / "MoCH" / split / sequence / f"{stem}.png"


@dataclass
class FrameItem:
    index: int
    count: int
    img_path: Path
    prev_img_path: Path
    gt_path: Path
    pred_path: Path
    prev_pred_path: Path
    dataset: str
    split: str
    sequence: str


class SourceRefinerDataset(Dataset):
    def __init__(
        self,
        moca_root: Path,
        cad_root: Path,
        base_root: Path,
        mode: str,
        size: int,
        max_items: int | None = None,
    ):
        self.size = size
        self.items: list[FrameItem] = []
        for seq in list_source_sequences(moca_root, cad_root, mode):
            frames = seq["frames"]
            count = len(frames)
            for idx, frame in enumerate(frames):
                pred = source_pred_path(base_root, frame)
                if not pred.exists():
                    continue
                prev_idx = max(0, idx - 1)
                prev_frame = frames[prev_idx]
                prev_pred = source_pred_path(base_root, prev_frame)
                if not prev_pred.exists():
                    prev_pred = pred
                self.items.append(
                    FrameItem(
                        index=idx,
                        count=count,
                        img_path=Path(frame["img_path"]),
                        prev_img_path=Path(prev_frame["img_path"]),
                        gt_path=Path(frame["gt_path"]),
                        pred_path=pred,
                        prev_pred_path=prev_pred,
                        dataset=frame["dataset"],
                        split=frame["split"],
                        sequence=frame["sequence"],
                    )
                )
        if max_items and len(self.items) > max_items:
            random.Random(17).shuffle(self.items)
            self.items = self.items[:max_items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        cur = read_rgb(it.img_path, self.size)
        prev = read_rgb(it.prev_img_path, self.size)
        raw = read_gray(it.pred_path, self.size)
        prev_raw = read_gray(it.prev_pred_path, self.size)
        gt = (read_gray(it.gt_path, self.size) >= 0.5).float()

        cur_unit = (cur * STD + MEAN).clamp(0, 1)
        prev_unit = (prev * STD + MEAN).clamp(0, 1)
        motion = (cur_unit - prev_unit).abs().mean(dim=0, keepdim=True)
        motion_norm = (motion / (motion.mean() + 2.0 * motion.std() + 1e-6)).clamp(0, 1)
        ratio = 0.0 if it.count <= 1 else it.index / float(it.count - 1)
        ratio_map = torch.full((1, self.size, self.size), ratio, dtype=torch.float32)
        early_map = torch.full((1, self.size, self.size), 1.0 if ratio < 1.0 / 3.0 else 0.0, dtype=torch.float32)
        x = torch.cat([cur, raw, prev_raw, motion_norm, ratio_map, early_map], dim=0)

        early_weight = 1.35 if ratio < 1.0 / 3.0 else 1.0
        low_motion_weight = 1.15 if float(motion.mean()) < 0.03 else 1.0
        return {
            "x": x,
            "gt": gt,
            "raw": raw,
            "prev_raw": prev_raw,
            "motion": motion_norm,
            "weight": torch.tensor(early_weight * low_motion_weight, dtype=torch.float32),
        }


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(4, cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(4, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CausalRefiner(nn.Module):
    def __init__(self, in_ch: int = 8, width: int = 32):
        super().__init__()
        self.e1 = ConvBlock(in_ch, width)
        self.e2 = ConvBlock(width, width * 2)
        self.e3 = ConvBlock(width * 2, width * 4)
        self.d2 = ConvBlock(width * 4 + width * 2, width * 2)
        self.d1 = ConvBlock(width * 2 + width, width)
        self.mask_head = nn.Conv2d(width, 1, 1)
        self.gate_head = nn.Conv2d(width, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.mask_head(d1), torch.sigmoid(self.gate_head(d1))


def dice_loss(prob, gt, eps=1e-6):
    dims = (1, 2, 3)
    inter = (prob * gt).sum(dim=dims)
    den = prob.sum(dim=dims) + gt.sum(dim=dims)
    return 1.0 - ((2.0 * inter + eps) / (den + eps))


def center_loss(prob, gt, eps=1e-6):
    b, _, h, w = prob.shape
    yy = torch.linspace(0, 1, h, device=prob.device).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, w, device=prob.device).view(1, 1, 1, w)
    pa = prob.sum(dim=(2, 3), keepdim=True) + eps
    ga = gt.sum(dim=(2, 3), keepdim=True) + eps
    pcx = (prob * xx).sum(dim=(2, 3), keepdim=True) / pa
    pcy = (prob * yy).sum(dim=(2, 3), keepdim=True) / pa
    gcx = (gt * xx).sum(dim=(2, 3), keepdim=True) / ga
    gcy = (gt * yy).sum(dim=(2, 3), keepdim=True) / ga
    has_gt = (gt.sum(dim=(1, 2, 3)) > 0).float()
    dist = torch.sqrt((pcx - gcx).pow(2) + (pcy - gcy).pow(2) + eps).view(b)
    return dist * has_gt


def boundary_map(mask):
    eroded = 1.0 - F.max_pool2d(1.0 - mask, 3, stride=1, padding=1)
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0, 1)


def loss_fn(correction, gate, batch):
    raw = batch["raw"]
    gt = batch["gt"]
    motion = batch["motion"]
    weight = batch["weight"]
    raw_logit = torch.logit(raw.clamp(1e-4, 1 - 1e-4))
    logits = gate * raw_logit + (1.0 - gate) * correction
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none").mean(dim=(1, 2, 3))
    dice = dice_loss(prob, gt)
    ctr = center_loss(prob, gt)
    area = (prob.mean(dim=(1, 2, 3)) - gt.mean(dim=(1, 2, 3))).abs()
    bdice = dice_loss(boundary_map(prob), boundary_map(gt))
    low_motion = (1.0 - motion.mean(dim=(1, 2, 3))).clamp(0, 1)
    temporal = ((prob - batch["prev_raw"]).abs().mean(dim=(1, 2, 3)) * low_motion)
    loss = bce + dice + 0.30 * ctr + 0.08 * area + 0.12 * bdice + 0.02 * temporal
    return (loss * weight).mean(), {
        "bce": float(bce.mean().detach().cpu()),
        "dice": float(dice.mean().detach().cpu()),
        "center": float(ctr.mean().detach().cpu()),
        "area": float(area.mean().detach().cpu()),
    }


def train(args) -> Path:
    device = torch.device(args.device)
    train_ds = SourceRefinerDataset(
        Path(args.moca_root),
        Path(args.cad_root),
        Path(args.source_base_root),
        mode="train",
        size=args.size,
        max_items=args.max_train_items or None,
    )
    val_ds = SourceRefinerDataset(
        Path(args.moca_root),
        Path(args.cad_root),
        Path(args.source_base_root),
        mode="val",
        size=args.size,
        max_items=args.max_val_items or None,
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"empty source dataset: train={len(train_ds)} val={len(val_ds)}")
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, drop_last=False)
    model = CausalRefiner(in_ch=8, width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best = {"val": 1e9, "epoch": -1}
    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)
    with (work_dir / "source_training_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strict_note": "MoCH is not used for training or validation. MoCH is apply/eval only.",
                "train_items": len(train_ds),
                "val_items": len(val_ds),
                "args": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    log_path = work_dir / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "bce", "dice", "center", "area"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            parts = {"bce": [], "dice": [], "center": [], "area": []}
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                loss, p = loss_fn(*model(batch["x"]), batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
                for k, v in p.items():
                    parts[k].append(v)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    loss, _ = loss_fn(*model(batch["x"]), batch)
                    val_losses.append(float(loss.detach().cpu()))
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
                "bce": float(np.mean(parts["bce"])) if parts["bce"] else 0.0,
                "dice": float(np.mean(parts["dice"])) if parts["dice"] else 0.0,
                "center": float(np.mean(parts["center"])) if parts["center"] else 0.0,
                "area": float(np.mean(parts["area"])) if parts["area"] else 0.0,
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps({"model": args.model_name, **row}, ensure_ascii=False), flush=True)
            if row["val_loss"] < best["val"]:
                best = {"val": row["val_loss"], "epoch": epoch}
                torch.save({"model": model.state_dict(), "args": vars(args), "best": best}, work_dir / "best.pth")
    print(json.dumps({"model": args.model_name, "best": best, "checkpoint": str(work_dir / "best.pth")}, ensure_ascii=False), flush=True)
    return work_dir / "best.pth"


def tensor_from_rgb(path: Path, size: int) -> torch.Tensor:
    return read_rgb(path, size).unsqueeze(0)


def tensor_from_gray(path: Path, size: int) -> torch.Tensor:
    return read_gray(path, size).unsqueeze(0)


def soft_blur_np(prob: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def apply(args, checkpoint: Path) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(checkpoint, map_location=device)
    model = CausalRefiner(in_ch=8, width=args.width).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    out_root = Path(args.output_root) / args.model_name
    base_root = Path(args.moch_base_root)
    manifest = []
    with torch.no_grad():
        for seq in list_moch_sequences(Path(args.moch_root), args.splits):
            frames = seq["frames"]
            count = len(frames)
            prev_belief_np = None
            prev_img = None
            motion_hist = []
            locked = False
            for idx, (img_path, gt_path) in enumerate(frames):
                pred_path = moch_pred_path(base_root, seq["split"], seq["sequence"], gt_path.stem)
                if not pred_path.exists():
                    continue
                cur = tensor_from_rgb(img_path, args.size).to(device)
                prev = tensor_from_rgb(prev_img or img_path, args.size).to(device)
                raw = tensor_from_gray(pred_path, args.size).to(device)
                if prev_belief_np is None:
                    prev_belief = raw.clone()
                else:
                    prev_belief = torch.from_numpy(prev_belief_np).float().view(1, 1, args.size, args.size).to(device)

                cur_unit = (cur.squeeze(0) * STD.to(device) + MEAN.to(device)).clamp(0, 1).unsqueeze(0)
                prev_unit = (prev.squeeze(0) * STD.to(device) + MEAN.to(device)).clamp(0, 1).unsqueeze(0)
                motion = (cur_unit - prev_unit).abs().mean(dim=1, keepdim=True)
                motion_score = float(motion.mean().detach().cpu())
                motion_hist.append(motion_score)
                hist = motion_hist[:-1]
                if len(hist) >= 3:
                    hmean = sum(hist) / len(hist)
                    hstd = math.sqrt(sum((m - hmean) ** 2 for m in hist) / max(1, len(hist) - 1))
                else:
                    hmean, hstd = motion_score, 0.0
                motion_onset = len(hist) >= 3 and motion_score > max(args.min_motion_onset, hmean + args.motion_std * hstd)
                motion_norm = (motion / (motion.mean() + 2.0 * motion.std() + 1e-6)).clamp(0, 1)
                ratio = 0.0 if count <= 1 else idx / float(count - 1)
                ratio_map = torch.full((1, 1, args.size, args.size), ratio, device=device)
                early_map = torch.full((1, 1, args.size, args.size), 1.0 if ratio < 1.0 / 3.0 else 0.0, device=device)
                x = torch.cat([cur, raw, prev_belief, motion_norm, ratio_map, early_map], dim=1)
                correction, gate = model(x)
                raw_logit = torch.logit(raw.clamp(1e-4, 1 - 1e-4))
                net = torch.sigmoid(gate * raw_logit + (1.0 - gate) * correction)

                raw_np = raw.squeeze().detach().cpu().numpy()
                net_np = net.squeeze().detach().cpu().numpy()
                prev_np = prev_belief.squeeze().detach().cpu().numpy()
                raw_conf = float(raw.max().detach().cpu())
                net_conf = float(net.max().detach().cpu())
                if motion_onset and max(raw_conf, net_conf) >= args.lock_conf:
                    locked = True
                if idx == 0:
                    final = args.first_net * net_np + (1.0 - args.first_net) * raw_np
                elif not locked and ratio < 1.0 / 3.0:
                    final = args.early_net * net_np + args.early_raw * raw_np + args.early_prev * soft_blur_np(prev_np, args.prev_blur)
                elif locked:
                    prev_w = args.locked_prev_low_motion if motion_score < hmean else args.locked_prev_high_motion
                    rest = 1.0 - prev_w
                    final = rest * (args.locked_net * net_np + (1.0 - args.locked_net) * raw_np) + prev_w * prev_np
                else:
                    final = args.mid_net * net_np + args.mid_raw * raw_np + args.mid_prev * prev_np
                final = np.clip(final, 0, 1)
                prev_belief_np = final.astype(np.float32)
                prev_img = img_path
                out_path = out_root / "MoCH" / seq["split"] / seq["sequence"] / f"{gt_path.stem}.png"
                save_prob(final, out_path, img_path)
                manifest.append(
                    {
                        "model": args.model_name,
                        "split": seq["split"],
                        "sequence": seq["sequence"],
                        "frame": idx,
                        "frame_name": gt_path.stem,
                        "motion_score": motion_score,
                        "motion_onset": int(motion_onset),
                        "locked": int(locked),
                        "raw_conf": raw_conf,
                        "net_conf": net_conf,
                        "output": str(out_path),
                    }
                )
            if args.progress and len(manifest) and len(manifest) % args.progress == 0:
                print(json.dumps({"model": args.model_name, "applied_frames": len(manifest)}, ensure_ascii=False), flush=True)
    with (Path(args.work_dir) / "moch_apply_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        if manifest:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)
    print(json.dumps({"model": args.model_name, "moch_frames": len(manifest), "output_root": str(out_root)}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--moca-root", default="MoCA_Video")
    ap.add_argument("--cad-root", default="CamouflagedAnimalDataset")
    ap.add_argument("--moch-root", default="MoCH")
    ap.add_argument("--source-base-root", required=True)
    ap.add_argument("--moch-base-root", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-train-items", type=int, default=0)
    ap.add_argument("--max-val-items", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--progress", type=int, default=500)
    ap.add_argument("--motion-std", type=float, default=1.0)
    ap.add_argument("--min-motion-onset", type=float, default=0.012)
    ap.add_argument("--lock-conf", type=float, default=0.62)
    ap.add_argument("--prev-blur", type=float, default=1.4)
    ap.add_argument("--first-net", type=float, default=0.65)
    ap.add_argument("--early-net", type=float, default=0.55)
    ap.add_argument("--early-raw", type=float, default=0.25)
    ap.add_argument("--early-prev", type=float, default=0.20)
    ap.add_argument("--mid-net", type=float, default=0.65)
    ap.add_argument("--mid-raw", type=float, default=0.25)
    ap.add_argument("--mid-prev", type=float, default=0.10)
    ap.add_argument("--locked-net", type=float, default=0.78)
    ap.add_argument("--locked-prev-low-motion", type=float, default=0.20)
    ap.add_argument("--locked-prev-high-motion", type=float, default=0.05)
    args = ap.parse_args()

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    ensure_dir(Path(args.work_dir))
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.work_dir) / "best.pth"
    if args.train:
        checkpoint = train(args)
    if args.apply:
        apply(args, checkpoint)


if __name__ == "__main__":
    main()
