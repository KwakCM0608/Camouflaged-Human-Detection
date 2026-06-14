# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_source_refiner_correct as src


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sorted_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in src.IMAGE_SUFFIXES)


def list_camotion_sequences(root: Path, split: str) -> list[dict]:
    base = root / "CAMotion" / split
    if not base.exists():
        base = root / split
    seqs: list[dict] = []
    if not base.exists():
        return seqs
    for seq_dir in sorted(p for p in base.iterdir() if p.is_dir()):
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
                        "dataset": "CAMotion",
                        "split": split,
                        "sequence": seq_dir.name,
                        "frame_name": gt.stem,
                        "img_path": img,
                        "gt_path": gt,
                    }
                )
        if frames:
            seqs.append({"dataset": "CAMotion", "split": split, "sequence": seq_dir.name, "frames": frames})
    return seqs


def flatten_source(args, mode: str) -> list[dict]:
    frames = []
    for seq in src.list_source_sequences(Path(args.moca_root), Path(args.cad_root), mode):
        frames.extend(seq["frames"])
    cam_split = "TrainDataset_per_sq" if mode == "train" else "TestDataset_per_sq"
    for seq in list_camotion_sequences(Path(args.camotion_root), cam_split):
        frames.extend(seq["frames"])

    usable, missing = [], 0
    for fr in frames:
        pred = source_pred_path(fr, args)
        if pred is None or not pred.exists():
            missing += 1
            continue
        fr = dict(fr)
        fr["zn_pred_path"] = pred
        usable.append(fr)
    print(json.dumps({"event": "source_frames", "mode": mode, "frames": len(frames), "usable": len(usable), "missing": missing}), flush=True)
    return usable


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def source_pred_path(frame: dict, args) -> Path | None:
    name = f"{frame['frame_name']}.png"
    if frame["dataset"] == "CAMotion":
        root = Path(args.zn_camotion_root)
        return first_existing(
            [
                root / "MoCA_Video" / frame["split"] / frame["sequence"] / name,
                root / "CAMotion" / frame["split"] / frame["sequence"] / name,
            ]
        )
    return src.source_pred_path(Path(args.zn_source_root), frame)


def moch_pred_path(root: Path, split: str, sequence: str, stem: str) -> Path:
    return root / "MoCH" / split / sequence / f"{stem}.png"


def source_out_path(root: Path, frame: dict) -> Path:
    return root / frame["dataset"] / frame["split"] / frame["sequence"] / f"{frame['frame_name']}.png"


def camotion_out_path(root: Path, frame: dict) -> Path:
    return root / "MoCA_Video" / frame["split"] / frame["sequence"] / f"{frame['frame_name']}.png"


def read_rgb_unit(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as raw:
        img = raw.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def read_gray_unit(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as raw:
        img = raw.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def retinex_maps(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y = (0.299 * rgb[0:1] + 0.587 * rgb[1:2] + 0.114 * rgb[2:3]).clamp(0, 1)
    illum = F.avg_pool2d(y.unsqueeze(0), kernel_size=31, stride=1, padding=15).squeeze(0)
    ret = torch.log(y + 1e-3) - torch.log(illum + 1e-3)
    ret = ((ret - ret.mean()) / (ret.std() + 1e-6)).clamp(-2.0, 2.0)
    ret = (ret + 2.0) / 4.0
    return illum, ret


def proto_features(rgb: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    illum, ret = retinex_maps(rgb)
    return torch.cat([rgb, illum, ret, raw], dim=0)


def bg_score_map(feat: torch.Tensor, prototypes: torch.Tensor, temp: float) -> torch.Tensor:
    c, h, w = feat.shape
    pix = feat.permute(1, 2, 0).reshape(-1, c)
    prot = prototypes.to(feat.device, dtype=feat.dtype)
    chunks = []
    for start in range(0, pix.shape[0], 32768):
        d2 = ((pix[start : start + 32768, None, :] - prot[None, :, :]) ** 2).sum(dim=2)
        chunks.append(d2.min(dim=1).values)
    mind = torch.cat(chunks, dim=0).view(1, h, w)
    return torch.exp(-mind / max(temp, 1e-6)).clamp(0, 1)


def build_input(
    img_path: Path,
    raw_path: Path,
    prototypes: torch.Tensor,
    hard_prototypes: torch.Tensor,
    size: int,
    temp: float,
    hard_temp: float,
) -> torch.Tensor:
    rgb = read_rgb_unit(img_path, size)
    raw = read_gray_unit(raw_path, size)
    feat = proto_features(rgb, raw)
    score = bg_score_map(feat, prototypes, temp)
    hard_score = bg_score_map(feat, hard_prototypes, hard_temp)
    reject = raw * (1.0 - score)
    hard_reject = raw * (1.0 - hard_score)
    illum, ret = retinex_maps(rgb)
    return torch.cat([rgb, raw, illum, ret, score, hard_score, reject, hard_reject], dim=0)


def save_prob(prob: np.ndarray, out_path: Path, ref_img: Path) -> None:
    with Image.open(ref_img) as ref:
        out_size = ref.size
    img = Image.fromarray(np.clip(prob * 255.0, 0, 255).astype(np.uint8), mode="L")
    if img.size != out_size:
        img = img.resize(out_size, Image.BILINEAR)
    ensure_dir(out_path.parent)
    img.save(out_path)


def collect_bg_samples(args, frames: list[dict]) -> torch.Tensor:
    rng = np.random.default_rng(17)
    samples = []
    per_frame = max(1, args.samples_per_frame)
    for idx, fr in enumerate(frames):
        if len(samples) * per_frame >= args.max_proto_samples:
            break
        rgb = read_rgb_unit(Path(fr["img_path"]), args.proto_size)
        raw = read_gray_unit(Path(fr["zn_pred_path"]), args.proto_size)
        gt = read_gray_unit(Path(fr["gt_path"]), args.proto_size) < 0.5
        feat = proto_features(rgb, raw).permute(1, 2, 0).reshape(-1, 6).numpy()
        bg = gt.view(-1).numpy().astype(bool)
        ids = np.flatnonzero(bg)
        if ids.size == 0:
            continue
        take = min(per_frame, ids.size)
        chosen = rng.choice(ids, size=take, replace=False)
        samples.append(feat[chosen])
        if idx % 1000 == 0:
            print(json.dumps({"event": "collect_bg", "frames_seen": idx, "sample_blocks": len(samples)}), flush=True)
    arr = np.concatenate(samples, axis=0)
    if arr.shape[0] > args.max_proto_samples:
        arr = arr[rng.choice(arr.shape[0], size=args.max_proto_samples, replace=False)]
    return torch.from_numpy(arr.astype(np.float32))


def collect_hard_bg_samples(args, frames: list[dict]) -> torch.Tensor:
    rng = np.random.default_rng(23)
    samples = []
    per_frame = max(1, args.hard_samples_per_frame)
    for idx, fr in enumerate(frames):
        if len(samples) * per_frame >= args.max_hard_proto_samples:
            break
        rgb = read_rgb_unit(Path(fr["img_path"]), args.proto_size)
        raw = read_gray_unit(Path(fr["zn_pred_path"]), args.proto_size)
        gt = read_gray_unit(Path(fr["gt_path"]), args.proto_size) < 0.5
        feat = proto_features(rgb, raw).permute(1, 2, 0).reshape(-1, 6).numpy()
        bg = gt.view(-1).numpy().astype(bool)
        raw_flat = raw.view(-1).numpy()
        hard = bg & (raw_flat >= args.hard_raw_threshold)
        ids = np.flatnonzero(hard)
        if ids.size == 0:
            bg_ids = np.flatnonzero(bg)
            if bg_ids.size == 0:
                continue
            take_pool = min(max(per_frame * 8, per_frame), bg_ids.size)
            top_local = np.argpartition(raw_flat[bg_ids], -take_pool)[-take_pool:]
            ids = bg_ids[top_local]
        take = min(per_frame, ids.size)
        chosen = rng.choice(ids, size=take, replace=False)
        samples.append(feat[chosen])
        if idx % 1000 == 0:
            print(json.dumps({"event": "collect_hard_bg", "frames_seen": idx, "sample_blocks": len(samples)}), flush=True)
    if not samples:
        return collect_bg_samples(args, frames)
    arr = np.concatenate(samples, axis=0)
    if arr.shape[0] > args.max_hard_proto_samples:
        arr = arr[rng.choice(arr.shape[0], size=args.max_hard_proto_samples, replace=False)]
    return torch.from_numpy(arr.astype(np.float32))


def kmeans(samples: torch.Tensor, k: int, iters: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(17)
    init = torch.randperm(samples.shape[0], generator=generator)[:k]
    centers = samples[init].clone()
    for _ in range(iters):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(k, dtype=torch.float32)
        for start in range(0, samples.shape[0], 65536):
            batch = samples[start : start + 65536]
            d2 = ((batch[:, None, :] - centers[None, :, :]) ** 2).sum(dim=2)
            labels = d2.argmin(dim=1)
            sums.index_add_(0, labels, batch)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        nz = counts > 0
        centers[nz] = sums[nz] / counts[nz, None]
    return centers


def build_or_load_prototypes(args, train_frames: list[dict]) -> torch.Tensor:
    work = Path(args.work_dir)
    ensure_dir(work)
    proto_path = work / "background_prototypes.npy"
    if proto_path.exists() and not args.rebuild_prototypes:
        return torch.from_numpy(np.load(proto_path).astype(np.float32))
    samples = collect_bg_samples(args, train_frames)
    centers = kmeans(samples, args.num_prototypes, args.kmeans_iters)
    np.save(proto_path, centers.numpy())
    print(json.dumps({"event": "prototypes_built", "samples": int(samples.shape[0]), "k": args.num_prototypes, "path": str(proto_path)}), flush=True)
    return centers


def build_or_load_hard_prototypes(args, train_frames: list[dict]) -> torch.Tensor:
    work = Path(args.work_dir)
    ensure_dir(work)
    proto_path = work / "hard_background_prototypes.npy"
    if proto_path.exists() and not args.rebuild_prototypes:
        return torch.from_numpy(np.load(proto_path).astype(np.float32))
    samples = collect_hard_bg_samples(args, train_frames)
    centers = kmeans(samples, args.num_hard_prototypes, args.kmeans_iters)
    np.save(proto_path, centers.numpy())
    print(
        json.dumps(
            {
                "event": "hard_prototypes_built",
                "samples": int(samples.shape[0]),
                "k": args.num_hard_prototypes,
                "raw_threshold": args.hard_raw_threshold,
                "path": str(proto_path),
            }
        ),
        flush=True,
    )
    return centers


class SourceDataset(Dataset):
    def __init__(self, frames: list[dict], prototypes: torch.Tensor, hard_prototypes: torch.Tensor, args, max_items: int = 0):
        self.frames = list(frames)
        self.prototypes = prototypes
        self.hard_prototypes = hard_prototypes
        self.args = args
        if max_items and len(self.frames) > max_items:
            random.Random(17).shuffle(self.frames)
            self.frames = self.frames[:max_items]

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        fr = self.frames[idx]
        x = build_input(
            Path(fr["img_path"]),
            Path(fr["zn_pred_path"]),
            self.prototypes,
            self.hard_prototypes,
            self.args.size,
            self.args.proto_temp,
            self.args.hard_proto_temp,
        )
        gt = (read_gray_unit(Path(fr["gt_path"]), self.args.size) >= 0.5).float()
        return x, gt


class HardBackgroundContrastiveGenerator(nn.Module):
    def __init__(self, width: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(4, width),
            nn.SiLU(inplace=True),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, x):
        raw = x[:, 3:4].clamp(1e-4, 1 - 1e-4)
        bg_score = x[:, 6:7]
        hard_score = x[:, 7:8]
        rejection_prior = raw * (1.0 - bg_score)
        hard_rejection_prior = raw * (1.0 - hard_score)
        prior = 0.52 * raw + 0.24 * rejection_prior + 0.24 * hard_rejection_prior
        return torch.logit(prior.clamp(1e-4, 1 - 1e-4)) + self.net(x)


def dice_loss(prob, gt):
    inter = (prob * gt).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + 1.0) / (den + 1.0)).mean()


def edge_smoothness(prob, x, edge_k: float):
    rgb = x[:, 0:3].clamp(0, 1)
    gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    pdx = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs()
    pdy = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs()
    gdx = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
    gdy = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
    return (pdx * torch.exp(-edge_k * gdx)).mean() + (pdy * torch.exp(-edge_k * gdy)).mean()


def objective(logits, gt, x, args):
    prob = torch.sigmoid(logits)
    raw = x[:, 3:4]
    bg_score = x[:, 6:7]
    hard_score = x[:, 7:8]
    bce = F.binary_cross_entropy_with_logits(logits, gt)
    dice = dice_loss(prob, gt)
    area = (prob.mean(dim=(1, 2, 3)) - gt.mean(dim=(1, 2, 3))).abs().mean()
    bg_suppress = (prob * bg_score * (1.0 - gt)).mean()
    hard_bg_suppress = (prob * hard_score * (1.0 - gt)).mean()
    hard_fg_preserve = ((1.0 - prob) * hard_score * gt).mean()
    anchor = F.mse_loss(prob, raw)
    smooth = edge_smoothness(prob, x, args.edge_k)
    return (
        bce
        + dice
        + args.area_weight * area
        + args.bg_weight * bg_suppress
        + args.hard_bg_weight * hard_bg_suppress
        + args.hard_fg_weight * hard_fg_preserve
        + args.anchor_weight * anchor
        + args.smooth_weight * smooth
    )


def batch_iou(prob, gt):
    pred = prob >= 0.5
    gt_b = gt >= 0.5
    inter = (pred & gt_b).float().sum(dim=(1, 2, 3))
    union = (pred | gt_b).float().sum(dim=(1, 2, 3))
    return ((inter + 1e-6) / (union + 1e-6)).mean()


def run_epoch(model, loader, opt, device, args):
    train = opt is not None
    model.train(train)
    losses, ious = [], []
    with torch.set_grad_enabled(train):
        for x, gt in loader:
            x = x.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            logits = model(x)
            loss = objective(logits, gt, x, args)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                losses.append(float(loss.detach().cpu()))
                ious.append(float(batch_iou(torch.sigmoid(logits), gt).detach().cpu()))
    return float(np.mean(losses)), float(np.mean(ious))


def train(args):
    device = torch.device(args.device)
    train_frames = flatten_source(args, "train")
    val_frames = flatten_source(args, "val")
    prototypes = build_or_load_prototypes(args, train_frames)
    hard_prototypes = build_or_load_hard_prototypes(args, train_frames)
    train_ds = SourceDataset(train_frames, prototypes, hard_prototypes, args, args.max_train_items)
    val_ds = SourceDataset(val_frames, prototypes, hard_prototypes, args, args.max_val_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = HardBackgroundContrastiveGenerator(width=args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    work = Path(args.work_dir)
    ensure_dir(work)
    best = {"val_loss": 1e9, "epoch": 0, "val_iou": 0.0}
    bad = 0
    with (work / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_iou", "val_loss", "val_iou", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            tr_loss, tr_iou = run_epoch(model, train_loader, opt, device, args)
            va_loss, va_iou = run_epoch(model, val_loader, None, device, args)
            row = {"epoch": epoch, "train_loss": tr_loss, "train_iou": tr_iou, "val_loss": va_loss, "val_iou": va_iou, "seconds": time.time() - start}
            writer.writerow(row)
            f.flush()
            print(json.dumps({"event": "epoch", **row}), flush=True)
            improved = va_iou > best["val_iou"] if args.select_metric == "val_iou" else va_loss < best["val_loss"]
            if improved:
                best = {"val_loss": va_loss, "epoch": epoch, "val_iou": va_iou}
                torch.save(
                    {
                        "model": model.state_dict(),
                        "best": best,
                        "args": vars(args),
                        "prototypes": prototypes.numpy(),
                        "hard_prototypes": hard_prototypes.numpy(),
                    },
                    work / "best.pth",
                )
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(json.dumps({"event": "early_stop", "best": best}), flush=True)
                    break
    print(json.dumps({"event": "trained", "best": best, "checkpoint": str(work / "best.pth")}), flush=True)
    return work / "best.pth"


def apply_moch(args, checkpoint: Path):
    device = torch.device(args.device)
    ckpt = torch.load(checkpoint, map_location=device)
    prototypes = torch.from_numpy(ckpt["prototypes"].astype(np.float32))
    hard_prototypes = torch.from_numpy(ckpt["hard_prototypes"].astype(np.float32))
    model = HardBackgroundContrastiveGenerator(width=args.width).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    out_root = Path(args.output_root) / args.model_name / "MoCH"
    raw_root = Path(args.zn_moch_root)
    done, missing = 0, 0
    with torch.no_grad():
        for seq in src.list_moch_sequences(Path(args.moch_root), args.splits):
            for img_path, gt_path in seq["frames"]:
                raw_path = moch_pred_path(raw_root, seq["split"], seq["sequence"], gt_path.stem)
                if not raw_path.exists():
                    missing += 1
                    continue
                x = build_input(img_path, raw_path, prototypes, hard_prototypes, args.size, args.proto_temp, args.hard_proto_temp).unsqueeze(0).to(device)
                prob = torch.sigmoid(model(x)).squeeze().detach().cpu().numpy()
                save_prob(prob, out_root / seq["split"] / seq["sequence"] / f"{gt_path.stem}.png", img_path)
                done += 1
    print(json.dumps({"event": "applied_moch", "frames": done, "missing": missing, "output_root": str(Path(args.output_root) / args.model_name)}), flush=True)


def apply_source(args, checkpoint: Path):
    device = torch.device(args.device)
    ckpt = torch.load(checkpoint, map_location=device)
    prototypes = torch.from_numpy(ckpt["prototypes"].astype(np.float32))
    hard_prototypes = torch.from_numpy(ckpt["hard_prototypes"].astype(np.float32))
    model = HardBackgroundContrastiveGenerator(width=args.width).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    root = Path(args.output_root) / args.model_name
    source_root = root / "source"
    camotion_root = root / "camotion"
    done_source, done_camotion, missing = 0, 0, 0

    source_frames: list[dict] = []
    for mode in ("train", "val"):
        for seq in src.list_source_sequences(Path(args.moca_root), Path(args.cad_root), mode):
            source_frames.extend(seq["frames"])

    camotion_frames: list[dict] = []
    for split in ("TrainDataset_per_sq", "TestDataset_per_sq"):
        for seq in list_camotion_sequences(Path(args.camotion_root), split):
            camotion_frames.extend(seq["frames"])

    with torch.no_grad():
        for fr in source_frames:
            raw_path = source_pred_path(fr, args)
            if raw_path is None or not raw_path.exists():
                missing += 1
                continue
            x = build_input(
                Path(fr["img_path"]),
                raw_path,
                prototypes,
                hard_prototypes,
                args.size,
                args.proto_temp,
                args.hard_proto_temp,
            ).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x)).squeeze().detach().cpu().numpy()
            save_prob(prob, source_out_path(source_root, fr), Path(fr["img_path"]))
            done_source += 1

        for fr in camotion_frames:
            raw_path = source_pred_path(fr, args)
            if raw_path is None or not raw_path.exists():
                missing += 1
                continue
            x = build_input(
                Path(fr["img_path"]),
                raw_path,
                prototypes,
                hard_prototypes,
                args.size,
                args.proto_temp,
                args.hard_proto_temp,
            ).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x)).squeeze().detach().cpu().numpy()
            save_prob(prob, camotion_out_path(camotion_root, fr), Path(fr["img_path"]))
            done_camotion += 1

    print(
        json.dumps(
            {
                "event": "applied_source",
                "source_frames": done_source,
                "camotion_frames": done_camotion,
                "missing": missing,
                "source_root": str(source_root),
                "camotion_root": str(camotion_root),
            }
        ),
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["train", "apply", "apply-source", "train-apply"], required=True)
    ap.add_argument("--model-name", default="HardBackgroundContrastiveGenerator")
    ap.add_argument("--moca-root", default="MoCA_Video")
    ap.add_argument("--cad-root", default="CamouflagedAnimalDataset")
    ap.add_argument("--camotion-root", default="CAMotion")
    ap.add_argument("--moch-root", default="MoCH")
    ap.add_argument("--zn-source-root", default="MoCH_Test/source_refiner_correct/source_predictions/ZoomNeXt_PvtV2B5")
    ap.add_argument("--zn-camotion-root", default="MoCH_Test/camotion_usage/camotion_raw_predictions/ZoomNeXt_PvtV2B5")
    ap.add_argument("--zn-moch-root", default="MoCH_Test/predictions_new_downloaded/ZoomNeXt_PvtV2B5")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-items", type=int, default=0)
    ap.add_argument("--max-val-items", type=int, default=0)
    ap.add_argument("--select-metric", choices=["val_loss", "val_iou"], default="val_iou")
    ap.add_argument("--proto-size", type=int, default=96)
    ap.add_argument("--num-prototypes", type=int, default=32)
    ap.add_argument("--max-proto-samples", type=int, default=160000)
    ap.add_argument("--samples-per-frame", type=int, default=48)
    ap.add_argument("--kmeans-iters", type=int, default=8)
    ap.add_argument("--proto-temp", type=float, default=0.05)
    ap.add_argument("--num-hard-prototypes", type=int, default=32)
    ap.add_argument("--max-hard-proto-samples", type=int, default=160000)
    ap.add_argument("--hard-samples-per-frame", type=int, default=48)
    ap.add_argument("--hard-raw-threshold", type=float, default=0.35)
    ap.add_argument("--hard-proto-temp", type=float, default=0.04)
    ap.add_argument("--rebuild-prototypes", action="store_true")
    ap.add_argument("--area-weight", type=float, default=0.06)
    ap.add_argument("--bg-weight", type=float, default=0.10)
    ap.add_argument("--hard-bg-weight", type=float, default=0.18)
    ap.add_argument("--hard-fg-weight", type=float, default=0.05)
    ap.add_argument("--anchor-weight", type=float, default=0.02)
    ap.add_argument("--smooth-weight", type=float, default=0.04)
    ap.add_argument("--edge-k", type=float, default=12.0)
    args = ap.parse_args()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    ckpt = Path(args.checkpoint) if args.checkpoint else Path(args.work_dir) / "best.pth"
    if args.action in {"train", "train-apply"}:
        ckpt = train(args)
    if args.action in {"apply", "train-apply"}:
        apply_moch(args, ckpt)
    if args.action == "apply-source":
        apply_source(args, ckpt)


if __name__ == "__main__":
    main()
