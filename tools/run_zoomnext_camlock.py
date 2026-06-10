# -*- coding: utf-8 -*-
"""Train/apply the single best ZoomNeXt-CamLock model.

Strict data rule:
  * MoCH is used only for final prediction/evaluation.
  * MoCA/CAD/CAMotion are source training or validation data.
  * The refiner is integrated directly with ZoomNeXt raw masks.  It does not
    consume a previous CamLock output as its input.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_source_refiner_correct as src  # noqa: E402
import run_camlock_multiframe_memory as mf  # noqa: E402


@dataclass(frozen=True)
class Experiment:
    name: str
    profile: str
    memory: bool = True
    edge_aux: bool = False
    hard_sampler: bool = False
    curriculum: bool = False
    pretrain_camotion: bool = False
    finetune_source: bool = False
    source_factor: int = 1
    camotion_factor: int = 1
    k: int = 3
    trajectory: bool = False
    quality_gate: bool = False
    low_prev_w: float = 0.25
    high_prev_w: float = 0.10


REFINER_EXPERIMENTS: list[Experiment] = [
    Experiment("ZoomNeXt-CamLock", "freeze_refiner", memory=False),
]

ZOOMNEXT_INTEGRATED: list[tuple[str, str]] = []


@dataclass
class Frame:
    dataset: str
    split: str
    sequence: str
    frame_name: str
    img_path: Path
    gt_path: Path
    edge_path: Path | None = None


@dataclass
class Seq:
    dataset: str
    split: str
    sequence: str
    frames: list[Frame]


@dataclass
class Item:
    index: int
    count: int
    frame: Frame
    prev_frame: Frame
    pred_path: Path
    prev_pred_path: Path
    memory_pred_paths: list[Path]
    memory_indices: list[int]
    weight: float
    hard_weight: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sorted_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in src.IMAGE_SUFFIXES)


def wrap_source_sequences(seqs: list[dict]) -> list[Seq]:
    out: list[Seq] = []
    for seq in seqs:
        frames = [
            Frame(
                dataset=str(fr["dataset"]),
                split=str(fr["split"]),
                sequence=str(fr["sequence"]),
                frame_name=str(fr["frame_name"]),
                img_path=Path(fr["img_path"]),
                gt_path=Path(fr["gt_path"]),
                edge_path=None,
            )
            for fr in seq["frames"]
        ]
        if frames:
            out.append(Seq(frames[0].dataset, frames[0].split, frames[0].sequence, frames))
    return out


def list_camotion_sequences(root: Path, split: str) -> list[Seq]:
    base = root / "CAMotion" / split
    if not base.exists():
        base = root / split
    seqs: list[Seq] = []
    for seq_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        img_dir = seq_dir / "Imgs"
        gt_dir = seq_dir / "GT"
        edge_dir = seq_dir / "Edge"
        if not img_dir.exists() or not gt_dir.exists():
            continue
        imgs = {p.stem: p for p in sorted_images(img_dir)}
        edges = {p.stem: p for p in sorted_images(edge_dir)} if edge_dir.exists() else {}
        frames: list[Frame] = []
        for gt in sorted_images(gt_dir):
            img = imgs.get(gt.stem)
            if img is None:
                continue
            frames.append(
                Frame(
                    dataset="CAMotion",
                    split=split,
                    sequence=seq_dir.name,
                    frame_name=gt.stem,
                    img_path=img,
                    gt_path=gt,
                    edge_path=edges.get(gt.stem),
                )
            )
        frames.sort(key=lambda f: f.frame_name)
        if frames:
            seqs.append(Seq("CAMotion", split, seq_dir.name, frames))
    return seqs


def pred_path(frame: Frame, source_base_root: Path, camotion_base_root: Path) -> Path:
    if frame.dataset == "CAMotion":
        # CAMotion raw masks are produced by run_source_zoomnext_predictions.py,
        # which stores frames in a MoCA-style layout.
        return camotion_base_root / "MoCA_Video" / frame.split / frame.sequence / f"{frame.frame_name}.png"
    return src.source_pred_path(source_base_root, frame.__dict__)


def build_sequences(args, mode: str, exp: Experiment, phase: str = "main") -> list[tuple[Seq, int]]:
    source_train = wrap_source_sequences(src.list_source_sequences(Path(args.moca_root), Path(args.cad_root), "train"))
    source_val = wrap_source_sequences(src.list_source_sequences(Path(args.moca_root), Path(args.cad_root), "val"))
    cam_train = list_camotion_sequences(Path(args.camotion_root), "TrainDataset_per_sq")
    cam_val = list_camotion_sequences(Path(args.camotion_root), "TestDataset_per_sq")
    if mode == "val":
        source, cam = source_val, cam_val
    else:
        source, cam = source_train, cam_train

    if exp.pretrain_camotion and phase == "pretrain":
        return [(s, 1) for s in cam]
    if exp.finetune_source and phase == "finetune":
        return [(s, 1) for s in source] + [(s, 1) for s in cam[: max(1, len(cam) // 4)]]
    if exp.profile == "camotion_only":
        return [(s, 1) for s in cam]

    out: list[tuple[Seq, int]] = []
    if exp.source_factor > 0:
        out.extend((s, exp.source_factor) for s in source)
    if exp.camotion_factor > 0:
        out.extend((s, exp.camotion_factor) for s in cam)
    return out


@lru_cache(maxsize=None)
def image_motion(cur_path: Path, prev_path: Path, size: int = 96) -> float:
    cur = cv2.imread(str(cur_path), cv2.IMREAD_COLOR)
    prev = cv2.imread(str(prev_path), cv2.IMREAD_COLOR)
    if cur is None or prev is None:
        return 0.0
    cur = cv2.resize(cur, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    prev = cv2.resize(prev, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return float(np.abs(cur - prev).mean())


def source_memory_prior(paths: list[Path], indices: list[int], current_idx: int, exp: Experiment, fallback: np.ndarray, size: int) -> np.ndarray:
    if not paths:
        return fallback
    mem: list[mf.MemoryItem] = []
    for idx, path in zip(indices, paths):
        prob = read_prob(path, size)
        q, area, comp = mf.memory_quality(prob, float(prob.max()), float(prob.max()))
        mem.append(mf.MemoryItem(idx, prob, mf.prob_center(prob), q, area, comp))
    state = mf.State(prev_belief=fallback, locked=True, onset_idx=indices[0] if indices else None, memory=mem)
    variant = mf.Variant(exp.name, "integrated", exp.k, exp.trajectory, exp.quality_gate, exp.low_prev_w, exp.high_prev_w)
    return mf.memory_prior(state, variant, current_idx, fallback)


def read_prob(path: Path, size: int) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise FileNotFoundError(path)
    if g.shape[:2] != (size, size):
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_LINEAR)
    return g.astype(np.float32) / 255.0


def build_items(args, exp: Experiment, mode: str, phase: str = "main") -> list[Item]:
    items: list[Item] = []
    target_limit = args.max_train_items if mode == "train" else args.max_val_items
    source_base = Path(args.source_base_root)
    cam_base = Path(args.camotion_base_root)
    for seq, factor in build_sequences(args, mode, exp, phase):
        motion_hist: list[float] = []
        locked = False
        memory_paths: list[tuple[int, Path]] = []
        for idx, fr in enumerate(seq.frames):
            pred = pred_path(fr, source_base, cam_base)
            if not pred.exists():
                continue
            prev_idx = max(0, idx - 1)
            prev_fr = seq.frames[prev_idx]
            prev_pred = pred_path(prev_fr, source_base, cam_base)
            if not prev_pred.exists():
                prev_pred = pred
            if args.slow_motion_build:
                motion = image_motion(fr.img_path, prev_fr.img_path)
                hist = motion_hist[:]
                motion_hist.append(motion)
                if len(hist) >= 3:
                    hmean = sum(hist) / len(hist)
                    hstd = math.sqrt(sum((m - hmean) ** 2 for m in hist) / max(1, len(hist) - 1))
                else:
                    hmean, hstd = motion, 0.0
                if len(hist) >= 3 and motion > max(args.min_motion_onset, hmean + args.motion_std * hstd):
                    locked = True
            else:
                # Training only needs a causal memory schedule.  Exact motion
                # onset is computed during MoCH inference from current/past
                # frames; avoiding full image-diff precompute keeps the GPU fed.
                motion = 0.0 if idx < args.assumed_source_onset_frame else args.min_motion_onset * 2.0
                if idx >= args.assumed_source_onset_frame:
                    locked = True
            mem = memory_paths[-exp.k :] if (exp.memory and locked) else []
            ratio = 0.0 if len(seq.frames) <= 1 else idx / float(len(seq.frames) - 1)
            hard = 1.0
            if ratio < 1.0 / 3.0:
                hard *= 1.35
            if motion < 0.02:
                hard *= 1.15
            if exp.curriculum and ratio < 1.0 / 3.0 and motion < 0.02:
                hard *= 1.35
            for _ in range(max(1, factor)):
                items.append(
                    Item(
                        index=idx,
                        count=len(seq.frames),
                        frame=fr,
                        prev_frame=prev_fr,
                        pred_path=pred,
                        prev_pred_path=prev_pred,
                        memory_pred_paths=[p for _i, p in mem],
                        memory_indices=[i for i, _p in mem],
                        weight=float(factor),
                        hard_weight=float(hard),
                    )
                )
                if target_limit and len(items) >= target_limit:
                    return items
            if exp.memory and locked:
                memory_paths.append((idx, pred))
                if len(memory_paths) > max(10, exp.k * 3):
                    memory_paths = memory_paths[-max(10, exp.k * 3) :]
    return items


class RefinerDataset(Dataset):
    def __init__(self, args, exp: Experiment, mode: str, phase: str = "main"):
        self.args = args
        self.exp = exp
        self.size = args.size
        self.items = build_items(args, exp, mode, phase)
        max_items = args.max_train_items if mode == "train" else args.max_val_items
        if max_items and len(self.items) > max_items:
            random.Random(17).shuffle(self.items)
            self.items = self.items[:max_items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        cur = src.read_rgb(it.frame.img_path, self.size)
        prev = src.read_rgb(it.prev_frame.img_path, self.size)
        raw = src.read_gray(it.pred_path, self.size)
        prev_raw = src.read_gray(it.prev_pred_path, self.size)
        gt = (src.read_gray(it.frame.gt_path, self.size) >= 0.5).float()

        if self.exp.memory:
            fallback = prev_raw.squeeze(0).numpy()
            mem_prior_np = source_memory_prior(it.memory_pred_paths, it.memory_indices, it.index, self.exp, fallback, self.size)
            mem_prior = torch.from_numpy(mem_prior_np.astype(np.float32)).unsqueeze(0)
        else:
            mem_prior = prev_raw

        cur_unit = (cur * src.STD + src.MEAN).clamp(0, 1)
        prev_unit = (prev * src.STD + src.MEAN).clamp(0, 1)
        motion = (cur_unit - prev_unit).abs().mean(dim=0, keepdim=True)
        motion_norm = (motion / (motion.mean() + 2.0 * motion.std() + 1e-6)).clamp(0, 1)
        ratio = 0.0 if it.count <= 1 else it.index / float(it.count - 1)
        ratio_map = torch.full((1, self.size, self.size), ratio, dtype=torch.float32)
        early_map = torch.full((1, self.size, self.size), 1.0 if ratio < 1.0 / 3.0 else 0.0, dtype=torch.float32)
        if self.exp.memory:
            x = torch.cat([cur, raw, prev_raw, mem_prior, motion_norm, ratio_map, early_map], dim=0)
        else:
            x = torch.cat([cur, raw, prev_raw, motion_norm, ratio_map, early_map], dim=0)

        edge = src.boundary_map(gt.unsqueeze(0)).squeeze(0)
        if it.frame.edge_path and it.frame.edge_path.exists():
            edge = (src.read_gray(it.frame.edge_path, self.size) >= 0.5).float()
        return {
            "x": x,
            "gt": gt,
            "raw": raw,
            "prev_raw": mem_prior if self.exp.memory else prev_raw,
            "motion": motion_norm,
            "edge": edge,
            "weight": torch.tensor(it.hard_weight, dtype=torch.float32),
            "sample_weight": torch.tensor(max(1.0, it.weight * it.hard_weight), dtype=torch.float32),
        }


def compute_loss(correction, gate, batch, edge_aux: bool) -> tuple[torch.Tensor, dict]:
    base_loss, parts = src.loss_fn(correction, gate, batch)
    if not edge_aux:
        return base_loss, parts
    raw_logit = torch.logit(batch["raw"].clamp(1e-4, 1 - 1e-4))
    logits = gate * raw_logit + (1.0 - gate) * correction
    prob = torch.sigmoid(logits)
    edge_prob = src.boundary_map(prob)
    edge_gt = batch["edge"].clamp(0, 1)
    edge_bce = F.binary_cross_entropy(edge_prob.clamp(1e-4, 1 - 1e-4), edge_gt)
    edge_dice = src.dice_loss(edge_prob, edge_gt).mean()
    loss = base_loss + 0.12 * edge_bce + 0.10 * edge_dice
    parts = {**parts, "edge_bce": float(edge_bce.detach().cpu()), "edge_dice": float(edge_dice.detach().cpu())}
    return loss, parts


def train_refiner(args, exp: Experiment, phase: str = "main", init_checkpoint: Path | None = None) -> Path:
    device = torch.device(args.device)
    train_ds = RefinerDataset(args, exp, "train", phase)
    val_ds = RefinerDataset(args, exp, "val", "main")
    print(json.dumps({"event": "dataset", "model": exp.name, "phase": phase, "train_items": len(train_ds), "val_items": len(val_ds)}, ensure_ascii=False), flush=True)
    if not train_ds or not val_ds:
        raise RuntimeError(f"empty dataset for {exp.name}: train={len(train_ds)} val={len(val_ds)}")
    in_ch = 9 if exp.memory else 8
    model = src.CausalRefiner(in_ch=in_ch, width=args.width).to(device)
    if init_checkpoint and init_checkpoint.exists():
        ckpt = torch.load(init_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if exp.hard_sampler:
        weights = [float(it.hard_weight) for it in train_ds.items]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=False)
    else:
        loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, drop_last=False)

    work = Path(args.work_dir) / exp.name
    if phase != "main":
        work = work / phase
    ensure_dir(work)
    with (work / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strict_note": "MoCH is prediction/evaluation only. This checkpoint is integrated with ZoomNeXt raw masks, not stacked on a prior CamLock output.",
                "experiment": exp.__dict__,
                "phase": phase,
                "args": vars(args),
                "train_items": len(train_ds),
                "val_items": len(val_ds),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    best = {"val": 1e9, "epoch": -1}
    bad = 0
    log_fields = ["epoch", "train_loss", "val_loss", "seconds"]
    with (work / "training_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            losses = []
            for batch in loader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                loss, _parts = compute_loss(*model(batch["x"]), batch, exp.edge_aux)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    loss, _parts = compute_loss(*model(batch["x"]), batch, exp.edge_aux)
                    vals.append(float(loss.detach().cpu()))
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "val_loss": float(np.mean(vals)) if vals else 0.0,
                "seconds": time.time() - t0,
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps({"event": "epoch", "model": exp.name, "phase": phase, **row}, ensure_ascii=False), flush=True)
            if row["val_loss"] < best["val"]:
                best = {"val": row["val_loss"], "epoch": epoch}
                bad = 0
                torch.save({"model": model.state_dict(), "experiment": exp.__dict__, "args": vars(args), "best": best}, work / "best.pth")
            else:
                bad += 1
                if args.early_stop_patience and bad >= args.early_stop_patience:
                    print(json.dumps({"event": "early_stop", "model": exp.name, "phase": phase, "epoch": epoch, "best": best}, ensure_ascii=False), flush=True)
                    break
    print(json.dumps({"event": "trained", "model": exp.name, "phase": phase, "best": best, "checkpoint": str(work / "best.pth")}, ensure_ascii=False), flush=True)
    return work / "best.pth"


def soft_blur_np(prob: np.ndarray, radius: float) -> np.ndarray:
    return src.soft_blur_np(prob, radius)


def apply_refiner(args, exp: Experiment, checkpoint: Path) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(checkpoint, map_location=device)
    in_ch = 9 if exp.memory else 8
    model = src.CausalRefiner(in_ch=in_ch, width=args.width).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    out_root = Path(args.output_root) / exp.name
    raw_root = Path(args.moch_raw_root)
    manifest = []
    mf_variant = mf.Variant(exp.name, "integrated", exp.k, exp.trajectory, exp.quality_gate, exp.low_prev_w, exp.high_prev_w)

    with torch.no_grad():
        for seq_idx, seq in enumerate(src.list_moch_sequences(Path(args.moch_root), args.splits), start=1):
            state = mf.State()
            motion_hist: list[float] = []
            prev_img = None
            frames = seq["frames"]
            count = len(frames)
            for idx, (img_path, gt_path) in enumerate(frames):
                raw_path = mf.pred_path(raw_root, seq["split"], seq["sequence"], gt_path.stem)
                if not raw_path.exists():
                    continue
                cur = src.tensor_from_rgb(img_path, args.size).to(device)
                prev = src.tensor_from_rgb(prev_img or img_path, args.size).to(device)
                raw = src.tensor_from_gray(raw_path, args.size).to(device)
                raw_np = raw.squeeze().detach().cpu().numpy()
                fallback = raw_np if state.prev_belief is None else state.prev_belief
                mem_prior_np = mf.memory_prior(state, mf_variant, idx, fallback) if (exp.memory and state.locked) else fallback
                prev_belief = torch.from_numpy(fallback.astype(np.float32)).view(1, 1, args.size, args.size).to(device)
                mem_prior = torch.from_numpy(mem_prior_np.astype(np.float32)).view(1, 1, args.size, args.size).to(device)

                cur_unit = (cur.squeeze(0) * src.STD.to(device) + src.MEAN.to(device)).clamp(0, 1).unsqueeze(0)
                prev_unit = (prev.squeeze(0) * src.STD.to(device) + src.MEAN.to(device)).clamp(0, 1).unsqueeze(0)
                motion = (cur_unit - prev_unit).abs().mean(dim=1, keepdim=True)
                motion_score = float(motion.mean().detach().cpu())
                hist = motion_hist[:]
                motion_hist.append(motion_score)
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
                if exp.memory:
                    x = torch.cat([cur, raw, prev_belief, mem_prior, motion_norm, ratio_map, early_map], dim=1)
                else:
                    x = torch.cat([cur, raw, prev_belief, motion_norm, ratio_map, early_map], dim=1)
                correction, gate = model(x)
                raw_logit = torch.logit(raw.clamp(1e-4, 1 - 1e-4))
                net = torch.sigmoid(gate * raw_logit + (1.0 - gate) * correction)
                net_np = net.squeeze().detach().cpu().numpy()
                raw_conf = float(raw.max().detach().cpu())
                net_conf = float(net.max().detach().cpu())
                if motion_onset and max(raw_conf, net_conf) >= args.lock_conf:
                    state.locked = True
                    if state.onset_idx is None:
                        state.onset_idx = idx
                if idx == 0:
                    final = args.first_net * net_np + (1.0 - args.first_net) * raw_np
                elif not state.locked and ratio < 1.0 / 3.0:
                    final = args.early_net * net_np + args.early_raw * raw_np + args.early_prev * soft_blur_np(fallback, args.prev_blur)
                elif state.locked:
                    prev_w = exp.low_prev_w if motion_score < hmean else exp.high_prev_w
                    rest = 1.0 - prev_w
                    final = rest * (args.locked_net * net_np + (1.0 - args.locked_net) * raw_np) + prev_w * mem_prior_np
                else:
                    final = args.mid_net * net_np + args.mid_raw * raw_np + args.mid_prev * fallback
                final = np.clip(final, 0.0, 1.0).astype(np.float32)
                state.prev_belief = final
                if exp.memory and state.locked and state.onset_idx is not None and idx >= state.onset_idx:
                    q, area, comp = mf.memory_quality(final, raw_conf, net_conf)
                    state.memory.append(mf.MemoryItem(idx, final.copy(), mf.prob_center(final), q, area, comp))
                    if len(state.memory) > max(10, exp.k * 3):
                        state.memory = state.memory[-max(10, exp.k * 3) :]
                out = out_root / "MoCH" / seq["split"] / seq["sequence"] / f"{gt_path.stem}.png"
                mf.save_prob(final, out, img_path)
                manifest.append(
                    {
                        "model": exp.name,
                        "split": seq["split"],
                        "sequence": seq["sequence"],
                        "frame": idx,
                        "motion_onset": int(motion_onset),
                        "locked": int(state.locked),
                        "memory_size": len(state.memory),
                        "output": str(out),
                    }
                )
                prev_img = img_path
            if args.progress and seq_idx % args.progress == 0:
                print(json.dumps({"event": "apply_progress", "model": exp.name, "done_sequences": seq_idx}, ensure_ascii=False), flush=True)
    work = Path(args.work_dir) / exp.name
    ensure_dir(work)
    with (work / "moch_apply_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        if manifest:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)
    print(json.dumps({"event": "applied", "model": exp.name, "frames": len(manifest), "output_root": str(out_root)}, ensure_ascii=False), flush=True)


def exp_by_name(name: str) -> Experiment:
    for exp in REFINER_EXPERIMENTS:
        if exp.name == name:
            return exp
    raise KeyError(name)


def train_apply_refiner(args, exp: Experiment) -> None:
    final_ckpt = Path(args.work_dir) / exp.name / "best.pth"
    if args.checkpoint:
        final_ckpt = Path(args.checkpoint)
    if args.train:
        if exp.pretrain_camotion and exp.finetune_source:
            pre = train_refiner(args, exp, "pretrain", None)
            final_ckpt = train_refiner(args, exp, "finetune", pre)
            target_dir = Path(args.work_dir) / exp.name
            ensure_dir(target_dir)
            torch.save(torch.load(final_ckpt, map_location="cpu"), target_dir / "best.pth")
            final_ckpt = target_dir / "best.pth"
        else:
            final_ckpt = train_refiner(args, exp)
    if args.apply:
        apply_refiner(args, exp, final_ckpt)


def run_zoomnext_integrated(args, name: str, kind: str) -> None:
    # Reuse the existing ZoomNeXt internal integration script.  It consumes a
    # MoCA-style source root, so this runner expects the caller to provide one.
    work = Path(args.work_dir) / name
    ensure_dir(work)
    if args.train:
        cmd = [
            sys.executable,
            "tools/run_moch5_zoomnext_augmented_models.py",
            "--action",
            "train",
            "--kind",
            kind,
            "--model-name",
            name,
            "--moca-root",
            args.combined_moca_root,
            "--cad-root",
            args.cad_root,
            "--work-dir",
            str(work),
            "--output-root",
            args.output_root,
            "--epochs",
            str(args.zoomnext_epochs),
            "--batch",
            str(args.zoomnext_batch),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
            "--size",
            str(args.size),
            "--lr",
            str(args.zoomnext_lr),
        ]
        subprocess.run(cmd, check=True)
    if args.apply:
        cmd = [
            sys.executable,
            "tools/run_moch5_zoomnext_augmented_models.py",
            "--action",
            "predict-moch",
            "--kind",
            kind,
            "--model-name",
            name,
            "--moca-root",
            args.combined_moca_root,
            "--cad-root",
            args.cad_root,
            "--work-dir",
            str(work),
            "--output-root",
            args.output_root,
            "--checkpoint",
            str(work / "best.pth"),
            "--splits",
            *args.splits,
            "--batch",
            str(args.zoomnext_batch),
            "--workers",
            str(args.workers),
            "--device",
            args.device,
            "--size",
            str(args.size),
        ]
        subprocess.run(cmd, check=True)


def prepare_combined_source(args) -> None:
    root = Path(args.combined_moca_root)
    train = root / "TrainDataset_per_sq"
    test = root / "TestDataset_per_sq"
    ensure_dir(train)
    ensure_dir(test)

    def link_many(src_root: Path, dst_root: Path, prefix: str = "") -> int:
        count = 0
        for seq in sorted(p for p in src_root.iterdir() if p.is_dir()):
            name = prefix + seq.name
            dst = dst_root / name
            if dst.exists():
                continue
            dst.symlink_to(seq.resolve(), target_is_directory=True)
            count += 1
        return count

    added = {
        "moca_train": link_many(Path(args.moca_root) / "TrainDataset_per_sq", train, ""),
        "moca_test": link_many(Path(args.moca_root) / "TestDataset_per_sq", test, ""),
        "camotion_train": link_many(Path(args.camotion_root) / "CAMotion" / "TrainDataset_per_sq", train, "CAMotionTR__"),
        "camotion_test": link_many(Path(args.camotion_root) / "CAMotion" / "TestDataset_per_sq", test, "CAMotionTE__"),
    }
    print(json.dumps({"event": "prepared_combined_source", "root": str(root), "added": added}, ensure_ascii=False), flush=True)


def evaluate_models(args, extra_models: list[str]) -> None:
    model_specs = [
        f"ZoomNeXt_Baseline={args.moch_raw_root}",
    ]
    for name in extra_models:
        model_specs.append(f"{name}={Path(args.output_root) / name}")
    cmd = [
        sys.executable,
        "tools/evaluate_moch_baselines.py",
        "--moch-root",
        args.moch_root,
        "--out-dir",
        args.analysis_dir,
        "--docx",
        args.docx,
        "--workers",
        str(args.eval_workers),
        "--splits",
        *args.splits,
    ]
    for spec in model_specs:
        cmd.extend(["--model", spec])
    subprocess.run(cmd, check=True)


def read_overall(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r.get("iou", 0)), reverse=True)
    print(json.dumps({"event": "overall_top", "rows": rows[:8]}, ensure_ascii=False), flush=True)


def selected_refiners(args) -> list[Experiment]:
    if args.models:
        return [exp_by_name(name) for name in args.models if name in {e.name for e in REFINER_EXPERIMENTS}]
    return REFINER_EXPERIMENTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["prepare", "train-apply", "evaluate", "run-all"], default="run-all")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--moca-root", default="MoCA_Video")
    ap.add_argument("--cad-root", default="CamouflagedAnimalDataset")
    ap.add_argument("--camotion-root", default="CAMotion")
    ap.add_argument("--combined-moca-root", default="work/combined_source/MoCA_Video")
    ap.add_argument("--source-base-root", default="data/source_raw/ZoomNeXt_PvtV2B5")
    ap.add_argument("--camotion-base-root", default="data/camotion_raw/ZoomNeXt_PvtV2B5")
    ap.add_argument("--moch-root", default="MoCH")
    ap.add_argument("--moch-raw-root", default="data/moch_raw/ZoomNeXt_PvtV2B5")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--output-root", default="predictions")
    ap.add_argument("--analysis-dir", default="results/ZoomNeXt-CamLock")
    ap.add_argument("--docx", default="results/ZoomNeXt-CamLock/evaluation.docx")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-items", type=int, default=0)
    ap.add_argument("--max-val-items", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--skip-zoomnext-integrated", action="store_true")
    ap.add_argument("--zoomnext-epochs", type=int, default=5)
    ap.add_argument("--zoomnext-batch", type=int, default=2)
    ap.add_argument("--zoomnext-lr", type=float, default=1e-4)
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--motion-std", type=float, default=1.0)
    ap.add_argument("--min-motion-onset", type=float, default=0.012)
    ap.add_argument("--assumed-source-onset-frame", type=int, default=3)
    ap.add_argument("--slow-motion-build", action="store_true")
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
    args = ap.parse_args()

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    if args.action == "prepare":
        prepare_combined_source(args)
        return
    if args.action in ("train-apply", "run-all") and not args.train and not args.apply:
        args.train = True
        args.apply = True

    if args.action in ("train-apply", "run-all"):
        trained: list[str] = []
        for exp in selected_refiners(args):
            train_apply_refiner(args, exp)
            trained.append(exp.name)
        if not args.skip_zoomnext_integrated:
            for name, kind in ZOOMNEXT_INTEGRATED:
                if args.models and name not in args.models:
                    continue
                run_zoomnext_integrated(args, name, kind)
                trained.append(name)
        if args.action == "run-all":
            evaluate_models(args, trained)
            read_overall(Path(args.analysis_dir) / "overall_metrics.csv")
        return

    if args.action == "evaluate":
        extra = args.models or [e.name for e in REFINER_EXPERIMENTS] + [n for n, _k in ZOOMNEXT_INTEGRATED]
        evaluate_models(args, extra)
        read_overall(Path(args.analysis_dir) / "overall_metrics.csv")


if __name__ == "__main__":
    main()
