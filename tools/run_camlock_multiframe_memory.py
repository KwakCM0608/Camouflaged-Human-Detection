# -*- coding: utf-8 -*-
"""Apply ZoomNeXt-CamLock with post-onset multi-frame memory variants.

No MoCH training is performed.  The refiner checkpoint is the source-trained
ZoomNeXt-CamLock checkpoint from augmented MoCA/CAD.  MoCH is used only for
prediction/evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_source_refiner_correct as src  # noqa: E402


@dataclass(frozen=True)
class Variant:
    name: str
    kind: str
    k: int = 1
    trajectory: bool = False
    quality_gate: bool = False
    low_prev_w: float = 0.20
    high_prev_w: float = 0.05


@dataclass
class MemoryItem:
    frame_idx: int
    prob: np.ndarray
    center: tuple[float, float] | None
    quality: float
    area_frac: float
    component_count: int


@dataclass
class State:
    prev_belief: np.ndarray | None = None
    locked: bool = False
    onset_idx: int | None = None
    memory: list[MemoryItem] = field(default_factory=list)


def variants() -> list[Variant]:
    return [
        Variant("CamLock_K1_Current", "k1", 1, False, False, 0.20, 0.05),
        Variant("CamLock_K3_EMA", "ema", 3, False, False, 0.25, 0.10),
        Variant("CamLock_K5_EMA", "ema", 5, False, False, 0.25, 0.10),
        Variant("CamLock_K3_Trajectory", "trajectory", 3, True, False, 0.25, 0.12),
        Variant("CamLock_K5_QualityTrajectory", "quality_trajectory", 5, True, True, 0.28, 0.12),
    ]


def pred_path(root: Path, split: str, seq: str, stem: str) -> Path:
    return root / "MoCH" / split / seq / f"{stem}.png"


def save_prob(prob: np.ndarray, out_path: Path, ref_path: Path) -> None:
    bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(ref_path)
    h, w = bgr.shape[:2]
    prob = np.nan_to_num(prob.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    if prob.shape[:2] != (h, w):
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.clip(prob * 255.0, 0, 255).astype(np.uint8))


def prob_center(prob: np.ndarray) -> tuple[float, float] | None:
    mask = prob >= 0.5
    if mask.any():
        ys, xs = np.nonzero(mask)
        return float(xs.mean()), float(ys.mean())
    total = float(prob.sum())
    if total <= 1e-6:
        return None
    yy, xx = np.indices(prob.shape)
    return float((prob * xx).sum() / total), float((prob * yy).sum() / total)


def component_count(prob: np.ndarray) -> int:
    mask = (prob >= 0.5).astype(np.uint8)
    num, _labels = cv2.connectedComponents(mask, 8)
    return max(0, int(num) - 1)


def memory_quality(prob: np.ndarray, raw_conf: float, net_conf: float) -> tuple[float, float, int]:
    mask = prob >= 0.5
    area_frac = float(mask.mean())
    comp = component_count(prob)
    fg_mean = float(prob[mask].mean()) if mask.any() else 0.0
    # raw/net max can be saturated after normalization, so foreground mean and
    # component stability are more useful for filtering memory states.
    comp_penalty = max(0.0, 1.0 - 0.05 * max(0, comp - 1))
    area_penalty = 1.0 if 0.0005 <= area_frac <= 0.65 else 0.4
    q = max(fg_mean, 0.5 * max(raw_conf, net_conf)) * comp_penalty * area_penalty
    return float(q), area_frac, comp


def shift_prob(prob: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = prob.shape
    mat = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        prob.astype(np.float32),
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def select_memory(st: State, v: Variant) -> list[MemoryItem]:
    mem = st.memory[-v.k :]
    if not v.quality_gate:
        return mem
    filtered = [
        item
        for item in mem
        if item.quality >= 0.50 and 0.0005 <= item.area_frac <= 0.55 and item.component_count <= 8
    ]
    return filtered if filtered else mem[-max(1, min(len(mem), 2)) :]


def estimate_velocity(mem: list[MemoryItem]) -> tuple[float, float]:
    valid = [m for m in mem if m.center is not None]
    if len(valid) < 2:
        return 0.0, 0.0
    first, last = valid[0], valid[-1]
    denom = max(1, last.frame_idx - first.frame_idx)
    return (last.center[0] - first.center[0]) / denom, (last.center[1] - first.center[1]) / denom


def memory_prior(st: State, v: Variant, current_idx: int, fallback: np.ndarray) -> np.ndarray:
    mem = select_memory(st, v)
    if not mem:
        return fallback
    vx, vy = estimate_velocity(mem) if v.trajectory else (0.0, 0.0)
    maps = []
    weights = []
    for item in mem:
        age = max(0, current_idx - item.frame_idx)
        shifted = shift_prob(item.prob, vx * age, vy * age) if v.trajectory else item.prob
        recency = math.exp(-age / max(1.0, v.k / 2.0))
        quality = max(0.05, item.quality) if v.quality_gate else 1.0
        maps.append(shifted)
        weights.append(recency * quality)
    wsum = max(1e-6, float(sum(weights)))
    prior = sum(m * w for m, w in zip(maps, weights)) / wsum
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def apply(args) -> None:
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = src.CausalRefiner(in_ch=8, width=args.width).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    var_list = variants()
    raw_root = Path(args.raw_root)
    out_root = Path(args.output_root)
    manifest = []

    with torch.no_grad():
        for seq_idx, seq in enumerate(src.list_moch_sequences(Path(args.moch_root), args.splits), start=1):
            states = {v.name: State() for v in var_list}
            motion_hist: list[float] = []
            prev_img = None
            frames = seq["frames"]
            count = len(frames)
            for idx, (img_path, gt_path) in enumerate(frames):
                raw_path = pred_path(raw_root, seq["split"], seq["sequence"], gt_path.stem)
                if not raw_path.exists():
                    continue
                cur = src.tensor_from_rgb(img_path, args.size).to(device)
                prev = src.tensor_from_rgb(prev_img or img_path, args.size).to(device)
                raw = src.tensor_from_gray(raw_path, args.size).to(device)

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

                xs = []
                prev_maps: dict[str, np.ndarray] = {}
                raw_np = raw.squeeze().detach().cpu().numpy()
                for v in var_list:
                    st = states[v.name]
                    fallback = raw_np if st.prev_belief is None else st.prev_belief
                    if v.kind == "k1" or not st.locked:
                        prev_np = fallback
                    else:
                        prev_np = memory_prior(st, v, idx, fallback)
                    prev_maps[v.name] = prev_np.astype(np.float32)
                    prev_belief = torch.from_numpy(prev_np).float().view(1, 1, args.size, args.size).to(device)
                    xs.append(torch.cat([cur, raw, prev_belief, motion_norm, ratio_map, early_map], dim=1))

                x = torch.cat(xs, dim=0)
                correction, gate = model(x)
                raw_logit = torch.logit(raw.clamp(1e-4, 1 - 1e-4)).repeat(len(var_list), 1, 1, 1)
                nets = torch.sigmoid(gate * raw_logit + (1.0 - gate) * correction).detach().cpu().numpy()[:, 0]
                raw_conf = float(raw.max().detach().cpu())

                for v, net_np in zip(var_list, nets):
                    st = states[v.name]
                    prev_np = prev_maps[v.name]
                    net_conf = float(net_np.max())
                    if motion_onset and max(raw_conf, net_conf) >= args.lock_conf:
                        st.locked = True
                        if st.onset_idx is None:
                            st.onset_idx = idx

                    if idx == 0:
                        final = args.first_net * net_np + (1.0 - args.first_net) * raw_np
                    elif not st.locked and ratio < 1.0 / 3.0:
                        final = args.early_net * net_np + args.early_raw * raw_np + args.early_prev * src.soft_blur_np(prev_np, args.prev_blur)
                    elif st.locked:
                        prev_w = v.low_prev_w if motion_score < hmean else v.high_prev_w
                        rest = 1.0 - prev_w
                        final = rest * (args.locked_net * net_np + (1.0 - args.locked_net) * raw_np) + prev_w * prev_np
                    else:
                        final = args.mid_net * net_np + args.mid_raw * raw_np + args.mid_prev * prev_np
                    final = np.clip(final, 0.0, 1.0).astype(np.float32)
                    st.prev_belief = final

                    # Only onset/post-onset frames are allowed into the memory
                    # bank.  The onset frame itself is added after its prediction,
                    # so it can influence later frames but not itself.
                    if st.locked and st.onset_idx is not None and idx >= st.onset_idx:
                        q, area, comp = memory_quality(final, raw_conf, net_conf)
                        st.memory.append(MemoryItem(idx, final.copy(), prob_center(final), q, area, comp))
                        if len(st.memory) > max(10, v.k * 3):
                            st.memory = st.memory[-max(10, v.k * 3) :]

                    out = out_root / v.name / "MoCH" / seq["split"] / seq["sequence"] / f"{gt_path.stem}.png"
                    save_prob(final, out, img_path)
                    manifest.append(
                        {
                            "model": v.name,
                            "split": seq["split"],
                            "sequence": seq["sequence"],
                            "frame_index": idx,
                            "frame_name": gt_path.stem,
                            "motion_score": motion_score,
                            "motion_onset": int(motion_onset),
                            "locked": int(st.locked),
                            "onset_idx": "" if st.onset_idx is None else st.onset_idx,
                            "memory_size": len(st.memory),
                            "output": str(out),
                        }
                    )
                prev_img = img_path
            if args.progress and seq_idx % args.progress == 0:
                print(json.dumps({"done_sequences": seq_idx, "total_sequences": len(src.list_moch_sequences(Path(args.moch_root), args.splits))}, ensure_ascii=False), flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "camlock_multiframe_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps({"variants": [v.name for v in var_list], "rows": len(manifest), "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="MoCH_Test/moch5_work/ZoomNeXt_AugFrozenRefiner/best.pth")
    ap.add_argument("--raw-root", default="MoCH_Test/predictions_new_downloaded/ZoomNeXt_PvtV2B5")
    ap.add_argument("--moch-root", default="MoCH")
    ap.add_argument("--output-root", default="MoCH_Test/camlock_multiframe_memory/predictions")
    ap.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--progress", type=int, default=10)
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
    args = ap.parse_args()
    apply(args)


if __name__ == "__main__":
    main()
