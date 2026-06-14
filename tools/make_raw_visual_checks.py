# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_source_refiner_correct as src


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pred_path(root: Path, split: str, sequence: str, stem: str) -> Path:
    return root / "MoCH" / split / sequence / f"{stem}.png"


def read_rgb(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size, Image.BILINEAR)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return np.asarray(img, dtype=np.uint8) >= 128


def read_prob(path: Path, size: tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + 1e-6) / (union + 1e-6))


def color_mask(mask: np.ndarray, size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    out = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    out[mask] = color
    return Image.fromarray(out)


def overlay(rgb: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    arr = np.asarray(rgb, dtype=np.float32).copy()
    tint = np.array(color, dtype=np.float32)
    arr[mask] = 0.45 * arr[mask] + 0.55 * tint
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def diff_panel(rgb: Image.Image, base: np.ndarray, hard: np.ndarray) -> Image.Image:
    arr = np.asarray(rgb, dtype=np.float32).copy()
    added = np.logical_and(hard, np.logical_not(base))
    removed = np.logical_and(base, np.logical_not(hard))
    both = np.logical_and(base, hard)
    arr[both] = 0.65 * arr[both] + 0.35 * np.array([255, 255, 255], dtype=np.float32)
    arr[added] = 0.35 * arr[added] + 0.65 * np.array([0, 220, 80], dtype=np.float32)
    arr[removed] = 0.35 * arr[removed] + 0.65 * np.array([255, 60, 60], dtype=np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def label(img: Image.Image, text: str, height: int = 26) -> Image.Image:
    font = ImageFont.load_default()
    out = Image.new("RGB", (img.width, img.height + height), "white")
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((4, 6), text, fill=(0, 0, 0), font=font)
    return out


def make_sheet(rows: list[dict], out_path: Path, title: str, cell: tuple[int, int]) -> None:
    cols = ["RGB", "GT", "ZoomNeXt raw", "HardBG raw", "Diff green=add red=remove"]
    panels = []
    for row in rows:
        rgb = read_rgb(Path(row["img_path"]), cell)
        gt = read_mask(Path(row["gt_path"]), cell)
        base = read_mask(Path(row["base_path"]), cell)
        hard = read_mask(Path(row["hard_path"]), cell)
        sub = [
            label(rgb, f"{row['sequence']} {row['frame']}"),
            label(color_mask(gt, cell, (255, 255, 255)), "GT"),
            label(overlay(rgb, base, (255, 80, 60)), f"Base IoU {float(row['base_iou']):.3f}"),
            label(overlay(rgb, hard, (0, 220, 80)), f"Hard IoU {float(row['hard_iou']):.3f}"),
            label(diff_panel(rgb, base, hard), f"delta {float(row['delta']):+.3f}"),
        ]
        panels.append(sub)

    tile_w, tile_h = panels[0][0].size
    header_h = 32
    out = Image.new("RGB", (tile_w * len(cols), header_h + tile_h * len(panels)), "white")
    draw = ImageDraw.Draw(out)
    draw.text((6, 8), title, fill=(0, 0, 0), font=ImageFont.load_default())
    for r, sub in enumerate(panels):
        for c, img in enumerate(sub):
            out.paste(img, (c * tile_w, header_h + r * tile_h))
    ensure_dir(out_path.parent)
    out.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moch-root", default="MoCH")
    ap.add_argument("--base-root", default="MoCH_Test/predictions_new_downloaded/ZoomNeXt_PvtV2B5")
    ap.add_argument("--hard-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["Train", "Validation", "Test"])
    ap.add_argument("--cell-w", type=int, default=180)
    ap.add_argument("--cell-h", type=int, default=128)
    ap.add_argument("--examples", type=int, default=10)
    args = ap.parse_args()

    base_root = Path(args.base_root)
    hard_root = Path(args.hard_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows = []
    for seq in src.list_moch_sequences(Path(args.moch_root), args.splits):
        for img_path, gt_path in seq["frames"]:
            b_path = pred_path(base_root, seq["split"], seq["sequence"], gt_path.stem)
            h_path = pred_path(hard_root, seq["split"], seq["sequence"], gt_path.stem)
            if not b_path.exists() or not h_path.exists():
                continue
            with Image.open(gt_path) as gt_img:
                size = gt_img.convert("L").size
            gt = read_mask(gt_path, size)
            base = read_mask(b_path, size)
            hard = read_mask(h_path, size)
            base_iou = iou(base, gt)
            hard_iou = iou(hard, gt)
            rows.append(
                {
                    "split": seq["split"],
                    "sequence": seq["sequence"],
                    "frame": gt_path.stem,
                    "img_path": str(img_path),
                    "gt_path": str(gt_path),
                    "base_path": str(b_path),
                    "hard_path": str(h_path),
                    "base_iou": base_iou,
                    "hard_iou": hard_iou,
                    "delta": hard_iou - base_iou,
                }
            )

    rows.sort(key=lambda r: r["delta"], reverse=True)
    with (out_dir / "raw_frame_delta.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = args.examples
    best = rows[:n]
    worst = list(reversed(rows[-n:]))
    test_rows = [r for r in rows if r["split"] == "Test"]
    random.Random(17).shuffle(test_rows)
    test_mix = sorted(test_rows[:n], key=lambda r: r["delta"], reverse=True)

    cell = (args.cell_w, args.cell_h)
    make_sheet(best, out_dir / "raw_top_improvements.png", "Top raw-mask improvements", cell)
    make_sheet(worst, out_dir / "raw_worst_regressions.png", "Worst raw-mask regressions", cell)
    make_sheet(test_mix, out_dir / "raw_test_random_samples.png", "Random Test split samples", cell)

    base_mean = float(np.mean([r["base_iou"] for r in rows]))
    hard_mean = float(np.mean([r["hard_iou"] for r in rows]))
    print({"frames": len(rows), "binary_frame_iou_base_mean": base_mean, "binary_frame_iou_hard_mean": hard_mean, "delta": hard_mean - base_mean})
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
