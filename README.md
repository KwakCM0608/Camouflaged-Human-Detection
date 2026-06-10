# ZoomNeXt-CamLock

This repository contains the best fair ZoomNeXt-CamLock checkpoint and the
minimal code needed to reproduce its first-mask + causal refiner workflow.

## Model

`ZoomNeXt-CamLock` uses:

1. Official ZoomNeXt PVTv2-B5 as the raw first-mask generator.
2. A small causal CamLock refiner trained on MoCA/CAD/CAMotion only.
3. MoCH only for final prediction/evaluation.

The committed refiner checkpoint is:

```text
checkpoints/ZoomNeXt-CamLock/best.pth
```

The official ZoomNeXt weight is not committed because it is large. Put it at:

```text
external_models/ZoomNeXt/weights/pvtv2-b5-zoomnext.pth
```

## MoCH Result

Saved fair MoCH result:

| model | frames | IoU | Dice | Failure | Boundary F1 |
|---|---:|---:|---:|---:|---:|
| ZoomNeXt_Baseline | 3742 | 0.388899 | 0.453324 | 0.492517 | 0.228861 |
| ZoomNeXt-CamLock | 3742 | 0.463299 | 0.547248 | 0.370925 | 0.251037 |

Detailed CSV files are in:

```text
results/ZoomNeXt-CamLock/
```

## Files

```text
tools/run_zoomnext_camlock.py          # train/apply/evaluate ZoomNeXt-CamLock
tools/run_moch_new_model_predictions.py # generate MoCH ZoomNeXt raw masks
tools/run_source_zoomnext_predictions.py # generate MoCA/CAD raw masks
tools/run_source_refiner_correct.py    # shared refiner/loss/data utilities
tools/evaluate_moch_baselines.py       # MoCH metric evaluation
external_models/ZoomNeXt/              # ZoomNeXt source code, weights excluded
checkpoints/ZoomNeXt-CamLock/          # committed CamLock refiner checkpoint
```

## Typical Usage

Generate MoCH raw ZoomNeXt masks:

```bash
python tools/run_moch_new_model_predictions.py \
  --models zoomnext \
  --moch-root MoCH \
  --out-root data/moch_raw \
  --device cuda
```

Generate MoCA/CAD source raw masks:

```bash
python tools/run_source_zoomnext_predictions.py \
  --moca-root MoCA_Video \
  --cad-root CamouflagedAnimalDataset \
  --out-root data/source_raw \
  --device cuda
```

Apply the committed refiner checkpoint to MoCH:

```bash
python tools/run_zoomnext_camlock.py \
  --action train-apply \
  --apply \
  --checkpoint checkpoints/ZoomNeXt-CamLock/best.pth \
  --source-base-root data/source_raw/ZoomNeXt_PvtV2B5 \
  --camotion-base-root data/camotion_raw/ZoomNeXt_PvtV2B5 \
  --moch-raw-root data/moch_raw/ZoomNeXt_PvtV2B5 \
  --work-dir work \
  --output-root predictions \
  --device cuda
```

Evaluate:

```bash
python tools/run_zoomnext_camlock.py \
  --action evaluate \
  --moch-root MoCH \
  --moch-raw-root data/moch_raw/ZoomNeXt_PvtV2B5 \
  --output-root predictions \
  --analysis-dir results/ZoomNeXt-CamLock \
  --device cuda
```

## Excluded

The repository intentionally excludes datasets, raw prediction PNGs, generated
MoCH prediction folders, Python caches, and official ZoomNeXt weight archives.
