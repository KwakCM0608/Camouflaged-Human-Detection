# HardBG Source-Only Experiments

이 문서는 `ZoomNeXt-CamLock`의 다음 단계 실험으로 진행한 source-only 1차 mask generator 실험을 정리한다.

핵심 목표는 CamLock refiner를 더 복잡하게 만드는 것이 아니라, CamLock에 들어가기 전의 1차 raw mask가 배경과 위장 객체를 덜 헷갈리도록 만드는 것이다.

## 공정성 규칙

이 실험은 기존 repository의 데이터 규칙을 그대로 따른다.

- 학습과 validation은 `MoCA`, `CAD`, `CAMotion` source 데이터만 사용한다.
- `MoCH`는 학습, validation, checkpoint 선택, threshold 선택, 비율 선택에 사용하지 않는다.
- `MoCH`는 마지막 prediction/evaluation에만 사용한다.
- 미래 frame, 미래 GT, 전체 sequence 후처리, oracle selection은 사용하지 않는다.
- 첫 frame prompt, bbox, click, text prompt, GT mask는 사용하지 않는다.

## 추가한 방법

### HardBGContrastiveGenerator

`HardBGContrastiveGenerator`는 source 데이터에서 두 종류의 배경 prototype을 만든다.

1. 일반 배경 prototype: source GT에서 background인 pixel을 모아 만든다.
2. hard background prototype: source GT에서는 background인데 기존 ZoomNeXt raw mask가 foreground처럼 높게 예측한 pixel을 모아 만든다.

즉 모델에게 "그럴듯하게 사람처럼 보이지만 실제로는 배경인 부분"을 따로 알려준다. 이 hard background bank를 이용해 1차 mask generator가 위장 배경 texture를 foreground로 끌고 오는 현상을 줄이는 것이 목적이다.

### HardBGWavefrontGenerator

`HardBGWavefrontGenerator`는 `HardBGContrastiveGenerator`에 fixed directional wavefront channel을 추가한 ablation이다.

RGB와 기존 raw mask, prototype distance만 쓰지 않고, Sobel/diagonal 방향 응답과 local contrast를 추가 입력으로 넣는다. 의도는 조각난 mask를 줄이고, 방향성 경계 단서를 더 보게 만드는 것이다.

결과적으로 raw mask의 IoU와 component count는 개선됐지만, 기존 CamLock을 붙인 최종 IoU는 `HardBGContrastiveGenerator`보다 낮았다. 따라서 최종 주력 후보는 `HardBGContrastiveGenerator + existing CamLock`으로 본다.

## MoCH 최종 평가

아래 평가는 모두 MoCH 전체 3742 annotated frame 기준이다. MoCH는 최종 평가에만 사용했다.

| model | frames | IoU | Dice | Precision | Recall | Failure | Boundary F1 | Component Count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZoomNeXt_Baseline | 3742 | 0.388899 | 0.453324 | 0.587854 | 0.420409 | 0.492517 | 0.228861 | 1.628274 |
| GitHub_CamLock | 3742 | 0.463299 | 0.547248 | 0.597844 | 0.566342 | 0.370925 | 0.251037 | 2.300107 |
| HardBGContrastive_Raw | 3742 | 0.422024 | 0.495423 | 0.568776 | 0.479385 | 0.439070 | 0.233055 | 4.018439 |
| HardBGContrastive_CamLock | 3742 | 0.466346 | 0.550431 | 0.580842 | 0.573787 | 0.370390 | 0.241273 | 4.052646 |
| HardBGWavefront_Raw | 3742 | 0.425623 | 0.498135 | 0.565146 | 0.487626 | 0.437199 | 0.233552 | 3.646179 |
| HardBGWavefront_CamLock | 3742 | 0.464788 | 0.547721 | 0.579325 | 0.570300 | 0.373597 | 0.236885 | 3.758418 |

## 해석

`HardBGContrastive_Raw`는 기존 `ZoomNeXt_Baseline`보다 raw 단계에서 다음만큼 좋아졌다.

| metric | ZoomNeXt raw | HardBGContrastive raw | delta |
|---|---:|---:|---:|
| IoU | 0.388899 | 0.422024 | +0.033125 |
| Dice | 0.453324 | 0.495423 | +0.042099 |
| Recall | 0.420409 | 0.479385 | +0.058977 |
| Failure | 0.492517 | 0.439070 | -0.053447 |

기존 `GitHub_CamLock`에 비해서도 `HardBGContrastive_CamLock`은 전체 IoU가 `0.463299`에서 `0.466346`으로 올라갔다.

다만 component count가 크게 증가했다. 이는 HardBG 계열 raw mask가 더 많은 객체 후보를 잡지만, 동시에 mask가 더 조각나는 경향이 있다는 뜻이다. 이 때문에 raw IoU 개선 폭이 CamLock 이후에는 일부만 살아남는다.

`HardBGWavefrontGenerator`는 raw mask IoU를 `0.425623`까지 올리고 component count를 `4.018439`에서 `3.646179`로 낮췄다. 하지만 CamLock 이후 최종 IoU는 `0.464788`로 `HardBGContrastive_CamLock`보다 낮았다. 따라서 wavefront는 조각 감소 ablation으로 남기고, 최종 후보는 `HardBGContrastiveGenerator`를 우선한다.

## 실행 예시

### HardBGContrastive 학습과 MoCH raw prediction

```bash
python tools/run_hard_background_contrastive_generator.py \
  --action train-apply \
  --model-name HardBackgroundContrastiveGenerator \
  --work-dir MoCH_Test/hard_bg_contrastive_generator/work/HardBackgroundContrastiveGenerator \
  --output-root MoCH_Test/hard_bg_contrastive_generator/raw_predictions \
  --epochs 8 \
  --patience 3 \
  --batch 24 \
  --workers 4 \
  --size 192 \
  --num-prototypes 32 \
  --num-hard-prototypes 32 \
  --max-proto-samples 160000 \
  --max-hard-proto-samples 160000 \
  --samples-per-frame 48 \
  --hard-samples-per-frame 48 \
  --hard-raw-threshold 0.35 \
  --select-metric val_iou
```

### 기존 GitHub CamLock 적용

```bash
python tools/run_zoomnext_camlock.py \
  --action train-apply \
  --models ZoomNeXt-CamLock \
  --apply \
  --checkpoint checkpoints/ZoomNeXt-CamLock/best.pth \
  --moch-raw-root MoCH_Test/hard_bg_contrastive_generator/raw_predictions/HardBackgroundContrastiveGenerator \
  --work-dir MoCH_Test/hard_bg_contrastive_generator/camlock_work \
  --output-root MoCH_Test/hard_bg_contrastive_generator/camlock_predictions \
  --analysis-dir MoCH_Test/hard_bg_contrastive_generator/analysis_unused \
  --splits Train Validation Test
```

### 평가

```bash
python tools/evaluate_moch_baselines.py \
  --moch-root MoCH \
  --out-dir MoCH_Test/hard_bg_contrastive_generator/analysis \
  --splits Train Validation Test \
  --model "ZoomNeXt_Baseline=MoCH_Test/predictions_new_downloaded/ZoomNeXt_PvtV2B5" \
  --model "GitHub_CamLock=MoCH_Test/camotion_usage/predictions/ZoomNeXt_Freeze_CAMotionRefiner" \
  --model "HardBGContrastive_Raw=MoCH_Test/hard_bg_contrastive_generator/raw_predictions/HardBackgroundContrastiveGenerator" \
  --model "HardBGContrastive_CamLock=MoCH_Test/hard_bg_contrastive_generator/camlock_predictions/ZoomNeXt-CamLock"
```

## 추가 파일

- `tools/run_hard_background_contrastive_generator.py`: source background/hard-background prototype 기반 1차 mask generator.
- `tools/run_hardbg_wavefront_generator.py`: HardBG에 fixed directional wavefront channel을 추가한 ablation.
- `tools/make_raw_visual_checks.py`: raw mask 개선/악화 frame을 시각적으로 비교하기 위한 contact sheet 생성 도구.
