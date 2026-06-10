# ZoomNeXt-CamLock

`ZoomNeXt-CamLock`은 위장 객체 추적에서 **초기 1차 mask가 흔들리는 문제**를 줄이기 위해 만든 모델이다. 공식 `ZoomNeXt PVTv2-B5`를 1차 mask 생성기로 사용하고, 그 위에 작은 causal refiner와 lock-in 정책을 붙여 최종 mask를 보정한다.

가장 중요한 데이터 규칙은 다음과 같다.

- 학습: `MoCA`, `CAD`, `CAMotion`
- 평가: `MoCH`
- `MoCH`는 학습/검증에 사용하지 않고, 최종 prediction/evaluation에만 사용한다.
- 전체 MoCH sequence를 볼 때도 미래 frame이나 미래 GT는 사용하지 않는다.

## 핵심 아이디어

위장 객체 영상에서는 초반 frame에서 객체와 배경이 거의 구분되지 않기 때문에, 단일 frame segmentation 모델이 객체를 너무 작게 잡거나 배경을 객체처럼 잡는 경우가 많다. 이 1차 mask를 그대로 쓰면 이후 onset 판단도 그 mask에 끌려가서 불안정해진다.

`ZoomNeXt-CamLock`의 핵심은 **1차 mask 자체를 버리지 않고, 시간적으로 causal한 정보로 보정하는 것**이다.

1. `ZoomNeXt`가 현재 frame만 보고 raw foreground probability mask를 만든다.
2. `CamLock refiner`가 현재 RGB, raw mask, 이전 belief, motion, frame 위치 정보를 함께 보고 mask를 보정한다.
3. motion onset이 감지되면 lock-in 상태로 들어가고, 이전 belief를 약하게 섞어 객체 위치가 갑자기 튀지 않도록 한다.
4. onset 전에는 보수적으로, onset 이후에는 현재 refiner 출력과 이전 belief를 함께 사용한다.

즉 이 모델은 새 backbone을 크게 학습하는 방식이 아니라, **검증된 ZoomNeXt의 1차 탐지 능력은 유지하면서 temporal 안정성과 onset 이후 추적성을 추가하는 방식**이다.

## 전체 흐름

```text
현재 frame
   |
   v
Official ZoomNeXt PVTv2-B5
   |
   v
raw 1차 mask
   |
   +-- 현재 RGB
   +-- 이전 frame / 이전 belief
   +-- motion map
   +-- frame ratio map
   +-- early phase map
   v
CamLock causal refiner
   |
   v
motion onset / lock-in 정책
   |
   v
최종 mask
```

## 왜 ZoomNeXt를 1차 mask 생성기로 쓰는가

`ZoomNeXt`는 camouflage object detection 계열에서 강한 single-frame mask를 제공한다. 이 프로젝트에서는 다음 이유로 baseline으로 사용했다.

- bbox가 아니라 pixel-level mask를 직접 출력한다.
- PVTv2-B5 기반 multi-scale 구조라 작은 단서와 넓은 문맥을 함께 본다.
- MoCH에서 raw baseline만으로도 비교적 안정적인 1차 후보를 만든다.
- CamLock은 raw mask를 완전히 대체하지 않고 보정하는 역할이므로, 강한 raw mask generator가 있을수록 유리하다.

공식 ZoomNeXt weight는 용량 문제로 repository에 포함하지 않았다. 아래 위치에 별도로 배치해야 한다.

```text
external_models/ZoomNeXt/weights/pvtv2-b5-zoomnext.pth
```

## CamLock Refiner 구조

CamLock refiner는 작은 encoder-decoder 형태의 convolutional refiner다.

- 입력 채널: `8`
- width: `32`
- 출력: `correction logit`, `gate`
- checkpoint: `checkpoints/ZoomNeXt-CamLock/best.pth`

입력은 다음 8채널이다.

| 입력 | 채널 | 의미 |
|---|---:|---|
| current RGB | 3 | 현재 frame appearance |
| current ZoomNeXt raw mask | 1 | 현재 frame의 1차 foreground probability |
| previous belief | 1 | 직전까지 유지한 최종 mask belief |
| motion map | 1 | 현재 frame과 이전 frame의 absolute RGB difference |
| frame ratio map | 1 | sequence 내 현재 위치 |
| early phase map | 1 | 초반 구간 여부 |

refiner는 raw mask를 무조건 덮어쓰지 않는다. raw logit과 refiner correction을 gate로 섞는다.

```text
final_refiner_logit = gate * raw_logit + (1 - gate) * correction
net_prob = sigmoid(final_refiner_logit)
```

이 구조 때문에 모델은 raw mask를 신뢰할 frame에서는 raw를 살리고, raw가 흔들릴 frame에서는 correction을 더 반영할 수 있다.

## Motion Onset과 Lock-in

추론은 sequence를 앞에서 뒤로 한 frame씩 처리한다. 미래 frame은 사용하지 않는다.

현재 frame과 이전 frame의 차이로 `motion_score`를 계산하고, 이전 motion history와 비교해 onset을 판단한다.

```text
motion_onset =
  len(history) >= 3
  and motion_score > max(min_motion_onset, mean(history) + motion_std * std(history))
```

현재 설정은 다음과 같다.

| 항목 | 값 |
|---|---:|
| `min_motion_onset` | `0.012` |
| `motion_std` | `1.0` |
| `lock_conf` | `0.62` |

onset이 감지되고 raw/refiner confidence가 충분하면 lock 상태로 들어간다. lock 이후에는 이전 belief를 약하게 섞어 위치와 면적이 급격히 흔들리지 않게 한다.

## 최종 Mask Blending 정책

CamLock은 refiner 출력만 그대로 저장하지 않고, frame 상태에 따라 raw mask, refiner 출력, 이전 belief를 섞는다.

| 상황 | 최종 mask 구성 |
|---|---|
| 첫 frame | `0.65 * net + 0.35 * raw` |
| onset 전 early frame | `0.55 * net + 0.25 * raw + 0.20 * blurred_prev` |
| onset 이후 locked 상태 | `prev_w`로 이전 belief를 섞고, 나머지는 `0.78 * net + 0.22 * raw` |
| 그 외 middle 상태 | `0.65 * net + 0.25 * raw + 0.10 * prev` |

locked 상태에서 previous belief weight는 motion이 낮을 때 더 크게 둔다.

| 상태 | previous belief weight |
|---|---:|
| low motion | `0.25` |
| high motion | `0.10` |

이 정책의 의도는 간단하다.

- 초반에는 성급한 확신을 줄인다.
- 움직임이 생기면 refiner 출력을 더 활용한다.
- lock 이후에는 객체 위치가 갑자기 배경 쪽으로 튀지 않게 이전 belief를 유지한다.

## 학습 방식

학습은 CamLock refiner만 수행한다. 공식 ZoomNeXt는 1차 raw mask generator로 사용하며, 이 repository에 포함된 checkpoint는 refiner checkpoint다.

학습에 사용한 source는 다음과 같다.

| 데이터 | 역할 |
|---|---|
| `MoCA` | source train/validation |
| `CAD` | source train/validation |
| `CAMotion` | motion/onset 상황 보강 |
| `MoCH` | 학습에는 사용하지 않음 |

학습 item 수는 저장된 config 기준 다음과 같다.

| 항목 | 개수 |
|---|---:|
| train items | `27390` |
| validation items | `7520` |

학습 설정:

| 항목 | 값 |
|---|---:|
| image size | `256` |
| batch size | `24` |
| epochs | `8` |
| optimizer | `AdamW` |
| learning rate | `0.0002` |
| early stop patience | `3` |
| best epoch | `1` |

실제 training log에서는 epoch 1 이후 validation loss가 계속 나빠졌기 때문에, best checkpoint는 epoch 1이다.

```text
epoch 1: val_loss 0.6113017758556233
epoch 2: val_loss 0.6202397368326309
epoch 3: val_loss 0.6573526193951346
epoch 4: val_loss 0.6590451747653591
```

## Loss 구성

CamLock refiner는 단순 BCE만 쓰지 않는다. mask 품질과 시간 안정성을 같이 보도록 구성했다.

```text
loss =
  BCE
  + Dice
  + 0.30 * center_loss
  + 0.08 * area_loss
  + 0.12 * boundary_dice
  + 0.02 * temporal_loss
```

각 항목의 역할은 다음과 같다.

| loss | 목적 |
|---|---|
| BCE | pixel-wise foreground/background 분류 |
| Dice | 전체 mask overlap 개선 |
| center loss | 객체 중심이 크게 벗어나는 현상 억제 |
| area loss | mask 면적이 과도하게 커지거나 작아지는 현상 억제 |
| boundary dice | 윤곽 품질 보조 |
| temporal loss | low-motion 구간에서 이전 mask와의 급격한 변화 억제 |

초반 frame과 low-motion frame에는 가중치를 더 준다. 위장 객체는 특히 onset 전후가 어려우므로, 이 구간에서 refiner가 너무 성급하게 변하지 않도록 하기 위한 장치다.

## MoCH 성능

공정 비교 기준에서 `MoCH` 전체 3742 frame에 대해 다음 결과를 얻었다.

| model | frames | IoU | Dice | Precision | Recall | Failure | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ZoomNeXt_Baseline | 3742 | 0.388899 | 0.453324 | 0.587854 | 0.420409 | 0.492517 | 0.228861 |
| ZoomNeXt-CamLock | 3742 | 0.463299 | 0.547248 | 0.597844 | 0.566342 | 0.370925 | 0.251037 |

개선 폭:

| metric | 변화 |
|---|---:|
| IoU | `+0.074399` |
| Dice | `+0.093925` |
| Recall | `+0.145933` |
| Failure | `-0.121593` |
| Boundary F1 | `+0.022176` |

상세 결과는 아래에 저장되어 있다.

```text
results/ZoomNeXt-CamLock/
```

## Repository 구성

```text
checkpoints/ZoomNeXt-CamLock/
  best.pth              # CamLock refiner checkpoint
  config.json           # 모델/학습/성능 요약
  training_log.csv      # 학습 로그

tools/
  run_zoomnext_camlock.py            # ZoomNeXt-CamLock train/apply/evaluate
  run_moch_new_model_predictions.py  # MoCH raw ZoomNeXt mask 생성
  run_source_zoomnext_predictions.py # MoCA/CAD raw ZoomNeXt mask 생성
  run_source_refiner_correct.py      # refiner/loss/data utility
  run_camlock_multiframe_memory.py   # memory helper
  evaluate_moch_baselines.py         # MoCH metric evaluation

external_models/ZoomNeXt/
  # 공식 ZoomNeXt source code
  # weights/ 는 포함하지 않음

results/ZoomNeXt-CamLock/
  overall_metrics.csv
  split_metrics.csv
  phase_metrics.csv
  first_frame_metrics.csv
  sequence_metrics.csv

docs/
  ZoomNeXt-CamLock_Explanation.docx
  ZoomNeXt-CamLock_Project_Summary.docx
```

## 실행 방법

필요한 Python package:

```bash
pip install -r requirements.txt
```

공식 ZoomNeXt weight를 아래 위치에 둔다.

```text
external_models/ZoomNeXt/weights/pvtv2-b5-zoomnext.pth
```

MoCH raw ZoomNeXt mask 생성:

```bash
python tools/run_moch_new_model_predictions.py \
  --models zoomnext \
  --moch-root MoCH \
  --out-root data/moch_raw \
  --device cuda
```

MoCA/CAD source raw ZoomNeXt mask 생성:

```bash
python tools/run_source_zoomnext_predictions.py \
  --moca-root MoCA_Video \
  --cad-root CamouflagedAnimalDataset \
  --out-root data/source_raw \
  --device cuda
```

저장된 checkpoint로 MoCH에 적용:

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

평가:

```bash
python tools/run_zoomnext_camlock.py \
  --action evaluate \
  --moch-root MoCH \
  --moch-raw-root data/moch_raw/ZoomNeXt_PvtV2B5 \
  --output-root predictions \
  --analysis-dir results/ZoomNeXt-CamLock \
  --device cuda
```

## 제외한 파일

repository를 가볍게 유지하기 위해 다음 파일은 포함하지 않았다.

- `MoCH`, `MoCA`, `CAD`, `CAMotion` dataset 원본
- MoCH 전체 prediction PNG
- MoCA/CAD/CAMotion raw prediction PNG
- 공식 ZoomNeXt weight와 `.7z` archive
- Python cache와 임시 산출물

즉 이 repository는 **최고 성능 ZoomNeXt-CamLock 모델을 설명하고 재현하기 위한 최소 코드/체크포인트/결과 요약**만 포함한다.
