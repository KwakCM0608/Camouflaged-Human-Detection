# ZoomNeXt-CamLock

`ZoomNeXt-CamLock`은 위장 객체 추적에서 **초기 1차 mask가 흔들리는 문제**를 줄이기 위해 만든 모델이다. 공식 `ZoomNeXt PVTv2-B5`를 1차 mask 생성기로 사용하고, 그 위에 작은 causal refiner와 lock-in 정책을 붙여 최종 mask를 보정한다.

가장 중요한 데이터 규칙은 다음과 같다.

- 학습: `MoCA`, `CAD`, `CAMotion`
- 평가: `MoCH`
- `MoCH`는 학습/검증에 사용하지 않고, 최종 prediction/evaluation에만 사용한다.
- 전체 MoCH sequence를 볼 때도 미래 frame이나 미래 GT는 사용하지 않는다.

## 프로젝트 제약사항

이 프로젝트는 일반적인 offline video segmentation이나 prompt 기반 tracking과 다르게, 다음 제약을 전제로 한다.

| 제약 | 내용 |
|---|---|
| 실시간성 | 현재 frame을 처리할 때 현재/과거 frame 정보만 사용한다. 미래 frame, 전체 sequence 후처리, oracle selection은 사용하지 않는다. |
| 첫 frame prompt 없음 | 첫 frame에서 bbox, point click, text prompt, GT mask 같은 사용자 prompt를 주지 않는다. 모델은 첫 frame부터 자동으로 mask를 생성해야 한다. |
| 자동 onset 판단 | motion onset은 GT나 수동 annotation이 아니라 현재/과거 frame의 motion history와 model confidence로만 판단한다. |
| mask 기반 추적 | 별도의 수동 object ID 지정 없이, ZoomNeXt raw mask와 CamLock belief를 기반으로 객체 후보를 유지한다. |

따라서 `ZoomNeXt-CamLock`은 첫 frame에 정답 mask나 box를 넣고 시작하는 semi-supervised VOS 모델이 아니라, **prompt 없이 시작해서 실시간 causal 방식으로 위장 객체 mask를 추정하는 모델**이다.

## MoCH

`MoCH`는 Moving Camouflaged Human Object Detection 데이터셋으로, 위장된 사람이 포함된 video sequence와 frame별 binary mask annotation을 제공한다. 전체 데이터는 98개 video sequence와 4,091개 human-annotated frame으로 구성되어 있고, `Train`, `Validation`, `Test` split을 가진다.

이 프로젝트에서 MoCH를 평가 데이터셋으로 사용한 이유는 다음과 같다.

| 이유 | 설명 |
|---|---|
| 실제 목표와 가장 가까운 평가 조건 | 정적인 단일 이미지보다, 움직임이 있는 video camouflaged object/human detection 상황을 직접 평가할 수 있다. |
| onset 문제를 보기 좋음 | 객체가 초반에는 배경에 숨어 있다가 움직임 이후 더 드러나는 sequence가 있어, onset 전/후 성능 차이를 분석하기 적합하다. |
| 실시간 causal 추론 검증 | frame 순서가 있는 video이므로 현재/과거 frame만 사용하는 실시간 추론 조건을 평가할 수 있다. |
| prompt-free 성능 검증 | 첫 frame prompt 없이 raw mask와 temporal belief만으로 객체를 찾는 설정이 실제로 동작하는지 확인할 수 있다. |
| source 학습셋과 분리 | 학습은 `MoCA`, `CAD`, `CAMotion`으로만 하고 MoCH는 평가에만 사용하여, 모델이 새로운 camouflaged video domain으로 일반화되는지 확인할 수 있다. |

즉 MoCH는 단순히 점수를 내기 위한 테스트셋이 아니라, `ZoomNeXt-CamLock`이 해결하려는 핵심 문제인 **prompt 없는 실시간 위장 객체 추적, onset 전후 안정성, 새로운 video domain 일반화**를 동시에 확인할 수 있는 평가셋이다.

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

## ZoomNeXt

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

onset 전/후로 나누면 개선 양상이 더 분명하다. 여기서는 `early` 구간을 onset 전, `middle+late` 구간을 onset 후로 묶었다.

| 구간 | model | frames | IoU | Dice | Recall | Failure | Boundary F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| onset 전 | ZoomNeXt_Baseline | 1248 | 0.302061 | 0.353429 | 0.324175 | 0.618590 | 0.186859 |
| onset 전 | ZoomNeXt-CamLock | 1248 | 0.340204 | 0.402856 | 0.394213 | 0.551282 | 0.198664 |
| onset 후 | ZoomNeXt_Baseline | 2494 | 0.432353 | 0.503311 | 0.468564 | 0.429431 | 0.249879 |
| onset 후 | ZoomNeXt-CamLock | 2494 | 0.524895 | 0.619502 | 0.652476 | 0.280674 | 0.277245 |

onset 전에도 IoU가 `+0.038143`, Recall이 `+0.070039` 개선되지만, onset 이후에는 개선 폭이 훨씬 커진다.

| 구간 | IoU 개선 | Dice 개선 | Recall 개선 | Failure 감소 |
|---|---:|---:|---:|---:|
| onset 전 | `+0.038143` | `+0.049427` | `+0.070039` | `-0.067308` |
| onset 후 | `+0.092542` | `+0.116191` | `+0.183911` | `-0.148757` |

이 결과가 `ZoomNeXt-CamLock`의 핵심 의미를 보여준다. 위장 객체는 onset 전에는 시각적 단서가 약해서 어떤 모델도 안정적으로 객체를 분리하기 어렵다. 하지만 움직임이 생긴 뒤에는 객체와 배경을 구분할 수 있는 단서가 늘어나고, CamLock은 이 시점부터 refiner 출력과 이전 belief를 함께 사용해 mask를 안정적으로 회복한다. 즉 이 모델의 강점은 단순히 전체 평균 IoU를 올리는 데 있지 않고, **onset 이후 객체가 드러나는 구간에서 raw baseline보다 훨씬 안정적으로 따라붙는 것**에 있다.

## 한계

가장 큰 한계는 onset 판단이 여전히 `ZoomNeXt`의 1차 mask 품질에 영향을 받는다는 점이다. CamLock은 현재 frame, raw mask, 이전 belief, motion 정보를 함께 사용하지만, 출발점이 되는 1차 mask가 배경과 위장 객체를 심하게 헷갈리면 onset 위치도 부정확해질 수 있다.

특히 raw baseline이 초반에 객체를 너무 작게 잡거나, 배경 texture를 foreground처럼 내보내면 다음 문제가 생긴다.

- onset 전후의 confidence가 실제 객체 움직임과 맞지 않을 수 있다.
- lock-in이 너무 늦게 걸리거나, 반대로 잘못된 위치에 걸릴 수 있다.
- refiner가 보정할 수 있는 범위를 넘어선 raw mask 오류는 최종 mask에도 남는다.
- onset 이후 성능은 크게 개선되지만, onset 자체를 정확히 잡는 능력은 1차 mask generator의 품질에 의해 제한된다.

따라서 다음 단계의 핵심은 `CamLock` 정책을 더 복잡하게 만드는 것보다, **1차 mask를 만드는 baseline을 배경과 위장 객체를 더 잘 구분하도록 개선하는 것**이다. raw mask 단계에서 객체 후보가 더 정확해지면 onset 판단도 안정되고, CamLock의 lock-in/refine 효과도 더 크게 살아날 수 있다.

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
