# 실험 히스토리

이 파일은 현재 작업공간에 **실험 결과 파일이 남아있는 항목**을 기준으로 정리한 것이다. 파일명은 요청을 그대로 따라 `histroy.md`로 둔다.

정리 기준:

- `overall_metrics.csv`가 있는 실험은 해당 CSV에서 IoU가 가장 높은 행을 대표 결과로 적었다.
- 같은 결과 파일 안에 여러 모델/variant가 있으면 `후보 수`에 행 개수를 적었다.
- `organized_outputs`의 과거 archive에는 MoCH가 아닌 MoCA/CAD 계열 결과가 섞여 있으므로, MoCH 결과와 점수를 직접 비교하면 안 된다.
- prediction image, checkpoint binary, GIF 같은 대용량 산출물은 metrics/log/manifest와 연결되는 경우만 간접적으로 기록했다.

## 현재 MoCH 평가 결과

| 결과 위치 | Dataset | 후보 수 | 대표 최고 모델 | IoU | Dice | Frames | 한줄요약 |
|---|---:|---:|---|---:|---:|---:|---|
| `MoCH_Test/analysis_base_combo` | MoCH | 6 | `EMIP_CLIP_CausalRefiner_BestPolicy` | 0.4718 | 0.5755 | 3742 | EMIP official/CLIP/causal refiner 조합을 한 번에 비교한 base-combination 평가. |
| `MoCH_Test/analysis_camlock_integrated_multiframe` | MoCH | 6 | `CamLockIntegrated_K3_EMA` | 0.4167 | 0.4929 | 3742 | ZoomNeXt-CamLock에 K3/K5 EMA 및 trajectory 기반 multi-frame memory를 붙인 비교. |
| `MoCH_Test/analysis_camlock_onset_policy_ablation` | MoCH | 9 | `OnsetABC_StrongTrust_PostOnly_DynamicMemory` | 0.4180 | 0.4945 | 3742 | StrongTrust, PostOnlyMemory, DynamicMemory 등 onset 이후 lock-in 정책 ablation. |
| `MoCH_Test/analysis_camlock_preonset_cleanup` | MoCH | 12 | `PreOnset_RawAnchoredCam` | 0.4205 | 0.4958 | 3742 | onset 전 raw mask를 보수적으로 anchor하고 이후 CamLock으로 보정하는 pre-onset cleanup 실험. |
| `MoCH_Test/analysis_causal_lockin_best` | MoCH | 4 | `CausalLockIn_Best` | 0.4718 | 0.5755 | 3742 | CausalLockIn 계열에서 best policy를 골라 전체 MoCH에 평가한 결과. |
| `MoCH_Test/analysis_causal_lockin_t10` | MoCH | 2 | `MoCH_CausalLockIn_T10` | 0.4713 | 0.5750 | 3742 | MoCH T10 후보에 causal lock-in을 붙이고 motion onset 분석까지 남긴 평가. |
| `MoCH_Test/analysis_clip_variant_full` | MoCH | 3 | `CLIPV6_MaxPromptSharp` | 0.4720 | 0.5757 | 3742 | CLIP prompt variant를 full MoCH에 평가한 비교; MoCH 개입 계열이라 최종 모델 근거로는 제외 대상. |
| `MoCH_Test/analysis_clip_variant_validation` | MoCH | 11 | `CLIPV6_MaxPromptSharp` | 0.4657 | 0.5619 | 472 | CLIP prompt variant를 Validation split에서 sweep한 비교; prompt 선택 경향 확인용. |
| `MoCH_Test/analysis_emip_causal_lockin_integrated` | MoCH | 4 | `EMIP_CausalLockIn` | 0.4713 | 0.5750 | 3742 | EMIP 기반 prediction에 CausalLockIn을 통합해 full MoCH에서 비교한 실험. |
| `MoCH_Test/analysis_emip_multihypothesis_full` | MoCH | 3 | `EMIP_CausalRefiner_Default` | 0.4690 | 0.5724 | 3742 | EMIP multi-hypothesis lock-in 후보 중 full MoCH 평가가 남아있는 실험. |
| `MoCH_Test/analysis_emip_multihypothesis_validation` | MoCH | 11 | `EMIP_CausalRefiner_Default` | 0.4601 | 0.5570 | 472 | EMIP multi-hypothesis lock-in 후보를 Validation split에서 비교한 실험. |
| `MoCH_Test/analysis_fair_top10` | MoCH | 3 | `EMIP` | 0.3639 | 0.4470 | 3742 | 초기 fair top-10 후보를 같은 MoCH 조건에서 비교한 baseline 정리. |
| `MoCH_Test/analysis_full_baseline_replacement` | MoCH | 12 | `ZoomNeXt-CamLock_Current` | 0.4395 | 0.5161 | 3742 | ZoomNeXt 1차 mask generator 자체를 decoder fine-tune 계열로 갈아끼운 뒤 CamLock refiner를 붙인 비교. |
| `MoCH_Test/analysis_rt01_now` | MoCH | 3 | `EMIP` | 0.3639 | 0.4470 | 3742 | RT01/EMIP 계열 current baseline을 MoCH에서 확인한 임시 평가. |
| `MoCH_Test/analysis_test_moch_1` | MoCH | 16 | `MoCH_T10_EMIP_OpenVocabulary_FiLM` | 0.3667 | 0.4491 | 3742 | MoCH T1-T10 foundation/open-vocabulary 후보군의 첫 전체 평가. |
| `MoCH_Test/analysis_test_moch_1_final` | MoCH | 16 | `MoCH_T10_EMIP_OpenVocabulary_FiLM` | 0.3667 | 0.4491 | 3742 | test_moch_1 결과를 final 형태로 정리한 평가본. |
| `MoCH_Test/analysis_test_moch_1_mp` | MoCH | 16 | `MoCH_T10_EMIP_OpenVocabulary_FiLM` | 0.3667 | 0.4491 | 3742 | test_moch_1을 multiprocessing 적용 경로로 다시 산출한 평가본. |
| `MoCH_Test/analysis_test_moch_2` | MoCH | 4 | `MoCH_CausalLockIn_T10` | 0.4713 | 0.5750 | 3742 | T10 기반 CausalLockIn 및 post-onset memory를 추가한 두 번째 MoCH 실험. |
| `MoCH_Test/analysis_test_moch_3` | MoCH | 7 | `CausalLockIn` | 0.4713 | 0.5750 | 3742 | CausalLockIn variants와 motion-triggered 계열을 확장 비교한 세 번째 MoCH 실험. |
| `MoCH_Test/analysis_test_moch_4` | MoCH | 5 | `ZoomNeXt_PvtV2B5` | 0.3889 | 0.4533 | 3742 | EMIP/TSP-SAM/ZoomNeXt/FastSAM/MobileSAM 등 외부 후보 baseline 비교. |
| `MoCH_Test/analysis_test_moch_4_partial` | MoCH | 3 | `ZoomNeXt_PvtV2B5` | 0.3889 | 0.4533 | 3742 | test_moch_4 후보 일부만 먼저 평가한 partial 결과. |
| `MoCH_Test/analysis_test_moch_5` | MoCH | 5 | `ZoomNeXt_AugFrozenRefiner` | 0.4158 | 0.4920 | 3742 | ZoomNeXt raw mask에 source-trained frozen/causal/pyramid refiner를 붙인 비교. |
| `MoCH_Test/camotion_usage/analysis` | MoCH | 13 | `ZoomNeXt_Freeze_CAMotionRefiner` | 0.4633 | 0.5472 | 3742 | CAMotion/MoCA/CAD source 조합을 바꿔 CamLock refiner를 학습한 source 구성 비교. |
| `MoCH_Test/object_motion_gate/analysis` | MoCH | 3 | `ZoomNeXt-CamLock` | 0.4633 | 0.5472 | 3742 | object-level motion gate를 CamLock에 추가해 실제 이득이 있는지 본 실험. |
| `MoCH_Test/object_motion_gate_frozen/analysis` | MoCH | 3 | `ZoomNeXt-CamLock` | 0.4633 | 0.5472 | 3742 | motion gate를 frozen/warm-start 조건으로 붙인 비교. |
| `MoCH_Test/object_motion_gate_frozen_tuned/analysis` | MoCH | 3 | `ZoomNeXt-CamLock` | 0.4633 | 0.5472 | 3742 | frozen motion gate 계열을 threshold/tuning까지 적용해 재평가한 비교. |
| `MoCH_Test/replacement_candidates_eval` | MoCH | 2 | `EMIP` | 0.3638 | 0.4468 | 3742 | EMIP와 TSP-SAM 등 교체 후보 baseline만 별도로 평가한 결과. |
| `MoCH_Test/source_refiner_correct/analysis` | MoCH | 5 | `ZoomNeXt_SourceRefiner` | 0.4059 | 0.4748 | 3742 | source-trained refiner가 MoCH에서 raw ZoomNeXt를 얼마나 보정하는지 확인한 실험. |
| `MoCH_Test/zoomnext_refiner_feature_matrix/analysis_final` | MoCH | 11 | `R1_PC1_ON1_PM1_CL0_PH0_PRE0_POST1_CUnone` | 0.4068 | 0.4758 | 3742 | refiner 입력 feature, previous memory, onset, cleanup 조합 matrix를 최종 평가한 ablation. |

## 추가 Metric/Manifest 결과

| 결과 위치 | 후보 수 | 대표 항목 | IoU | Dice | 한줄요약 |
|---|---:|---|---:|---:|---|
| `MoCH_Test/zoomnext_refiner_opt/combo_overall_metrics.csv` | 263 | `static_a1.00_t0.40_small` | 0.4094 | 0.4806 | ZoomNeXt refiner blending alpha/threshold/cleanup 조합 grid search. |
| `MoCH_Test/zoomnext_refiner_opt/combo_first_metrics_top.csv` | 20 | `static_a1.00_t0.40_none` | 0.3312 | 0.3918 | refiner grid search 중 첫 frame 성능이 높은 조합을 따로 모은 결과. |
| `MoCH_Test/zoomnext_refiner_opt/combo_phase_metrics_top.csv` | 60 | `static_a0.85_t0.40_none` | 0.5282 | 0.6119 | refiner grid search의 early/middle/late phase별 상위 조합 정리. |
| `MoCH_Test/zoomnext_refiner_feature_matrix/core_feature_matrix_metrics.csv` | 49 | `R1_PC1_ON1_PM1_CL0_PH0_PRE0_POST1` | 0.3969 | 0.4694 | refiner 핵심 입력 feature 조합만으로 만든 ablation matrix. |
| `MoCH_Test/zoomnext_refiner_feature_matrix/final_feature_matrix_metrics.csv` | 9 | `R1_PC1_ON1_PM1_CL0_PH0_PRE0_POST1_CUnone` | 0.3969 | 0.4694 | cleanup까지 포함한 refiner feature matrix 최종 후보 결과. |
| `MoCH_Test/policy_grid_emip_causal_lockin/full_summary.csv` | 12 | `T0334_early_net_strong_mid_raw_more_locked_stable_prev_onset_fast_blur_strong_first_default` | 0.4718 | 0.5755 | EMIP+causal lock-in의 full policy grid search 결과. |
| `MoCH_Test/policy_grid_emip_causal_lockin/single_summary.csv` | 26 | `P20_mid_raw_more` | 0.4647 | 0.5610 | EMIP+causal lock-in single-change policy ablation 결과. |
| `MoCH_Test/policy_grid_emip_causal_lockin/targeted_summary.csv` | 881 | `T0334_early_net_strong_mid_raw_more_locked_stable_prev_onset_fast_blur_strong_first_default` | 0.4663 | 0.5625 | EMIP+causal lock-in targeted policy 후보 결과. |
| `MoCH_Test/cad_top_moca2moch/summary.json` | 3 | `dino, thin, stage2` | - | - | CAD/MoCA source에서 MoCH로 넘어가는 DINO/thin/stage2 cascade 실험 manifest. |
| `organized_outputs/experiment_results/cleanup_records/cleanup_top10_experiments.json` | 10 | `Exp2_FusionAvg` | 0.8109 | 0.8620 | 과거 archive 정리 시 IoU 기준 상위 10개 실험을 남긴 기록. |

## 과거 Archive 평가 결과

아래 항목은 `organized_outputs/experiment_results/analysis_result_dirs`에 보존된 과거 결과다. 대부분 MoCA/CAD 중심 실험이므로 현재 최종 MoCH 평가표와 같은 선상에서 비교하지 않는다.

| 결과 위치 | Dataset | 후보 수 | 대표 최고 모델 | IoU | Dice | Frames | 한줄요약 |
|---|---:|---:|---|---:|---:|---:|---|
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results` | MoCA_Video | 4 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | 초기 EMIP/TSP-SAM 및 fusion 후보 baseline archive. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_a1_teacher_blend` | MoCA_Video | 18 | `A1TeacherBlend_025` | 0.8071 | 0.8590 | 4691 | teacher blend 비율을 바꿔 pseudo/teacher output을 섞은 archive 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_baseline_moch` | MoCH | 2 | `EMIP` | 0.3639 | 0.4470 | 3742 | MoCH baseline EMIP/TSP-SAM 비교를 보존한 archive. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_baseline_moch_emip_only` | MoCH | 1 | `EMIP` | 0.3639 | 0.4470 | 3742 | MoCH baseline 중 EMIP 단독 결과를 보존한 archive. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_emip_compare_prelim` | MoCA_Video | 4 | `EMIP` | 0.7867 | 0.8426 | 4691 | EMIP 재현/비교 preliminary 결과. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_emip_dagger_prelim` | MoCA_Video | 2 | `EMIP† 재현시도` | 0.7641 | 0.8224 | 4691 | EMIP dagger-style 재현 시도 preliminary 결과. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test11` | MoCA_Video | 54 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | TSP-SAM/EMIP 및 후처리 후보들을 대량 비교한 Test11 archive. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test11_partial_k` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test11 candidate bank 중 일부 후보만 남은 partial 결과. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12` | MoCA_Video | 42 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | shared stem/cross FiLM/hybrid 등 대규모 Test12 후보 sweep. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_fast` | MoCA_Video | 42 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 full sweep를 fast 경로로 재산출한 결과. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_AB` | MoCA_Video | 8 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_ABC` | MoCA_Video | 10 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_ABCD` | MoCA_Video | 12 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 A-D 계열 partial 후보 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_ABCDE` | MoCA_Video | 14 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 A-E 계열 partial 후보 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_EFGH` | MoCA_Video | 12 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 E-H 계열 partial 후보 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_J` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_K` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_M` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_N` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_Q` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_R` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_S` | MoCA_Video | 2 | `Test12_S_SharedStem_CrossFiLM_Hybrid` | 0.6736 | 0.7555 | 4691 | Test12 개별 candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test12_partial_baseline_T` | MoCA_Video | 6 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test12 baseline 후보 보존 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13` | MoCA_Video | 42 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | prototype token bank, VideoMAE, SAM adapter, DINO 계열 Test13 sweep. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_C` | MoCA_Video | 2 | `Test13_C_TSP_SAM_ProtoTokenBank` | 0.7013 | 0.7782 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_D` | MoCA_Video | 2 | `Test13_D_VideoMAE_DistilledCamoStudent` | 0.6520 | 0.7360 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_G` | MoCA_Video | 2 | `Test13_G_EMIP_ProtoContrastBank` | 0.7216 | 0.8014 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_I` | MoCA_Video | 2 | `Test13_I_EMIP_VideoMAEAppearance` | 0.7394 | 0.8097 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_K` | MoCA_Video | 2 | `Test13_K_SAM_AdapterTemporalSideNet` | 0.6671 | 0.7524 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_L` | MoCA_Video | 2 | `Test13_L_SAM_TokenMotionSelector` | 0.7114 | 0.7883 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_M` | MoCA_Video | 2 | `Test13_M_SAM_PromptPoolRetriever` | 0.6693 | 0.7537 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_O` | MoCA_Video | 2 | `Test13_O_SAM_TextParallelPrompt` | 0.7023 | 0.7855 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_Q` | MoCA_Video | 2 | `Test13_Q_ConvNeXtV2_PixelContrast` | 0.5126 | 0.6107 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_R` | MoCA_Video | 2 | `Test13_R_MSN_ViT_TemporalStudent` | 0.7119 | 0.7847 | 4691 | Test13 개별 foundation/model candidate partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test13_partial_baseline` | MoCA_Video | 4 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test13 baseline 후보 보존 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test14` | MoCA_Video | 14 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test14 hybrid teacher, query bridge, DINO-XMem 계열 통합 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test14_partial_H1` | MoCA_Video | 2 | `Test14_HybridTeacherDistilledSegFormer` | 0.7120 | 0.7872 | 4691 | Test14 hybrid teacher/query bridge/auto prompt memory 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test14_partial_M1` | CAD | 2 | `Test14_EMIPNextMotionAppearanceStudent` | 0.1229 | 0.1709 | 191 | Test14 hybrid teacher/query bridge/auto prompt memory 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test14_partial_Q1` | MoCA_Video | 2 | `Test14_CrossFoundationQueryBridge` | 0.7600 | 0.8269 | 4691 | Test14 hybrid teacher/query bridge/auto prompt memory 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test14_partial_S1` | MoCA_Video | 2 | `Test14_AutoPromptMemorySAM` | 0.6048 | 0.7073 | 4691 | Test14 hybrid teacher/query bridge/auto prompt memory 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test14_partial_X1` | MoCA_Video | 2 | `Test14_DINOXMemCalibratedStudent` | 0.7071 | 0.7891 | 4691 | Test14 hybrid teacher/query bridge/auto prompt memory 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test14_partial_baseline` | MoCA_Video | 4 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test14 baseline 후보 보존 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test15` | MoCA_Video | 14 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test15 late exploratory 후보군 통합 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test15_partial_P1` | MoCA_Video | 2 | `Test15_P1_QBridgeV2_SAMDINO_QueryMemory` | 0.1455 | 0.2231 | 4691 | Test15 query-memory/slot-memory/delayed fusion 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test15_partial_P2` | MoCA_Video | 2 | `Test15_P2_DINO_CutieSlotMemory` | 0.0845 | 0.1433 | 4691 | Test15 query-memory/slot-memory/delayed fusion 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test15_partial_P3` | MoCA_Video | 2 | `Test15_P3_SegFormerB1_DeAOTPermanence` | 0.1943 | 0.2771 | 4691 | Test15 query-memory/slot-memory/delayed fusion 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test15_partial_P4` | MoCA_Video | 2 | `Test15_P4_DINOMSN_TemporalAdapter` | 0.0839 | 0.1408 | 4691 | Test15 query-memory/slot-memory/delayed fusion 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test15_partial_P5` | CAD | 2 | `Test15_P5_VideoMAE_ConvNeXt_DelayedFusion` | 0.2103 | 0.3087 | 191 | Test15 query-memory/slot-memory/delayed fusion 후보 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test15_partial_baseline` | MoCA_Video | 4 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | Test15 baseline 후보 보존 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test2_cpu` | MoCA_Video | 14 | `Exp2_FusionAvg` | 0.8109 | 0.8620 | 4691 | FusionAvg/FusionQualityGate/FusionRecallMax 등 초기 fusion 계열 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test2_cpu_fast` | MoCA_Video | 14 | `Exp2_FusionAvg` | 0.8109 | 0.8620 | 4691 | FusionAvg/FusionQualityGate/FusionRecallMax 등 초기 fusion 계열 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test2_final` | MoCA_Video | 24 | `Exp2_FusionAvg` | 0.8109 | 0.8620 | 4691 | FusionAvg/FusionQualityGate/FusionRecallMax 등 초기 fusion 계열 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test3` | MoCA_Video | 18 | `ExpRef_FusionAvg` | 0.8109 | 0.8620 | 4691 | test2 fusion 계열을 reference/parallel 경로로 재확인한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test3_cpu` | MoCA_Video | 14 | `ExpRef_FusionAvg` | 0.8109 | 0.8620 | 4691 | test2 fusion 계열을 reference/parallel 경로로 재확인한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test3_parallel` | MoCA_Video | 18 | `ExpRef_FusionAvg` | 0.8109 | 0.8620 | 4691 | test2 fusion 계열을 reference/parallel 경로로 재확인한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4` | MoCA_Video | 36 | `ExpC_ConstrainedPixelGate` | 0.8109 | 0.8601 | 4691 | constrained pixel gate 중심의 broader fusion/gating sweep. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_base2` | MoCA_Video | 16 | `Exp5_SparseFusionGate` | 0.8030 | 0.8559 | 4691 | base2 sparse/deferral 계열을 묶어 비교한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_base2_parts/cpu_done` | MoCA_Video | 6 | `Exp1_AdaptiveCalibratedTSP` | 0.7967 | 0.8469 | 4691 | sparse fusion gate, adaptive calibrated TSP, selective prompt deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_base2_parts/gpu_done` | MoCA_Video | 8 | `Exp5_SparseFusionGate` | 0.8030 | 0.8559 | 4691 | sparse fusion gate, adaptive calibrated TSP, selective prompt deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_base2_parts/selective` | MoCA_Video | 2 | `Exp2_SelectivePromptDeferral` | 0.7948 | 0.8440 | 4691 | sparse fusion gate, adaptive calibrated TSP, selective prompt deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_parts/part1` | MoCA_Video | 10 | `ExpC_ConstrainedPixelGate` | 0.8109 | 0.8601 | 4691 | constrained pixel gate, recall-max fusion, calibrated TSP, boundary topology refiner를 나눠 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_parts/part2` | MoCA_Video | 10 | `ExpH_FusionRecallMax` | 0.8014 | 0.8550 | 4691 | constrained pixel gate, recall-max fusion, calibrated TSP, boundary topology refiner를 나눠 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_parts/part3` | MoCA_Video | 10 | `ExpJ_CalibratedTSP` | 0.7957 | 0.8462 | 4691 | constrained pixel gate, recall-max fusion, calibrated TSP, boundary topology refiner를 나눠 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test4_parts/part4` | MoCA_Video | 6 | `ExpM_BoundaryTopologyRefiner` | 0.7928 | 0.8457 | 4691 | constrained pixel gate, recall-max fusion, calibrated TSP, boundary topology refiner를 나눠 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test6` | MoCA_Video | 16 | `Test6_SparseGateV2` | 0.8035 | 0.8561 | 4691 | Test6 sparse gate/prototype memory 계열 통합 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test6_parts/base_sparse` | MoCA_Video | 6 | `Test6_SparseGateV2` | 0.8035 | 0.8561 | 4691 | sparse gate v2, calibrated static, prototype memory cleanup, selective deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test6_parts/calibrated_static` | MoCA_Video | 4 | `Test6_AdaptiveCalibratedTSP` | 0.7967 | 0.8469 | 4691 | sparse gate v2, calibrated static, prototype memory cleanup, selective deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test6_parts/proto_memory` | MoCA_Video | 2 | `Test6_PrototypeMemoryCleanup` | 0.7954 | 0.8451 | 4691 | sparse gate v2, calibrated static, prototype memory cleanup, selective deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test6_parts/selective_proto_memory` | MoCA_Video | 2 | `Test6_SelectiveDeferralProtoMemory` | 0.7937 | 0.8429 | 4691 | sparse gate v2, calibrated static, prototype memory cleanup, selective deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test6_parts/selective_raw` | MoCA_Video | 2 | `Test6_SelectiveDeferralV2` | 0.7948 | 0.8440 | 4691 | sparse gate v2, calibrated static, prototype memory cleanup, selective deferral을 부분 평가한 실험. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test7` | MoCA_Video | 18 | `Test7_SeqMoEViterbi` | 0.8083 | 0.8610 | 4691 | SeqMoE/Viterbi 및 tri-expert sparse gate 계열 archive 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test8` | MoCA_Video | 16 | `Test8_MonotoneRegionGate` | 0.8053 | 0.8573 | 4691 | monotone region gate와 proto graph decoder를 포함한 Test8 통합 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test8_part1` | MoCA_Video | 10 | `Test8_PhaseConformalDeferralPP` | 0.7953 | 0.8454 | 4691 | phase conformal deferral/post-processing 계열 partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test8_part2_monotone` | MoCA_Video | 2 | `Test8_MonotoneRegionGate` | 0.8053 | 0.8573 | 4691 | monotone region gate 단독/부분 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test8_part3_stage2` | MoCA_Video | 4 | `Test8_ProtoGraphDecoder` | 0.8015 | 0.8532 | 4691 | proto graph decoder stage2 계열 부분 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9` | MoCA_Video | 14 | `Test9_D_GraphReasonerD3Topology` | 0.8000 | 0.8519 | 4691 | graph reasoner, dual-teacher, MoE student 계열 Test9 통합 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_final_subset` | MoCA_Video | 24 | `Test9_D_GraphReasonerD3Topology` | 0.8000 | 0.8519 | 4691 | test9 후보 중 final subset으로 추린 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_md_full_baseline` | MoCA_Video | 4 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | test9 motion/dynamics full run의 baseline 비교. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_md_full_d` | MoCA_Video | 6 | `Test9MD_D1_SpatialGNN` | 0.7951 | 0.8436 | 4691 | test9 motion/dynamics D계열 spatial GNN 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_md_full_partial_a` | MoCA_Video | 6 | `Test9MD_A3_SkeletonBoundary` | 0.7280 | 0.7895 | 4691 | test9 motion/dynamics A계열 skeleton/boundary partial 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_original_md` | MoCA_Video | 16 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | test9 original motion/dynamics 전체 후보 archive. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_original_md_baseline` | MoCA_Video | 4 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | test9 original motion/dynamics baseline 보존본. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_part_ab` | MoCA_Video | 8 | `Test9_A_DualTeacherTemporalStudent` | 0.7966 | 0.8470 | 4691 | dual-teacher temporal student 등 test9 A/B계열 부분 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_part_d2` | MoCA_Video | 2 | `Test9_D_GraphReasonerD2` | 0.8000 | 0.8519 | 4691 | graph reasoner D2 topology 계열 부분 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_part_d3` | MoCA_Video | 2 | `Test9_D_GraphReasonerD3Topology` | 0.8000 | 0.8519 | 4691 | graph reasoner D3 topology 계열 부분 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_part_e` | MoCA_Video | 2 | `Test9_E_SharedExpertMoEStudent` | 0.7738 | 0.8328 | 4691 | shared expert MoE student 계열 부분 평가. |
| `organized_outputs/experiment_results/analysis_result_dirs/analysis_results_test9_strict` | MoCA_Video | 4 | `TSP-SAM` | 0.7910 | 0.8430 | 4691 | test9 strict 조건 baseline/후보 평가. |

## 학습 로그가 남아있는 실험

아래는 평가 CSV와 별개로 `training_log.csv`가 남아있는 학습 실험이다. 성능 비교가 아니라 어떤 학습을 실제로 돌렸는지 확인하기 위한 목록이다.

| 실험 | 로그 위치 | 한줄요약 |
|---|---|---|
| `ZN_BaseAdapter_Std` | `MoCH_Test/baseline_replacement/work/baseline_adapters/ZN_BaseAdapter_Std/training_log.csv` | 학습 로그가 남아있는 refiner/adapter 학습 실험. |
| `CamLockIntegrated_K3_EMA` | `MoCH_Test/camlock_integrated_multiframe_memory/smoke_work/CamLockIntegrated_K3_EMA/training_log.csv` | multi-frame memory 통합 CamLock refiner 학습 로그. |
| `CamLockIntegrated_K3_EMA` | `MoCH_Test/camlock_integrated_multiframe_memory/work/CamLockIntegrated_K3_EMA/training_log.csv` | multi-frame memory 통합 CamLock refiner 학습 로그. |
| `CamLockIntegrated_K3_Trajectory` | `MoCH_Test/camlock_integrated_multiframe_memory/work/CamLockIntegrated_K3_Trajectory/training_log.csv` | multi-frame memory 통합 CamLock refiner 학습 로그. |
| `CamLockIntegrated_K5_EMA` | `MoCH_Test/camlock_integrated_multiframe_memory/work/CamLockIntegrated_K5_EMA/training_log.csv` | multi-frame memory 통합 CamLock refiner 학습 로그. |
| `CamLockIntegrated_K5_QualityTrajectory` | `MoCH_Test/camlock_integrated_multiframe_memory/work/CamLockIntegrated_K5_QualityTrajectory/training_log.csv` | multi-frame memory 통합 CamLock refiner 학습 로그. |
| `ZoomNeXt_Freeze_CAMotionRefiner` | `MoCH_Test/camotion_usage/smoke_work/ZoomNeXt_Freeze_CAMotionRefiner/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotionMix_CamLock` | `MoCH_Test/camotion_usage/work/CAMotionMix_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotionOnly_CamLock` | `MoCH_Test/camotion_usage/work/CAMotionOnly_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotionPretrain_MoCADFinetune_CamLock/finetune` | `MoCH_Test/camotion_usage/work/CAMotionPretrain_MoCADFinetune_CamLock/finetune/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotionPretrain_MoCADFinetune_CamLock/pretrain` | `MoCH_Test/camotion_usage/work/CAMotionPretrain_MoCADFinetune_CamLock/pretrain/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotionPrimary_EdgeAux_CamLock` | `MoCH_Test/camotion_usage/work/CAMotionPrimary_EdgeAux_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotionPrimary_HardSampler_CamLock` | `MoCH_Test/camotion_usage/work/CAMotionPrimary_HardSampler_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotionPrimary_MoCADAux_CamLock` | `MoCH_Test/camotion_usage/work/CAMotionPrimary_MoCADAux_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `CAMotion_OnsetCurriculum_CamLock` | `MoCH_Test/camotion_usage/work/CAMotion_OnsetCurriculum_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `MoCADPrimary_CAMotionAux_CamLock` | `MoCH_Test/camotion_usage/work/MoCADPrimary_CAMotionAux_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `MoCAD_CAMotion_EdgeAux_CamLock` | `MoCH_Test/camotion_usage/work/MoCAD_CAMotion_EdgeAux_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `MoCAD_CAMotion_HardSampler_CamLock` | `MoCH_Test/camotion_usage/work/MoCAD_CAMotion_HardSampler_CamLock/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `ZoomNeXt_Feature_CAMotionIntegrated` | `MoCH_Test/camotion_usage/work/ZoomNeXt_Feature_CAMotionIntegrated/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `ZoomNeXt_Freeze_CAMotionRefiner` | `MoCH_Test/camotion_usage/work/ZoomNeXt_Freeze_CAMotionRefiner/training_log.csv` | MoCA/CAD/CAMotion source 조합으로 CamLock refiner를 학습한 로그. |
| `causal_lockin_t10` | `MoCH_Test/causal_lockin_t10/training_log.csv` | causal lock-in/motion-triggered refiner 학습 로그. |
| `causal_lockin_t10_smoke` | `MoCH_Test/causal_lockin_t10_smoke/training_log.csv` | causal lock-in/motion-triggered refiner 학습 로그. |
| `ZN_FullDecoderFT` | `MoCH_Test/full_baseline_replacement/work/full_zoomnext/ZN_FullDecoderFT/training_log.csv` | ZoomNeXt decoder/full baseline replacement 쪽 학습 로그. |
| `ZN_FullDecoderFT_Contrastive` | `MoCH_Test/full_baseline_replacement/work/full_zoomnext/ZN_FullDecoderFT_Contrastive/training_log.csv` | ZoomNeXt decoder/full baseline replacement 쪽 학습 로그. |
| `ZN_FullDecoderFT_EdgeAux` | `MoCH_Test/full_baseline_replacement/work/full_zoomnext/ZN_FullDecoderFT_EdgeAux/training_log.csv` | ZoomNeXt decoder/full baseline replacement 쪽 학습 로그. |
| `ZN_FullDecoderFT_HardNeg` | `MoCH_Test/full_baseline_replacement/work/full_zoomnext/ZN_FullDecoderFT_HardNeg/training_log.csv` | ZoomNeXt decoder/full baseline replacement 쪽 학습 로그. |
| `ZN_FullDecoderFT_Quality` | `MoCH_Test/full_baseline_replacement/work/full_zoomnext/ZN_FullDecoderFT_Quality/training_log.csv` | ZoomNeXt decoder/full baseline replacement 쪽 학습 로그. |
| `CAMotionMix_CamLock` | `MoCH_Test/full_baseline_replacement/work/refiners/ZN_FullDecoderFT/CAMotionMix_CamLock/training_log.csv` | 교체한 1차 mask 위에 CamLock refiner를 다시 학습한 로그. |
| `CAMotionMix_CamLock` | `MoCH_Test/full_baseline_replacement/work/refiners/ZN_FullDecoderFT_Contrastive/CAMotionMix_CamLock/training_log.csv` | 교체한 1차 mask 위에 CamLock refiner를 다시 학습한 로그. |
| `CAMotionMix_CamLock` | `MoCH_Test/full_baseline_replacement/work/refiners/ZN_FullDecoderFT_EdgeAux/CAMotionMix_CamLock/training_log.csv` | 교체한 1차 mask 위에 CamLock refiner를 다시 학습한 로그. |
| `CAMotionMix_CamLock` | `MoCH_Test/full_baseline_replacement/work/refiners/ZN_FullDecoderFT_HardNeg/CAMotionMix_CamLock/training_log.csv` | 교체한 1차 mask 위에 CamLock refiner를 다시 학습한 로그. |
| `CAMotionMix_CamLock` | `MoCH_Test/full_baseline_replacement/work/refiners/ZN_FullDecoderFT_Quality/CAMotionMix_CamLock/training_log.csv` | 교체한 1차 mask 위에 CamLock refiner를 다시 학습한 로그. |
| `ZoomNeXt_AugFrozenRefiner` | `MoCH_Test/moch5_work/ZoomNeXt_AugFrozenRefiner/training_log.csv` | 학습 로그가 남아있는 refiner/adapter 학습 실험. |
| `motion_triggered_t10` | `MoCH_Test/motion_triggered_t10/training_log.csv` | causal lock-in/motion-triggered refiner 학습 로그. |
| `motion_triggered_t10_smoke` | `MoCH_Test/motion_triggered_t10_smoke/training_log.csv` | causal lock-in/motion-triggered refiner 학습 로그. |
| `ZoomNeXt-CamLock_ObjectMotionGate` | `MoCH_Test/object_motion_gate/work/ZoomNeXt-CamLock_ObjectMotionGate/training_log.csv` | object motion gate를 붙인 CamLock 계열 학습 로그. |
| `ZoomNeXt-CamLock_ObjectMotionGate` | `MoCH_Test/object_motion_gate_frozen/work/ZoomNeXt-CamLock_ObjectMotionGate/training_log.csv` | object motion gate를 붙인 CamLock 계열 학습 로그. |
| `EMIP_SourceRefiner` | `MoCH_Test/source_refiner_correct/work/EMIP_SourceRefiner/training_log.csv` | source prediction correction refiner 학습 로그. |
| `ZoomNeXt_SourceRefiner` | `MoCH_Test/source_refiner_correct/work/ZoomNeXt_SourceRefiner/training_log.csv` | source prediction correction refiner 학습 로그. |
| `zoomnext_causal_refiner` | `MoCH_Test/zoomnext_causal_refiner/training_log.csv` | 학습 로그가 남아있는 refiner/adapter 학습 실험. |

## 해석 메모

- 현재 제약 조건에 맞는 핵심 축은 `ZoomNeXt` 1차 mask와 `CamLock` causal refiner를 결합한 계열이다.
- CLIP/EMIP CausalLockIn 계열은 높은 MoCH 점수가 남아 있지만, MoCH 학습 또는 prompt/selection 개입 가능성이 있어 최종 prompt-free 실시간 모델 근거로는 분리해서 봐야 한다.
- 최근 실험의 핵심 관찰은 1차 mask 품질이 onset 위치와 refiner 성능의 상한을 크게 좌우한다는 점이다.
- 따라서 이후 개선 방향은 ZoomNeXt-CamLock 구조를 유지하되, 1차 mask generator가 배경과 위장 객체를 덜 헷갈리도록 source 학습과 hard negative/edge/quality 신호를 강화하는 것이다.
