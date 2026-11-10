# 강화학습 모델 보존과 관측 랜덤화의 한계

2026-11-09 사용자 결정에 따라 완료된 3090 실험을 정리하고 Transformer와 MLP 계열을 함께 보존한다.
추가 학습 없이 기존 가중치와 각 가중치의 normalizer를 보관한다.
A 복귀는 현재 정책 선정의 필수 조건에서 제외한다.

## 완료된 비교 결과

성공은 ① `initial_a`에서 시작해 ④ 정상 개방·그릇 안 안정 안착을 달성한 이력이다.
실패 이전의 `metric_stage_place_complete`를 집계하고, 10cm 이상 낙하의 성공 종료 예외는 제외한다.
각 행은 384회, 최대20초 평가다. 직접 초기화한 각 단계의 성공률과 구분한다.

| 보존 모델 | 관측 랜덤화 | 큐브·그릇 스폰 | ①→④ 정상 놓기 |
| --- | --- | --- | ---: |
| 원본 Transformer+LSTM 16배 | 꺼짐 | x250~350mm, y±200mm | 380/384 · 99.0% |
| 원본 MLP+LSTM 16배 | 꺼짐 | x250~350mm, y±200mm | 372/384 · 96.9% |
| 시간 Transformer KV64 18배 | 켜짐 | x100~500mm, y±300mm | 353/384 · 91.9% |
| MLP+LSTM 24배 | 켜짐 | x100~500mm, y±300mm | 351/384 · 91.4% |
| 실기 사용 MLP+LSTM 14+2배 | 켜짐 | x100~500mm, y±300mm | 294/384 · 76.6% |

물리 랜덤화는 모든 행에서 켜져 있다. 처음 두 행은 기존 조건의 참고 평가이며,
나머지는 관측 랜덤화·확장 스폰 조건의 평가다. Transformer+LSTM과 시간 Transformer KV64는 서로 다른 구조다.
현재 재평가와 보관 평가의 결과는 [선정 비교표](https://github.com/redEddie/piper-sim-lab/blob/a6de40735528e7ae9550ba6730fed0d24dbc1833/docs/show115/placement-selection/ranking.md)에 연결한다.

## 한계로 남기는 내용

- 관측 랜덤화를 포함한 확장 조건에서 정상 놓기 성공률이 낮게 관측됐다. 해당 조건의 최고 재평가91.9%로 목표98%에 미달했다.
- 관측 잡음·지연·크기 입력과 스폰 범위·평가 bank가 함께 달라졌다. 관측 랜덤화 하나의 인과 효과를 분리한 실험은 수행하지 않았다. 두 Transformer의 구조와 학습 이력도 다르다.
- 새 크기 관측 분포에 대한 normalizer 적응도 함께 발생했다. 원본 normalizer와 새 가중치를 혼합하지 않는다.
- 학습 seed는 구조별 한 번이며 고정 평가 bank의 점추정이다. 같은 체크포인트의 재평가에서도 건수 차이가 있었다. MLP24는 보관349회에서 재평가351회, 시간 Transformer12는352회에서346회였다.
- 지연 모델은 에피소드별 고정 지연과 보간이다. 실제 프레임 홀드·유실·가변 지연·장기 가림의 전체 동작은 포함하지 않는다.
- 자세 A 복귀는 보상·critic·제어 조건을 바꿔 실험했으나 모든 조건에서 일관된 완료를 얻지 못했다. 관련 이력은 SHOW-111에 보존한다.
- 현재 선정 지표는 ④까지다. 보존 가중치에는 과거 A 목표의 학습 이력이 남아 있다. ④ 완료 후 실행기의 종료·제어 전환 연결은 별도 작업이다.

이 결론은 완료된 3090 실험 범위다. SHOW-115에서 별도로 진행 중인 Thor 학습분 재평가는 그 작업의 기록과 상태를 유지한다.

## 저장 자료와 파일 구조

[통합 모델 보존 Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-model-archive-20261109)에
위 다섯 모델의 독립 아카이브, 파일별 SHA256, 평가 집계와 실기 출처를 보존한다.
[모델 목록](model-archive-20261109.json)에 원본 학습 commit, 체크포인트·아카이브 해시와 기존 Release를 기록한다.
기존 Release와 원본 체크포인트도 유지한다.

```bash
gh release download show108-model-archive-20261109 \
  --repo redEddie/piper-sim-lab --dir downloads/model-archive
(cd downloads/model-archive && sha256sum -c SHA256SUMS)
mkdir -p runs
tar -xzf downloads/model-archive/original-transformer-lstm-16u.tar.gz -C runs
```

```text
original-transformer-lstm-16u/
├── training.yaml
└── training/
    ├── config.json
    ├── provenance.json / parameters.json / reward-settings.json
    ├── reset-bank.json / initial-resets.json / summary.json
    └── checkpoints/000134217728/
        ├── learner.msgpack
        └── metadata.json
```

각 모델의 `learner.msgpack`에는 정책·가치망·normalizer·optimizer·RNG·step을 포함한다.
config와 checkpoints의 상대 구조를 유지한다.
JAX `piper_sim_lab.recurrent.load_actor_checkpoint`로 읽고 각 스텝의 LSTM 또는 K/V 상태를 이어받는다.
에피소드 시작 시 상태를 초기화한다. 원래 학습을 재현할 때는 목록에 기록한 각 학습 commit을 사용한다.

## Thor에서 실제 사용한 체크포인트

2026-11-01 실행 기록과 영상으로 아래 모델을 특정했다.

| 항목 | 기록 |
| --- | --- |
| 모델 | MLP+LSTM, 원본14배에서 관측 DR·확장 스폰으로 추가2배 |
| 원본 run | `show108-pipeline-dr-wide-from14u-plus2u-20261101` |
| 체크포인트 | `training/checkpoints/000016777216/learner.msgpack` |
| 누적 학습량 | 134,217,728전이, 16배 |
| 배포 policy ID | `ee9da8b8f493eaa56c3c` |
| learner SHA256 | `19438a198e31345d04e4234073028c1e0d9f20b48b90f9cb917af48b55f09c2c` |
| 배포 weights SHA256 | `91d1fa9ab188c66fb9e195164330d8c4f16155287cba8828f362c956849938b4` |

[실기 기록](https://github.com/redEddie/showonce-studio/blob/main/docs/REAL-ROBOT-DEMO-20261101.md)과
[실행 요약](real-demo-reference-20261101.json)의 checkpoint SHA256이 로컬 원본 learner와 일치했다.
실행 완료·98회 추론·10초 정책 실행과 파지·운반·개방·그릇 안 배치의 영상 확인 기록이 있다.
단일 시도이며 실기 성공률이나 완전한 A 복귀를 검증한 것은 아니다.
이 식별은 보관된11월1일 실행에 해당한다. 현재 Thor의 선택 정책이나 다른 날의 모든 실행까지 확인한 것은 아니다.

[원본 체크포인트 Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-obsdr-wide-14plus2u-20261101)와
[실기 영상·실행 기록 Release](https://github.com/redEddie/showonce-studio/releases/tag/show108-obsdr-wide-real-demo-20261101)를 연결한다.
