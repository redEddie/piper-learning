# SHOW-103 — 현재 결과와 실행 안내

2026-11-01 기준: Transformer+마지막 LSTM, 진행형 보상 v5, 누적16배까지 학습·평가·영상 생성을 완료했습니다.
현재 작업 브랜치는 `codex/show103-mjx-3090`입니다. [README](../../README.md)의 결과·조건과
[14→16배 원자료](mjx-3090/progress-reward-v5-continuation-16u.json)를 기준으로 사용합니다.
이전 구현 전 인계서는 [이력](history/3090-handoff-before-20261101.txt)으로 보존했습니다.

## 확인한 결과

초기화별384회에서①~④ 단계 내100%,⑤20.6%(79/384),① 시작 전체 완료52.1%(200/384)입니다.
⑤ 최종 최대 관절 오차 중앙값은16.1°입니다. 성공 이력과 최종 관절 오차는 별개로 기록합니다.
4,096환경·초기화 각20%·위치 랜덤화·동역학DR·기존 v5 보상과 완료 조건을 유지합니다.

학습 코드 commit은 `2177b9ec1107585c23df830fe671cd2cedb487e2`입니다. 문서 정리로 보상·제어·모델·설정을 바꾸지 않습니다.
누적16배는134,217,728전이이며, 마지막 실행 폴더의 체크포인트 이름16,777,216은 이번 추가2배 양입니다.
현재16배에는 초기8배 v2,추가2배 v3,이후6배 v5의 이력이 있습니다. 처음부터 동일v5를16배 학습한 실행은 아닙니다.

## 설치와 체크포인트

설치·다운로드·평가 명령은 [README](../../README.md#설치평가-재현)를 따릅니다.
[Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show103-v5-16u-20261101)의
`show103-v5-16u-checkpoint.tar.gz`를 `runs/`에 풀면 아래 구조입니다.

```text
runs/show103-v5-16u/
  training.yaml
  training/
    config.json
    provenance.json
    reward-settings.json
    reset-bank.json
    continuation.json
    summary.json
    checkpoints/000016777216/
      learner.msgpack
      metadata.json
```

`SHA256SUMS`와 [명세](mjx-3090/release-16u-manifest.json)로 파일을 확인합니다.
`learner.msgpack`에는 normalizer·optimizer·RNG도 포함합니다. 부모14배와 원본 정책은 별도 보존했습니다.
`training/config.json`이 없거나 다른 구조의 모델인 경우 로더가 복원을 거부합니다.

## 추가 학습이 필요할 때

아래는 다음 추가2배(누적18배) 예시입니다. 이 정리 작업에서는 실행하지 않았습니다.

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/run_stage_campaign.py \
  --config configs/show103/stage-reward-v5-transformer-lstm-progress-plus2.yaml \
  --warm-start runs/show103-v5-16u/training/checkpoints/000016777216 \
  --parent-units 16 --units 2 \
  --output runs/show103-v5-from16u-plus2u
```

모델·가치망·정규화·optimizer·RNG를 복원하고 simulator episode/history는 새로 시작합니다.
현재 설정은4096환경·5초기화 각20%,팔0.25/그리퍼0.5 탐색 최소항입니다. config 파일의 오래된
부모10배 주석 대신 CLI의 `--warm-start`, `--parent-units`와 checkpoint metadata로 실제 부모를 확인합니다.
캠페인은 학습 후 각 초기화3seed×128회 평가, 비교표, 초기화별 첫 사례 영상5개를 생성합니다.

## 주요 파일

- `src/piper_sim_lab/recurrent_train.py`: recurrent PPO와 full learner 복원.
- `src/piper_sim_lab/transformer_lstm.py`: Transformer+마지막 LSTM.
- `src/piper_sim_lab/five_start_stage.py`, `stage_reward_env.py`: 실제 접촉·개방·안착과 초기화.
- `src/piper_sim_lab/stage_reward_v5.py`: 완료 점수+현재 진행−상황 비용.
- `scripts/show103/evaluate_stage_env.py`: 첫 에피소드의 단계 달성·전체 완료·A 오차 기록.
- `scripts/show103/render_stage_env.py`: 기록된 trace 재생. 표시한 scene을 사용하며 성공 선별 여부를 구분.
- `uv.lock`, `scripts/run.sh`, `validated-environment-16u.json`: 환경 재현.

기존 v1~v4 보상 파일은 v5의 실제 의존성입니다. 원본·레거시 정책을 임의로 삭제하지 않습니다.
실제 로봇은 구동하지 않습니다. Transformer의 portable NumPy export는 현재 지원 범위가 아닙니다.
후속 구조 비교는 SHOW-108에서 별도로 진행합니다.
