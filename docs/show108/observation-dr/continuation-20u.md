# 관측 랜덤화 MLP+LSTM 누적20배 연장

2026-11-08 사용자 지시로 원본14배에서 관측 DR·확장 스폰으로 추가2배 학습한 모델에서4배를 더 학습했다.
2026-11-09 학습·누적18/20배 평가·기존 조건 재평가·최종 첫 사례 영상5개를 완료했다.
결과는 [평가표](continuation-20u-results.md),체크포인트는 [다운로드 안내](download-20u.md)를 따른다.
브랜치는 `codex/show108-obsdr-joint-a-20u`다.

| 항목 | 이번 실행 |
| --- | --- |
| 부모 | 원본14배 + 관측 DR·확장 스폰2배,누적16배 |
| 추가 학습 | 33,554,432전이,4배 |
| 저장·평가 | 추가2/4배,누적18/20배 |
| Actor | 관측49 → SiLU256 → SiLU64 → LayerNorm → LSTM128 → 선형14,행동7 |
| Critic | privileged66 → MLP256×256 → 값1 |
| 보상 | stage_reward_v5,완료 단계 점수 누적과 현재 목표 진행 점수 |
| A 목표 | 관절 A,각 관절 오차≤0.2rad |
| 환경 | 4,096환경,5개 초기화 각20%,20초 |
| PPO | 16스텝,미니배치256×16,4회 업데이트,학습률0.0003 |
| 관측·물리 | 부모의 관측·물리 랜덤화와 넓은 스폰 |
| 복원 | 정책·가치망·normalizer·optimizer·RNG 전체 |

소스 기준은 기존 관측 DR 게시 브랜치의 `c3d76e52c4d697003adad155881202ed99a524fa`다.
실행 코드는 부모 학습 소스 `0f5aa042416f49c7a8839396179e230a0e853242`와 같다.
설정 변경은 추가 학습 예산과 저장 간격이다. 새 실행의 시뮬레이터 에피소드와 LSTM carry는 초기화한다.

부모는 [관측 DR Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-obsdr-wide-14plus2u-20261101)의
`training/checkpoints/000016777216`이다. 이 step은 이전 추가량이며 누적 전이는134,217,728이다.
이번 자식 step16,777,216과33,554,432는 누적150,994,944와167,772,160에 해당한다.

## 학습과 자동 평가

부모 경로는 Release를 `runs/`에 압축 해제한 구조에 맞춘다.

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/run_stage_campaign.py \
  --config configs/show108/stage-reward-v5-mlp-lstm-observation-dr-wide-4096-plus4.yaml \
  --warm-start runs/show108-mlp-lstm-obsdr-wide-14plus2u/training/checkpoints/000016777216 \
  --parent-units 16 --units 2 4 \
  --output runs/show108-obsdr-joint-a-16plus4u-20261108
```

학습 후 두 체크포인트를 같은 독립 평가 bank와seed2026/2027/2028에서 초기화별384회 평가한다.
단계 내 성공률과 시작점부터 실제 A 완료까지의 비율,⑤ 최대 관절 오차를 함께 집계한다.
원본14+2배의 기존 평가를 같은 조건의 추가0배 기준으로 사용하고 bank·case ID를 대조한다.
실패는 분모에 포함하며10cm 이상 그릇 내 낙하 종료 예외는 실제 A 완료에서 제외한다.
자동 렌더링은 최종과 단계 최소 성공률 기준 후보의1~5 첫 사례를 기록한다.
성공은 에피소드 중200ms 유지 달성 이력이며 마지막 오차는 별도 지표다.

원래 좁은 스폰·관측 DR 없는 조건에서도 자식을 재평가해 기존 조건에서의 수행을 확인한다.
자식 가중치·normalizer로 원본 평가 환경을 실행한다.

## 완료 기록

학습33,554,432전이·51분50초,정상상태11,028전이/초다.
학습 비유한·용량 초과0,제어 실패85전이는 종료·리셋 처리했다.
GPU 보드 메모리 표본 중앙9,494MiB·최대20,509MiB,util 중앙100%였다.
이번 자식의두체크포인트×두조건×5초기화×384회=7,680회 평가에서 비유한·용량 초과0,제어 실패1회였다.
기존 조건의 부모를 추가1,920회 평가했다.부모의새조건 평가는 기존 저장 기록을 사용했다.
아카이브를 재추출해 파일별 해시 일치를 확인했고5영상각200프레임·20초를 검증했다.
첫 시작 시도는 학습기 초기화 전에 종료돼 관리 세션에서 재실행했다.

[부모 평가](results.md) · [관측 설정](preparation.md) · [v5 보상 계약](../../show103/mjx-3090/progress-reward-v5.json)
관절 A와 후속 TCP 보상 계열의 관계는 [목표 변경 이력](../../show111/home-objective-history.md)에 정리한다.
