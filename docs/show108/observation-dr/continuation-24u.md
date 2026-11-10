# 관측 DR MLP+LSTM20→24배 연장

2026-11-09 사용자의 추가 학습 지시에 따라 에이전트가 직전과 같은4배를 첫 구간으로 정했다.
누적20배에서2/4배를 더 학습한 누적22/24배를 저장·평가했다.
학습·평가·영상 생성을완료했다. [전체 평가표](continuation-24u-results.md)와[다운로드](download-24u.md)를 제공한다.
브랜치는 `codex/show108-obsdr-joint-a-24u`다.

| 항목 | 실행 조건 |
| --- | --- |
| 부모 | 원본14배 + 관측 DR2배 +4배,누적20배 |
| 추가 학습 | 33,554,432전이,누적24배201,326,592전이 |
| Actor·critic | MLP256→64→LSTM128 actor49,MLP256×256 critic66 |
| 목표 | v5 관절 A,각 관절 오차≤0.2rad와기존200ms 유지 조건 |
| 환경·초기화 | 4096환경·다섯 초기화 각20%·에피소드200스텝 |
| PPO | 구간16,미니배치256×16,4회 업데이트,학습률0.0003 |
| 랜덤화 | 부모와같은관측·물리 DR와x100~500mm/y±300mm 스폰 |
| 복원 | 정책·가치망·normalizer·optimizer·RNG 전체 |

부모 learner의 SHA256을 [20배 Release 명세](continuation-20u-release-manifest.json)와 대조했다.
기존 plus4 YAML을 재사용한다. 새 시뮬레이터 에피소드와LSTM carry로 시작한다.

## 실행

[부모20배 Release](download-20u.md)를 다운로드·압축 해제한 구조를 사용한다.

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/run_stage_campaign.py \
  --config configs/show108/stage-reward-v5-mlp-lstm-observation-dr-wide-4096-plus4.yaml \
  --warm-start runs/show108-mlp-lstm-obsdr-wide-20u/training/checkpoints/000033554432 \
  --parent-units 20 --units 2 4 \
  --output runs/show108-obsdr-joint-a-20plus4u-20261109
```

자식 체크포인트 `000016777216`은누적22배,`000033554432`는누적24배다.
동일독립평가bank·seed2026/2027/2028,초기화별384회로부모20배와비교한다.
실제 A 전체 완료는 높은낙하 종료 예외를 제외한다. 완료 달성 이력과 종료 관절 오차를 함께 기록한다.
기존 관측·좁은 스폰에서도 자식들을 평가하고 최종·후보의1~5 첫 사례 영상을 생성한다.

## 결과 집계

집계기는 부모 누적량·부모 run의checkpoint 배수·기존조건 평가 경로를 명시적으로 받는다.
기존16→20배 데이터를 재집계해 JSON 전체와CSV 바이트 일치를 확인했다.

```bash
PYTHONPATH=src:runtime/src .venv/bin/python scripts/show108/summarize_observation_dr_continuation.py \
  --run runs/show108-obsdr-joint-a-20plus4u-20261109 \
  --parent-run runs/show108-obsdr-joint-a-16plus4u-20261108 \
  --parent-units 20 --parent-checkpoint-units 4 \
  --original-parent-run runs/show108-obsdr-joint-a-16plus4u-20261108/original-conditions \
  --output docs/show108/observation-dr
```

[이전20배 결과](continuation-20u-results.md) · Plane SHOW-114

## 완료 기록

추가33,554,432전이·51분9초,정상상태11,050전이/초다.
학습비유한·용량초과0,제어실패56전이는종료·리셋했다.GPU보드표본메모리최대9,639MiB·중앙9,428MiB다.
새조건·기존조건의자식2체크포인트×초기화5개×384회=7,680회평가를완료했다.
평가비유한·용량초과0,제어실패1회였다.부모20배의같은조건저장평가를재사용하고bank·case ID를대조했다.
아카이브재추출파일해시와최종5영상각200프레임·20초를확인했다.
