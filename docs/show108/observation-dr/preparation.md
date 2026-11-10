# 관측 오차·지연 및 넓은 스폰 범위: 다음 학습 준비

상태: 구현·물리 검증 및14배에서 추가2배 학습·평가·5영상까지 완료했다. [결과](results.md)를 먼저 확인한다. 실행 상태는 `runs/show108-pipeline-dr-wide-from14u-plus2u-20261101/campaign.json`을 따른다. 기존 SHOW-108 선정 MLP+LSTM16배와 Release는 보존한다.
작업 브랜치: `codex/show108-observation-dr-wide-spawn`.

## 사용자 제공 실측과 적용 범위

사용자가 최근7회 실행의 초기2초 정지 구간에서 확인한 값:

- 큐브 입력 XY 표준편차 약0.04~0.50mm. 반복 변동이며 절대 위치 정확도의 측정값은 아니다.
- 큐브 마지막 영상 관측 나이: 중앙71ms,95백분위115ms.
- 그릇 마지막 영상 관측 나이: 중앙141ms,95백분위272ms.

원본 실행 로그를 이 작업에서 다시 분석한 값은 아니다. 이번 실패의 원인을 관측 랜덤화 부족 하나로 확정하지 않는다.

| 항목 | 적용한 초기 실험값 | 근거·제한 |
| --- | --- | --- |
| 큐브 위치 오차 | 매 스텝 축별 Gaussian σ3mm, ±9mm에서 clip | 사용자 제안2~3mm 중 상단. 실측 반복 변동보다 보수적이며 Z 오차는 별도 실측되지 않았다. |
| 큐브 회전 오차 | 임의3D축, 각도 ±5° | yaw 전용이 아니다. |
| 그릇 위치 오차 | 에피소드별 Gaussian σ5mm, ±15mm에서 clip | 기존 그릇 전용 고정 오차 유지. 큐브·그릇 공통 카메라 보정 편향은 아니다. |
| 인식 큐브 크기 | 실제50mm의 ±5% | 실제 물체 크기는 고정하고 인식된 크기만 변경. |
| 큐브 관측 지연 | 에피소드별 균등0~200ms | 기존3개 시점 이력에서 보간. |
| 그릇 관측 지연 | 에피소드별 균등0~300ms | 새 독립4시점 위치 이력. 실제95백분위272ms를 포함하는 초기 시험 범위. |
| 로봇 상태 지연 | 에피소드별 균등0~50ms | 측정·식별된 실기 지연값은 아니다. |
| 행동 실행 지연 | 에피소드별 균등0~50ms | 그리퍼가 물리적으로 닫히는 시간과 구별. 팔과 그리퍼 모두 적용. |

새 큐브 고정 위치 편향은 추가하지 않았다. 지연 범위는 실측 분포를 재현하거나 최적화한 값이 아니라 초기 시험 범위다.
에피소드 내에서 일정한 지연과 시점 간 보간을 사용하므로 실제 프레임 홀드·유실·가변 지연·긴 가림을 그대로 재생하는 구현은 아니다.
actor49에는 지연·오차가 반영된 위치와 그 위치로부터 계산한 상대 벡터가 들어간다. Critic66의 실제 위치 필드와 보상·완료 판정은 물리 상태를 유지한다.

## 스폰 범위·물리 초기화

큐브·그릇 모두 로봇 베이스 XY 기준 **x=0.1~0.5m,y=−0.3~0.3m**.
기존400mm 반경 컷을600mm로 조정해 사각형 모서리(약583mm)를 자르지 않는다. 자유 큐브와 그릇의 중심 간 최소140mm는 유지한다.

- ① 빈손 접근 초기화는 요청 사각형 전체에서 독립 연속 위치를 샘플링한다.
- ②~⑤는 실제 IK·파지·해제 이력이 성립하는 결합 자세 bank를 사용한다. 넓은 사각형에서 후보를 생성하고 유효한 자세만 보존한다.
- 결합 자세의 추가 yaw도 객체가 사각형 밖으로 나가지 않는 후보만 쓴다. 잡은 큐브만 독립적으로 이동시키지 않는다.
- ②~⑤의 샘플은 현재 생성기의 자세·IK seed·접촉 검사에 통과한 부분집합이다. 사각형 전체가 모든 단계에서 도달 가능하거나 균등하게 덮였다는 뜻은 아니다.
- train/eval의 독립128개×4단계 native bank에서 각64개×5단계 bank를 만든다. ⑤는 실제 파지→하강→개방→20ms분리 이력을 기록한다.
- 기존 물리 DR·보상v5·완료 조건·제어기·4,096환경·초기화별20%는 유지한다.

## 이어 학습할 부모 후보

‘처음 50%를 넘는 성공’은 **①에서 시작해 A 복귀까지 실제로 완료한 비율**로 해석한다.
높은낙하 종료 예외를 실제 A 완료에 포함하지 않는다. 비교 가능한 저장 checkpoint 중 첫 후보는 **MLP+LSTM14배**다.

| 기존 학습량 | ① 시작 실제 A 전체 완료 | ⑤ A 복귀 달성 |
| --- | ---: | ---: |
| 12배 |111/384 (28.9%)|86/384 (22.4%)|
| **14배** |**240/384 (62.5%)**|**297/384 (77.3%)**|
| 16배 |289/384 (75.3%)|369/384 (96.1%)|

14배는5초기화 각각의 실제 A 전체 완료율도 모두50%를 넘는다. 원본 경로:
`runs/show108-matched-v5-mlp-vs-transformer-16u-20261101/mlp_lstm/training/checkpoints/000117440512`.
이 수치는 기존 좁은 위치 범위·관측 DR 없는 조건의 결과다. 새 조건의 성공률로 해석하지 않는다.
14배가16배보다 적응을 잘한다는 실험적 근거는 아직 없다.

현재 `configs/show108/stage-reward-v5-mlp-lstm-observation-dr-wide-4096-plus2.yaml`은 추가2배의 확정 설정이다.
사용자는14배 부모에서 추가2배를 확정했다. 아래 명령은 실행 전 계획 확인용이다. 실제 캠페인은 별도 고정 worktree에서 `--plan-only` 없이 실행한다.

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/run_stage_campaign.py \
  --config configs/show108/stage-reward-v5-mlp-lstm-observation-dr-wide-4096-plus2.yaml \
  --warm-start runs/show108-matched-v5-mlp-vs-transformer-16u-20261101/mlp_lstm/training/checkpoints/000117440512 \
  --parent-units 14 --units 2 --output runs/show108-pipeline-dr-wide-from14u-plus2u \
  --plan-only
```

## 재현·검증 기록

[protocol.json](protocol.json)에 설정·부모·bank 해시를, [validation.json](validation.json)에 단위 검사와 train/eval 각각4,096환경 물리 검증을 기록했다.
물리 검증은 고정된 명령의 연결·수치·초기화 검사이며 정책 성공률 평가나 본학습 결과가 아니다.

Bank 생성의 물리 설정 원본은 [bank-generation-config.yaml](bank-generation-config.yaml)에 보존했다.
관측 오차·지연값을 실측 제안으로 갱신하기 전에 생성했지만, 생성기는 관측 DR를 끄고 실제 물리만 사용하므로 두 설정의 물리·기구학은 같다.

```bash
scripts/run.sh .venv/bin/python scripts/show103/build_position_banks.py \
  --config docs/show108/observation-dr/bank-generation-config.yaml \
  --output-prefix show108-wide-position-random --count 128 \
  --train-seed 61108 --eval-seed 71108 --x-range .1 .5 --y-range -.3 .3 --max-radius .6
scripts/run.sh .venv/bin/python scripts/show103/build_five_start_banks.py \
  --config docs/show108/observation-dr/bank-generation-config.yaml \
  --source-prefix show108-wide-position-random --output-prefix show108-wide-five-start --count 64
```

이미 생성된 bank는 덮어쓰지 않고 별도 이름을 지정한다. 실제 로봇·SDK/CAN은 구동하지 않았다.
