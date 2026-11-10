# SHOW-103: 3090 MJX 연결과 학습 준비 — 2026-10-30

최신 모델 연결은 [Flax Transformer → 마지막 LSTM 구조·검증](transformer-lstm.json)을 참고한다.
4096환경의 짧은 새 모델 학습과5초기화 평가 연결을 완료했다. 사용자가 이 구조의 처음부터2/4/6/8배 본학습을 지시했다.
실행 상태는 `runs/show103-transformer-lstm-4096-s7-2-4-6-8u-20261030/campaign.json`을 확인한다.

최신 합의는 [v2 다섯 초기화·4배 부모 모델에서 이어 학습](five-start-v2.json)이다.
현재 위치 정렬·최단거리 이탈 비용·낙하 즉시 리셋을 검증하고, 초기화 각20%로 추가2/4/6/8배를 비교한다.
아래 처음부터 학습·네 초기화 조건과 준비 상태는 이전 실험의 기록이다.

최신 사용자 실행 지시는 [위치 랜덤화 후 처음부터2/4/6/8배 학습](random-positions-v1.md)이다.
이전 [고정 위치 학습·비교](campaign-2-4-6-8.md)는 전체 완료0%로 끝났으며 보존한다.
아래 약839만 전이 단기안은 준비 단계의 기록이며 최신 예산은 누적67,108,864전이다.

새 `stage_reward_pick_place` task, 네 가지 고정 초기화, 4,096환경 GPU 물리 및 짧은 PPO 연결 검증을 완료했다.
**정식 8,388,608전이 학습과 난이도 확대는 아직 실행하지 않았다.**
SHOW-76/100 원본 코드·정책은 보존했고 실제 로봇·SDK/CAN 명령은 실행하지 않았다.

## 구현

- `stage_reward_env.py`가 기존 `stage_reward.py`의 순수 수식을 실제 측정값에 연결한다.
  정책 주기마다 `start_interval` 1회, 5ms 물리 이력 `advance_memory` 20회, `reward_terms` 1회다.
- 접촉은 world별 contact sensor에서 읽는다. Warp 공유 contact buffer를 world별 접촉으로 읽지 않는다.
- 자세를 반영한 큐브 최저점, 그릇 회전을 반영한 상대 좌표·속도, 실제 관절 A 오차·속도를 계산한다.
- SHOW-100의 actor49/critic66/action7, 최종 개폐 목표 관측, 열기/유지/닫기 및 개폐 속도 제한을 유지한다.
- 기존 GPU `checked_plan` DLS+QP를 사용한다. 팔/그리퍼 동시 동작을 허용하며 MOVE/BRAKE/GRIP 조정기를 넣지 않는다.
- 최종 학습 보상은 새 수식으로 대체한다. 이전 +5/+2 등은 합산하지 않는다.
- `python -m piper_sim_lab.train`에 실제 진입점을 추가했다.
  새 실행에는 reset bank·보상 계수·resolved config·코드 revision·uv.lock 해시를 기록한다.

## 초기 난이도

새 MLP256×256, seed7, **4,096환경**, 빈손 A / 열린 파지 직전 / 들고 있음 / 기존 그릇 위 비율 40/20/20/20%.
대표 배치·자세를 고정하고 동역학 DR만 적용한다. 초기 관측 DR와 가까운 놓기 초기화는 사용하지 않는다.

탁자 좌표계의 큐브 시작 XY는 [0.30, −0.12]m, 그릇은 [0.25, 0.12]m다.
명목 TCP 높이는 파지 직전 65mm, 들고 있음 100mm, 그릇 위 125mm다.
관절 한계를 유지하는 도달 가능 자세를 IK로 구하고 C MuJoCo에서 300ms 안정화했다.
실제 물체·관절 위치와 접촉은 [reset-bank-v1.json](reset-bank-v1.json)에 고정 기록했다.
각 MJX reset은 해당 world의 실제 DR 동역학으로 100ms 더 안정화한다.
보상 메모리는 이후 비어 있는 상태에서 시작하며 파지·운반·안착 타이머를 미리 채우지 않는다.

v=1rad/s, a=5rad/s², jerk=100rad/s³, TCP 배율0.1m/s·0.5rad/s,
TCP link6 z=138.5mm, 탁자 z=0·base z=15mm를 보존했다.

## 검증

[실제 GPU 물리 결과](physics-validation.json): 동역학 DR, 각 24정책 스텝(2.4초),
열기→닫기→유지와 작은 팔 이동을 함께 실행했다.

| 환경 수 | 결과 | 비유한/용량 초과/제어 오류 | 파지 reset 실패 | 보드 GPU 최대 MiB |
| --- | --- | --- | --- | --- |
| 128 | 통과 | 0/0/0 | 0 | 1197 |
| 1,024 | 통과 | 0/0/0 | 0 | 1777 |
| 4,096 | 통과 | 0/0/0 | 0 | 5049 |

20개 5ms 측정 샘플을 원래 순수 함수에 다시 넣어 보상·메모리가 일치함을 확인했다.
새 episode reset 후 해제·보너스 이력도 초기화됐다. GPU 메모리는 샘플링한 보드 전체 사용량이다.
고정 행동 검증은 학습된 과제 성공이나 모든 가능한 동작의 물리적 성립을 증명하지 않는다.
128/1,024 검사 뒤 기본 계수를 YAML에 명시했으며 물리·제어 값은 동일하다. 4,096 결과는 명시된 설정으로 실행했다.

[단위 검사](tests.txt): 보상 수식·좌표/속도/회전 형상·실접촉 초기화·그리퍼·제어 계획·export 검사 **52개 통과**.

[PPO 연결 검사](ppo-smoke-summary.json)는 같은 4,096환경에서 처음부터 131,072전이(약91초)를 실행했다.
정책 L2 변화 약0.987과 체크포인트 `000000131072` 저장·재로딩을 확인했다.
관측된 비유한/용량 초과/제어 오류는 0이다. 짧은 실행의 KL은 약1593으로 크며,
이 검사는 학습 수렴을 뜻하지 않는다. 본학습 첫 구간에서 정책 변화·가치 손실·진척을 함께 살핀다.
초기 상태 A와 held를 seed2026 각4회 평가한 전체 완료는 각각0/4였다.
원시 평가·영상은 `runs/show103-evaluator-smoke/`에 보존했다. 영상은 성공 선택 없이 첫 사례를 1배속으로 녹화했다.

## 첫 본학습과 평가

저장소 루트, 새 output 경로에서 실행한다. 처음에는 warm-start를 사용하지 않는다.

```bash
uv sync --locked --extra gpu --extra test --extra control --python 3.12
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python -m piper_sim_lab.train \
  --config configs/show103/stage-reward-v1-4096.yaml \
  --output runs/show103-stage-v1-fixed-4096-s7 --platform gpu
```

첫 예산은 **8,388,608전이**이며 4,194,304/8,388,608 체크포인트를 남긴다.
이후 약50,331,648전이까지의 연장 여부는 수치·물리 상태와 실제 진척을 보고 판단한다.
초기 전체 완료0% 하나만으로 실패를 단정하지 않는다.

```bash
scripts/run.sh .venv/bin/python scripts/show103/evaluate_stage_env.py \
  --run runs/show103-stage-v1-fixed-4096-s7 \
  --checkpoint runs/show103-stage-v1-fixed-4096-s7/checkpoints/000008388608 \
  --output runs/show103-stage-v1-fixed-evaluation
```

기본 평가 seed2026/2027/2028, 각128회, 네 초기화별로 평가한다.
최종 inside / 합격 낮은 해제 / 안착 / 철수 / A 복귀 전체 완료를 구분한다.
autoreset을 사용하지 않고 첫 실패를 누적해 성공 집계에서 제외한다.
raw action, 관측, 최종 목표·실제 폭·reference, TCP/큐브/그릇, 모든 보상 항과 5ms 측정 이력을 NPZ로 저장한다.

```bash
scripts/run.sh .venv/bin/python scripts/show103/render_stage_env.py \
  --run runs/show103-stage-v1-fixed-4096-s7 \
  --trace runs/show103-stage-v1-fixed-evaluation/initial_a-s2026.npz \
  --output runs/show103-stage-v1-fixed-evaluation/initial-a-first.mp4 --scene 0
```

held도 같은 방식으로 첫 사례를 녹화한다. 성공 사례를 따로 선택하면 scene 번호와 선택 기준을 기록한다.

## 난이도 승급

원래 조건과 초기화별 전체 완료율을 확인한 뒤 위치 범위 → 상대 방향·파지 깊이 → 파지한 관절 자세를
한 종류씩 확대한다. 좋은 체크포인트에서 policy/critic/normalizer를 이어받고 원래 조건 평가도 유지한다.
현재 warm-start는 optimizer/RNG를 재시작한다. 난이도 변경과 보상·QP 제약 변경을 한 실험으로 섞지 않는다.
수치 승급 임계값과 다음 실제 범위는 첫 결과를 보고 정한다. 지금 구현을 위해 추가 사용자 결정은 필요하지 않다.
