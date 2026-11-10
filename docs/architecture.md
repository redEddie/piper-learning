# Piper Sim Lab 구조와 개발 경계

`piper_sim_lab`은 Piper 강화학습을 반복 개발하기 위한 독립 Python 패키지다. RTX 3090에서 JAX로 여러 환경을 벡터화하고 MJX-Warp로 물리를 계산하며, 각 환경의 Piper 한 대가 같은 정책을 사용한다. 환경끼리 상태나 보상을 공유하는 협동 다중 에이전트 학습은 포함하지 않는다.

이 저장소는 단독 reach 실험용이다. MuJoCo Playground 사용은 이 실험의 선택이며, Dex-One2Many 재현 프로젝트의 시뮬레이터 변경을 승인하거나 원 논문의 재현을 완료했다는 뜻은 아니다. 기존 `one2many` 저장소와 별도로 개발한다.

## 패키지 경계

| 위치 / 구성 요소 | 담당 내용 |
| --- | --- |
| `src/piper_sim_lab/env.py` | `PiperReach` 환경, reset/step 및 기본 환경 설정 |
| `src/piper_sim_lab/train.py` | Brax PPO 설정 연결, 학습 기록과 체크포인트 |
| `src/piper_sim_lab/evaluate.py` | 저장된 정책과 zero/random 동작의 동일 초기 조건 비교 |
| `src/piper_sim_lab/benchmark.py` | PPO·렌더링을 제외한 벡터 물리 실행의 별도 계측 |
| `src/piper_sim_lab/wrappers.py` | 오류 flag 검사, upstream 래퍼 조합 및 episode 경계 정보 보존 |
| `src/piper_sim_lab/assets/` | 출처가 고정된 Piper MJCF·메시와 파일 해시 목록 |
| `configs/` | 환경 수, 시간 간격, 보상, 네트워크, PPO 및 평가 설정 |
| `scripts/` | 설치·실행 편의를 위한 얇은 진입점; 환경과 학습 로직은 패키지에 둔다 |
| `tests/` | 모델 계약, reset/step, 설정 오류, 학습 연동의 의미 있는 검사 |
| `docs/` | 구조, 실행 증거와 변경 시 검토할 계약 |

환경은 `src/piper_sim_lab/env.py`의 `PiperReach`이며 MuJoCo Playground의 `MjxEnv`를 상속한다. `default_config()`, `reset(rng)`와 `step(state, action)`을 제공하고 상태·관측·보상·종료 정보와 지표를 반환한다. 벡터화와 학습 연결에는 Brax/Playground의 래퍼를 사용한다. 로컬 `wrappers.py`는 새 reset에서 사라지는 시간제한·episode 통계를 공식 preserve-info 훅으로 보존하며, 물리나 PPO를 재구현하지 않는다. PPO의 rollout, advantage 계산, clipped loss 및 optimizer는 외부 Brax 구현에 맡긴다. 이전 프로젝트의 자체 PPO 코드를 복사해 별도 학습기를 유지하지 않는다.

환경 코드는 PPO 설정이나 체크포인트 저장 경로를 알 필요가 없다. 학습 코드는 환경을 생성하고 외부 PPO를 구성하며 실행 기록을 남긴다. 평가는 저장된 정책과 같은 환경 계약을 사용하되 학습과 분리한 seed로 수행한다. 학습 래퍼의 `full_reset=True`는 episode마다 새 초기 상태와 목표를 샘플링한다. 처음부터 작은 저장소 여러 개로 분리하지 않고 이 패키지 안에서 모듈 경계를 유지한다.

`wrap_for_training`은 Vmap → `RuntimeSafety` → upstream Episode → 경계 정보 capture → upstream AutoReset → 경계 정보 restore 순서로 구성한다. `RuntimeSafety`는 벡터화된 reset/step의 `nonfinite`·`capacity_overflow`를 검사해 실패 분기에서만 host callback으로 예외를 발생시킨다. 정상 기본 경로에서는 GPU의 flag reduction과 조건 분기만 수행한다. 자동 reset 이전에 검사하므로 새 reset과 action repeat 안의 실행도 대상이다. `strict_physics=False`는 고의 실패를 주입하는 reset 테스트에만 사용한다.

## 첫 환경의 계약

| 항목 | 계약 |
| --- | --- |
| 작업 | 한 Piper의 말단 기준점을 지정한 3차원 목표점으로 이동하는 reach |
| 좌표 / 단위 | 목표와 말단 위치는 같은 시뮬레이션 world frame의 m; 팔 관절은 rad, 시간은 s |
| 말단 기준점 | 양쪽 distal finger pad 중심의 중점; `link6` local frame의 `z = 0.09003 m`에 둔 TCP site |
| 관측 | `obs['state']` 30차원: 팔 관절 위치 6, 속도 6, TCP 위치 3, 목표점−TCP 위치 3, 팔 제어 목표−실제 관절 위치 6, 이전 action 6 |
| 정책 동작 | 팔 6개 관절의 목표 위치 변화량; 정규화·변화량 배율·관절 한계 처리는 환경 설정과 구현에 명시 |
| 그리퍼 | 정책이 제어하지 않는다. `joint7` 목표를 `+0.025 m`로 유지하며 equality로 `joint8`은 반대 방향을 따른다. 두 손가락의 명목 개구는 50 mm다 |
| 모델 차원 | 8개 관절 위치/속도 좌표, 7개 액추에이터; 손가락 equality가 있으므로 8개의 독립 제어축이라는 의미는 아니다 |
| 충돌 | 원본의 팔 capsule 및 손가락 box 충돌 형상을 유지한다. 자세한 근사와 제외 조건은 [ASSETS.md](../ASSETS.md)에 기록 |
| 성공 | 말단 기준점과 목표점 사이 거리 `< 0.03 m`; 자세 정렬, 집기, 유지 시간 또는 실물 작업 성공을 뜻하지 않는다 |
| 시간 간격 | 기본 제어 간격 `0.05 s`, 물리 간격 `0.005 s`; 설정 변경 시 실행 기록에 남긴다 |
| 목표 분포 | 샘플링한 목표 관절 자세의 forward kinematics로 TCP 목표점을 생성한다 |
| 종료 / 제한 시간 | 성공은 metric이며 성공만으로 종료하지 않는다. 비유한 상태·용량 초과는 환경의 `done`이며 기본 학습 래퍼가 실행을 중단한다. episode 길이는 upstream wrapper가 처리한다 |

고정 그리퍼란 매 step 고정 목표를 주는 position control이다. 접촉이나 제약 오차가 있어도 실제 손가락 좌표가 항상 정확히 그 값이라는 보장은 아니다. 관측 벡터의 순서·크기, 말단 site 위치, 목표 분포, action scale, 제어 주기와 physics substep은 환경 구현과 실행 설정을 기준으로 추적한다. 이 중 하나를 바꾸면 기존 정책의 호환성을 다시 확인한다. FK로 만든 목표도 현재 관절·접촉 제약 아래에서 해당 episode에 도달 가능하다고 보장되지는 않는다.

각 벡터 환경은 별도 상태와 RNG를 갖는다. 초기 상태·목표 샘플링과 개별 episode reset을 검증해야 한다. 환경 수를 늘리는 것은 병렬 표본 수를 늘리는 것이며, 물리 공간에 여러 팔을 배치하는 작업과는 다르다.

## Warp 배치 용량

Warp의 접촉·CCD 공유 배열은 `vmap`만으로 환경 수에 맞게 자동 확장되지 않는다. `batch_capacity`는 실제 배치 환경 수로 지정한다. `train.py`는 학습과 내장 평가의 환경 수를 각각 주입하고, `benchmark.py`와 `evaluate.py`도 실행 배치를 반영한다. 직접 환경을 벡터화하는 호출자는 이 값을 명시해야 한다.

| 설정 | 기본값과 의미 |
| --- | --- |
| `batch_capacity` | 실행 배치의 환경 수 `N` |
| `contacts_per_env` | 160; 공유 `naconmax = 160 × N` |
| `max_constraints_per_env` | 512; 환경별 `njmax` |
| `ccd_pairs_per_env` | 4; 공유 `naccdmax = 4 × N` |

이 용량은 현재 Piper·장면의 형상과 제약에 대한 설정이다. 모든 미래 작업의 상한은 아니다. `capacity_overflow`는 reset의 forward와 한 제어 step의 모든 physics substep에서 발생한 Warp 용량 초과를 기록한다. solver/line-search의 반복 한도 도달은 이 용량 초과와 별도로 해석한다. JAX 물리 경로의 해당 metric은 0이며, Warp 용량이 검증됐다는 의미는 아니다. 유효한 GPU 실행 결과는 모든 환경·모든 시간에서 용량 초과와 비유한 상태가 없어야 한다.

## 세 사람의 작업 경계

| 담당 | 소유 범위 | 다른 담당에게 전달할 결과 |
| --- | --- | --- |
| 환경 / 보상 | Piper 모델 로딩, 초기화, action·observation 계약, 보상·종료, 목표 분포, 충돌 설정 | 환경 설정, seed가 고정된 rollout, reward 항목과 성공 지표 정의 |
| 훈련 / 평가 | Brax PPO 설정, 학습 실행, 체크포인트, 별도 seed 평가, 메모리·속도·성공률 기록 | 재실행 가능한 설정과 명령, 정책 산출물, 평가 결과 및 실패 사례 |
| 통합 / 입출력 | 패키징, 자산 출처, 설치 재현, 외부 입출력 어댑터와 메인 프로젝트 의존성 연결 | 설치 절차, 버전·해시 기록, 좌표·단위·정책 호환성 확인 |

보상이나 관측 변경은 환경 담당과 훈련 담당이 함께 검토한다. main 프로젝트 연결 전에는 환경 담당과 통합 담당이 좌표·단위·동작 의미를 함께 확인한다. 사람 이름은 팀이 정하며 위 경계를 역할 배정의 기준으로 사용한다.

## 의존성과 후속 통합

MuJoCo, `mujoco-mjx`, JAX, MuJoCo Playground, Brax의 버전은 `pyproject.toml`과 `uv.lock`으로 추적하고, 실행 결과에도 실제 설치 버전을 남긴다. 재현 설치는 lock 파일을 사용한다. Playground/Brax 내부 API에 의존하는 코드는 버전 갱신 때 import·환경 rollout·PPO 연동을 함께 검증한다. 공유 프로젝트 환경을 임의로 갱신하지 않고 이 저장소의 환경으로 설치한다.

Playground는 `d59156e099a8785dd58fbce2e25a6ee1f7b5800c`, Brax는 `6d039374b34a266107bd1e952180eea33ac4eab8`로 고정한다. 설치된 배포판의 버전 문자열에 더해 `direct_url.json`의 실제 소스 commit을 `provenance.sources`에 기록한다. 소스 파일 해시도 함께 남긴다. Brax PyPI 0.14.2의 삭제된 JAX API 사용 경로로 되돌리지 않는다.

컴파일 캐시는 기본 `.cache/jax`와 `.cache/warp`에 지속된다. 환경 변수 `JAX_COMPILATION_CACHE_DIR`, `WARP_CACHE_PATH`가 있으면 그 경로를 사용한다. 캐시는 학습 체크포인트와 다르며, 최초 호출 시간 비교에는 캐시 상태를 고려한다.

패키지 루트에서 개발 설치하는 표준 형태는 다음과 같다. 실행할 Python 환경을 먼저 활성화해야 한다. 이 명령 자체는 transitive dependency를 lock 버전으로 설치하지 않으므로 정확한 환경 재현에는 README의 lock 기반 설치 절차를 사용한다.

```bash
python -m pip install -e .
```

추후 메인 파이프라인에서는 같은 패키지를 editable 설치해 함께 개발하거나, 원격 저장소가 마련된 뒤 검증한 Git commit을 고정해 의존성으로 설치한다. 그때도 환경·학습 코드를 메인 저장소에 복사하지 않는다. 메인 저장소와의 연결은 후속 작업이며, 이 문서는 연결이 이미 구현됐다고 주장하지 않는다.

팀 공유 bare 저장소의 위치는 `/data/projects/piper-sim-lab/piper-sim-lab.git`이다. 실제 공유 상태·검증 revision은 검증 문서에서 별도로 기록하며, 경로 지정만으로 배포 완료로 간주하지 않는다.

현재 Brax 체크포인트는 policy, value, observation normalizer를 저장한다. `--warm-start`로 이를 가져올 수 있지만 optimizer 상태와 RNG는 새로 시작한다. 중단 지점에서 모든 학습 상태를 그대로 복원하는 기능은 지원하지 않는다. 체크포인트를 평가할 때는 해당 학습 출력의 `config.json`을 함께 사용한다.

`checkpoint.py`는 upstream이 저장한 선택적 `null` initializer를 보존해 설정을 읽고, 관측 shape·action 크기를 현재 환경과 대조한 뒤 upstream 네트워크 구성·파라미터 로더로 복원한다. 체크포인트 파일은 변경하지 않는다. 24차원 관측 정책을 30차원 환경에 넣는 경우는 거부한다. 크기가 같아도 관측 순서나 의미가 달라질 수 있으므로 환경 설정 호환성 검토는 별도로 필요하다.

## 검증과 결과 해석

검증은 아래 단계를 나누어 기록한다. 이 문서의 기준 자체는 통과 증거가 아니며, 수치는 실제 실행 기록에서만 가져온다.

1. **모델·환경 계약:** 자산 해시, 8개 관절 좌표와 7개 액추에이터, finger equality, 6차원 정책 동작, 관측 shape, 유한한 reset/step, 관절 한계 및 독립 RNG/reset을 확인한다.
2. **PPO 연동:** 작은 환경 수로 실제 rollout과 optimizer update를 실행하고, loss·관측·보상의 유한성과 정책 파라미터 변경을 확인한다. 물리 step만 통과한 것을 PPO 통과로 기록하지 않는다.
3. **3090 병렬 실행:** 환경 수와 rollout 길이·batch·network·Warp 용량을 명시하고 GPU 메모리, JIT/초기 준비 시간과 학습 실행 시간을 구분한다. 모든 physics substep의 Warp 용량 초과와 제어 step 경계의 비유한 상태를 확인한다. 실측 전에는 환경 수의 상한이나 처리량을 보장하지 않는다. 용량 수정 전 측정값을 현재 유효 성능으로 인용하지 않는다. 같은 프로세스의 순차 배치 측정과 독립 프로세스 측정도 구별한다.
4. **학습 결과:** 저장된 정책을 별도 seed와 목표에서 평가한다. 성공률·말단 거리·episode return 및 대표 rollout을 같은 설정의 무학습 정책 또는 기준 정책과 비교한다. 처리량이나 파라미터 변경만으로 reach 학습 성공을 주장하지 않는다. Brax가 합산하는 `eval/episode_success`는 성공 step 수다. episode 성공률은 별도 평가의 `final_success_rate`와 `ever_success_rate`로 구별한다.
5. **재실행:** 코드 revision, 자산 revision/해시, 설치 버전, 전체 설정, seed, 명령, 체크포인트를 함께 남긴다. 추론용 정책 저장과 optimizer/RNG를 포함한 중단 후 학습 재개 지원은 별도 기능으로 표시한다.

reach 학습·평가의 실측 결과와 남은 한계는 [검증 기록](validation.md)에 정리한다. 물체 조작, perception 연결, 실물 제어와 논문 수준의 재현은 별도 계약·검증이 필요한 후속 범위다.

## 집어서 놓기 환경 계약 (`pick_place.py`, 2026-10-20)

`configs/pick_place.yaml`의 `environment.task: pick_place`로 선택한다(`tasks.py`). reach 환경은 바꾸지 않았다.

| 항목 | 계약 |
| --- | --- |
| 작업 | 테이블 위 블록(freejoint box)을 지름 15 cm 그릇(mocap body, 바닥 박스 1 + 기울어진 벽 박스 12) 안에 넣는다. 술어 사슬 pre_grasp → grasp → pre_inside → inside |
| 물체 | `block_half`·`block_mass`·`block_rgba` 설정값. `configs/pick_place.yaml`: 5 cm 초록 PLA 큐브 약 80 g(팀이 준비한 실물, SHOW-20). `configs/pick_place_block30.yaml`: 3 cm 빨강 20 g(파이프라인 데이 물체, pp-full-01/02). 블록 폭 + 4 mm는 그리퍼 개방폭 70 mm보다 작아야 하고 `block_half < bowl_r_bottom`이어야 한다(생성 시 검사) |
| 배치 | 블록 x 0.25–0.40, y −0.15–−0.02, yaw 랜덤; 그릇 x 0.25–0.35, y 0.12–0.20 (좌우 분리, 겹치지 않음) |
| 말단 기준점 | `piper_grasp` site, `link6` local z = 0.12 m (손끝판 중심). reach의 `piper_tcp`(z 0.09, 중간 패드)와 다르다 |
| 행동 (7) | 베이스 좌표계 말단 선속도 3·각속도 3 (정규화 [-1,1] × 0.10 m/s, 1.0 rad/s) + 그리퍼 목표 1 ([-1,1] → 손가락 joint7 목표 0–35 mm, 즉 개방폭 0–70 mm; 스텝당 10 mm 속도 제한) |
| IK | `mjx.jac`(site, link6) → damped least squares λ 0.05 → 관절 속도 clip ±1.5 rad/s → 서보 목표 누적, 측정 관절 ±0.15 rad 안으로 제한, 관절 한계 clip |
| 관측 `state` (51) | 관절 6, 관절 속도 6, 서보 목표−관절 6, TCP 위치 3, TCP 회전벡터 3, 그리퍼 폭 1, 블록 위치 3, 블록 회전벡터 3, 그릇 위치 3, 블록−TCP 3, 블록−그릇 3, 단계 one-hot 4, 이전 행동 7. 논문의 actor 관측은 노드·엣지 특징뿐이고 단계 입력이 없다. `stage_obs: false`로 빼면 47차원이 되는데, v0 리셋·보상에서는 같은 예산(16.7M)에서 놓기를 못 배웠다(pp-full-05: 0% vs 단계 포함 pp-full-04: 82.8%). pre_inside 리셋과 placed/release 보상을 넣은 뒤에는 단계 입력 없이도 93.0%(pp-full-08, `configs/pick_place_nostage.yaml`) — 단계 입력은 선택 사항이 됐다. 실기에서는 단계를 같은 술어 코드로 FP++ 물체 자세·SDK 그리퍼 폭·FK TCP에서 계산해 넣는다(숨은 입력이 아님). 문턱 근처 깜빡임은 관측 잡음 랜덤화·히스테리시스로 다룰 1주차 항목 |
| 관측 `privileged` (58) | `state` + 블록 선속도 3 + 술어 4 (critic 전용, `value_obs_key: privileged`) |
| 보상 v0 | (1−tanh 5·d(TCP,블록)) + grasp 1 + grasp·(1−tanh 5·d_xy(블록,그릇)) + `inside_reward` 5·placed(블록이 그릇 안, 쥐고 있어도) + `release_bonus` 2·inside(놓고 TCP가 2 cm 이상 떨어짐; 성공 판정) − 행동 비용. 2026-10-21 전에는 placed 항 없이 inside 5만 있었다(pp-full-04~07). 단계는 매 스텝 술어로 다시 계산한다(정책 하나가 전체 순서를 수행; 단계별 정책을 잇지 않는다) |
| 단계별 리셋 | `reset_stage_probs` [0.4, 0.2, 0.2, 0.2]: 빈손 / 블록 3 cm 위 pre_grasp / 블록을 쥔 채 8 cm 들어 올린 grasp / 블록을 쥔 채 그릇 위(테두리 + 3 cm) pre_inside 상태에서 시작(논문처럼 모든 단계에서 리셋; 2026-10-21 추가). 생성 시 MuJoCo IK로 푼 top-down 관절 표(블록 영역·그릇 영역)에서 샘플링. 3개짜리 목록도 받는다(pre_inside 0) |
| 그리퍼 서보 | 결정(2026-10-21): 그리퍼는 열림/닫힘 두 상태이고, 닫히면 작업별 힘 상한 `gripper_force`(N, 손가락당; 기본 3 N)로 물체를 그냥 쥔다. 시뮬에서는 액추에이터 forcerange = ±gripper_force, kp 400(몇 mm 오차에서 포화)이라 물체 크기와 무관하게 쥐는 힘 = 상한. `gripper_binary: true`면 행동 7번째의 부호만 쓴다(>0 열림, ≤0 닫힘); 목표는 여전히 스텝당 10 mm만 이동. 실기는 `GripperCtrl(angle 0 또는 70000, effort=힘 상한에 해당하는 값)`로 대응하고 effort↔손끝 힘 변환은 저울로 실측(SHOW-63). 이전 값(kp 100, 연속 폭)으로 학습한 run은 pp-full-04/06까지. 80 g PLA 큐브: 마찰 0.5 가정 시 손가락당 0.8 N 필요 → 3 N은 4배 여유. 실기 piper_sdk `GripperCtrl(gripper_angle[0.001 mm], gripper_effort[0.001 N·m, 0–5000])`는 위치 + 토크 상한 방식이므로 실기에서 폭·힘·닫힘 속도를 재서(SHOW-63) kp·힘 상한·`gripper_rate`를 맞추고 ×0.8–1.25로 랜덤화한다. piper_sdk 확인(2026-10-21): 임피던스(MIT) 모드 `JointMitCtrl(motor_num 1–6, pos_ref, vel_ref, kp 0–500, kd −5–5, t_ref −8–8 N·m)`는 팔 관절 1–6 전용이고 그리퍼에는 토크·전류 모드가 없다. 따라서 실기 그리퍼의 강성은 바꿀 수 없고 effort(0–5 N·m) 값만 조절 가능하다. 시뮬의 kp는 그리퍼 내부 서보 강성을 흉내내는 값이며, 실측 전까지는 미확정 |
| 충돌 비트 | 그릇 2/2, 블록·손가락 3/3, 나머지 1/1: 그릇은 블록·손가락과만 충돌하고 팔·테이블과는 충돌하지 않는다(v0 제한, 접촉 쌍 감소). `bowl_arm_collision: true`(2026-10-21, `configs/pick_place_dr.yaml`)면 그릇 conaffinity 3으로 팔 캡슐(link1–6)과도 충돌하고, 그릇–바닥·그릇–베이스 쌍은 exclude로 뺀다. 1024 env에서 팔을 그릇으로 밀어 넣는 스트레스 시험: 최대 총 접촉 11,499개(예산 393,216), 넘침·발산 0. 기본값 false는 옛 run 설정 재현용 |
| 도메인 랜덤화 | 논문 Table 10 범위(2026-10-21). 기본값은 모두 꺼짐, `configs/pick_place_dr.yaml`에서 켠다. **관측·행동**(`obs_randomization`, env.step 안): 블록 위치 가우시안 σ 1 cm(축당 상한 3 cm, 매 스텝), 블록 회전 무작위 축 ±5°(매 스텝), 그릇 위치 σ 0.5 cm(상한 1.5 cm, 에피소드당 한 번), 인지 크기 ×(1±0.1)(에피소드당), 블록 자세 지연 0–150 ms·로봇 상태 지연 0–50 ms(에피소드당 뽑고 제어 스텝 사이 보간), 행동 지연 0–50 ms(앞 0–10 서브스텝 동안 이전 명령 유지). 논문의 크기 항목은 물리 크기가 아니라 **관측된 크기**라서 `size_obs: true`가 관측에 블록 반폭 1차원을 더한다(state 48, privileged 65: critic은 참 블록 위치·회전·그릇 위치·반폭을 따로 받는다). 보상·술어·단계는 참 상태로 계산한다. **물리**(`physics_randomization`, Playground `BraxDomainRandomizationVmapWrapper`, env 슬롯마다 한 번 뽑아 학습 내내 유지): 블록 질량 64–96 g(관성 비례), 블록 마찰 ×(1±0.1), 테이블 마찰 0.3–0.6, 그리퍼 강성·감쇠 ×0.8–1.25. Warp에서 슬롯별 값이 실제로 반영되는지 극단값으로 확인(2 kg·마찰 0.05 블록은 떨어지고 게인 0이면 손가락이 풀림). 블록 geom priority 1로 블록 접촉은 블록 마찰을 쓴다(MuJoCo 기본은 둘 중 큰 값). 빠진 항목: 테이블 질량(테이블이 정적 평면), 동마찰(MuJoCo는 마찰 계수 하나) |

| 시간 | 제어 0.1 s, 물리 0.005 s (substep 20), episode 80 스텝 |
| Warp 용량 | env당 접촉 384, 제약 1024, CCD 쌍 16 (`contacts_per_env`, `max_constraints_per_env`, `ccd_pairs_per_env`). 박스–박스 쌍(블록–그릇 벽·바닥, 블록–손가락 패드)은 CCD 경로를 쓰므로 블록이 그릇 안에 앉으면 쌍이 급증한다. CCD 쌍 4로는 5 cm 블록의 deterministic eval(모든 env가 같은 스텝에 inside 도달)에서 넘쳐 pp-full-03이 중단됐다(`docs/evidence/pick-place/pp-full-03-block50-ccd-overflow.txt`). `runtime.configure`는 GPU에서도 `JAX_PLATFORMS=cuda,cpu`로 두어 RuntimeSafety 콜백이 실제 원인을 보고하게 한다. 용량 의심 시 `scripts/diag_capacity_rollout.py` |
| 접촉 solver | SHOW-62: `solver_iterations=8`, `solver_ls_iterations=20`. 기존 line search 한도 4에서는 수렴 전 종료로 C 접촉 폭발·MJX-JAX 큰 진동이 발생했다. 주 반복만 늘려 해결되지 않았다. 20은 검증한 조건의 통과값이며 새로운 접촉 조건에서는 넉넉한 예산(예: 50)과 비교한다 |
| 검증 | SHOW-62 변경 후 5 cm·80 g 블록, 빈손 시작 scripted grasp: C 8/8, Warp 8/8, CPU MJX-JAX 1/1. C의 변경 전 같은 시험은 0/8. 정책 성공률과 구분한다. `tests/test_contact_solver.py`는 두 블록 규격의 C 유지 회귀 검사, `scripts/validate_scripted_grasp.py`는 전체 집기, `scripts/benchmark_line_search.py`는 컴파일 제외 성능 비교. 근거: `docs/evidence/contact-solver/` |
| 학습 결과 | 1024 env, 16.7M 스텝, 단계별 리셋. 3 cm 블록(pp-full-02): 빈손 시작 성공률 97.7%. 5 cm 80 g 블록(pp-full-04): 빈손 82.8%, 혼합 88.3%, zero/random 0% (128 env, `docs/evidence/pick-place/`). 단계 one-hot 제거(pp-full-05-block50-nostage): 보상 189, 성공 0%. SHOW-62 line search 20 재학습(pp-full-06-block50-ls20, 연속 폭 그리퍼 kp 100): 빈손 82.0% — 패치가 학습을 바꾸지 않음. 힘 제한 2상태 그리퍼(pp-full-07-block50-forcecap, 3 N, kp 400, ls 20): 빈손 **96.9%**(ever 97.7%) — 연속 폭 그리퍼보다 잡기가 안정적. **단계 입력 없음**(pp-full-08-nostage-preinside: `stage_obs false` + pre_inside 리셋 + placed 5/release 2 보상): 빈손 **93.0%**(ever 96.9%), 혼합 93.8% — pp-full-05의 0%에서 회복. 혼합 시작의 random 13%는 pre_inside 시작에서 무작위로 손이 열려 떨어지는 경우라 빈손 수치를 기준으로 본다. 보드 전체 GPU 메모리는 학습 중 1.4–2 GB(run 01의 19.5 GB는 시작 10초 스파이크). **도메인 랜덤화 + 그릇–팔 충돌**(pp-full-09-dr, `configs/pick_place_dr.yaml`, 기본 정책 = 단계 없음): 랜덤화 켠 환경 빈손 78.9%(seed 3개 82.0/78.9/75.8), 끈 환경 83.1%. 같은 랜덤화 환경에서 pp-full-08은 0%(원인: 블록 위치 잡음 1 cm; 물리·지연만으로는 93%). 이 정책은 놓은 뒤 손목(link6)을 그릇 테두리에 걸치는 접촉이 에피소드 66–80%에서 생긴다(깊이 ≤ 2 mm, 그릇이 mocap이라 비용 없음). `docs/evidence/pick-place/pp-full-09-dr/RESULTS.md`. **학습량 4배**(pp-full-10-dr-4x, `configs/pick_place_dr_long.yaml`, 67.1M, 43분, 한 실행의 중간 체크포인트): 랜덤화 켠 환경 1배 86.2% / 2배 93.8% / 4배 95.3%, 끈 환경 91.4 / 96.6 / 90.1%. 학습 중 평가 곡선은 2배 이후 평평. 같은 설정 pp-full-09의 1배 78.9%와 차이는 실행 간 편차라서 학습 seed 여러 개가 필요하다. 이 실행은 1·2·4배 모두 그릇–팔 접촉 0(손목 걸치기는 pp-full-09만의 버릇). `docs/evidence/pick-place/pp-full-10-dr-4x/RESULTS.md` |

### 새 물체·새 환경을 만들 때 (2026-10-20)

같은 작업에서 물체만 바뀌면 클래스를 만들지 않는다. YAML의 `environment:`에 `block_half`, `block_mass`, `block_rgba`만 적는다(`pick_place.yaml` ↔ `pick_place_block30.yaml`이 그 예). 술어 문턱(closed 폭, inside 반지름), 리셋 표의 높이, 관측의 블록 위치는 모두 `block_half`에서 계산되므로 따로 손대지 않는다. 정책은 물체별로 다시 학습한다(1024 env 기준 약 10분).

말단 기준점(TCP)은 로봇의 속성이지 물체의 속성이 아니다. `piper_grasp`는 손끝 패드(distal pad) 중심인 `link6` z = 0.12 m이고, 패드는 z 0.075–0.135 m(근위 패드 3 cm + 원위 패드 3 cm), 손바닥은 z ≈ 0.07 m에 있다. 5 cm 블록의 중심을 TCP에 두면 블록은 z 0.095–0.145 m을 차지해 손바닥과 닿지 않는다. 더 큰 물체(한 변 > 6.6 cm)는 그리퍼에 들어가지 않으므로 생성 시 오류가 난다.

새 작업(다른 술어 사슬, 다른 물체 종류)은 다음 순서로 만든다.

1. `pick_place.py`를 복사해 `_add_block`/`_add_bowl`(물체), `_predicates`·`_stage`(술어 사슬), `_reward`, `reset`의 단계별 시작 상태를 바꾼다. 관측 크기 상수와 `PROGRESS_METRIC`·`SUCCESS_DEFINITION`을 맞춘다.
2. `tasks.py`의 `make_env`에 이름을 등록하고 `configs/<task>.yaml`에 `environment.task`를 적는다.
3. `tests/test_<task>.py`로 CPU(`impl: jax`, `bowl_segments` 축소)에서 모양·유한성·행동 매핑을 검사하고, 집기처럼 접촉이 핵심인 동작은 `scripts/scripted_grasp_warp.py`처럼 Warp(GPU)에서 스크립트로 먼저 확인한다.
4. `configs/<task>_smoke.yaml`(256 env, 2M 스텝)로 보상이 오르는지 본 뒤 전체 학습을 돌리고, `evaluate.py`로 성공률·영상을 남긴다.
