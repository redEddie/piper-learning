# 최종 배포 기준 — 관측 랜덤화 MLP+LSTM24배

2026-11-09 사용자 결정으로 누적24배 MLP+LSTM을 향후 실기 사용을 위한 최종 코드·모델 기준으로 선정했다.

- 코드 브랜치는 `codex/show108-mlp-lstm-24u-final`이다.
- [최종 Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-24u-final-20261109)에 전체 체크포인트와 NumPy 추론 패키지를 제공한다.
- [모델 명세](final-model.json)에 원본 학습 commit·체크포인트 해시·평가 조건·실기 검증 범위를 기록했다.
- Transformer와 원본 MLP 모델은 [모델 보존 Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-model-archive-20261109)에 보존했다.

## 선정 결과와 한계

| 같은 관측 DR·확장 스폰 평가 | ① 접근 → ④ 정상 놓기·안착 |
| --- | ---: |
| 실기 사용14+2배 MLP+LSTM | 294/384 · 76.6% |
| 최종24배 MLP+LSTM | 351/384 · 91.4% |

초기화별384회, seed2026/2027/2028, 최대20초 조건이다. 정상 놓기는 최초 실패 전에 달성한 이력으로 집계한다.
A 복귀와10cm 이상 낙하 성공 종료 예외를 제외한다. 24배의 보관 평가는349/384였고 최근 재평가는351/384였다.
학습 seed는1개이며98% 목표는 미달이다. 관측 랜덤화·확장 스폰 조건의 성능 한계는 [실험 정리](model-archive-20261109.md)에 남긴다.

24배 모델은 원본14배에서 관측 DR로 추가2배 학습한 정책을 이어 총24배까지 학습한 모델이다.
원본 learner SHA256은 `84e064bf50d34c30b346f08c109a387b0d4b1ca58fdeb30b93db638380d44306`이다.
정책·normalizer를 세트로 배포한다. 14+2배의 normalizer를24배 가중치와 섞지 않는다.
보상과 제어의 학습 원본은 `d377cae0afdc87a389b09071eae6bcde72465e86`이며 이 브랜치는 해당 코드에서 파생했다.

11월1일 실기 기록은14+2배 모델에 해당한다. 배포 policy ID는 `ee9da8b8f493eaa56c3c`다.
24배의 실기 작업 성공률은 미검증이다. [부모 모델의 실기 기록](real-demo-reference-20261101.json)을 함께 제공한다.

## 다운로드와 구조

```bash
git clone --branch codex/show108-mlp-lstm-24u-final https://github.com/redEddie/piper-sim-lab.git
cd piper-sim-lab
gh release download show108-mlp-lstm-24u-final-20261109 \
  --repo redEddie/piper-sim-lab --dir downloads/final24
(cd downloads/final24 && sha256sum -c SHA256SUMS)
mkdir -p runs
tar -xzf downloads/final24/show108-mlp-lstm-24u-final-checkpoint.tar.gz -C runs
tar -xzf downloads/final24/show108-mlp-lstm-24u-final-inference.tar.gz
```

```text
runs/show108-mlp-lstm-24u-final/
├── training.yaml
└── training/
    ├── config.json / provenance.json / continuation.json
    ├── reward-settings.json / reset-bank.json / initial-resets.json
    └── checkpoints/000033554432/
        ├── learner.msgpack
        └── metadata.json

show108-mlp-lstm-24u-final-inference/
├── policy/
│   ├── policy.json / weights.npz
│   ├── golden.npz / recorded-golden.npz
│   ├── piper.xml / training-config.json
│   └── export-validation.json
├── runtime/
│   ├── pyproject.toml
│   └── src/piper_policy_runtime/
├── final-model.json / final-release.md
└── validation.json
```

`000033554432`는 마지막 실행에서 추가 학습한4배다. 누적 학습량은201,326,592전이,24배다.
전체 learner는 정책·가치망·normalizer·optimizer·RNG를 포함한다.
학습 재개 시 상위 `training/config.json`과 checkpoints의 상대 구조를 유지한다.
NumPy 추론 패키지는 actor와 고정 normalizer를 사용한다.

## 로봇 연결 없는 추론 확인

NumPy가 설치된 Python3.10 이상 환경에서 다음 명령을 실행한다.

```bash
PYTHONPATH=show108-mlp-lstm-24u-final-inference/runtime/src \
  python3 -m piper_policy_runtime.check \
  --policy show108-mlp-lstm-24u-final-inference/policy
```

`golden.npz`는 원본 JAX 모델의 합성 관측·부분 reset·행동·logits·LSTM cell/hidden 기준값이다.
`recorded-golden.npz`는 다섯 초기화의 저장된 시뮬레이션 관측을 재생한 기준값이다.
이 검사는 파일·수식·메모리 전달을 확인하며 작업 성공률 평가는 위384회 평가에서 확인한다.

```python
from piper_policy_runtime import Policy

policy = Policy("show108-mlp-lstm-24u-final-inference/policy")
policy.reset()
action = policy.step(observation_49d)
# 다음 정책 주기에도 같은 policy 객체를 사용해 LSTM 상태를 이어받는다.
# 새 에피소드에서는 policy.reset()을 호출한다.
```

`Policy.step`은49차원 원시 관측을 받아7차원 정규화 행동을 반환한다. 정규화는 객체 내부에서 한 번 수행한다.

## 실기 연결 규약

기존14+2배의 공통 Runtime 제어 프로필 `show108-mlp-lstm-v1`을 사용한다.
[piper-runtime](https://github.com/redEddie/piper-runtime)의 검증 기준 commit은 `e28046050f12f7aa98e392baf089184a20d60827`이다.
학습 저장소의 NumPy Policy와 replay checker는 이 공통 Runtime의 구현을 재사용했다.
최종 패키지를 공통 Runtime의 `contract(metadata)`로 검사하고 같은 NumPy 추론 결과를 확인했다.

| 항목 | 규약 |
| --- | --- |
| Actor | 관측49 → SiLU256 → SiLU64 → LayerNorm → LSTM128 → 선형14 |
| 행동 | 베이스 좌표계 TCP 선속도3·각속도3·래치 그리퍼1, 정규화 범위−1~1 |
| TCP 명령 배율 | 선속도 최대0.1m/s, 각속도 최대0.5rad/s |
| 시간 | 정책10Hz, 제어200Hz |
| 제어 | DLS+checked-plan QP, 속도1rad/s·가속도5rad/s²·jerk100rad/s³ |
| TCP | piper_v100, 로컬z138.5mm, 베이스 높이15mm |
| 그리퍼 | 최종 OPEN/CLOSE 목표를 래치하고 정책 주기마다 폭 reference를 제한 |
| 학습 랜덤화 | 관측·물리 랜덤화 켜짐, 큐브·그릇 x100~500mm·y±300mm |
| 실기 배포 차이 | TCP 바닥 여유10mm, SDK 그리퍼 force0.5는 공통 Runtime의 기존 실기 설정 |

49차원 관측의 필드·좌표계·단위·오프셋은 `policy/policy.json`의 `observation_layout`을 따른다.
관절 reference−실측 위치, 직전 정책 raw action과 최종 그리퍼 목표를 각각 보존한다.
그리퍼 최종 목표와 속도 제한된 reference, 실측 폭은 별도의 상태다.
파지 후 물체 상태는 실기 상태 추정 경로에서 갱신한다. 부모 실기 실행은 `showonce_scene_state_v2`를 사용했다.

향후 연결 시 공통 Runtime으로 이 policy 디렉터리를 읽고 실제 관측·기구학·제어 상태를 같은 규약으로 구성한다.
Studio의 기존 특정 Release 내보내기 스크립트는 원본16배와14+2배에 pin되어 있다.
24배용 추론 파일은 이번 최종 Release에서 받아 사용하고, 실제 Studio 등록·모델 전환은 해당 통합 작업에서 수행한다.

현재 선정 목표는④ 완료까지다. 보존 가중치에는 기존⑤ A 복귀 학습 이력이 남아 있다.
실기 실행기는 물체가 그릇 안에 안착한 것을 확인한 뒤 종료하거나 다음 제어로 전환해야 한다.
실기 완료 검출과 정상 감속·제어 전환의 연결은 통합 시 확인할 항목이다.

## 전체 체크포인트 평가와 추론 파일 재생성

```bash
uv sync --locked --extra gpu --extra test --extra control --python 3.12
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/evaluate_stage_env.py \
  --run runs/show108-mlp-lstm-24u-final/training \
  --checkpoint runs/show108-mlp-lstm-24u-final/training/checkpoints/000033554432 \
  --output runs/show108-final24-recheck
```

원래 평가기의 `success`는 학습 당시 A 완료다. ④까지 정상 놓기 선정 지표와 구분한다.
최종 Release에는 최근 재평가의①~④ 집계와 각 단계 초기화 결과를 별도로 포함했다.

기본 NumPy 추론 파일은 다음 명령으로 재생성한다.

```bash
scripts/run.sh env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/deployment/export_policy.py \
  --run runs/show108-mlp-lstm-24u-final/training \
  --checkpoint runs/show108-mlp-lstm-24u-final/training/checkpoints/000033554432 \
  --cumulative-training-steps 201326592 --output exports/final24
```

공통 Runtime용 metadata도 재생성할 때는 위에서 고정한 piper-runtime 소스를 준비하고
그 `src`를 `PYTHONPATH`에 추가한 뒤 `--runtime-profile show108-mlp-lstm-v1` 옵션을 함께 지정한다.
해당 옵션은 설치된 공통 Runtime의 `contract` 검증과 소스 identity 기록을 수행한다.
