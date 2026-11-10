# SHOW-76 v1 MLP — 누적 8× 확정 정책

선택 모델은 **누적 134,217,728 transitions(8×)** MLP입니다.
`pre_grasp` 이전 초기 상태에서 시작하여 `pre_grasp → grasp → pre_inside → inside`를 수행합니다.
80번째 제어 step의 `inside` 성공률은 **DR 켬 371/384 = 96.61%, 끔 324/384 = 84.38%**입니다.
실기 폐루프 성공률과 동작 평활화는 이 결과에 포함되지 않습니다.

## 1×부터 8×까지

1× = 16,777,216 transitions입니다. 동일한 선택 학습 경로의 체크포인트를 비교했습니다.

| 누적 학습량 | DR 켬 최종 성공 | DR 끔 최종 성공 |
| --- | --- | --- |
| 1× | 0/384 = 0.00% | 0/384 = 0.00% |
| 2× | 146/384 = 38.02% | 138/384 = 35.94% |
| 3× | 334/384 = 86.98% | 296/384 = 77.08% |
| 4× | 350/384 = 91.15% | 미평가 |
| 5× | 369/384 = 96.09% | 미평가 |
| 6× | 363/384 = 94.53% | 미평가 |
| 7× | 356/384 = 92.71% | 미평가 |
| **8× 선택** | **371/384 = 96.61%** | **324/384 = 84.38%** |

평가 seed 2026·2027·2028 × 128환경, deterministic policy, 80 control steps,
`reset_stage_probs=[1,0,0,0]`, GPU-Warp 조건입니다. 주지표는 `final_success_rate`이며
`ever_success_rate`나 학습 중 성공 step 합과 구별합니다.
[CSV](evidence/show76-8x/history-1x-to8x.csv), [요약 JSON](evidence/show76-8x/summary.json)을 함께 제공합니다.

학습 seed는 7 하나이며 평가 seed 세 개는 독립 학습 반복이 아닙니다.
8×와 5×의 DR 차이는 384회 중 2회로 통계적 우월성이나 수렴을 확정하지 않습니다.
정책 선택에 같은 평가 seed를 사용했으며 새로운 holdout 최종 평가는 아직 없습니다.

## 시작 비율을 조절한 이유

시작 상태의 비율을 초기 상태 / `pre_grasp` 근처 / `grasp` 상태 / `pre_inside` 상태 순서로
40/20/20/20에서 **50/20/20/10**으로 조절했습니다.
`pre_inside`에서 시작하면 이미 든 블록을 그릇 위에서 놓는 쉬운 후반 동작만으로 보상을 얻기 쉬워,
초기 상태부터 집기까지의 연습 비중을 높이려는 목적입니다. 보상식은 유지했습니다.
이전 확률적 학습 실행에서 가치함수 손실이 발산한 적이 있으며 원인은 확정하지 않았습니다.
다음 실험도 2×를 먼저 평가하고 개선되는 설정만 연장합니다.

## 버전과 산출물

학습·export 코드 commit은 `b40fa4df0c8ef301421edab65250f9e3ed70037a`입니다.
기존 tag `show76-v1-20261023-r1`은 이 코드와 당시 **3×** 기록을 보존합니다.
그 태그를 8× 가중치 태그로 해석하지 않습니다. 이번 정리는 같은 코드의 8× 결과·배포 문서를 보완합니다.

| 배포 파일 내부 경로 | 용도 |
| --- | --- |
| `policies/show76-v1-mlp/` | policy.json, weights.npz, golden.npz, export 검증 |
| `artifacts/training/checkpoints/000083886080/` | 원본 Orbax 체크포인트 13파일 |
| `artifacts/training/config.json` | 원본 학습 설정 |
| `artifacts/training/lineage.json` | 부모·누적 학습량 기록 |
| `artifacts/evaluation/` | 평가·기하 검증 기록 |
| `runtime/src/piper_policy_runtime/` | 해당 정책과 검증한 NumPy 런타임 |
| `release.json` | 파일별 SHA-256 manifest |

체크포인트 디렉터리 숫자는 마지막 실행의 83,886,080 transitions이며 누적량은 134,217,728입니다.
모델·normalizer·원본 checkpoint는 코드 Git 이력 대신 별도 release 파일로 보관합니다.
아카이브는 `thor-show76-v1-8x-20261023-r2.tar.gz`이며 SHA-256은
`125ecf7df4daa4bf311667d34f5f5e5163f4319b5c7c1393db4d4674fc7c4c85`입니다.
추론 weights.npz SHA-256은
`087abda903449401c553f6e57bbb48d7e555a05e33292bb5e86b6d1dbb11670a`입니다.
배포 방식은 [GitHub 배포 안내](github_distribution.md)를 참고하세요.

[8× GitHub Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show76-v1-8x-20261023-r2)에서
압축파일과 SHA-256 파일을 받습니다. 이 Release의 tag는 배포 문서·시연이 포함된 코드 버전이며
실제 학습 당시 commit은 위에 기록한 `b40fa4d`입니다.

## 학습·평가 재현

새 clone에 `uv sync --locked --extra gpu --extra test --python 3.11`을 실행하고 가상환경을 활성화합니다.
저장소 루트에서 실행하며 output은 매번 새 경로를 사용합니다.

```bash
export PYTHONPATH="$PWD/src:$PWD/runtime/src"

# 처음부터 2×
scripts/run.sh piper-train --config configs/pick_place_v1.yaml --output runs/my-v1-2x --platform gpu

# 2×에서 초기 상태 비중을 높여 +1×
scripts/run.sh piper-train --config configs/pick_place_v1_empty50.yaml --warm-start runs/my-v1-2x/checkpoints/000033554432 --output runs/my-v1-3x --platform gpu

# 성공률 확인
scripts/run.sh python scripts/evaluate_empty_hand.py --run runs/my-v1-3x --output runs/my-v1-3x-evaluation

# 개선을 확인한 정책에서 +5×, 누적 8×
scripts/run.sh piper-train --config configs/pick_place_v1_to8x.yaml --warm-start runs/my-v1-3x/checkpoints/000016777216 --output runs/my-v1-8x --platform gpu

scripts/run.sh python scripts/evaluate_empty_hand.py --run runs/my-v1-8x --checkpoint runs/my-v1-8x/checkpoints/000083886080 --output runs/my-v1-8x-evaluation

scripts/run.sh python scripts/deployment/export_policy.py --run runs/my-v1-8x --checkpoint runs/my-v1-8x/checkpoints/000083886080 --cumulative-training-steps 134217728 --output runs/my-v1-8x-export
```

Warm-start는 정책·가치망·normalizer를 불러오는 기능이며 optimizer를 포함한 정확한 중단 재개는 아닙니다.
동일 seed에서도 GPU 실행의 bitwise 동일 결과를 보장하지 않습니다.
기본 평가 seed는 위 정책 선택에 이미 사용한 비교용 seed입니다.

## 소품·관측·제어 조건과 한계

아래 [시연 영상](#시연-영상)은 이 조건에서 새로 실행한 한 에피소드입니다.

- 제조사 URDF 기반 `piper_v100`, TCP link6 로컬 [0,0,0.1358] m, 전체 개구 100 mm, base 높이 15 mm.
- 블록: PLA 50 mm 큐브, 질량 70 g, DR 범위 56–84 g. 실제 형상은 고정입니다.
- 그릇: 자유 물체 110 g, 높이 70 mm(바닥 3 + 벽 67), 내측 바닥 반경 42 mm,
  윗변 반경 75 mm, 벽·바닥 3 mm. 바닥 1개와 기울어진 벽 12개 box로 근사합니다.
- 그릇 바닥 box 모서리 돌출, 근사 관성·마찰·충돌형상, 고정 그릇 질량이 한계입니다.
- 크기 관측 잡음은 실제 소품 형상 랜덤화와 다릅니다. 카메라 관측 유실·재획득은 검증하지 않았습니다.
- actor 관측 48차원, stage one-hot 없음, privileged critic 65차원. frame은 base_link, 단위 m·rad입니다.
- 출력 7차원: 최대 0.1 m/s 선속도, 1 rad/s 각속도, 그리퍼 양수 열기·그 외 닫기.
- 정책 10 Hz, 물리 200 Hz. DLS IK·관절 속도 제한·목표 누적을 사용하며 명시적 가속도·jerk 제한은 없습니다.

48차원 순서는 export policy.json을 따릅니다. 정책 출력은 관절각이나 SDK packet이 아닙니다.
관측·행동 계약은 SHOW-24, 실기 명령 적용은 SHOW-85, 폐루프 실기는 SHOW-75를 참조합니다.
SHOW-87의 v1 LSTM·평활화는 후속 계획이며 기존 v0 LSTM 비교와 구별합니다.

## Thor에서 사용

8× 설치는 2026-10-23 완료했습니다. 이전 3× r1은 보존했습니다.

```bash
release="$HOME/robot/releases/thor-show76-v1-8x-20261023-r2"
export PYTHONPATH="$release/runtime/src"
export PYTHONDONTWRITEBYTECODE=1
thor_python="$HOME/robot/envs/policy-r1/bin/python"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$thor_python" -m piper_policy_runtime.check --policy "$release/policies/show76-v1-mlp"
```

138파일 checksum, 원본 checkpoint 13파일, NumPy 정책 출력 일치를 확인했습니다.
Thor 최대 출력 차이는 1.04e-6, FK/Jacobian 100자세 최대 차이는 2.22e-16입니다.
정책 단독 CPU 지연 p50 0.05463 ms / p95 0.056149 ms는 전체 제어 루프 지연이 아닙니다.
실기 명령 전송이나 실행 대상 활성화는 하지 않았습니다.
과거 `docs/evidence/show76/`는 3× r1 기록이며 최신 집계는 `docs/evidence/show76-8x/`입니다.

## 시연 영상

[MP4](media/show76-v1-8x-demo.mp4)는 2026-10-28에 누적 8× 체크포인트로 새로 실행한
첫 번째 에피소드입니다. seed 2026, 1환경, DR 켬, 초기 상태 시작 100%, deterministic,
80 control steps 조건에서 마지막 `inside` 성공과 물리 오류 0을 확인했습니다.
이는 한 에피소드의 시연이며 위 384회 집계나 실기 성공률을 대체하지 않습니다.

원본 평가기의 2×2 화면에서 첫 번째 640×480 패널만 잘라 H.264 MP4로 저장했습니다.
10 fps, 8초, 실제 시간 속도로 재생합니다. 시뮬레이션 제어 step별 상태를 렌더링한 영상입니다.
[전체 실행 보고서](evidence/show76-8x/demo-evaluation.json),
[상태 궤적](evidence/show76-8x/demo-trajectory.npz),
[영상 해시·조건](evidence/show76-8x/demo.json)을 보관합니다.

재현하려면 먼저 위 학습·평가 재현 절차로 8× 실행을 준비하고, 초기 상태 평가 설정을 만듭니다.

```bash
python - <<'PY'
import json
from pathlib import Path
config = json.loads(Path('runs/my-v1-8x/config.json').read_text())
env = config.get('resolved_environment', config['environment'])
env.update(reset_stage_probs=[1, 0, 0, 0], obs_randomization=True, physics_randomization=True)
Path('runs/demo-config.json').write_text(json.dumps(config, indent=2))
PY
scripts/run.sh python -c 'from piper_sim_lab.evaluate import main; main()' \
  --checkpoint runs/my-v1-8x/checkpoints/000083886080 \
  --config runs/demo-config.json --output runs/demo-8x \
  --num-envs 1 --seed 2026 --video --platform gpu
```

재학습 없이 배포한 원본 checkpoint를 사용할 때는 `--checkpoint`를 배포 파일의
`artifacts/training/checkpoints/000083886080`으로 바꿉니다. 위 Python 코드가 읽는 설정 경로도
배포 파일의 `artifacts/training/config.json`으로 바꿉니다.
생성된 `policy.mp4`는 기본 2×2 화면입니다. 저장소의 영상처럼 첫 패널만 저장하려면:

```bash
python - <<'PY'
import subprocess
import imageio_ffmpeg
subprocess.run([
    imageio_ffmpeg.get_ffmpeg_exe(), '-i', 'runs/demo-8x/policy.mp4',
    '-vf', 'crop=640:480:0:0', '-c:v', 'libx264', '-crf', '18',
    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an',
    'runs/demo-8x/single-episode.mp4',
], check=True)
PY
```
