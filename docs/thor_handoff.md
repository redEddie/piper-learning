# Jetson Thor 이전: 코드와 모델을 나누어 버전으로 배포

최신 v1 8× 모델·설치 경로·평가 결과는 [SHOW-76 v1 릴리스](show76_v1_release.md)를 따른다.
아래는 2026-10-21의 최초 이전 기록이며 당시 세 정책과 경로를 보존한다.

2026-10-21 강화학습 release를 Thor 홈에 전송·설치하고 세 정책의 수치 일치와 CPU 추론 속도를 검증했다.
실제 결과와 설치 경로는 [Thor 검증 기록](thor_validation.md)에 있다.
비전 환경과 대형 모델은 그대로 유지했다. 실카메라→정책→명령 후보 연결은 아직 미완료다.
추가 강화학습은 실행하지 않았으며 기존 학습 결과를 그대로 보관한다.

## 완료한 코드 정리

- `piper-sim-lab`: 기존 미커밋 DR 코드·실험 기록을 백업 후 커밋하고 LSTM 브랜치를 `main`에 merge.
- `one2many`: 코드 변경 없이 기본 브랜치 이름을 `main`으로 통일. 공유 원격의 `master`는 삭제하지 않았다.
- 두 저장소의 원격은 현재 `/data/projects/.../*.git`인 로컬 공유 저장소다. GitHub 저장소는 아직 미정이다.
- 합치기 전 백업: `/data/projects/piper-sim-lab/artifacts/pre-thor-main-backup-20261021/`.

## 배포의 단위

코드만 `git pull`하고 가중치를 아무 경로에 덮어쓰는 방식을 피한다. 하나의 release에는
Git commit, policy 구조, observation/action 계약, normalizer, weights checksum,
수치 일치 검증 입력/출력을 같이 고정한다.

```text
3090: 학습/튜닝 → Git main 코드 + 모델 export → release 생성
                        ↓ SFTP (처음) / GitHub Releases 등 (이후)
Thor: incoming → checksum 검증 → 새 release 디렉터리 → 추론 검사 → 실행 대상 선택
```

전송 수단이 바뀌어도 release 형식은 같다. 모델은 Git 저장소에 직접 넣지 않고
SFTP나 release asset/object storage로 옮긴다. 이전 버전은 남겨서 코드·모델 쌍으로 돌아갈 수 있다.
자동으로 실로봇 제어를 시작하거나 실행 중인 파일을 갱신하는 기능은 넣지 않았다.

## 이번에 보관할 정책

`/data/projects/piper-sim-lab/artifacts/thor-policy-exports-20261021/`:

| 디렉터리 | 모델 | 학습 예산 |
| --- | --- | --- |
| `lstm128` | LSTM 1층·폭128 + MLP | 16,777,216 step |
| `matched-mlp` | 같은 비교 학습기의 MLP | 16,777,216 step |
| `p10-mlp-67m` | 기존 Brax의 마지막 DR MLP | 67,108,864 step |

`runtime/`는 독립 설치 가능한 **NumPy 추론 패키지**다. JAX, MuJoCo, CUDA, PyTorch,
Piper SDK를 import하지 않는다. CPU로 작은 정책을 실행하므로 Thor의 학습용 JAX/CUDA 설치가
첫 이전의 필수 조건이 되지 않는다. GPU/TensorRT 최적화는 Thor에서 병목을 측정한 뒤 판단한다.
학습 코드는 기존 `src/piper_sim_lab/`에 남는다.

각 정책에는 `policy.json`, `weights.npz`, `golden.npz`, `export-validation.json`이 있다.
가중치와 관측 정규화를 함께 가져오고 LSTM cell/hidden을 episode 사이에 명시적으로 초기화한다.
`golden`은 원래 JAX 정책과 NumPy 정책의 연산 일치를 검사하는 합성 입력 80 step ×4개다.
이는 실제 파지 성공률 평가나 Thor 성능 측정이 아니다. Thor에서 다시 검사해야 한다.
원래 학습 체크포인트도 별도 artifact로 보관해 export를 다시 만들 수 있다.

**입력 계약:** 현재 모델은 48차원 관측을 요구한다. 관절각·속도·서보 목표 오차·TCP 자세·
그리퍼 폭·물체/그릇 자세·상대 위치·관측 크기·이전 행동의 정확한 순서는 `policy.json`에 있다.
단위와 base frame, 관절/회전 표현이 같아야 한다. 출력 7개는 정규화한 선속도/각속도 명령과
그리퍼 행동이며, 관절 목표각이나 CAN 명령 자체가 아니다. 기존 IK/서보·SDK adapter로 변환해야 한다.
실기 관측 생성/카메라 좌표계/CAN 제어는 이 이전 준비에서 새로 구현하거나 검증하지 않았다.

## 1. Thor 버전 확인 — 설치 전

준비한 `thor_preflight.py`를 SFTP로 Thor에 넣고 그곳에서 실행한다.

```bash
python3 thor_preflight.py --output thor-preflight.json
```

JetPack/L4T·CUDA·TensorRT 패키지, Python, GPU, 여유 공간, 카메라/CAN 노드를 읽기만 한다.
2026-10-21 사용자 제공 결과:

| 항목 | 확인 결과 |
| --- | --- |
| OS / 아키텍처 | Ubuntu 24.04.5 / aarch64 |
| L4T / 커널 | 39.2.1 / 6.8.12-1021-tegra |
| GPU / 드라이버 | NVIDIA Thor / 595.78 |
| CUDA 컴파일러 | 13.2, V13.2.86 |
| TensorRT native 패키지 | 10.16.2.10-1+cuda13.2 |
| 현재 Python | Conda base에서 3.14.7 |
| 홈 여유 공간 | 약 414.4 GiB |
| SSH/SFTP | pi_lab@100.99.100.99 |

L4T 39.2.1은 NVIDIA 공식 JetPack 7.2.1의 기반 버전과 일치한다. 공식 배포표의 CUDA는
13.2.2이므로 현재 nvcc 출력만으로 CUDA 전체 패키지의 패치 버전까지 같다고 확정하지 않는다.
`nvidia-jetpack` 메타패키지가 없다는 이유로 JetPack 구성 요소 전체가 없다고 판단하지 않는다.
TensorRT native 라이브러리는 설치되어 있으며 Python `tensorrt: null`은 현재 Python에서
배포 메타데이터가 발견되지 않았다는 뜻이다. 다른 `packages: null`도 같은 의미다.
GPU 메모리 `[N/A]`는 해당 조회값이 제공되지 않았다는 뜻이며 GPU 부재를 뜻하지 않는다.
카메라 노드 6개와 CAN 인터페이스 5개의 존재만 확인했다. 실제 장치 수·연결·bitrate·제어는 미검증이다.

첫 단계의 NumPy 정책 검사를 위해 CUDA/TensorRT를 재설치할 필요는 없다.
Conda base는 유지하고 별도의 Python 3.12 환경을 사용한다. 실제 설치에서는 시스템 Python 3.12.3의 venv를 사용했다. 추론 패키지는 Python >=3.10을
지원하지만 기존 학습 프로젝트의 Python 범위(>=3.11,<3.13)도 고려해 3.12를 선택한다.
인식 모듈의 ARM64/JetPack 호환 환경과 Python 버전은 각 의존성 검증 후 별도로 정한다.
현재 x86의 `.venv`, Conda, CUDA 라이브러리, JIT cache는 복사해서 사용하지 않는다.

NVIDIA 근거: [Thor Docker setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html),
[JetPack downloads](https://developer.nvidia.com/embedded/jetpack/downloads).
인식 모듈에 필요한 CUDA extension과 wheel의 호환성을 확인한 뒤 컨테이너 tag를 고정한다.
`one2many/pyproject.toml`과 기존 CUDA12 학습 extras를 Thor에서 한 번에 설치하는 명령도 쓰지 않는다.

## 2. SFTP로 작은 release 먼저 옮기기

release 위치: `/data/projects/piper-sim-lab/releases/thor-20261021-r1/`.
사용자명은 `pi_lab`, 접속 주소는 `100.99.100.99`다. 아래 명령은 설치 절차의 참고이며 이미 설치한 r1을 다시 덮어쓰지 않는다. 비밀번호나 개인키를 bundle에 넣지 않는다.

```bash
# 3090에서: 첫 전송용 batch는 put을 사용한다. r1의 put -a batch는 미존재 파일에서 실패했다.
sftp -b /data/projects/piper-sim-lab/releases/thor-20261021-r1/upload-thor-first-transfer.sftp pi_lab@100.99.100.99
```

Thor에서:

```bash
python3 ~/robot-incoming/thor_preflight.py --output ~/thor-preflight.json
python3 ~/robot-incoming/verify_release.py \
  ~/robot-incoming/thor-20261021-r1.tar.gz \
  --extract-to ~/robot/releases/thor-20261021-r1

release="$HOME/robot/releases/thor-20261021-r1"
# 실제 설치에 사용한 환경: 시스템 Python 3.12.3의 독립 venv
/usr/bin/python3 -m venv "$HOME/robot/envs/policy-r1"
source "$HOME/robot/envs/policy-r1/bin/activate"
python -m pip install --no-cache-dir numpy==1.26.4
# 빌드 산출물이 release 안에 생기지 않도록 소스를 임시 경로에 복사한다.
policy_build_dir=$(mktemp -d "$HOME/robot/policy-build.XXXXXX")
cp -a "$release/runtime" "$policy_build_dir/runtime"
python -m pip install --no-deps "$policy_build_dir/runtime"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m piper_policy_runtime.check \
 --policy "$release/policies/lstm128"
# 기존 MLP도 같은 방법으로 비교할 수 있다.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 python -m piper_policy_runtime.check \
 --policy "$release/policies/p10-mlp-67m"
```

NumPy 설치에는 ARM64용 패키지가 필요하다. 인터넷이 없는 Thor라면 새 환경의 Python 3.12에
맞춰 ARM64 wheel을 별도로 준비한다. 위의 검사 결과는 정책 단독 CPU 지연이며 VLM·pose·SDK를
포함한 전체 제어 루프 지연이 아니다. 현재 3090 호스트에서 잰 CPU 지연을 Thor 수치로 인용하지 않는다.

코드 저장소 복구도 네트워크 없이 가능하다:

```bash
mkdir -p ~/robot/repos
git clone --branch main "$release/repos/piper-sim-lab.bundle" ~/robot/repos/piper-sim-lab
git clone --branch main "$release/repos/one2many.bundle" ~/robot/repos/one2many
```

검증/설치는 실행 대상을 바꾸지 않는다. 새 release에서 추론·실기 관측 연결을 확인한 뒤
실행 프로세스를 재시작할 때 그 버전의 경로를 선택한다. 문제가 생기면 이전 release 경로를 선택한다.

## 3. 큰 인식 모델 — 이번에는 전송하지 않음

Thor의 `/home/pi_lab/songjisu/models/`에 있는 SAM3 PT 및 FoundationPose++/Cutie 체크포인트
5개가 3090 원본과 SHA-256 일치함을 확인했다. 사용자 요청에 따라 기존 비전 파일·환경은
유지했으며 아래 대형 모델 전송 절차는 이번에 실행하지 않았다. SAM3.1/Qwen은 필요가
정해졌을 때 추가한다. `sam3/model.safetensors`도 기존 PT 경로가 충분하면 중복 설치하지 않는다.

기존 `/data/models`의 SAM3, SAM3.1, Qwen3-VL-8B, FoundationPose++는 다시 받지 않는다.
용량이 크므로 작은 정책 release와 분리했다. 현재 파일을 읽어 만든 파일별 SHA-256 manifest와
재개 가능한 SFTP batch는 다음 위치에 있다:

`/data/projects/piper-sim-lab/artifacts/thor-model-transfer-20261021/`

```bash
# 3090에서
sftp -b /data/projects/piper-sim-lab/artifacts/thor-model-transfer-20261021/models.sftp pi_lab@THOR_IP
# Thor에서: 전송 결과의 파일 크기와 SHA-256을 확인한다.
python3 ~/robot-models/20261021/verify_models.py ~/robot-models/20261021
```

전송 도중 끊기면 같은 batch를 다시 실행한다(`put -a` 재개). 같은 원본/버전의 파일을 이어받는
용도이며 다른 모델로 변경할 때는 새 버전 경로를 쓴다. 전송 후 checksum 불일치 파일은 지우고
그 파일을 처음부터 다시 전송한다. 기존 모델 파일은 수정하지 않았다.

SAM/FP++의 source commit과 준비 기록도 release에 보관한다. 추론 라이브러리·CUDA extension은
Thor에서 다시 설치/빌드해야 하며 큰 weights를 옮겼다는 이유로 해당 모듈이 실행 검증된 것은 아니다.
카메라 calibration, 로봇 base frame, CAN 인터페이스/bitrate와 그리퍼 calibration도 Thor에서 확인한다.

## 4. GitHub가 정해진 뒤의 원격 튜닝 흐름

현재 공유 원격 `origin`은 유지하고 GitHub를 별도 원격으로 추가할 수 있다:

```bash
# 각 저장소에서 해당 GitHub URL을 사용한다.
git remote add github git@github.com:ORG/piper-sim-lab.git
git push -u github main
# Thor의 bundle clone은 GitHub를 origin으로 바꾼다.
git remote set-url origin git@github.com:ORG/piper-sim-lab.git
git fetch origin
```

튜닝 브랜치에서 코드 변경 → 테스트 → `main` merge → 새 체크포인트 학습 → 모델 export/검증 →
새 release 생성 순서로 한다. 코드만 바뀌고 모델이 그대로인지, 모델/normalizer도 바뀌는지
release manifest로 구분한다. Thor는 검증한 commit/모델 쌍을 가져오며 실행 중인 코드를 자동 pull하지 않는다.
GitHub 업로드는 코드 전달이고 Thor에서 원격으로 명령을 실행하는 기능은 SSH 별도 영역이다.
GitHub 저장소가 정해지면 수동 release 생성을 GitHub Actions/Release asset 또는 object storage로
연결할 수 있다. private repository 인증·CI 배포는 아직 구성하지 않았다.

현재 `thor-20261021-r1` 묶음은 이미 생성한 commit과 파일 checksum을 그대로 유지한다.
이 문서의 target 점검 결과와 설치 절차 보완으로 기존 archive를 덮어쓰지 않는다.

## 재생성

```bash
# 3090의 학습 환경, 추가 학습 없이 기존 checkpoint를 export한다.
env -u LD_LIBRARY_PATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:runtime/src \
 /data/envs/venv/piper-sim-lab/bin/python scripts/deployment/export_policy.py \
 --run /data/projects/piper-sim-lab/artifacts/show69-lstm128-20261021/lstm \
 --output /새로운/artifact/정책디렉터리
```

`build_release.py --help`에서 repo/policy/artifact를 이름=경로로 추가한다. 저장소는 clean 상태여야 하고
이미 존재하는 release/정책 디렉터리는 덮어쓰지 않는다. `verify_release.py`는 checksum을 확인하고
새 디렉터리에만 푼다. 테스트에는 변조 검출과 경로 이탈 거부도 포함한다.
