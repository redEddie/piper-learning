# Piper Learning — ShowOnce 최종24배 학습 코드

사용자 확정 [piper-sim-lab 최종 브랜치](https://github.com/redEddie/piper-sim-lab/tree/codex/show108-mlp-lstm-24u-final)의 환경·보상·관측·학습·평가 코드를 유지하고, 공통 제어/추론은 별도 `piper-runtime==0.5.0`에서 가져옵니다.

- 최종 모델: 관측 DR·확장 스폰 MLP256→64→LayerNorm→LSTM128, 누적201,326,592전이(24배).
- 선택 checkpoint: 추가4배 `000033554432`, normalizer는 해당 learner와 함께 사용합니다.
- 동일 관측 DR·확장 스폰에서 접근→개방·안착351/384(91.4%). A 복귀·높은낙하종료예외 제외. 최종24배 실기는 2026-11-10 사용자 확인10회 시도·10회 성공으로 기록했습니다. [현장 결과](https://github.com/redEddie/showonce-studio/blob/main/docs/FINAL24-REAL-ROBOT-20261110.md)를 따릅니다.
- [최종 모델 다운로드](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-24u-final-20261109) · [3저장소 연결·재현](docs/FINAL24-INTEGRATION.md).

## 세 저장소 설치

형제 폴더로 `piper-runtime`, `piper-learning`, `showonce-studio`를 준비합니다. 세 저장소는 GitHub에 공개되어 있으며 [공개 범위와 라이선스 상태](docs/PUBLICATION.md)를 확인하세요.

```bash
python -m pip install -e './piper-runtime[training]'
python -m pip install -e './piper-learning[test,control]'
cd piper-learning
piper-runtime-info --lock piper-runtime.lock.json
```

또는 이 폴더에서 `uv sync --locked --extra test --extra control --extra gpu --python 3.12`를 사용합니다. CUDA/RealSense는 머신별로 준비하며 인식은 Studio의 별도 환경에서 수행합니다. 설치만으로 학습·로봇을 시작하지 않습니다.

## 기존 학습/평가 코드

`configs/show108/`, `scripts/show103/run_stage_campaign.py`, `scripts/show103/evaluate_stage_env.py`가 최종 소스와 같습니다. `src/piper_sim_lab/stage_reward_v5.py`와 참조하는v1–v4 수식을 보존합니다. `deployment/show108-24u-final.json`에 확정 SHA와 패키지 명세를 기록했습니다.

정책 내보내기에는 설치한 공통 Runtime을 사용합니다.

```bash
scripts/run.sh env JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/deployment/export_policy.py \
  --run runs/show108-mlp-lstm-24u-final/training \
  --checkpoint runs/show108-mlp-lstm-24u-final/training/checkpoints/000033554432 \
  --cumulative-training-steps 201326592 --runtime-profile show108-mlp-lstm-v1 \
  --output exports/final24
```

원본 `docs/show108/final-release.md`의 `runtime/src` 지정은 단일 저장소 당시 명령입니다. 이 분리본에서는 공통 패키지를 설치하고 그 경로를 수동 삽입하지 않습니다. 가중치·정규화 통계·LSTM 상태·TCP138.5mm·팔1/5/100·정책10Hz/제어200Hz를 유지합니다.

공개 라이선스는 아직 소유자 결정 전이며 외부 자산 출처는 ASSETS.md를 따릅니다.
