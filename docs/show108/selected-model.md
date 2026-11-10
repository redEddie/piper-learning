> 2026-11-09 최종 배포 기준은 관측 DR MLP+LSTM24배다. [최종 코드·체크포인트·실기 연결 안내](final-release.md)를 따른다. 아래는 당시 실험·게시 기록이다.

# SHOW-108 선정 모델: MLP+마지막 LSTM 16배

2026-11-01 사용자 결정으로 **MLP+마지막 LSTM 16배**를 현재 시뮬레이션 선정 모델로 확정했다.
브랜치: `codex/show108-mlp-lstm`. 학습 소스: `49267d34714545266fbdf85c77540307ca5c2f51`.
선정·배포 정리는 문서와 자료 연결이며 보상·모델·제어 코드는 바꾸지 않았다.

- [GitHub Release: 체크포인트·비교 영상·평가 원자료·SHA256SUMS](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-16u-20261101)
- Plane SHOW-108
- [선정 모델·체크포인트 파일 해시](selected-model.json) · [영상별 해시·사례 정보](paired-videos-16u.json)
- [2~16배 비교 JSON](matched-comparison.json) · [CSV](matched-comparison.csv) · [실험 프로토콜](experiment.json)

## 선정 근거

두 모델 모두 최종 v5 보상에서 처음부터 134,217,728전이(16배)를 학습했다.
4,096환경, 5개 초기화 각20%, 위치 랜덤화·동역학 DR·완료 조건·제어기를 유지했다.
독립 평가 bank와 seed2026/2027/2028, 초기화별384회, 최대20초 결과다.

| 지표 | 선정 MLP+LSTM | fresh Transformer+LSTM 대조군 |
| --- | ---: | ---: |
| ① 접근 | 100% | 100% |
| ② 파지·들기 | 100% | 100% |
| ③ 운반 | 100% | 100% |
| ④ 개방·놓기 | 100% | 100% |
| ⑤ 안착 후 후퇴·A 복귀 | **96.1% (369/384)** | 0% (0/384) |
| ① 시작 실제 A 전체 완료 | **75.3% (289/384)** | 0% (0/384) |
| ⑤ 최종 최대 A 관절 오차 중앙값 | **12.18°** | 25.78° |
| ⑤ 에피소드 중 최소 최대관절 오차 중앙값 | 7.80° | 21.47° |

성공은 기존 조건을 에피소드 중 달성한 이력으로 유지된다. 마지막까지 계속 A에 머무는 비율과
최종 오차는 별도 지표다. Transformer의 높은낙하 예외 성공2회는 실제 A 전체 완료에서 제외했다.
각 구조의 학습은 seed7 한 번이므로 구조 전반의 우월성·반복 수렴을 보장하지 않는다.
원본 SHOW103 Transformer 정책과 이번 fresh 대조군은 보존했다.

## 체크포인트 다운로드·평가

```bash
git clone --branch codex/show108-mlp-lstm https://github.com/redEddie/piper-sim-lab.git
cd piper-sim-lab
uv sync --locked --extra gpu --extra test --extra control --python 3.12
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
gh release download show108-mlp-lstm-16u-20261101 --repo redEddie/piper-sim-lab --dir downloads/show108
(cd downloads/show108 && sha256sum -c SHA256SUMS)
mkdir -p runs
tar -xzf downloads/show108/show108-mlp-lstm-16u-checkpoint.tar.gz -C runs
scripts/run.sh .venv/bin/python scripts/show103/evaluate_stage_env.py \
  --run runs/show108-mlp-lstm-16u/training \
  --checkpoint runs/show108-mlp-lstm-16u/training/checkpoints/000134217728 \
  --output runs/show108-mlp-lstm-16u-recheck
```

비공개 저장소이므로 접근 권한이 있는 GitHub 계정으로 로그인해야 한다. 기록된 환경 버전은
아카이브의 `training/provenance.json`과 저장소 `uv.lock`을 따른다.

```text
show108-mlp-lstm-16u/
├── training.yaml
└── training/
    ├── config.json
    ├── provenance.json
    ├── reward-settings.json
    ├── reset-bank.json
    ├── initial-resets.json
    ├── summary.json / parameters.json / gpu.json
    └── checkpoints/000134217728/
        ├── learner.msgpack
        └── metadata.json
```

`learner.msgpack`에는 정책·가치망·관측 정규화·optimizer·RNG·step이 함께 있다.
설정과 체크포인트의 상대 구조를 유지한다. JAX `recurrent.load_actor_checkpoint`를 사용하며,
이 encoder-first `mlp_lstm`은 기존 `--actor lstm`(LSTM→MLP)과 다르다.
현재 portable NumPy 추론기는 새 `mlp_lstm`을 지원하지 않는다. 실기 로봇 배포·Thor export는 별도 작업이다.

## 초기화별 비교 영상

모든 영상은 **왼쪽 MLP+LSTM / 오른쪽 Transformer+LSTM**, seed2026·scene0의 첫 사례다.
성공 사례를 골라낸 영상이 아니며, 두 모델의 동일 평가 시각을 1배속으로 비교한다(각20초).
③ 첫 사례는 MLP도 A 완료에 이르지 못한다. 대표 성능은 위384회 평가표로 판단한다.

| 초기화 | MP4 |
| --- | --- |
| ① 자세 A에서 접근 | [01-initial_a](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-16u-20261101/show108-16u-comparison-01-initial_a.mp4) |
| ② 파지 직전 | [02-pre_grasp](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-16u-20261101/show108-16u-comparison-02-pre_grasp.mp4) |
| ③ 큐브를 든 상태 | [03-held](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-16u-20261101/show108-16u-comparison-03-held.mp4) |
| ④ 그릇 위 | [04-over_bowl](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-16u-20261101/show108-16u-comparison-04-over_bowl.mp4) |
| ⑤ 해제 후 안착·후퇴·A 복귀 | [05-post_release](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-16u-20261101/show108-16u-comparison-05-post_release.mp4) |

⑤ 첫 사례에서 MLP는4.4초에 A 완료 조건을 만족했고 Transformer는20초 동안 완료하지 못했다.
각 영상의 완료 시각·마지막 오차·파일 해시는 영상 manifest에 기록했다.
