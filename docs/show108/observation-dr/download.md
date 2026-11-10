# 관측 DR·넓은 스폰14+2배 부모 다운로드

이 페이지는 누적20배 연장의 부모 체크포인트 안내다. [새 누적18/20배 다운로드](download-20u.md)를 함께 제공한다.

이 체크포인트는 **원본 MLP+LSTM14배에서 새 조건으로 추가2배** 학습한 모델이다.
누적 학습량은16배지만, 기존 관측 DR 없는16배 모델과 다른 체크포인트다.

- [GitHub Release: 체크포인트·영상5개·결과·SHA256SUMS](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-obsdr-wide-14plus2u-20261101)
- [체크포인트 파일 해시·모델 명세](release-manifest.json)
- [동일 새 조건의 부모/추가학습 비교](results.md)

**정책 가중치와 정규화 통계를 이 체크포인트의 것으로 함께 교체한다.** `learner.msgpack`에는
정책·가치망·갱신된 관측 정규화·optimizer·RNG가 모두 포함되어 있다. 이전 normalizer와 혼합하지 않는다.

## 다운로드·파일 구조

```bash
git clone --branch codex/show108-observation-dr-wide-spawn https://github.com/redEddie/piper-sim-lab.git
cd piper-sim-lab
gh release download show108-mlp-lstm-obsdr-wide-14plus2u-20261101 \
  --repo redEddie/piper-sim-lab --dir downloads/show108-obsdr
(cd downloads/show108-obsdr && sha256sum -c SHA256SUMS)
mkdir -p runs
tar -xzf downloads/show108-obsdr/show108-mlp-lstm-obsdr-wide-14plus2u-checkpoint.tar.gz -C runs
```

저장소가 비공개이므로 접근 권한이 있는 GitHub 계정으로 로그인한다.

```text
show108-mlp-lstm-obsdr-wide-14plus2u/
├── training.yaml
└── training/
    ├── config.json
    ├── provenance.json
    ├── continuation.json
    ├── reward-settings.json / reset-bank.json
    ├── initial-resets.json / summary.json / parameters.json
    └── checkpoints/000016777216/
        ├── learner.msgpack
        └── metadata.json
```

`000016777216`은 **추가2배의 전이 수**이며 metadata에는 부모117440512·누적134217728전이가 기록되어 있다.
`training/config.json`과 체크포인트의 상대 경로를 유지한다. 새 reset bank와 그릇 지연 코드는 위 브랜치에 포함되어 있다.

## 3090 평가 재현

```bash
uv sync --locked --extra gpu --extra test --extra control --python 3.12
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/evaluate_stage_env.py \
  --run runs/show108-mlp-lstm-obsdr-wide-14plus2u/training \
  --checkpoint runs/show108-mlp-lstm-obsdr-wide-14plus2u/training/checkpoints/000016777216 \
  --output runs/show108-obsdr-recheck
```

JAX `recurrent.load_actor_checkpoint`를 사용한다. actor49→MLP256→64→LayerNorm→LSTM128→선형14이며
기존 `--actor lstm`과는 구조가 다르다. 저장소의 portable NumPy runtime은 새 `mlp_lstm`을 지원하지 않는다.
실제 파이프라인의 모델 어댑터에서도 이 정책과 갱신된 normalizer를 같은 세트로 사용한다.

## 이번 모델의 평가 범위

초기화별384회에서 단계 성공률은 **98.2/98.7/99.5/99.7/10.2%**다.
① 시작→그릇 안 놓기76.6%, 실제 A 전체 완료5.5%이며 A 복귀가 남은 병목이다.
기존16배 Release의 높은 A 복귀 수치는 관측 DR 없는 이전 조건의 결과이므로 직접 비교하지 않는다.
이 Release는 시뮬레이션 후속 실험 결과이며, 실기 성공률을 측정한 결과는 아니다.

각 영상은 seed2026·scene0의 첫 사례,20초·1배속이다.

| 초기화 | 영상 다운로드 |
| --- | --- |
| ① 접근 | [MP4](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-obsdr-wide-14plus2u-20261101/show108-pipeline-dr-wide-from14u-2u-initial_a-first.mp4) |
| ② 파지·들기 | [MP4](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-obsdr-wide-14plus2u-20261101/show108-pipeline-dr-wide-from14u-2u-pre_grasp-first.mp4) |
| ③ 운반 | [MP4](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-obsdr-wide-14plus2u-20261101/show108-pipeline-dr-wide-from14u-2u-held-first.mp4) |
| ④ 개방·놓기 | [MP4](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-obsdr-wide-14plus2u-20261101/show108-pipeline-dr-wide-from14u-2u-over_bowl-first.mp4) |
| ⑤ 안착·후퇴·A 복귀 | [MP4](https://github.com/redEddie/piper-sim-lab/releases/download/show108-mlp-lstm-obsdr-wide-14plus2u-20261101/show108-pipeline-dr-wide-from14u-2u-post_release-first.mp4) |
