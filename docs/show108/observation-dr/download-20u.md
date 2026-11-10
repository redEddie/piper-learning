# 관측 DR·확장 스폰 누적20배 체크포인트

2026-11-09 원본14배→관측 DR2배→추가4배의 누적20배 학습·평가를 완료했다.
v5 관절 A 목표의 MLP+마지막 LSTM이며,누적18/20배 체크포인트를 함께 제공한다.

- [체크포인트·영상5개·평가·SHA256SUMS Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-obsdr-wide-20u-20261109)
- [16/18/20배 평가표](continuation-20u-results.md)
- [파일별 해시·모델·영상 명세](continuation-20u-release-manifest.json)
- Plane SHOW-114

## 다운로드

```bash
git clone --branch codex/show108-obsdr-joint-a-20u https://github.com/redEddie/piper-sim-lab.git
cd piper-sim-lab
gh release download show108-mlp-lstm-obsdr-wide-20u-20261109 \
  --repo redEddie/piper-sim-lab --dir downloads/show108-obsdr20
(cd downloads/show108-obsdr20 && sha256sum -c SHA256SUMS)
mkdir -p runs
tar -xzf downloads/show108-obsdr20/show108-mlp-lstm-obsdr-wide-20u-checkpoints.tar.gz -C runs
```

저장소 접근 권한이 있는 GitHub 계정으로 로그인한다.

```text
show108-mlp-lstm-obsdr-wide-20u/
├── training.yaml
└── training/
    ├── config.json / provenance.json / continuation.json
    ├── reward-settings.json / reset-bank.json / initial-resets.json
    ├── summary.json / parameters.json / gpu.json
    └── checkpoints/
        ├── 000016777216/  # 이번 추가2배,누적18배
        │   ├── learner.msgpack
        │   └── metadata.json
        └── 000033554432/  # 이번 추가4배,누적20배
            ├── learner.msgpack
            └── metadata.json
```

정책·가치망·normalizer·optimizer·RNG는 각 learner에 함께 있다.
해당 체크포인트와 상위 config의 상대 구조를 보존한다. JAX `recurrent.load_actor_checkpoint`로 로딩한다.

## 평가 재현

```bash
uv sync --locked --extra gpu --extra test --extra control --python 3.12
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/evaluate_stage_env.py \
  --run runs/show108-mlp-lstm-obsdr-wide-20u/training \
  --checkpoint runs/show108-mlp-lstm-obsdr-wide-20u/training/checkpoints/000033554432 \
  --output runs/show108-obsdr20-recheck
```

## 관측 DR·확장 스폰 조건의 최종 수치

초기화별384회에서 단계 내 성공률은99.7/96.9/99.2/100/68.5%다.
① 시작 실제 A 전체 완료는161/384,41.9%이며⑤ 마지막 최대 관절 A 오차 중앙값은13.87°다.
성공은 에피소드 중200ms 연속 조건을 달성한 이력이다.
⑤ 첫 영상은5.4초에 A 완료를 달성했지만20초 종료의 최대 관절 오차는20.92°였다.
완료 후 자세를 계속 유지하는 성능은 남은 한계로 기록한다.

## 첫 사례 영상

모두seed2026·scene0,20초·200프레임이다. 성공 사례를 선별하지 않았다.

| 초기화 | Release 파일 |
| --- | --- |
| ① initial_a | show108-mlp-lstm-obsdr-wide-20u-01-initial_a.mp4 |
| ② pre_grasp | show108-mlp-lstm-obsdr-wide-20u-02-pre_grasp.mp4 |
| ③ held | show108-mlp-lstm-obsdr-wide-20u-03-held.mp4 |
| ④ over_bowl | show108-mlp-lstm-obsdr-wide-20u-04-over_bowl.mp4 |
| ⑤ post_release | show108-mlp-lstm-obsdr-wide-20u-05-post_release.mp4 |

영상별 A 달성 시각·종료 오차·파일 해시는 Release manifest에 있다.
평가 집계 JSON/CSV와6개 조건·체크포인트 평가 summary도 Release에 포함한다.
