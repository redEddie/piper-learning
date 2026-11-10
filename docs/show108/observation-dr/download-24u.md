> 2026-11-09 최종 배포 기준은 관측 DR MLP+LSTM24배다. [최종 코드·체크포인트·실기 연결 안내](../final-release.md)를 따른다. 아래는 당시 실험·게시 기록이다.

# 관측 DR·확장 스폰 누적22/24배 체크포인트

2026-11-09 누적20배에서4배 이어 학습하고22/24배 평가·기존조건 재평가·최종첫사례영상5개를완료했다.
v5 관절 A 목표·관측 DR·MLP+마지막 LSTM을 사용한다.

- [체크포인트·영상·평가·SHA256SUMS Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show108-mlp-lstm-obsdr-wide-24u-20261109)
- [20/22/24배 전체 평가표](continuation-24u-results.md)
- [모델·파일·영상 해시 명세](continuation-24u-release-manifest.json)
- [학습 조건과 명령](continuation-24u.md)

## 다운로드·파일 구조

```bash
git clone --branch codex/show108-obsdr-joint-a-24u https://github.com/redEddie/piper-sim-lab.git
cd piper-sim-lab
gh release download show108-mlp-lstm-obsdr-wide-24u-20261109 \
  --repo redEddie/piper-sim-lab --dir downloads/show108-obsdr24
(cd downloads/show108-obsdr24 && sha256sum -c SHA256SUMS)
mkdir -p runs
tar -xzf downloads/show108-obsdr24/show108-mlp-lstm-obsdr-wide-24u-checkpoints.tar.gz -C runs
```

```text
show108-mlp-lstm-obsdr-wide-24u/
├── training.yaml
└── training/
    ├── config.json / provenance.json / continuation.json
    ├── reward-settings.json / reset-bank.json / initial-resets.json
    ├── summary.json / parameters.json / gpu.json
    └── checkpoints/
        ├── 000016777216/  # 이번 추가2배,누적22배
        │   ├── learner.msgpack
        │   └── metadata.json
        └── 000033554432/  # 이번 추가4배,누적24배
            ├── learner.msgpack
            └── metadata.json
```

저장소 접근 권한이 있는 계정으로 로그인한다.가중치와normalizer는 각 learner의것을 함께 사용한다.
상위 config와체크포인트 상대구조를 보존한다.로더는 JAX `recurrent.load_actor_checkpoint`다.

## 평가 재현

```bash
uv sync --locked --extra gpu --extra test --extra control --python 3.12
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
scripts/run.sh .venv/bin/python scripts/show103/evaluate_stage_env.py \
  --run runs/show108-mlp-lstm-obsdr-wide-24u/training \
  --checkpoint runs/show108-mlp-lstm-obsdr-wide-24u/training/checkpoints/000033554432 \
  --output runs/show108-obsdr24-recheck
```

## 결과와 해석

초기화별384회·최대20초 평가다.새조건24배의단계 내 성공률은100/99.7/100/100/96.9%다.
① 시작 실제 A 전체는22배191/384(49.7%),24배175/384(45.6%)였다.
⑤ 종료 최대 관절 A 오차 중앙값은20배13.87°→22배12.55°→24배10.51°다.
기존 관측·좁은 스폰에서24배⑤는384/384(100%),① 시작 전체는273/384(71.1%)였다.

A 완료는200ms 유지 달성 이력이다.최종24배⑤ 첫영상은5.3초에완료했고20초종료오차는11.60°였다.
관절 허용오차0.2rad는약11.46°이므로완료 후 이탈한사례도있다.종료 오차와달성이력을 구분한다.
학습seed는1개다. 전체 연결 수행과직접 단계 초기화의성공률은각각제공한다.

## 첫 사례 영상

Release의 MP4는 `show108-mlp-lstm-obsdr-wide-24u-01-initial_a.mp4`부터
`02-pre_grasp`, `03-held`, `04-over_bowl`, `05-post_release`까지다.
모두seed2026·scene0 첫 사례,20초·200프레임이다.성공 사례를선별하지않았다.
⑤와④·① 첫사례는 A를달성했으며②·③ 첫사례는20초안에A를달성하지못했다.
영상별 달성시각·종료오차·파일해시는Release manifest에 기록했다.
