# Piper Sim Lab — SHOW-103

SHOW-100 성공본을 바탕으로 **단계 보상·낮은 위치 해제·안착·철수·A 복귀**를 연구합니다.
학습은 **RTX 3090**, 선정된 정책의 추론·실기 적용은 **Thor**로 분리합니다.

**현재는 수식과 시간 이력 검증만 완료했습니다. 새 MJX task·고정 초기화·GPU 물리 연결과 학습은 아직입니다.**

- [3090에서 이어갈 지침서](docs/show103/3090-handoff.md)
- [보상 수식·판정 계약](docs/show103/stage-reward-v1.md)
- [수치 검증](docs/show103/stage-reward-v1/audit.json) · [실행 계획](docs/show103/3090-execution-plan.json)
- 담당 Plane: SHOW-103

## 현재 설계

| 항목 | 설정 |
| --- | --- |
| 학습 | 새 MLP 256×256, 처음부터 4,096환경, seed 7 |
| 초기화 4개 | 빈손 A / 파지 직전 / 들고 있음 / 그릇 위 = 40/20/20/20% |
| 초기 난이도 | 각 시작점의 대표 배치·자세 고정, 동역학 DR만 적용 |
| 보상 구간 5개 | 접근 0~1 → 파지·들기 1~2 → 운반 2~3 → 놓기 준비 3~3.5·실제 해제 4 → 안착·철수·A 복귀 4~5 |
| 보조 항 | 면 정렬 최대 +0.5, XY 정렬 +0.5/스텝, 저속 유지 +0.5·최대 두 번 |
| 비용 | 약한 큐브 밀림, 해제 후 재접촉 −0.5/스텝, 기존 행동/행동 변화 비용 |
| 제어 | DLS+QP: v=1rad/s, a=5rad/s², jerk=100rad/s³. SHOW-100 그리퍼 3상태·개폐 속도 제한 |
| 주기·관측 | 정책 100ms / 물리·제어 5ms, actor49 / critic66 / action7, 20초 |
| 목표 | TCP–그릇 내부 바닥 50~60mm에서 해제·안착 후 접촉 없이 A 복귀 |

원본 +5/+2 상태 보상에 새 보상을 더하지 않습니다. 새 보상으로 대체하며 운반 중 높이 벌점·깔때기·
순차 개폐 조정기·과제별 강제 액션 보정은 넣지 않습니다. 첫 학습 성공 후 위치·상대 자세·관절 자세를
한 종류씩 다양화하고 매 수준의 범위·성공률·체크포인트를 기록합니다.

## 3090에서 시작

```bash
git clone --branch show-103 https://github.com/redEddie/piper-sim-lab.git piper-sim-lab-show103
cd piper-sim-lab-show103
uv sync --locked --extra gpu --extra test --extra control --python 3.12
export PYTHONPATH="$PWD/src:$PWD/runtime/src"
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -q tests/test_stage_reward.py tests/test_export_show100_layout.py
JAX_PLATFORMS=cpu .venv/bin/python scripts/show103/audit_stage_reward.py --output runs/show103-formula-audit.json
```

**다음 작업은 학습 실행이 아니라 새 보상을 시뮬레이터에 연결하는 것입니다.** 과거
`configs/show103/*.yaml`은 이전 보상·개폐 방식의 기록이므로 최신 설계의 학습 설정으로 쓰지 않습니다.
연결·4,096환경 물리 검증·단기 학습·본 학습·최종 Thor 전달은 지침서 순서를 따릅니다.

## 보존한 기준과 이력

| 위치 | 내용 |
| --- | --- |
| [show-76](https://github.com/redEddie/piper-sim-lab/tree/show-76) | 기존 main의 SHOW-76 8× 성공본 |
| [show-100](https://github.com/redEddie/piper-sim-lab/tree/show-100) | 5,033만 전이 빈손 전체 완료 351/384(91.4%) 성공본 |
| [SHOW-100 Release](https://github.com/redEddie/piper-sim-lab/releases/tag/show100-50m-20261029) | 원본 체크포인트·normalizer·추론 가중치·평가·영상 |
| [이전 SHOW-103 본문](docs/show103/history/plane-before-20261029.html) | 최신 정리 전 설계·실험 기록 |
| [최근 이전 실험 평가](docs/show103/history/direct-gripper-50m-evaluation.json) | 기존 직접 개폐 5,033만 정책. 새 보상 실험의 결과가 아님 |

Thor throttle 메시지는 사용자 보고이며 원인 진단 결과가 아닙니다. 사용자는 비전의 큐브·그릇 좌표가
올바르면 웹앱·액션 스크립트에서 시뮬레이션과 유사하게 움직이는 것을 확인했습니다.
이번 새 보상의 정책은 아직 학습하지 않았으므로 그 정책의 실기 성공을 의미하지 않습니다.
최종 이전에는 선택 checkpoint·normalizer·설정·코드/형상 해시와 검증된 추론 가중치를 함께 보관합니다.
학습용 가상환경·CUDA/JIT 캐시는 머신 간 복사하지 않습니다.

자산 출처와 원본 고지는 [ASSETS.md](ASSETS.md), SHOW-100 백업 안내는
[docs/show100_release.md](docs/show100_release.md)에 있습니다. 기존 main·show-76·show-100은 변경하지 않습니다.
