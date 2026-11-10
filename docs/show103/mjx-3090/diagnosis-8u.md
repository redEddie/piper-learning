# 8배 정책의 초기화별 진단 영상 — 2026-10-30

학습67,108,864전이와 네 체크포인트 평가를 완료했다. 각 초기화384회,
총6,144회에서 네 모델 모두 전체 완료0%였다. [비교표](four-checkpoint-comparison.csv).

[![네 초기화 동시 비교](../../../media/show103-8u-four-starts-diagnostic.png)](../../../media/show103-8u-four-starts-diagnostic.mp4)

각 영상은 **최종8배 정책, 평가 seed2026의 첫 사례0, 20초·1배속**이다.
저장된 실제 GPU 평가 궤적을 같은 카메라로 렌더링했다. 성공/실패 장면을 골라내지 않았다.
화면은 위 왼쪽 빈손A, 위 오른쪽 파지 직전, 아래 왼쪽 held, 아래 오른쪽 그릇 위다.

| 초기화 | 개별 MP4 |
| --- | --- |
| 빈손 A | [영상](../../../media/show103-8u-initial_a-diagnostic.mp4) |
| 열린 파지 직전 | [영상](../../../media/show103-8u-pre_grasp-diagnostic.mp4) |
| 들고 있음 | [영상](../../../media/show103-8u-held-diagnostic.mp4) |
| 그릇 위 | [영상](../../../media/show103-8u-over_bowl-diagnostic.mp4) |

HUD는 raw 그리퍼 요청과 ±0.5 threshold, 최종 목표/적용 reference/실제 폭,
TCP 높이(합격50~60mm), 큐브–그릇XY(합격≤17mm), 양쪽 접촉, 주/총보상과 완료를 표시한다.

## 확인한 현상

그릇 위에 진입한 뒤 큐브를 계속 쥔 채 멀리 이탈하고, 높이 약50~60mm에서 정지한다.
"그릇 위 도달100%"는 도중에 한 번 해당 조건을 만족했다는 뜻이다. 현재 그릇 위에 있다는 뜻이 아니다.

8배 모델 전체 평가384회씩의 마지막 상태 중앙값:

| 초기화 | TCP 높이 mm | 큐브–그릇XY mm | 주보상 | 전체 에피소드 중 열기 요청 경험 |
| --- | --- | --- | --- | --- |
| 빈손 A | 52.6 | 437.6 | 3.5 | 384/384 |
| 파지 직전 | 52.6 | 424.6 | 3.5 | 384/384 |
| 들고 있음 | 53.0 | 424.6 | 3.5 | 0/384 |
| 그릇 위 | 51.1 | 422.3 | 3.5 | 0/384 |

초기화 모두 마지막 양쪽 손가락 접촉384/384, 합격 해제0/384다.
빈손A/파지 직전의 "열기 요청 경험"은 파지 전 요청까지 포함하며 해제 성공이 아니다.
그리퍼 목표0mm·실제 폭약50mm는 큐브를 쥔 상태다.
[원시 통계 요약](8u-video-trace-summary.json).

## 보상식에서 확인한 허용 경로

`stage_reward.py:advance_memory`의 `bowl_reached`는 한 번 도달하면 유지된다.
`section`은 그 이력으로 놓기 준비 구간3을 유지하고, `main_score`의 `prepare`는
현재 XY와 관계없이 TCP 높이 오차만으로3~3.5를 계산한다.
XY 정렬 보조 항은 멀어지면 사라지지만 구간3의 주보상3.5는 유지할 수 있다.

따라서 그릇에서 약42cm 이탈한 채 목표 높이만 맞춰 보상을 유지하는 경로가 수식에 존재하며
평가 궤적에서 그 동작을 관측했다. 이 결과만으로 다른 병목까지 배제하지는 않는다.
다음 보상 설계 검토는 현재 그릇 근처에 머무는 조건과 실제 해제 진척을 어떻게 연결할지가 핵심이다.

## 재현

```bash
scripts/run.sh .venv/bin/python scripts/show103/render_stage_env.py \
  --run runs/show103-fixed-4096-s7-2-4-6-8u-20261030/training \
  --trace runs/show103-fixed-4096-s7-2-4-6-8u-20261030/evaluation/8u/held-s2026.npz \
  --output runs/held-diagnostic-new.mp4 --scene 0 --diagnostic --label '8배 정책 · 들고 있음 시작'
```
