# SHOW-100 성공본 백업

SHOW-76 기반의 **그리퍼 3상태·목표 관측·DLS+QP 통합 성공본**입니다.
후속 놓기 실험의 가중치나 보상으로 교체하지 않았습니다.

## 버전과 출처

| 항목 | 값 |
| --- | --- |
| 학습·평가 코드 기준 | `f0c11617a050b7e6f42dc3b3bbdca03038ea7c64` |
| 브랜치 / Release tag | `show-100` / `show100-50m-20261029` |
| 배포 파일 | `thor-show100-50m-20261029.tar.gz`와 `.sha256` |
| 정책·학습 | MLP 256×256, seed 7, 처음부터 50,331,648전이, 1,024환경 |
| 체크포인트 | `artifacts/training/checkpoints/000050331648` |
| 관측/행동 | actor 49 / critic 66 / action 7 |

`show-76`은 백업 시작 시 GitHub main의 `965b21c4c27368333d75ae6f4853ce99e07ce675`를
그대로 가리킵니다. 기존 8× Release와 tag도 변경하지 않았습니다.
`show-100`은 그 GitHub 기록에 SHOW-100 코드 이력을 병합하고 안내 문서를 추가했습니다.
학습 코드·모델·SHOW-100 설정은 기준 commit과 동일합니다.

## 최종 설정

| 항목 | 성공본 설정 |
| --- | --- |
| 시작 자세 A | `[0, 0.523, -0.450, 0, 0, 0]` rad |
| 초기화 | 빈손 / 파지 직전 / 들고 있음 / 그릇 위 = 50/20/20/10% |
| 에피소드 | 80 step × 0.1초 = 8초 |
| 팔 제어 | TCP 속도 10Hz → DLS → recovering QP 200Hz → 관절 위치 참조 |
| 관절 참조 상한 | 속도 1.5rad/s · 가속도 8rad/s². 후속 jerk 제한은 포함하지 않음 |
| 정책 이동·회전 배율 | 병진 0.1m/s/축 · 회전 1rad/s/축 |
| 그리퍼 | a≤−0.5 닫기, a≥+0.5 열기, 사이 값은 이전 최종 목표 유지 |
| 개폐 목표 | 닫기 0mm / 열기 100mm. 기존 개폐 속도·힘 제한 유지 |
| 목표 관측 | 최종 개방 목표/100mm를 실제 폭과 별도로 관측 |
| 형상·좌표 | piper_v100, TCP link6 z=138.5mm, 탁자 z=0·베이스 z=15mm, 손가락 충돌 공백 보완 |
| 물체 | 50mm·70g 큐브, 110g 자유 그릇 |
| 보상·판정 | SHOW-76 접근·들기·운반·투입·해제와 행동 비용 유지 |

QP가 팔 전체의 충돌을 방지하는 것은 아닙니다. 이 정책에는 송지수의 XY 보정·수직 자세 강제·
개방 높이 제한과 후속 감속/개폐 조정기를 추가하지 않았습니다.

## 학습과 결과

| 누적 전이 | 빈손 A 전체 완료 |
| --- | --- |
| 8,388,608 | 0/128 |
| 16,777,216 | 0/128 |
| 25,165,824 | 0/128 |
| 33,554,432 | 122/384 · 31.8% |
| 41,943,040 | 272/384 · 70.8% |
| 50,331,648 | **351/384 · 91.4%** |

최종 혼합 초기화 전체 완료는 360/384(93.75%)입니다.
후반·최종 평가는 seed 2026/2027/2028 각 128회이며 초기 중간 평가는 seed 2026 한 개입니다.
성공 기준은 마지막 제어 step의 기존 `inside`입니다. 높이서 떨어뜨리는 투입도 성공할 수 있습니다.
일반 완료는 부드러운 놓기·안정 파지·실기 성공의 증명이 아닙니다.

원본 집계는 [final-evaluation.json](show100/final-evaluation.json),
학습 기록은 [training-summary.json](show100/training-summary.json)에 있습니다.
학습 중 `eval/episode_success`는 성공 step의 합이며 성공률이 아닙니다.
체크포인트는 정책·가치망·정규화를 포함하지만 optimizer/RNG까지 저장한 정확한 재개본은 아닙니다.

## 배포 파일 구성

```text
release.json                         전체 파일 SHA-256 manifest
repos/piper-sim-lab.bundle            show-100 코드와 Git 이력
runtime/                             NumPy 추론 런타임
policies/show100-mlp/                 policy.json, weights.npz, golden.npz, 검증 기록
artifacts/training/                  선택 체크포인트, config, metrics, summary, provenance
artifacts/evaluation/                최종 평가 JSON과 원시 NPZ
artifacts/videos/                    첫 네 사례·선택 성공 사례 영상과 선택 기준
tools/verify_release.py              아카이브·개별 파일 검증
```

NumPy export는 동일한 50,331,648 체크포인트에서 별도로 생성·검증한 파일을 그대로 보존합니다.
export의 `code_commit`은 exporter 실행 당시 기록이며 학습 코드 기준과 다릅니다.
그리퍼 목표 관측과 TCP override를 명시한 메타데이터를 포함합니다.
기준 체크포인트와 일치하는지는 저장된 golden 벡터로 재검증합니다.
원본 checkpoint·가중치·메타데이터를 새 export로 덮어쓰지 않았습니다.

다운로드·검증은 [README](../README.md)의 명령을 따릅니다.
코드 Git bundle도 새 디렉터리로 복원할 수 있습니다.

```bash
git clone -b show-100 releases/show100-50m/repos/piper-sim-lab.bundle restored-show100
```

원본 절대 경로는 provenance로 보존합니다. 다른 머신에서는 추출한 release의 실제 경로를 사용합니다.
캐시·가상환경·인증 파일·원본 전체 runs와 후속 실험은 배포 파일에 넣지 않습니다.

## 보존 검증

- 선택 체크포인트와 export를 원본 파일별 SHA-256으로 비교합니다.
- 학습 코드·모델·설정이 SHOW-100 기준 commit과 일치하는지 비교합니다.
- 그리퍼·기구학·QP와 기존 보상 보존 검사를 실행합니다.
- NumPy export 출력 일치, 아카이브 manifest와 원격 다운로드 해시를 확인합니다.
- 원본 로컬 코드·release·체크포인트, 기존 GitHub main·SHOW-76 Release를 보존합니다.

검증 결과와 파일별 해시는 [github-backup.json](show100/github-backup.json)에 기록합니다.
