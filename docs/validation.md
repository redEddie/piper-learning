# RTX 3090 검증 기록

검증일: 2026-10-20 (KST). NVIDIA RTX 3090 24 GiB, driver 580.178.04, Python 3.11.17. 전용 venv `/data/envs/venv/piper-sim-lab`을 사용했다. JAX 0.10.2, MuJoCo/MJX/Warp 3.16.0, Warp-lang 1.18.0이며 Playground·Brax Git revision과 전체 소스 해시는 각 provenance에 남겼다.

## 병렬 실행 속도

[원시 결과와 provenance](evidence/benchmark-rtx3090.json). 최종 Warp 용량 설정과 `strict_physics=True`를 사용했다.

| 환경 수 | transitions/s (warm 중앙값) | GPU 전체 메모리 관측 피크 | 첫 reset | 첫 128-step rollout |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 28,012 | 705 MiB | 2.01 s | 4.14 s |
| 1,024 | 172,603 | 1,161 MiB | 1.42 s | 4.55 s |
| 4,096 | 389,158 | 2,739 MiB | 1.50 s | 5.41 s |

각 transition은 제어 1회(0.05 s), 물리 10회(각 0.005 s)다. 렌더링·정책 추론·PPO 업데이트를 제외한 headless 물리 실행이며 zero action을 사용했다. 최초 rollout 이후 128 제어 step을 3회 측정했고 GPU 완료를 동기화했다. 환경별 최소 5회 fresh autoreset을 포함했다. 모든 구간에서 nonfinite·capacity overflow가 0이었다.

같은 프로세스에서 128 → 1024 → 4096 순서로 실행했다. 후속 메모리에는 앞선 JIT 캐시가 남을 수 있고, `nvidia-smi`의 0.5초 간격 GPU 전체 메모리에는 디스플레이·다른 프로세스도 포함된다. 프로세스 전용 할당 피크가 아니다. 첫 호출 시간은 기존 Warp 커널 캐시가 있는 상태이며 설치 직후의 완전한 cold start 시간을 뜻하지 않는다. 초기 커널 구축 때는 약 1분이 추가로 걸렸다.

**기본 학습은 1024개 환경으로 시작한다.** 4096은 물리 처리량 비교를 통과한 설정이며 해당 수에서 PPO 학습까지 검증했다는 뜻은 아니다. 정책·물체·접촉·카메라가 추가되면 다시 측정한다.

## 충돌 용량

[정적 후보 및 CPU 검사](evidence/warp-capacity-cpu-audit.json), [GPU 검사](evidence/warp-capacity-gpu.json)를 보존했다. 현재 모델에서 전체 contact/CCD 용량을 환경 수에 비례해 `160 × N`, `4 × N`으로 잡고 constraint는 환경마다 512행을 둔다. CLI가 실제 배치 크기를 주입한다.

GPU에서 관절 전체 범위의 1024개 무작위 자세를 forward 및 10 물리 step 동안 검사했다. contact 최대 8672/163840, 환경별 constraint 최대 156/512, 용량 overflow·nonfinite는 0이었다. 일부 자세는 심하게 겹친 stress 입력이며, solver/line-search 반복 상한 비트는 기록하되 용량 실패와 구분했다. 이 검사로 모든 충돌 자세의 물리 수렴을 입증하지 않는다. 물체나 충돌 설정을 바꾸면 용량 상한도 다시 산정해야 한다.

## CPU 계약과 실제 PPO 연결

[환경·wrapper 테스트](evidence/cpu-tests.txt) 19개와 [실제 checkpoint 저장/복원 테스트](evidence/checkpoint-tests.txt) 3개가 통과했다. 독립 RNG·목표 FK·TCP·제어 한계·리셋 격리·시간제한 통계·오류 검사·30차원 관측의 제어 목표 정보·복원 전후 행동 일치를 확인했다. JAXopt 경고 1개는 외부 의존성에서 발생했다. subprocess의 Orbax 제한 때문에 checkpoint 테스트는 실제 호스트에서 별도 실행했다.

[패키지 검사](evidence/package-check.json): wheel에서 89개 자산을 해시 검증하고 native MuJoCo로 로드했다(nq=8, nv=8, nu=7). [의존성 검사](evidence/dependencies.txt)도 통과했다.

## 실제 학습과 평가

`configs/reach.yaml`의 seed7, 1024개 환경, 16,777,216 transitions를 학습했다. [설정](evidence/reach-config.json), [로그](evidence/reach-metrics.jsonl), [결과](evidence/reach-summary.json), [소스/라이브러리 provenance](evidence/reach-provenance.json)를 남겼다. 컴파일·중간 평가를 포함한 실행은 **166.7초**, GPU 전체 메모리 관측 피크는 **1405 MiB**였다. 마지막 epoch의 PPO 포함 처리량은 약149,699 transitions/s였으며 초기 컴파일 시간을 포함한 전체 평균과는 다르다.

체크포인트를 별도 프로세스에서 복원하고 seed2026, 128개 목표, 각100제어 step(5초)으로 평가했다. 성공은 마지막 TCP 위치가 목표의3cm 이내인 경우다. 같은 초기 상태·목표에 기준 행동도 적용했다. [평가 원본](evidence/evaluation.json).

| 정책 | 최종 성공률 | 한 번 이상 성공 | 최종 평균 거리 |
| --- | ---: | ---: | ---: |
| 무동작 | 1.56% | 1.56% | 11.60cm |
| 무작위 | 1.56% | 19.53% | 18.91cm |
| 학습 정책 | **64.84%** | **67.19%** | **2.70cm** |

모든 비교에서 비유한 물리 상태·실패 전이는0이었다. 영상은 이128개 중 처음4개를 그대로 재생한다. 한 학습 seed와 한 목표 분포의 결과이며 여러 seed의 평균, 완전 수렴, 물체 집기, 원 논문 재현 성능으로 주장하지 않는다. 목표 자세나 그리퍼 제어 없이 위치만 맞추는 reach 과제다.

`configs/refine.yaml`로 위 정책·가치함수·관측 정규화를 불러와419만 transitions를 추가하는 warm-start도 실제 통과했다. [복원 실행 정보](evidence/warm-start-invocation.json)와 [결과](evidence/warm-start-summary.json)를 보존했다. optimizer/RNG/환경 상태를 복구하는 정확한 중단 재개는 아니다. 추가 학습 뒤 같은 seed의 최종 성공률은 61.72%, 평균 거리는 3.08cm였다([기록](evidence/evaluation-warm-start.json)). 성능 향상은 보장하지 않으며 기본 배포 데모는 위1678만-step 정책이다.

## 공유와 재실행

독립 공유 저장소: `/data/projects/piper-sim-lab/piper-sim-lab.git` (`labusers` 그룹). 이 checkout의 origin으로 연결하고 main에 업로드한다. GitHub 저장소는 생성하지 않았다. 기존 one2many 저장소와 환경은 수정하지 않았다.

- 모델·소스·YAML·lock·작은 검증 기록은 Git으로 공유한다.
- 학습 결과·영상·컴파일 캐시는 `runs/`, `.cache/`로 분리하고 Git에서 제외한다.
- 배포 데모는 이 저장소 옆의 `piper-sim-lab-demo/`에 config·checkpoint·평가·영상으로 별도 보관한다.
- 설치와 재실행은 [README](../README.md), 역할 분담과 확장은 [구조 문서](architecture.md)를 따른다.
