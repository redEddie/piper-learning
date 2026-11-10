# Piper Sim Lab — SHOW-103 인계

- 이 브랜치는 `show-103`이다. 현재 상태와 이어갈 작업은 `docs/show103/3090-handoff.md`가 기준이다.
- 최신 수식은 `src/piper_sim_lab/stage_reward.py`, 상세 계약은 `docs/show103/stage-reward-v1.md`다.
- 수식 검증만 완료됐다. 새로운 MJX task·4종 고정 초기화·4,096 GPU 물리 연결은 아직 구현되지 않았다.
  과거 `configs/show103/*.yaml`을 새 설계의 학습 설정으로 실행하지 않는다.
- SHOW-76·SHOW-100 원본과 선택 정책을 보존한다. 새 실행 경로를 사용하고 실험/수식 버전을 기록한다.
- 학습/평가는 3090에서 진행할 계획이며, 이번 인계 정리에서는 학습을 시작하지 않는다.
  이후 학습 실행은 사용자의 최신 지시에 따른다. 새 학습은 4,096환경·처음부터가 합의 사항이다.
- 실제 로봇·SDK/CAN 명령은 실행하지 않는다. 새 정책은 먼저 시뮬레이션에서 평가한다.
- actor 49 / critic 66 / action 7, SHOW-100 그리퍼 3상태·최종 목표와 실제 폭 구분을 보존한다.
- 저수준 팔 제어는 v1/a5/jerk100 DLS+QP, gripper는 SHOW-100 개폐 속도 제한을 사용한다.
  MOVE/BRAKE/GRIP 조정기와 과제별 출력 덮어쓰기를 다시 넣지 않는다.
- Python/가상환경/CUDA 캐시는 머신별로 설치한다. 체크포인트와 normalizer·설정·코드 revision을 함께 관리한다.
- Plane 본문은 현재 설계/결과로 유지하고 과거 결과는 history 문서로 구분한다. 댓글·관계는 임의로 변경하지 않는다.

## 세 저장소 분리 작업본

- 이 작업본의 공통 실행 의존성은 형제 저장소 `piper-runtime==0.5.0`이다.
- `runtime/` 복제본은 제거했으며 제어 커널/운동학은 설치한 패키지를 import한다.
- 현재 설치 방법은 루트 README와 `piper-runtime.lock.json`을 따른다. 과거 문서의 runtime/src 경로는 역사 기록이다.
- 추출은 원본 프로젝트와 운영 중인 Studio를 변경하지 않는다. 새로운 학습/실기 동작을 자동 시작하지 않는다.
