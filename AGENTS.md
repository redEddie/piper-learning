# Piper Learning — 최종 SHOW-108 24배 호환 작업본

- 사용자 확정 기준은 piper-sim-lab의 codex/show108-mlp-lstm-24u-final, snapshot939bdb623a6bd641152388429396b8ad74e1e70a이다.
- 학습 원본 d377cae0afdc87a389b09071eae6bcde72465e86, MLP256→64→LayerNorm→LSTM128, actor49/critic66/action7이다.
- 선택 learner SHA256은84e064bf50d34c30b346f08c109a387b0d4b1ca58fdeb30b93db638380d44306, 추가 step33554432, 누적201326592전이다.
- docs/FINAL24-INTEGRATION.md와 deployment/show108-24u-final.json을 따른다. 학습·보상·관측·제어 수식과 reset bank는 보존한다.
- 공통 제어/NumPy 추론은 piper-runtime==0.5.0에서 import한다. runtime/ 복사본을 다시 만들지 않는다. 버전과 내용 hash를 piper-runtime.lock.json으로 확인한다.
- 정상 놓기 시뮬레이션351/384(91.4%)와 실기 확인을 구분한다. 최종24배는 2026-11-10 사용자 현장 확인10회 시도·10회 성공으로 기록했으며 자동 집계 결과로 표시하지 않는다.
- 가중치·자신의 normalizer·LSTM cell/hidden을 함께 사용하며 새 에피소드에서만 memory를 초기화한다.
- A 복귀와10cm 이상낙하종료예외는선정지표에서제외한다. 보존 가중치에A행동이남아있다. 실기완료감지는별도검증대상이다.
- 실제 로봇·SDK/CAN 명령과 추가 학습은 사용자가 명시적으로 요청했을 때만 수행한다.
- 기존 체크포인트, SHOW-76/100 정책과 운영 데이터는 보존한다. 이전 인계 지침은 docs/show103/history/AGENTS-split-before-final24-20261109.md에 보관한다.

## Plane 참조 공개 규칙

Plane은 개인 서버입니다. GitHub에 커밋하는 코드·문서·JSON·HTML에는 개인 Plane 서버 주소나 링크를 넣지 않습니다. 이슈를 언급할 때는 `SHOW-75`처럼 번호와 설명을 일반 텍스트로만 기록합니다. 원본 자료를 복사할 때도 링크를 제거하고, 이슈 내용과 결과는 필요한 범위에서 보존합니다.
