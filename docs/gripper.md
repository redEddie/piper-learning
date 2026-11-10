# Piper 그리퍼: 확인된 사실, 시뮬 모델링 가정, 실측 절차 (2026-10-21)

이 문서는 그리퍼에 관해 **확인된 것**, **시뮬레이터에서 가정한 것**, **아직 모르는 것**을 구분한다.
시뮬 값은 실기 검증 전의 모델링 가정이며, SDK의 N·m 값을 시뮬 슬라이드 관절의 N 값에 그대로 넣지 않는다.

## 1. 확인된 사실

| 항목 | 내용 | 출처 |
| --- | --- | --- |
| 기구 | 평행 2지, 손가락 joint7(0–35 mm)·joint8(−35–0 mm)을 등식 제약으로 묶음 → 개방폭 0–70 mm | Menagerie `agilex_piper/piper.xml` (`src/piper_sim_lab/assets/agilex_piper`) |
| 손가락 패드 | link6 기준 z 0.075–0.105 m(근위 패드 3 cm) + 0.105–0.135 m(원위 패드 3 cm), 손바닥 ≈ 0.07 m. 집기 TCP `piper_grasp` = z 0.12 m(원위 패드 중심) | 같은 모델, `pick_place.py` |
| SDK 명령 | `GripperCtrl(gripper_angle: int, gripper_effort: int, gripper_code, set_zero)`. `gripper_angle` 단위 0.001 mm. `gripper_effort` 단위 0.001 N·m, 범위 0–5000(0–5 N·m). `gripper_code` 0x00 비활성/0x01 활성/0x02·0x03 오류 해제. `set_zero` 0xAE = 현재 위치를 영점으로 | piper_sdk `piper_interface_v2.py` (2026-10-21 열람) |
| SDK 피드백 | `GetArmGripperMsgs()` → `grippers_angle`(0.001 mm), `grippers_effort`(0.001 N·m), 상태 코드. `GetArmGripperCtrl()` → 마지막 명령(0x159) | 같은 파일 |
| 모드 | 그리퍼에는 토크·전류 제어 모드가 없다. 임피던스(MIT) 모드 `JointMitCtrl(motor_num 1–6, pos_ref, vel_ref, kp 0–500, kd −5–5, t_ref ±8)`는 **팔 관절 1–6 전용** | 같은 파일 |
| 속도 설정 | 그리퍼 속도 파라미터는 없다(`MotorMaxSpdSet`은 관절 1–6, `EndSpdAndAccParamSet`은 말단) | 같은 파일 |
| 기타 파라미터 | `GripperTeachingPendantParamConfig(teaching_range_per 100–200, max_range_config [0,70,100], teaching_friction)`, `SetSDKGripperRangeParam(min, max)`(문서 없음), 생성자 `start_sdk_gripper_limit` | 같은 파일 |

## 2. 아직 모르는 것 (실측·문서 확인 필요)

1. **effort의 의미**: 내부적으로 토크 **상한**인지 토크 **목표값**인지. SDK 문서는 "gripper torque"라고만 적는다. 제조사의 해당 펌웨어 버전 설명을 확인해야 확정된다.
2. **effort(N·m) ↔ 손끝 힘(N)** 변환: 감속비·리드가 공개되어 있지 않다. 저울/힘 게이지로 직접 잰다.
3. 닫힘 속도 프로파일과 effort 의존성, 명령→동작 지연, 피드백 주기.
4. 접촉 후 거동: 목표 폭에 못 미칠 때 계속 밀고 있는지(상한 거동), 멈추는지, 진동하는지.
5. 위치 정확도·반복성, 영점 드리프트.
6. **SDK 버전과 그리퍼 펌웨어 버전** — 모든 실측 기록에 같이 적는다.

## 3. 시뮬레이터 모델링 가정 (현재 `pick_place.py`)

| 항목 | 값 | 성격 |
| --- | --- | --- |
| 제어 형태 | 위치 서보 + 힘 제한. 액추에이터 `kp 400`, `forcerange ±gripper_force` | 가정. 실기 SDK가 "폭 + effort"를 받는 구조를 흉내낸 것 |
| 힘 상한 `gripper_force` | 3 N (손가락당, 작업별 설정값) | 가정. 80 g PLA 큐브, 마찰계수 0.5 가정 시 손가락당 0.8 N 필요 → 약 4배 여유. 에셋 상한 10 N 미만 |
| 열림/닫힘 | `gripper_binary: true`: 행동 7번째 > 0 열림(70 mm), ≤ 0 닫힘(0 mm) | 결정(2026-10-21). 닫히면 힘 상한까지 그냥 쥔다 |
| 닫힘 속도 | 목표가 스텝(0.1 s)당 10 mm 이동 → 약 0.1 m/s | 가정. 실측값으로 교체 |
| 강성·감쇠 | kp 400 N/m (오차 7.5 mm에서 3 N 포화), kv 5 | 가정. 실기 강성은 조절 불가하므로 실측 강성에 맞춘다. `physics_randomization`이 켜지면 kp·kv를 env마다 ×0.8–1.25로 랜덤화(2026-10-21 구현, 논문 Table 10의 hand stiffness/damping) |
| 마찰 | 블록 geom friction 1.0, 패드는 Menagerie 기본 | 가정. PLA–패드 실측으로 교체 |
| 이전 값 | kp 100, 연속 폭 행동(pp-full-04/06까지의 학습) | 기록용 |

주의: `gripper_force`(N)는 SDK `gripper_effort`(0.001 N·m)와 **다른 양**이다. 둘 사이의 표는 4절의 실측으로 만든다.

## 4. 실측 절차 (SHOW-63)

준비: Piper 전원·CAN 연결, piper_sdk 설치본, 5 cm PLA 큐브(80 g), 저울 또는 힘 게이지(0–20 N), 자(폭 확인), 녹화용 카메라. 기록 파일마다 SDK 버전(`pip show piper_sdk`), 그리퍼 펌웨어 버전(SDK 조회 함수가 있으면 그 값, 없으면 제조사 도구), 날짜를 적는다.

공통: 피드백을 50–100 Hz로 로깅한다(`GetArmGripperMsgs()` 각도·effort·상태, 명령 시각). 안전: effort는 낮은 값부터 올리고, 손가락 사이에 손을 넣지 않는다.

| # | 실험 | 명령 | 기록 | 알 수 있는 것 |
| --- | --- | --- | --- | --- |
| A | 빈손 닫힘 | 폭 70 → 0 mm, effort ∈ {500, 1000, 2000, 3000, 5000} 각 3회 | 폭(t), effort(t), 명령→첫 움직임 지연 | 닫힘 속도 프로파일, effort 의존성, 지연 → `gripper_rate` |
| B | 큐브 접촉 | 큐브를 손가락 사이에 두고 A와 같은 명령 | 접촉 전·접촉 순간·정지 후의 폭과 effort | 접촉 후 거동(계속 밀기/정지/진동), 정지 폭(= 큐브 폭인지), effort 피드백의 수렴값 |
| C | 힘 변환 | 손가락 사이에 저울/게이지, 폭 0 mm 명령, effort 단계별 | 정상 상태 힘(N) vs effort | **effort → 손끝 힘 표**. 상한 거동이면 힘이 effort에 비례해 포화, 목표 거동이면 다른 모양 |
| D | 미끄럼 | 큐브를 쥐고 들어올린 뒤 게이지로 당김, effort 단계별 | 미끄러지기 시작하는 힘 | 마찰계수 추정, 작업별 힘 상한의 하한 |
| E | 미달 목표 | 큐브(50 mm) 상태에서 폭 30 mm 명령 | 폭·effort 정상 상태 | 폭 목표에 못 미칠 때 effort가 상한으로 작동하는지 |
| F | 열림 | 0 → 70 mm, effort 단계별 | 폭(t) | 열림 속도, 비대칭 여부 |

결과 반영: C의 표로 작업별 `gripper_force` → 실기 effort 값을 정한다. A·F로 `gripper_rate`, C의 힘–오차 관계로 kp(강성), D로 마찰·안전 여유를 정하고, 실측 산포로 랜덤화 범위를 잡는다. B·E의 결과가 "상한 거동"이 아니면 시뮬 모델(위치 서보 + 힘 제한)을 바꿔야 한다.

## 5. 작업별 힘 설정 방식

- 시뮬: 작업 YAML의 `environment.gripper_force`(N). 기본 3 N(PLA 큐브). 연약한 물체는 낮추고 무거운 물체는 올린다.
- 실기: 4절 C의 표로 변환한 effort를 작업 설정에 적는다. 표가 없으면 실기에서 낮은 effort부터 올려 D(미끄럼)를 통과하는 최소값의 1.5–2배를 쓴다.
- one2many 파이프라인: 작업 JSON에 `gripper_force_N` 필드를 두고, 시뮬과 실기 양쪽이 같은 값을 읽어 각자 변환하는 구조로 한다(후속).

## 6. 상태

- 시뮬: 위 가정으로 구현·테스트됨(`tests/test_pick_place.py::test_gripper_force_cap_and_binary_action`, `scripts/scripted_grasp_warp.py` 7/8, `docs/evidence/pick-place/scripted-grasp-warp-block50-forcecap.txt`).
- 실기: 미실측. SDK·펌웨어 버전 미기록. effort 의미 미확정.
