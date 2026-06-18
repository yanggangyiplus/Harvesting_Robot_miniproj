# Simulation Interface Specification - Strawberry Harvest Robot

기준일: 2026-06-18

목적: 시뮬레이션 담당자가 현재 실기 수확 파이프라인을 재현하거나,
runtime JSONL을 replay할 수 있도록 ROS 입출력, 모듈 책임, motion sequence,
성공/실패 판정 기준을 한 문서에 정리한다.

## 1. 시스템 범위

현재 시스템은 Doosan E0509, RH-P12-RN-A gripper, RealSense eye-in-hand 카메라를
사용하는 모형 딸기 수확 실험이다.

현재 검증 상태:

| 항목 | 상태 |
| --- | --- |
| SW 단일딸기 pick | 육안 기준 줄기 파지/분리 성공 사례 확보 |
| SW 기준 runtime | `20260609T160052-da5edd5a`, 약 `36.4s` |
| NW 잎/줄기 가림 셀 | 실기 안정화 중 |
| NW 2026-06-17 문제 | J4 equivalent rewrite로 큰 회전 위험 발생 |
| NW 2026-06-18 문제 | TOOL 상대 MoveLine이 success를 반환했지만 실제 final approach 미동작 |
| Place | Slot0/1/3/4 성공, Slot2 보정 후 도달 관찰, Slot5 row2 line deviation 차단 |
| Grasp 자동 판정 | SafeGrasp position/current 로그 가능, 얇은 줄기 최종 성공 판정은 수기 라벨 병행 |

중요:

```text
pick_complete != 수확 성공
grasp_detected != 줄기 파지 성공
GRASP_POSE_REACHED != 실제 파지 성공
```

## 2. 전체 런타임 파이프라인

```text
RealSense RGB-D
 -> strawberry_fusion_node
    -> YOLO segmentation: ripe / unripe / sick
    -> YOLO pose: KP0 / KP1 / KP2 stem keypoints
    -> mask-keypoint matching
    -> depth + hand-eye + robot FK
    -> stable base_link grasp target
 -> /strawberry/detection/pick_pose
 -> scan_executor_node
    -> cell/subcell scan pose 이동
    -> target buffering / duplicate filtering
    -> selected target forwarding
 -> /dsr01/curobo/pick_pose
 -> curobo_planner_node
    -> neighbor obstacle registration
    -> cuRobo pre-approach planning
    -> Doosan MoveSplineJoint execution
    -> final approach
    -> open-stem descent
    -> gripper close / SafeGrasp
    -> BASE -Z detach pull
    -> retreat
    -> optional tray place
    -> /dsr01/curobo/pick_complete
 -> scan_executor_node continues or finishes
```

## 3. Main Modules

### 3.1 `strawberry_fusion_node.py`

역할:

- RealSense RGB-D 기반 검출 및 3D target 생성
- YOLO seg + YOLO pose 결과 결합
- KP0/KP1/KP2 depth와 geometry guard
- stable target window 기반 target 발행

입력:

| Interface | Type | 설명 |
| --- | --- | --- |
| `/dsr01/joint_states` | `sensor_msgs/msg/JointState` | eye-in-hand 변환용 현재 관절 |
| RealSense image/depth | camera topics | launch 설정에 따름 |

출력:

| Interface | Type | 설명 |
| --- | --- | --- |
| `/strawberry/detection/pick_pose` | `geometry_msgs/msg/PoseStamped` | `base_link` 기준 안정화된 줄기 파지 target |
| `/strawberry/detection/scene_positions` | `std_msgs/msg/Float64MultiArray` | 주변 ripe 과실 중심 `[x,y,z,...]` |
| runtime JSONL | file | detection/rejection/target metadata |

주요 JSONL 이벤트:

- `node_start`
- `scene_positions_published`
- `pick_target_rejected`
- `stable_pick_target_published`

### 3.2 `scan_executor_node.py` (`strawberry_motion`)

역할:

- workspace scan pose 실행
- `root/nw`, `root/sw` 등 target cell 검증
- detection target을 한 번에 하나씩 planner에 전달
- planner의 `/pick_complete`를 기다린 뒤 다음 target/cell 진행

입력:

| Interface | Type | 설명 |
| --- | --- | --- |
| `/strawberry/scan/start` | `std_srvs/srv/Trigger` | scan 시작 요청 |
| `/dsr01/joint_states` | `sensor_msgs/msg/JointState` | 현재 scan pose 확인 |
| `/strawberry/detection/pick_pose` | `geometry_msgs/msg/PoseStamped` | fusion target |
| `/dsr01/curobo/pick_complete` | `std_msgs/msg/Empty` | planner sequence 종료 이벤트 |

출력:

| Interface | Type | 설명 |
| --- | --- | --- |
| `/dsr01/curobo/pick_pose` | `geometry_msgs/msg/PoseStamped` | planner로 전달하는 최종 target |
| `/strawberry/scan/status` | `std_msgs/msg/String` | scan 진행 상태 |
| `/dsr01/gripper/position_cmd` | `std_msgs/msg/Int32` | legacy visualization/초기화용 |

로봇 실행 서비스:

| Service | Type | 설명 |
| --- | --- | --- |
| `/dsr01/motion/move_spline_joint` | `dsr_msgs2/srv/MoveSplineJoint` | scan pose 이동 |
| `/dsr01/motion/move_joint` | `dsr_msgs2/srv/MoveJoint` | overview/pose 이동 |

주의:

- overview gate는 J1/J4/J6의 `±360deg` equivalent를 고려해야 한다.
- launch 파라미터는 `scan_movej_vel_deg_s`, `scan_movej_acc_deg_s2`,
  `overview_return_vel_deg_s`, `overview_return_acc_deg_s2`를 사용한다.

### 3.3 `curobo_planner_node.py`

역할:

- `/dsr01/curobo/pick_pose` 수신 후 pick/place sequence 실행
- cuRobo MotionGen 기반 pre-approach 계획
- Doosan motion service 호출
- gripper close / SafeGrasp 결과 기록
- runtime JSONL 기록

입력:

| Interface | Type | 설명 |
| --- | --- | --- |
| `/dsr01/joint_states` | `sensor_msgs/msg/JointState` | planner start state |
| `/dsr01/curobo/pick_pose` | `geometry_msgs/msg/PoseStamped` | scan executor가 선택한 target |
| `/dsr01/curobo/target_pose` | `geometry_msgs/msg/PoseStamped` | legacy/manual target |
| `/dsr01/curobo/obstacles` | `std_msgs/msg/String` | legacy dynamic obstacle JSON |
| `/strawberry/detection/scene_positions` | `std_msgs/msg/Float64MultiArray` | neighbor sphere obstacle |

출력:

| Interface | Type | 설명 |
| --- | --- | --- |
| `/dsr01/curobo/pick_complete` | `std_msgs/msg/Empty` | sequence 종료 알림. 성공률 아님 |
| runtime JSONL | file | planning/motion/grasp/place event |

사용 서비스:

| Service | Type | 설명 |
| --- | --- | --- |
| `/dsr01/motion/move_spline_joint` | `dsr_msgs2/srv/MoveSplineJoint` | cuRobo trajectory 실행 |
| `/dsr01/motion/move_joint` | `dsr_msgs2/srv/MoveJoint` | fixed pose/return fallback |
| `/dsr01/motion/move_line` | `dsr_msgs2/srv/MoveLine` | BASE relative approach/descent/detach/retreat/place |
| `/dsr01/motion/change_operation_speed` | `dsr_msgs2/srv/ChangeOperationSpeed` | operation speed 설정 |
| `/gripper_service/set_position` | `dsr_gripper_tcp_interfaces/srv/SetPosition` | gripper open/release fallback |
| `/gripper_service/get_state` | `dsr_gripper_tcp_interfaces/srv/GetState` | gripper state fallback |

사용 action:

| Action | Type | 설명 |
| --- | --- | --- |
| `/gripper_service/safe_grasp` | `dsr_gripper_tcp_interfaces/action/SafeGrasp` | close + current/position feedback |

## 4. Current Pick Motion Contract

현재 measured TCP 기준 sequence:

```text
1. gripper set_position(600)
2. neighbor obstacles 등록
3. cuRobo plan: current scan pose -> pre-approach
4. MoveSplineJoint: pre-approach 실행
5. final approach:
   - 2026-06-18 이전: TOOL relative +Z MoveLine
   - 2026-06-18 수정 후: approach_dir 기반 BASE relative MoveLine
6. OPEN_STEM_DESCENT:
   - gripper open 600 유지
   - BASE -Z 30mm
7. close + SafeGrasp
8. DETACH_PULL_DOWN:
   - BASE -Z 40mm
9. retreat:
   - measured TCP: approach_dir 반대 BASE relative MoveLine
10. place disabled이면 pick-start scan pose 복귀
11. /dsr01/curobo/pick_complete publish
```

현재 주요 상수:

| 이름 | 값 | 의미 |
| --- | ---: | --- |
| `GRIPPER_APPROACH_POS` | `600` | 접근/줄기 하강 시 open |
| `GRIPPER_PLACE_RELEASE_POS` | `600` | place release 시 제한 개방 |
| `PRE_APPROACH_OFFSET` | `0.06m` | pre-approach offset |
| `CRANE_Z_OFFSET_M` | `0.030m` | open stem descent |
| `DETACH_PULL_DOWN_MM` | `40mm` | 파지 후 아래 방향 분리 |
| `FINAL_APPROACH_VEL_MM_S` | `15mm/s` | 실기 안정화용 final approach 속도 |
| `RETREAT_VEL_MM_S` | `24mm/s` | retreat 속도 |
| `GRASP_EMPTY_POSITION_THRESHOLD` | `700` | 목표까지 완전히 닫히면 empty 후보 |

## 5. Doosan Motion Service Details

### 5.1 `MoveSplineJoint`

사용 구간:

- scan pose -> pre-approach
- pick 완료 후 pick-start scan pose 복귀
- tray slot above 이동

요청 필드:

```text
pos: Float64MultiArray[] joint waypoints, degree
vel: [deg/s] * 6
acc: [deg/s^2] * 6
time: requested execution time
mode: 0
sync_type: 0
```

### 5.2 `MoveLine`

서비스 타입:

```text
dsr_msgs2/srv/MoveLine
pos: float64[6]  # [x,y,z,rx,ry,rz], mm/deg
vel: float64[2]  # [mm/s, deg/s]
acc: float64[2]  # [mm/s^2, deg/s^2]
time: float64
radius: float64
ref: int8        # DR_BASE=0, DR_TOOL=1, DR_WORLD=2
mode: int8       # ABS=0, REL=1
blend_type: int8
sync_type: int8  # SYNC=0, ASYNC=1
---
success: bool
```

2026-06-18 관찰:

- TOOL relative MoveLine이 `success=True`를 빠르게 반환했지만 실제 관절 이동이
  거의 없었다.
- 따라서 measured TCP final approach/retreat은 BASE relative vector 방식으로
  우회했다.
- 코드는 MoveLine success 후에도 예상시간과 joint delta를 확인해 fake success를
  실패로 처리한다.

## 6. Gripper / SafeGrasp Contract

주요 state 값:

| Field | 의미 |
| --- | --- |
| `present_position` | 현재 그리퍼 position. `600` open, `700` close target |
| `goal_position` | 목표 position |
| `present_current` | 현재 전류 raw |
| `current_limit` | SafeGrasp max current |
| `grasp_detected` | SafeGrasp 접촉 후보 |
| `object_lost` | object lost 후보 |
| `status_text` | `ok`일 때 신뢰 가능 |

현재 판단:

```text
present_position >= 700 -> GRASP_EMPTY 후보
present_position < 700 + current spike -> 접촉 후보
SafeGrasp result timeout 또는 -1 -> GRASP_UNVERIFIED
```

주의:

- 얇은 모형 줄기는 전류 변화가 작아 current threshold 하나로 최종 성공 판정 불가.
- 잎이나 과실 접촉도 `grasp_detected=true`로 오인될 수 있다.
- 최종 KPI는 수기 라벨과 비교해 확정한다.

## 7. Runtime JSONL Replay Contract

저장 위치:

```text
logs/runtime/YYYY-MM-DD/
```

공통 필드:

```json
{
  "schema_version": "strawberry_runtime_event.v1",
  "timestamp": "2026-06-18T10:41:51.000+09:00",
  "monotonic_sec": 12345.67,
  "run_id": "20260618T104151-1b85c213",
  "node": "curobo_planner_node",
  "git_commit": "af7f0e4",
  "event": "motion_command",
  "data": {}
}
```

시뮬레이션에 필요한 핵심 이벤트:

| Event | 사용처 |
| --- | --- |
| `node_start` | 파라미터/모델/캘리브레이션 추적 |
| `stable_pick_target_published` | perception target 재현 |
| `scene_positions_published` | 주변 과실 obstacle 재현 |
| `pick_sequence_start` | pick attempt 시작 |
| `pick_target_prepared` | clamp/Z-bias 후 target |
| `collision_world_update` | cuRobo world 상태 |
| `curobo_plan_success` | joint trajectory replay |
| `curobo_plan_fail` | IK/planning 실패 분석 |
| `curobo_plan_rejected` | branch/spline jump/limit rejection |
| `motion_command` | MoveSpline/MoveLine 실행 명령 |
| `motion_result` | controller result |
| `verify_grasp` | 자동 파지 후보 결과 |
| `pick_sequence_complete` | sequence 종료. 성공 아님 |

Replay 최소 절차:

```text
1. node_start로 parameter profile 확인
2. scene_positions_published로 neighbor obstacle 구성
3. pick_target_prepared 또는 stable_pick_target_published로 target 설정
4. curobo_plan_success trajectory_rad를 joint trajectory로 재생
5. motion_command의 MoveLine relative vector를 TCP/BASE command로 재생
6. verify_grasp와 수기 라벨을 비교
```

## 8. SW Successful Baseline Evidence

SW 단일딸기 기준 문서:

```text
docs/sw_single_strawberry_harvest_notion_20260609.md
docs/harvest_motion_session_20260609.md
docs/experiment_results.md
```

대표 runtime log:

```text
logs/runtime/2026-06-09/curobo_planner_node_20260609T160052-da5edd5a.jsonl
```

확인된 값:

| 항목 | 값 |
| --- | --- |
| target 수신 -> scan pose 복귀 | 약 `36.4초` |
| final approach | MoveLine 직선 접근 |
| detach | BASE `-Z 40mm` |
| 자동 결과 | `GRASP_UNVERIFIED` |
| 사람 관찰 | 줄기 파지/분리 성공 사례 |

주의:

- 이 log는 "육안 성공 사례"의 증거이며, 자동 성공률 통계가 아니다.
- `GRASP_UNVERIFIED`인 이유는 당시 gripper state read가 안정적이지 않았기 때문이다.

## 9. NW Current Failure Case

2026-06-18 대표 실패:

```text
logs/runtime/2026-06-18/curobo_planner_node_20260618T104934-1baf93e6.jsonl
```

관찰:

```text
FINAL_APPROACH_STRAIGHT TOOL +Z 184.6mm vel=15.0mm/s
MoveLine returned early: 0.15s < expected 7.38s
joints barely moved: max_delta=0.01deg
ABORT: 직선 진입 실패
```

해석:

- cuRobo pre-approach는 성공했다.
- J4 large rewrite는 나타나지 않았다.
- 문제는 TOOL relative MoveLine이 실제 이동 없이 성공을 반환한 것이다.

조치:

- MoveLine fake success guard 추가
- measured TCP final approach/retreat을 BASE relative vector 방식으로 변경

다음 테스트에서 기대 로그:

```text
FINAL_APPROACH_STRAIGHT_BASE BASE REL xyz=[..., ..., ...]mm
```

더 이상 `FINAL_APPROACH_STRAIGHT TOOL +Z ...`가 나오면 빌드/source가 반영되지
않은 것이다.

## 10. KPI And Manual Labels

자동 집계 도구:

```bash
python3 scripts/prepare_harvest_label_sheet.py --cell root/nw
python3 scripts/check_harvest_logging.py --cell root/nw
python3 scripts/summarize_runtime_kpis.py --cell root/nw
python3 scripts/generate_harvest_kpi_report.py --cell root/nw
```

수기 라벨 파일:

```text
reports/harvest_kpi/manual_labels_root_nw.csv
```

수기 입력 필드:

| Field | 의미 |
| --- | --- |
| `stem_grasp` | 실제 줄기를 잡았는지 |
| `detach` | 줄기에서 분리됐는지 |
| `retention` | retreat 후에도 유지됐는지 |
| `non_target_contact` | 잎/과실/파츠 비목표 접촉 |
| `human_intervention` | 사람이 멈춤/재시작/수동 조작했는지 |
| `place` | 계란판 place 성공 여부 |
| `notes` | 실패 원인 메모 |

현재 NW는 모션 안정화 전이므로 KPI는 "측정 구조 준비" 단계다.

## 11. 시뮬팀에 전달할 최소 파일 목록

필수:

```text
docs/SIMULATION_INTERFACE_SPEC_20260618.md
docs/runtime_pipeline_and_simulation_logs.md
docs/system_architecture.md
docs/HANDOFF_20260617_NW_MOTION_GRIPPER_STATUS.md
logs/runtime/2026-06-09/curobo_planner_node_20260609T160052-da5edd5a.jsonl
logs/runtime/2026-06-18/curobo_planner_node_20260618T104934-1baf93e6.jsonl
reports/harvest_kpi/manual_labels_root_nw.csv
```

선택:

```text
config/environment.yaml
config/curobo/e0509_gripper_measured_tcp.yml
config/curobo/e0509_gripper.urdf
urdf/e0509_with_gripper.urdf.xacro
docs/sw_single_strawberry_harvest_notion_20260609.md
docs/GRIPPER_BIDIRECTIONAL_DIAGNOSIS_20260615.md
```
