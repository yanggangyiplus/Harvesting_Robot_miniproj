# Harvesting Robot Project

벽면 재배 딸기를 **인식 → 파지 → 적재**까지 자동으로 수행하는 협동로봇 수확 시스템입니다.  
두산 E0509, RealSense RGB-D, YOLO 기반 비전, cuRobo 모션 플래닝을 하나의 파이프라인으로 묶었습니다.

## 시연 영상

약 4분 30초 분량의 pick & place 시연입니다.

<video src="media/strawberry_harvest_demo.mp4" controls width="720" poster="media/strawberry_harvest_demo_thumb.jpg">
  <a href="media/strawberry_harvest_demo.mp4">시연 영상 다운로드 (mp4)</a>
</video>

---

## 프로젝트 소개

과수·시설원예 현장에서 인력 의존도가 높은 딸기 수확을, **인지–계획–실행**이 연결된 로봇 파이프라인으로 재현하는 것이 목표입니다.

현재 프로토타입은 모형·실험 셀 기준으로 다음을 검증했습니다.

- ripe / unripe 딸기 검출과 줄기 그립점 추정
- eye-in-hand 좌표 변환 후 `base_link` pick target 생성
- cuRobo 기반 approach → grasp → retreat → place
- Gazebo / Isaac Sim 환경에서의 동일 인터페이스 재현

실기 SW 단일 딸기 기준 runtime은 약 **36초** 수준이며, NW(잎·줄기 가림) 셀은 모션 안정화 단계입니다.

**개발 기간:** 2026.05 ~ 2026.06 (프로토타입 / 실기 검증)

---

## 주요 기능

| 기능 | 설명 |
| --- | --- |
| Hand-eye 캘리브레이션 | Eye-in-hand / Eye-to-hand ArUco 기반 변환 행렬 추정 |
| 딸기 비전 | YOLO26m ripe/unripe 검출, 줄기 ROI 세그멘테이션, RealSense 실시간 추론 |
| 모션 플래닝 | cuRobo pick & place, 그리퍼 soft-close, 계란판 place slot 티칭 |
| 스캔·퓨전 | cell 스캔, seg+pose 결합 target, duplicate filtering |
| 시뮬레이션 | Gazebo bringup, Isaac Sim 5.1 Docker, runtime JSONL 리플레이 스펙 |
| VLA (실험) | SmolVLA FastAPI 서버로 이미지+instruction → action 추론 |
| 대시보드 | 수확 통계·카메라 피드 웹 UI |

---

## 프로젝트 아키텍처

```text
┌─────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  RealSense  │────▶│  Vision / Fusion     │────▶│  Motion Planning    │
│  RGB-D      │     │  YOLO + stem / KP    │     │  cuRobo + Doosan    │
└─────────────┘     │  Calibration (T)     │     │  Gripper / Place    │
                    └──────────────────────┘     └──────────┬──────────┘
                                                            │
                    ┌──────────────────────┐                │
                    │  Simulation / VLA    │◀───────────────┘
                    │  Isaac · Gazebo ·    │
                    │  SmolVLA (optional)  │
                    └──────────────────────┘
```

런타임 데이터 흐름 (실기 기준):

```text
RealSense
  → strawberry detection / fusion
  → /strawberry/detection/pick_pose  (또는 YOLO 노드가 직접 발행)
  → scan_executor (선택)
  → /dsr01/curobo/pick_pose
  → curobo_planner_node
       open → approach → grasp → retreat → place → home
  → /dsr01/curobo/pick_complete
```

모듈은 feature 브랜치에서 개발 후 통합합니다.

| 모듈 | 브랜치 | 경로 |
| --- | --- | --- |
| Calibration | `feature/calibration` | `calibration/` |
| Vision | `feature/vision_yolo26m_strawberry` | `vision_yolo/` |
| Motion | `feature/motion`, `motion_main` | `scripts/`, `launch/`, `config/curobo/` |
| Docker / VLA / Isaac | `feature/docker-env-setup` | `docker/` |
| Dashboard | `feature/dashboard` | `dashboard/` |

---

## 기술 스택

**하드웨어**

- Doosan E0509 (6축)
- ROBOTIS RH-P12-RN(A) 그리퍼
- Intel RealSense D4xx (eye-in-hand)

**소프트웨어**

| 영역 | 기술 |
| --- | --- |
| 미들웨어 | ROS2 Humble |
| 비전 | Ultralytics YOLO26m, OpenCV, pyrealsense2 |
| 플래닝 | NVIDIA cuRobo |
| 시뮬 | Isaac Sim 5.1, Gazebo |
| VLA | LeRobot SmolVLA, FastAPI |
| 인프라 | Docker Compose, NVIDIA Container Toolkit |
| 대시보드 | FastAPI, WebSocket, MJPEG |

---

## 개발 환경

| 항목 | 권장 |
| --- | --- |
| OS | Ubuntu 22.04 (ROS2 Humble) |
| GPU | NVIDIA (YOLO / cuRobo / Isaac) |
| 컨테이너 | Docker + nvidia-container-toolkit |
| 로봇 워크스페이스 | `~/doosan_ws` (DSR ROS2 패키지) |
| 공용 마운트 | `~/curobo`, `~/models` |

팀원이 같은 호스트를 쓸 때는 `.env`의 `USER_ID`, `ROS_DOMAIN_ID`로 컨테이너·DDS를 분리합니다. 실로봇 연동 시 `ROS_DOMAIN_ID=66`을 사용합니다.

---

## 개발자 소개 (역할)

커밋·브랜치 기여를 기준으로 정리한 역할입니다.

| 개발자 | GitHub / 메일 | 역할 |
| --- | --- | --- |
| **한가영 (HANGAYEONG)** | [yanggangyiplus](https://github.com/yanggangyiplus) | 프로젝트 리포 운영, Vision(YOLO26m) 파이프라인, Docker/문서 정리, Motion README |
| **곽정원 (jeongwonkwak)** | jeongwonkwak | Eye-in/to-hand 캘리브레이션, 수확 대시보드 |
| **djdoss1234** | djdoss1234 | Motion planning (cuRobo pick & place), fusion/scan 런타임, 시뮬 인터페이스 스펙 |
| **OceanShape** | OceanShape | Docker 환경·라인엔딩 등 인프라 정리 |

세부 담당 모듈

- **Calibration** — ArUco 샘플 수집, `T_cam_to_base` / `T_cam_to_gripper`, Z offset 보정
- **Vision** — 데이터셋 통합·학습, RealSense stem pipeline, 배포 가중치
- **Motion** — planner 시퀀스, 그리퍼, place slot 티칭, NW/SW 셀 안정화
- **Simulation / VLA** — Isaac·emulator compose profile, SmolVLA 서버 스켈레톤
- **Dashboard** — 실시간 통계·카메라 피드

---

## 시작하기

모듈은 브랜치별로 존재합니다. 필요한 기능을 checkout하거나 `dev-mini` 통합본을 참고하세요.

```bash
git clone https://github.com/yanggangyiplus/harvesting_robot_proj.git
cd harvesting_robot_proj
git fetch origin
```

### 1) Docker 개발 환경 (권장 공통 베이스)

브랜치: `feature/docker-env-setup`

```bash
git checkout feature/docker-env-setup
cd docker
cp .env.example .env          # USER_ID, ROS_DOMAIN_ID 설정
export DOCKER_BUILDKIT=1

# 사전: nvidia-smi, Container Toolkit, xhost +local:docker
# Isaac 이미지 사용 시: docker login nvcr.io

docker compose --profile dev up -d                 # ros2-lab + DSR emulator
docker compose --profile sim --profile dev up -d   # + Isaac Sim
docker compose --profile vla up -d                 # SmolVLA 서버
```

자세한 절차는 `docker/README_DOCKER.md`를 참고하세요.

### 2) Vision (YOLO)

브랜치: `feature/vision_yolo26m_strawberry`

```bash
git checkout feature/vision_yolo26m_strawberry
cd vision_yolo
pip install -r requirements.txt

python scripts/realsense_stem_pipeline.py \
  --weights-det runs/detect/runs/strawberry/yolo26m_unified_832b8/weights/best.pt \
  --weights-stem runs/segment/runs/strawberry/yolo26m_stem_roi_128b16/weights/best.pt \
  --imgsz-det 832 --imgsz-stem 128
```

학습·데이터셋 상세: `vision_yolo/README.md`

### 3) Calibration

브랜치: `feature/calibration`

```bash
git checkout feature/calibration

# Eye-to-hand
cd calibration/eye_to_hand
python calibrate_eye_to_hand.py --robot-ip 192.168.137.100 --marker-size 0.1

# Eye-in-hand (자동 수집)
cd ../eye_in_hand
python eye_in_hand_auto.py --robot-ip 192.168.137.100 --marker-size 0.1
```

결과는 `config/<timestamp>/*.npz`에 저장됩니다. Motion 쪽에서는 로컬 경로에 복사해 사용합니다.

### 4) Motion (실기)

브랜치: `feature/motion`  
패키지명: `e0509_gripper_description`

```bash
# 워크스페이스에 패키지 배치 후
cd ~/doosan_ws
colcon build --packages-select e0509_gripper_description
source install/setup.bash

ros2 launch e0509_gripper_description bringup.launch.py mode:=real host:=<robot_ip>
ros2 run e0509_gripper_description curobo_planner_node.py
ros2 run e0509_gripper_description strawberry_yolo_node.py
```

YOLO 창 조작: `1~9` lock → `s` pick 전송 → `a` auto / `h` home / `q` 종료

로컬에만 두는 파일:

```text
models/best.pt
config/calibration_eye_in_hand_*.npz
logs/
```

### 5) Dashboard

브랜치: `feature/dashboard`

```bash
cd dashboard
docker compose up -d
# http://localhost:8765
```

---

## 모듈 가이드 (요약)

### Calibration

고정/핸드 마운트 카메라와 로봇 베이스를 맞춥니다. Eye-to-hand는 `T_cam_to_base`, Eye-in-hand는 `T_cam_to_gripper`를 추정합니다. 포즈는 마커가 카메라에 평행하도록 J3/J6 고정·J4/J5 소각도를 권장합니다.

### Vision

통합 데이터셋 약 934장, 클래스 `ripe_strawberry` / `unripe_strawberry`. RealSense에서는 detect → ROI → stem seg → 그립점까지 한 스크립트로 처리합니다. 줄기 처리는 기본 ripe만 대상입니다.

### Motion Planning

`strawberry_yolo_node` 또는 `strawberry_fusion_node`가 pick pose를 내고, `curobo_planner_node`가 시퀀스를 실행합니다. Place는 `config/place_slots.yaml`의 티칭된 joint pose를 사용합니다.  
`motion_main`의 `SIMULATION_INTERFACE_SPEC_20260618.md`에 토픽·성공 판정·JSONL 스키마가 정의되어 있습니다.

> `pick_complete` ≠ 수확 성공. 줄기 파지 여부는 수기 라벨(`stem_grasp`, `detach`, `retention`)과 병행합니다.

### VLA

`docker/smolvla/serve_smolvla.py`가 `POST /predict`로 action을 반환하는 스켈레톤입니다. 모델은 `~/models/vla-model`에 두고 `SMOLVLA_URL=http://127.0.0.1:8000`으로 연동합니다.

### Simulation

- **Isaac:** compose profile `sim`
- **Gazebo:** `bringup_gazebo.launch.py` 등 (`feature/motion`)
- **스펙:** runtime JSONL 리플레이용 인터페이스 문서 (`motion_main`)

---

## FAQ

**Q. 브랜치를 전부 합쳐야 하나요?**  
A. 필수는 아닙니다. 작업 모듈 브랜치만 checkout하거나, 필요한 디렉터리만 가져와 통합하면 됩니다.

**Q. 가중치·캘리브 `.npz`가 없어요.**  
A. 용량·장비 차로 Git에서 제외된 경우가 많습니다. Vision은 `share/` 또는 학습 산출물을, Calibration은 현장 재수집 결과를 로컬에 두세요.

**Q. RealSense가 EBUSY예요.**  
A. 다른 프로세스가 카메라를 점유 중입니다. `fuser -v /dev/video*`로 확인 후 종료하세요.

**Q. Isaac / SmolVLA 빌드가 실패해요.**  
A. BuildKit 활성화(`DOCKER_BUILDKIT=1`), NGC 로그인, 디스크 여유(약 40~50GB)를 확인하세요.

**Q. Place가 실패하거나 벽과 충돌해요.**  
A. 현재 place는 고정 joint 티칭 기반입니다. 계란판을 옮기면 재티칭이 필요하고, `retreat → bin` 직행은 검증 전 사용을 피하세요.

**Q. VLA가 바로 수확에 쓰이나요?**  
A. 아직 실험용 서버입니다. 주 경로는 YOLO + cuRobo입니다.

---

## 알려진 이슈 · 디버깅

| 증상 | 확인 포인트 |
| --- | --- |
| YOLO GUI 미표시 | `opencv-python`과 `opencv-python-headless` 충돌 → headless 제거 또는 `--headless` |
| cuRobo `IK_FAIL` | wall orientation 고정, 물리적으로 닿는 과실 각도 |
| MoveLine이 성공인데 안 움직임 | TOOL 상대 이동 fake success — BASE 상대 vector 방식 로그 확인 |
| 그리퍼 stroke 신뢰 불가 | 일부 노드는 피드백이 아닌 명령값일 수 있음 |
| 줄기 파지 불안정 | finger tip 형상·마찰 한계 → 현재는 몸통 soft grasp도 병행 |

런타임 로그는 `logs/runtime/` JSONL을 우선 확인하세요. KPI 도구·수기 라벨 형식은 `motion_main` 스펙 문서를 따릅니다.

이슈 제보: GitHub Issues — [yanggangyiplus/harvesting_robot_proj](https://github.com/yanggangyiplus/harvesting_robot_proj/issues)

---

## 버전

| 버전 | 시기 | 내용 |
| --- | --- | --- |
| 0.1 | 2026.05 | Docker 환경, Calibration, Vision YOLO26m 초기 통합 |
| 0.2 | 2026.05 | cuRobo pick & place, Motion README, Dashboard |
| 0.3 | 2026.06 | fusion/scan 런타임, 시뮬 인터페이스 스펙, 실기 KPI 로깅 |

현재 저장소 `main`은 문서·미디어 중심이며, 실행 코드는 feature 브랜치에 있습니다.

---

## 참고 · 출처

- [Ultralytics YOLO](https://docs.ultralytics.com/) — YOLO26m 검출·세그멘테이션
- [NVIDIA cuRobo](https://curobo.org/) — GPU 모션 플래닝
- [Doosan Robotics ROS2](https://github.com/doosan-robotics) — E0509 제어
- [Intel RealSense](https://www.intelrealsense.com/) — RGB-D
- [LeRobot / SmolVLA](https://github.com/huggingface/lerobot) — VLA 실험
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim) — 시뮬레이션
- Vision 독립 개발 repo: [yolo26m_strawberry](https://github.com/yanggangyiplus/yolo26m_strawberry)
- 데이터셋: Qin2006, UniqueData ripe-strawberries 등 HuggingFace 공개셋 + 자체 촬영

모듈별 상세 문서

- `calibration/eye_to_hand/README_calibration.md`
- `vision_yolo/README.md`
- `README_motion.md` (`feature/motion`)
- `docker/README_DOCKER.md`
- `SIMULATION_INTERFACE_SPEC_20260618.md` (`motion_main`)

---

## 라이선스

[MIT License](LICENSE) — Copyright (c) 2026 HANGAYEONG

소프트웨어는 있는 그대로 제공되며, 로봇 실기 운용에 따른 안전 책임은 사용자에게 있습니다. 실로봇 실행 전 속도·작업공간·비상정지를 반드시 확인하세요.
