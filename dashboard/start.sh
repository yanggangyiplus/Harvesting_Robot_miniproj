#!/bin/bash
# 딸기 수확 대시보드 + 텔레오퍼레이션 수집 통합 실행
# 브라우저: http://localhost:8765
#
# 시리얼 번호 확인: rs-enumerate-devices | grep "Serial Number"

# ── 여기만 수정하세요 ────────────────────────────────────────────────────────
SERIAL_CAM0="215122254786"   # YOLO 인식 카메라 시리얼 (D455)
SERIAL_CAM1="342622303457"   # 전경 카메라 시리얼      (D455F)
HOME_POSE="top_left"         # top_left / top_right / bottom_left / bottom_right
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -f "$WS_DIR/install/setup.bash" ]]; then
    source "$WS_DIR/install/setup.bash"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
fi

cleanup() {
    echo "종료 중..."
    [[ -n "${TELEOP_PID:-}" ]] && kill "$TELEOP_PID" 2>/dev/null || true
    [[ -n "${CAM0_PID:-}"   ]] && kill "$CAM0_PID"   2>/dev/null || true
    [[ -n "${CAM1_PID:-}"   ]] && kill "$CAM1_PID"   2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "  딸기 수확 대시보드 + 텔레오퍼레이션 수집"
echo "  → http://localhost:8765"
echo ""

# [1] 카메라 ROS2 노드 (bag 녹화 + 대시보드 MJPEG 소스)
echo "[1/3] 카메라 시작..."
ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera camera_name:=camera \
    enable_color:=true enable_depth:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 \
    align_depth.enable:=true \
    "serial_no:='${SERIAL_CAM0}'" &
CAM0_PID=$!

ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera2 camera_name:=camera2 \
    enable_color:=true enable_depth:=false \
    rgb_camera.color_profile:=640x480x30 \
    "serial_no:='${SERIAL_CAM1}'" &
CAM1_PID=$!

echo "  카메라 워밍업 대기 (8초)..."
sleep 8

# [2] 텔레오퍼레이션 API 서버 (포트 8767, 백그라운드)
#     로봇 제어 + bag 녹화 관리 + 카메라 MJPEG 스트림 서빙
echo "[2/3] 텔레오퍼레이션 API 서버 시작 (포트 8767)..."
conda run -n robot_env --no-capture-output \
    python3 "$WS_DIR/src/teleop_api_server.py" \
    --home-pose "$HOME_POSE" &
TELEOP_PID=$!
sleep 3

# [3] 대시보드 (포트 8765) — 카메라는 8767 MJPEG 스트림 사용
echo "[3/3] 대시보드 시작 (포트 8765)..."
echo ""
conda run -n robot_env --no-capture-output \
    python3 "$SCRIPT_DIR/harvest_dashboard.py" \
    --host 0.0.0.0 --port 8765 \
    --camera-url-0 "http://localhost:8767/camera/0" \
    --camera-url-1 "http://localhost:8767/camera/1"
