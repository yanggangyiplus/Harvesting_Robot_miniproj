#!/usr/bin/env bash
# 데이터 수집 시작 스크립트
# 카메라 ROS2 노드 → teleop_api_server (웹 UI 포함) 순서로 실행.
#
# 사용법:
#   bash src/dashboard/start_collection.sh
#
# 브라우저에서 http://localhost:8767 접속

# ── 여기만 수정하세요 ────────────────────────────────────────────────────────
SERIAL_CAM1="215122254786"   # YOLO 인식 카메라 시리얼
SERIAL_CAM2="342622303457"   # 전경 카메라 시리얼
HOME_POSE="top_left"         # top_left / top_right / bottom_left / bottom_right
# ────────────────────────────────────────────────────────────────────────────

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -f "$WS_DIR/install/setup.bash" ]]; then
    source "$WS_DIR/install/setup.bash"
elif [[ -f /opt/ros/humble/setup.bash ]]; then
    source /opt/ros/humble/setup.bash
fi

cleanup() {
    echo "종료 중..."
    [[ -n "${CAM1_PID:-}" ]] && kill "$CAM1_PID" 2>/dev/null || true
    [[ -n "${CAM2_PID:-}" ]] && kill "$CAM2_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo ""
echo "=================================================="
echo " 텔레오퍼레이션 데이터 수집"
echo " 브라우저: http://localhost:8767"
echo "=================================================="
echo ""

# [1] 카메라 ROS2 노드 시작
echo "[1/2] RealSense 카메라 시작..."
ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera camera_name:=camera \
    enable_color:=true enable_depth:=true \
    rgb_camera.color_profile:=640x480x30 \
    depth_module.depth_profile:=640x480x30 \
    align_depth.enable:=true \
    "serial_no:='${SERIAL_CAM1}'" &
CAM1_PID=$!

ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera2 camera_name:=camera2 \
    enable_color:=true enable_depth:=false \
    rgb_camera.color_profile:=640x480x30 \
    "serial_no:='${SERIAL_CAM2}'" &
CAM2_PID=$!

echo "  카메라 워밍업 대기 (8초)..."
sleep 8

# [2] 텔레오퍼레이션 웹 서버 시작 (UI + API + 카메라 스트림 통합)
echo "[2/2] 텔레오퍼레이션 서버 시작 (포트 8767)..."
echo ""
conda run -n robot_env --no-capture-output \
    python3 "$WS_DIR/src/teleop_api_server.py" \
    --home-pose "$HOME_POSE"
