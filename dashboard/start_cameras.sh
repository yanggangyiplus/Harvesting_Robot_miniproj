#!/usr/bin/env bash
# RealSense 카메라 2대를 호스트에서 실행
# (컨테이너 내 librealsense는 D455 펌웨어와 호환되지 않아 호스트에서 띄운다)
#
# 사용법:
#   bash src/dashboard/start_cameras.sh
#
# 이 스크립트를 켜둔 상태에서 docker compose up 하면 대시보드에 카메라가 표시된다.

# ── 카메라 시리얼 ─────────────────────────────────────────────────────────────
SERIAL_CAM1="215122254786"   # YOLO 인식 카메라 (D455)
SERIAL_CAM2="342622303457"   # 전경 카메라 (D455F)
# ────────────────────────────────────────────────────────────────────────────

source /opt/ros/humble/setup.bash
# 로봇(restart.sh) 및 도커 컨테이너와 동일한 도메인이어야 토픽이 보인다
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-99}"

cleanup() {
    echo "카메라 종료 중..."
    [[ -n "${CAM1_PID:-}" ]] && kill "$CAM1_PID" 2>/dev/null || true
    [[ -n "${CAM2_PID:-}" ]] && kill "$CAM2_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "RealSense 카메라 2대 시작 중 (color-only)..."
# 주의: 카메라가 USB 2.0(480M)으로 연결되어 있어 depth+color 동시 스트리밍은
# 대역폭 부족으로 프레임이 안 나온다. color-only 로만 띄운다.
# depth 가 필요하면 카메라를 USB 3.0 포트(파란색)에 다시 연결할 것.

# 카메라 1: YOLO 인식 (color only)
ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera camera_name:=camera \
    enable_color:=true enable_depth:=false \
    rgb_camera.color_profile:=640x480x30 \
    "serial_no:='${SERIAL_CAM1}'" &
CAM1_PID=$!

# 카메라 2: 전경 (color only)
ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=camera2 camera_name:=camera2 \
    enable_color:=true enable_depth:=false \
    rgb_camera.color_profile:=640x480x30 \
    "serial_no:='${SERIAL_CAM2}'" &
CAM2_PID=$!

echo "카메라 시작됨. 종료하려면 Ctrl+C."
echo "  - YOLO 카메라: /camera/camera/color/image_raw"
echo "  - 전경 카메라: /camera2/camera2/color/image_raw"
wait
