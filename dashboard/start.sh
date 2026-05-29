#!/bin/bash
# 딸기 수확 대시보드 실행 스크립트 (conda robot_env)
# RealSense + LG 웹캠 동시 스트리밍

# ── 여기만 수정하세요 ────────────────────────────────────────────────────────
# 장치 번호 확인: v4l2-ctl --list-devices  또는  ls /dev/video*
YOLO_CAM_ID=6    # 딸기 인식 카메라 장치 번호 (/dev/videoN) — 보통 RealSense
FRONT_CAM_ID=4   # 전경 카메라 장치 번호     (/dev/videoN) — 보통 LG 웹캠
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$SCRIPT_DIR/data/harvest_state.json"
mkdir -p "$SCRIPT_DIR/data"

export HARVEST_STATE_FILE="$STATE_FILE"

echo ""
echo "  딸기 수확 대시보드 시작"
echo "  → http://localhost:8765"
echo "  → YOLO 카메라: /dev/video${YOLO_CAM_ID}"
echo "  → 전경 카메라: /dev/video${FRONT_CAM_ID}"
echo ""

conda run -n robot_env --no-capture-output \
    python3 "$SCRIPT_DIR/harvest_dashboard.py" \
    --host 0.0.0.0 --port 8765 \
    --camera-id "$YOLO_CAM_ID" \
    --camera-id-1 "$FRONT_CAM_ID"
