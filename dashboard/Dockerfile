FROM python:3.10-slim

LABEL description="딸기 수확 로봇 웹 대시보드"

WORKDIR /app

# opencv-python-headless: GUI 없는 경량 빌드 (libGL 불필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libglib2.0-0 libsm6 libxext6 \
      libusb-1.0-0 \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" websockets opencv-python-headless pyrealsense2

COPY harvest_dashboard.py .

RUN mkdir -p /data
ENV HARVEST_STATE_FILE=/data/harvest_state.json

EXPOSE 8765

CMD ["sh", "-c", "python3 harvest_dashboard.py --host 0.0.0.0 --port 8765 --camera-url-0 \"${CAMERA_URL_0:-http://localhost:8766/cam0}\" --camera-url-1 \"${CAMERA_URL_1:-http://localhost:8766/cam1}\""]
