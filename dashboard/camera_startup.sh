#!/bin/bash
source /opt/ros/humble/setup.bash

echo "Starting YOLO camera..."
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=camera \
  camera_name:=camera \
  enable_color:=true \
  enable_depth:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  align_depth.enable:=true \
  "serial_no:='215122254786'" &

sleep 5

echo "Starting overview camera..."
ros2 launch realsense2_camera rs_launch.py \
  camera_namespace:=camera2 \
  camera_name:=camera2 \
  enable_color:=true \
  enable_depth:=false \
  rgb_camera.color_profile:=640x480x30 \
  "serial_no:='342622303457'"

wait
