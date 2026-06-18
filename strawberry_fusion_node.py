#!/usr/bin/env python3
"""
Strawberry Seg+Pose Fusion Detection Node

Seg 모델(과실 마스크·익음도) + Pose 모델(줄기 3-키포인트)을 결합하여
ripe 딸기의 줄기 파지 위치를 계산하고 수확 후보를 퍼블리시합니다.

Fusion 규칙: Pose keypoint(KP0/KP1 우선)와 Seg 마스크를 매칭.
수확 후보: class=ripe + Pose 매칭 성공 + stable track lock.

Published topics:
  /strawberry/detection/pick_pose      (PoseStamped)           — ripe 후보 1개씩
  /strawberry/detection/scene_positions (Float64MultiArray)    — 모든 ripe 중심 [x,y,z, ...]
"""

import os
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as ScipyR
from runtime_jsonl_logger import RuntimeJsonlLogger

# ── Joint names (Doosan E0509) ────────────────────────────────────────────────
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# ── Seg model class IDs ───────────────────────────────────────────────────────
RIPE_CLASS_ID = 0   # 0=ripe, 1=unripe, 2=sick
RIPE_RED_RATIO_MIN = 0.28
RIPE_STRONG_RED_RATIO_MIN = 0.12
RIPE_SAT_MEAN_MIN = 105.0
RED_HSV_RANGES = (
    (np.array([0, 90, 50], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)),
    (np.array([168, 90, 50], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
)
STRONG_RED_HSV_RANGES = (
    (np.array([0, 120, 60], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8)),
    (np.array([168, 120, 60], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
)

# ── Visualization ─────────────────────────────────────────────────────────────
COLOR_RIPE   = (0, 255, 0)
COLOR_UNRIPE = (0, 220, 220)
COLOR_SICK   = (0, 0, 220)
COLOR_MATCH  = (0, 255, 255)
COLOR_KP0    = (0, 140, 255)   # stem_base — orange
COLOR_KP1    = (0, 0, 255)     # stem_mid  — red
COLOR_KP2    = (0, 200, 0)     # stem_tip  — green
SEG_ALPHA    = 0.35


# ── E0509 FK (calibration-identical) ─────────────────────────────────────────
def _T(xyz, rpy, q=0.0):
    M = np.eye(4)
    M[:3, 3] = xyz
    R_fixed = ScipyR.from_euler('xyz', rpy).as_matrix()
    R_joint  = ScipyR.from_euler('z',   q  ).as_matrix()
    M[:3, :3] = R_fixed @ R_joint
    return M


def e0509_fk(q_rad):
    T = np.eye(4)
    T = T @ _T([0,      0,       0.2045], [0,         0,           0      ], q_rad[0])
    T = T @ _T([0,      0,       0     ], [0,        -np.pi/2,    -np.pi/2], q_rad[1])
    T = T @ _T([0.373,  0,       0     ], [0,         0,           np.pi/2], q_rad[2])
    T = T @ _T([0,     -0.373,   0     ], [np.pi/2,   0,           0      ], q_rad[3])
    T = T @ _T([0,      0,       0     ], [-np.pi/2,  0,           0      ], q_rad[4])
    T = T @ _T([0,     -0.1725,  0     ], [np.pi/2,   0,           0      ], q_rad[5])
    T = T @ _T([0,      0,       0     ], [np.pi,    -np.pi/2,     0      ])  # TCP fixed
    return T


def stem_vec_to_quat_xyzw(stem_vec_3d: np.ndarray) -> np.ndarray:
    """Quaternion [x,y,z,w] that aligns gripper Z-axis with stem direction.

    stem_vec_3d: KP2_3d - KP0_3d in base_link frame.
    Falls back to identity when vector is degenerate.
    NOTE: curobo_planner_node currently uses its own WALL_QUAT — this
    orientation is stored in pick_pose for future dynamic stem grasping.
    """
    v = stem_vec_3d.copy().astype(float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    v /= n

    z_ref = np.array([0.0, 0.0, 1.0])
    axis  = np.cross(z_ref, v)
    axis_n = np.linalg.norm(axis)
    if axis_n < 1e-6:
        # Parallel → identity; Anti-parallel → 180° around Y
        return np.array([0.0, 0.0, 0.0, 1.0]) if np.dot(z_ref, v) > 0 \
               else np.array([0.0, 1.0, 0.0, 0.0])

    axis /= axis_n
    angle = np.arccos(np.clip(np.dot(z_ref, v), -1.0, 1.0))
    return ScipyR.from_rotvec(angle * axis).as_quat()  # [x,y,z,w]


def hsv_mask_bgr(image_bgr, ranges):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lower, upper in ranges:
        part = cv2.inRange(hsv, lower, upper)
        mask = part if mask is None else (mask | part)
    return mask, hsv


def ripe_metrics_in_polygon(image_bgr, polygon):
    """Red evidence inside the segmentation mask.

    Seg class can flicker close-up; require visible red pixels before allowing
    a ripe pick target to be published.
    """
    h, w = image_bgr.shape[:2]
    if len(polygon) < 3:
        return {"red_ratio": 0.0, "strong_red_ratio": 0.0, "sat_mean": 0.0}

    poly = polygon.astype(np.int32)
    mask_poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_poly, [poly], 255)
    area = float(np.count_nonzero(mask_poly))
    if area < 1.0:
        return {"red_ratio": 0.0, "strong_red_ratio": 0.0, "sat_mean": 0.0}

    red_mask, hsv = hsv_mask_bgr(image_bgr, RED_HSV_RANGES)
    strong_mask, _ = hsv_mask_bgr(image_bgr, STRONG_RED_HSV_RANGES)

    b, g, r = cv2.split(image_bgr)
    dominance = ((r.astype(np.float32) > g.astype(np.float32) * 1.20) &
                 (r.astype(np.float32) > b.astype(np.float32) * 1.20))
    strong_mask = strong_mask & dominance.astype(np.uint8) * 255

    in_poly = mask_poly > 0
    red_in_poly = (red_mask > 0) & in_poly
    strong_in_poly = (strong_mask > 0) & in_poly
    sat_mean = float(np.mean(hsv[:, :, 1][red_in_poly])) if np.any(red_in_poly) else 0.0
    return {
        "red_ratio": float(np.count_nonzero(red_in_poly)) / area,
        "strong_red_ratio": float(np.count_nonzero(strong_in_poly)) / area,
        "sat_mean": sat_mean,
    }


def ripe_metrics_pass(metrics):
    return (
        metrics["red_ratio"] >= RIPE_RED_RATIO_MIN
        and metrics["strong_red_ratio"] >= RIPE_STRONG_RED_RATIO_MIN
        and metrics["sat_mean"] >= RIPE_SAT_MEAN_MIN
    )


# ─────────────────────────────────────────────────────────────────────────────
class StrawberryFusionNode(Node):

    def __init__(self):
        super().__init__("strawberry_fusion_node")
        self.runtime_log = RuntimeJsonlLogger(self.get_name())

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter(
            "seg_model",
            "~/Downloads/share_yolo/share_yolo/strawberry_seg_best.pt")
        self.declare_parameter(
            "pose_model",
            "~/Downloads/share_yolo/share_yolo/strawberry_pose_best.pt")
        self.declare_parameter(
            "calib_npz",
            "~/doosan_ws/src/e0509_gripper_description/config/calibration_eye_in_hand_1.npz")
        self.declare_parameter("yolo_conf",    0.25)
        self.declare_parameter("kp_conf_min",  0.40)   # keypoint visibility threshold
        self.declare_parameter("pick_kp_conf_min", 0.60)
        self.declare_parameter("require_all_stem_keypoints", True)
        self.declare_parameter("stem_segment_min_m", 0.005)
        self.declare_parameter("stem_segment_max_m", 0.100)
        self.declare_parameter("stem_total_max_m", 0.160)
        # KP0 바로 위는 과실/잎과 닿기 쉽고, 꺾인 줄기는 전체 chord를 따르면
        # 측방으로 빗나간다. KP0->KP1 국소 구간의 80% 지점을 목표로 삼는다.
        # 아래 계산에서 segment 길이의 0.8배로 cap되므로 0.080m는 상한값이다.
        self.declare_parameter("stem_grasp_offset_from_kp0_m", 0.080)
        self.declare_parameter("stem_grasp_direction_mode", "kp0_to_kp1")
        self.declare_parameter("grasp_target_base_z_trim_m", 0.000)
        self.declare_parameter("infer_every",  3)       # run inference every N camera frames
        self.declare_parameter("stable_hits_required", 4)
        self.declare_parameter("target_position_window_size", 9)
        self.declare_parameter("target_position_min_samples", 7)
        self.declare_parameter("target_position_max_spread_m", 0.012)
        self.declare_parameter("track_match_distance_m", 0.035)
        self.declare_parameter("track_ttl_sec", 1.5)
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("target_lock_enabled", True)
        self.declare_parameter("target_lock_ttl_sec", 3.0)
        self.declare_parameter("target_switch_distance_m", 0.090)
        self.declare_parameter("show_display", True)
        self.declare_parameter("show_cell_grid", True)

        seg_path   = os.path.expanduser(self.get_parameter("seg_model").value)
        pose_path  = os.path.expanduser(self.get_parameter("pose_model").value)
        calib_path = os.path.expanduser(self.get_parameter("calib_npz").value)
        self._conf    = self.get_parameter("yolo_conf").value
        self._kp_min  = self.get_parameter("kp_conf_min").value
        self._pick_kp_min = float(self.get_parameter("pick_kp_conf_min").value)
        self._require_all_stem_kps = bool(
            self.get_parameter("require_all_stem_keypoints").value)
        self._stem_segment_min_m = float(
            self.get_parameter("stem_segment_min_m").value)
        self._stem_segment_max_m = float(
            self.get_parameter("stem_segment_max_m").value)
        self._stem_total_max_m = float(
            self.get_parameter("stem_total_max_m").value)
        self._stem_grasp_offset_m = max(
            0.0, float(self.get_parameter("stem_grasp_offset_from_kp0_m").value))
        self._stem_grasp_direction_mode = str(
            self.get_parameter("stem_grasp_direction_mode").value)
        self._grasp_target_base_z_trim_m = float(
            self.get_parameter("grasp_target_base_z_trim_m").value)
        self._infer_n = max(1, self.get_parameter("infer_every").value)
        self._stable_hits = max(1, int(self.get_parameter("stable_hits_required").value))
        self._position_window_size = max(
            3, int(self.get_parameter("target_position_window_size").value))
        self._position_min_samples = min(
            self._position_window_size,
            max(3, int(self.get_parameter("target_position_min_samples").value)),
        )
        self._position_max_spread_m = float(
            self.get_parameter("target_position_max_spread_m").value)
        self._track_match_dist = float(self.get_parameter("track_match_distance_m").value)
        self._track_ttl_sec = float(self.get_parameter("track_ttl_sec").value)
        self._publish_period_sec = float(self.get_parameter("publish_period_sec").value)
        self._target_lock_enabled = bool(self.get_parameter("target_lock_enabled").value)
        self._target_lock_ttl_sec = float(self.get_parameter("target_lock_ttl_sec").value)
        self._target_switch_dist = float(self.get_parameter("target_switch_distance_m").value)
        self._display = self.get_parameter("show_display").value
        self._show_cell_grid = bool(self.get_parameter("show_cell_grid").value)

        # ── calibration ───────────────────────────────────────────────────────
        self.get_logger().info(f"Loading calibration: {calib_path}")
        calib = np.load(calib_path)
        self.T_cam2gripper = calib['T_cam_to_gripper']
        self.get_logger().info(
            f"T_cam2gripper translation(mm): {self.T_cam2gripper[:3,3]*1000}")

        # ── joint state ───────────────────────────────────────────────────────
        self.current_joints = None
        self._jlock = threading.Lock()

        # ── YOLO ──────────────────────────────────────────────────────────────
        self.get_logger().info(f"Loading seg model:  {seg_path}")
        self.seg_model = YOLO(seg_path)
        self.get_logger().info(f"Loading pose model: {pose_path}")
        self.pose_model = YOLO(pose_path)
        self.get_logger().info(
            f"Seg classes: {self.seg_model.names} | "
            f"Pose classes: {self.pose_model.names}")

        # ── RealSense ─────────────────────────────────────────────────────────
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        for attempt in range(1, 6):
            try:
                self.pipeline.start(cfg)
                break
            except RuntimeError as exc:
                if attempt == 5:
                    raise
                self.get_logger().warn(
                    f"RealSense busy (attempt {attempt}/5): {exc} — retrying in 2s")
                import time as _t; _t.sleep(2.0)
        self.align = rs.align(rs.stream.color)
        self.get_logger().info("RealSense started.")

        # ── ROS2 I/O ──────────────────────────────────────────────────────────
        self.joint_sub = self.create_subscription(
            JointState, "/dsr01/joint_states", self._joint_cb, 10)
        self.pick_pub  = self.create_publisher(
            PoseStamped, "/strawberry/detection/pick_pose", 20)
        self.scene_pub = self.create_publisher(
            Float64MultiArray, "/strawberry/detection/scene_positions", 10)

        self._frame_n = 0
        self._tracks = {}
        self._next_track_id = 1
        self._active_track_id = None
        self._active_last_seen = 0.0
        self._last_reject_log = {}
        self._last_seg_items = []   # cached from last inference for smooth viz
        self._last_pose_items = []
        self.timer = self.create_timer(1.0 / 30.0, self._loop)
        self.get_logger().info(
            "StrawberryFusionNode ready.  q=quit in display window")
        self.get_logger().info(f"Runtime JSONL: {self.runtime_log.path}")
        self.get_logger().info(
            f"Target stabilization: median window={self._position_window_size}, "
            f"min_samples={self._position_min_samples}, "
            f"max_spread={self._position_max_spread_m*1000:.0f}mm")
        self.get_logger().info(
            f"Stem quality guard: kp_conf>={self._pick_kp_min:.2f}, "
            f"all_keypoints={self._require_all_stem_kps}, "
            f"segment={self._stem_segment_min_m*1000:.0f}.."
            f"{self._stem_segment_max_m*1000:.0f}mm")
        self.get_logger().info(
            f"Stem grasp target: mode={self._stem_grasp_direction_mode}, "
            f"offset up to {self._stem_grasp_offset_m*1000:.0f}mm, "
            f"base-Z trim={self._grasp_target_base_z_trim_m*1000:+.0f}mm")
        self.runtime_log.log(
            "node_start",
            pipeline_role="seg_pose_fusion_and_target_generation",
            models={"seg": seg_path, "pose": pose_path},
            calibration_npz=calib_path,
            parameters={
                "yolo_conf": self._conf,
                "kp_conf_min": self._kp_min,
                "pick_kp_conf_min": self._pick_kp_min,
                "require_all_stem_keypoints": self._require_all_stem_kps,
                "stem_segment_min_m": self._stem_segment_min_m,
                "stem_segment_max_m": self._stem_segment_max_m,
                "stem_total_max_m": self._stem_total_max_m,
                "stem_grasp_offset_from_kp0_m": self._stem_grasp_offset_m,
                "stem_grasp_direction_mode": self._stem_grasp_direction_mode,
                "grasp_target_base_z_trim_m": self._grasp_target_base_z_trim_m,
                "infer_every": self._infer_n,
                "stable_hits_required": self._stable_hits,
                "target_position_window_size": self._position_window_size,
                "target_position_min_samples": self._position_min_samples,
                "target_position_max_spread_m": self._position_max_spread_m,
                "track_match_distance_m": self._track_match_dist,
            },
        )

    def _update_track(self, pos_base: np.ndarray, quat_xyzw: np.ndarray, quality: dict):
        """Simple 3-D nearest-neighbor tracker to suppress frame flicker."""
        now = time.monotonic()
        stale_ids = [
            tid for tid, track in self._tracks.items()
            if now - track["last_seen"] > self._track_ttl_sec
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        best_id = None
        best_dist = float("inf")
        for tid, track in self._tracks.items():
            dist = float(np.linalg.norm(pos_base - track["pos"]))
            if dist < best_dist:
                best_id = tid
                best_dist = dist

        if best_id is None or best_dist > self._track_match_dist:
            best_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[best_id] = {
                "pos": pos_base.astype(float),
                "quat": quat_xyzw.astype(float),
                "position_history": [],
                "position_spread_m": float("inf"),
                "hits": 0,
                "last_seen": now,
                "last_pub": 0.0,
                "quality": {},
            }

        track = self._tracks[best_id]
        history = track["position_history"]
        history.append(pos_base.astype(float))
        del history[:-self._position_window_size]
        history_array = np.asarray(history)
        median_pos = np.median(history_array, axis=0)
        track["pos"] = median_pos
        track["position_spread_m"] = float(
            np.max(np.linalg.norm(history_array - median_pos, axis=1)))
        track["quat"] = quat_xyzw.astype(float)
        track["quality"] = quality
        track["hits"] += 1
        track["last_seen"] = now
        return best_id, track

    def _track_is_stable(self, track) -> bool:
        return (
            track["hits"] >= self._stable_hits
            and len(track["position_history"]) >= self._position_min_samples
            and track["position_spread_m"] <= self._position_max_spread_m
        )

    def _should_publish_track(self, track) -> bool:
        now = time.monotonic()
        if not self._track_is_stable(track):
            return False
        if now - track["last_pub"] < self._publish_period_sec:
            return False
        track["last_pub"] = now
        return True

    def _select_active_track(self, candidates):
        """Pick one stable target and hold it briefly to suppress target swaps."""
        now = time.monotonic()
        candidates = [c for c in candidates if self._track_is_stable(c["track"])]
        if not candidates:
            if now - self._active_last_seen > self._target_lock_ttl_sec:
                self._active_track_id = None
            return None

        if not self._target_lock_enabled:
            return min(candidates, key=lambda c: c["center_dist_px"])

        active = None
        if self._active_track_id is not None:
            for c in candidates:
                if c["track_id"] == self._active_track_id:
                    active = c
                    break
            if active is not None:
                self._active_last_seen = now
                return active

        # If the detector temporarily changed the track id, keep a nearby stable
        # candidate instead of jumping to another fruit in the same close-up view.
        if self._active_track_id in self._tracks:
            prev = self._tracks[self._active_track_id]["pos"]
            nearby = [
                c for c in candidates
                if float(np.linalg.norm(c["track"]["pos"] - prev)) <= self._target_switch_dist
            ]
            if nearby:
                active = min(nearby, key=lambda c: c["center_dist_px"])
                self._active_track_id = active["track_id"]
                self._active_last_seen = now
                return active

        active = min(candidates, key=lambda c: c["center_dist_px"])
        self._active_track_id = active["track_id"]
        self._active_last_seen = now
        return active

    def _publish_pick_track(self, track):
        pmsg = PoseStamped()
        pmsg.header.frame_id    = "base_link"
        pmsg.header.stamp       = self.get_clock().now().to_msg()
        pmsg.pose.position.x    = float(track["pos"][0])
        pmsg.pose.position.y    = float(track["pos"][1])
        pmsg.pose.position.z    = float(track["pos"][2])
        pmsg.pose.orientation.x = float(track["quat"][0])
        pmsg.pose.orientation.y = float(track["quat"][1])
        pmsg.pose.orientation.z = float(track["quat"][2])
        pmsg.pose.orientation.w = float(track["quat"][3])
        self.pick_pub.publish(pmsg)
        self.runtime_log.log(
            "stable_pick_target_published",
            topic="/strawberry/detection/pick_pose",
            frame_id="base_link",
            target_pos_m=track["pos"],
            target_quat_xyzw=track["quat"],
            samples=len(track["position_history"]),
            spread_m=track["position_spread_m"],
            target_quality=track["quality"],
        )
        self.get_logger().info(
            "Published stable pick target "
            f"xyz=({track['pos'][0]:.3f},{track['pos'][1]:.3f},{track['pos'][2]:.3f})m "
            f"samples={len(track['position_history'])} "
            f"spread={track['position_spread_m']*1000:.1f}mm "
            f"stem_offset={track['quality'].get('grasp_offset_from_kp0_m', 0.0)*1000:.1f}mm "
            f"kp_conf={track['quality'].get('kp_conf', [])}")

    def _reject_target(self, reason: str, **data):
        """Rate-limited diagnostics for unsafe or ambiguous stem targets."""
        now = time.monotonic()
        if now - self._last_reject_log.get(reason, 0.0) < 1.0:
            return
        self._last_reject_log[reason] = now
        self.runtime_log.log("pick_target_rejected", reason=reason, **data)
        self.get_logger().warn(f"Pick target rejected: {reason}")

    # ── joint callback ────────────────────────────────────────────────────────
    def _joint_cb(self, msg: JointState):
        jmap = {n: p for n, p in zip(msg.name, msg.position)}
        try:
            with self._jlock:
                self.current_joints = [jmap[n] for n in JOINT_NAMES]
        except KeyError:
            pass

    # ── coordinate transform ──────────────────────────────────────────────────
    def _cam_to_base(self, pt_cam_xyz):
        with self._jlock:
            joints = self.current_joints
        if joints is None:
            return None
        T_g2b   = e0509_fk(joints)
        T_total = T_g2b @ self.T_cam2gripper
        return (T_total @ np.array([*pt_cam_xyz, 1.0]))[:3]

    # ── depth helpers ─────────────────────────────────────────────────────────
    def _depth_at_px(self, depth_frame, u, v, radius=4) -> float | None:
        """Median depth in a small region around pixel (u, v)."""
        fw, fh = depth_frame.get_width(), depth_frame.get_height()
        samples = []
        for dv in range(-radius, radius + 1):
            for du in range(-radius, radius + 1):
                pu, pv = int(u) + du, int(v) + dv
                if 0 <= pu < fw and 0 <= pv < fh:
                    d = depth_frame.get_distance(pu, pv)
                    if 0.05 < d < 3.0:
                        samples.append(d)
        return float(np.median(samples)) if samples else None

    def _px_to_3d(self, depth_frame, intr, u, v, radius=4):
        """Return 3-D point in base_link, or None on failure."""
        d = self._depth_at_px(depth_frame, u, v, radius)
        if d is None:
            return None
        pt_cam = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], d)
        return self._cam_to_base(pt_cam)

    def _polygon_centroid_3d(self, depth_frame, intr, polygon):
        """3-D centroid of a seg polygon (via moment centroid pixel)."""
        if len(polygon) < 3:
            return None
        M = cv2.moments(polygon.astype(np.float32))
        if M["m00"] < 1e-6:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return self._px_to_3d(depth_frame, intr, cx, cy)

    @staticmethod
    def _polygon_centroid_px(polygon):
        if len(polygon) < 3:
            return None
        M = cv2.moments(polygon.astype(np.float32))
        if M["m00"] < 1e-6:
            return None
        return np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=float)

    def _match_seg_for_pose(self, pose_bbox, kps_np, seg_items):
        """Match a pose detection to one seg mask.

        Close-up views often contain overlapping strawberries/leaves.  The old
        rule used only the pose bbox center, which can jump between adjacent
        masks. Keypoint containment is useful matching evidence, but stem-side
        KP0/KP1 are normally outside a fruit-body segmentation mask and must
        not be required for accepting an otherwise valid ripe match.
        """
        pose_cx = (pose_bbox[0] + pose_bbox[2]) / 2.0
        pose_cy = (pose_bbox[1] + pose_bbox[3]) / 2.0
        pose_center = np.array([pose_cx, pose_cy], dtype=float)

        points = [("center", pose_center, 1.0)]
        for ki, weight in ((0, 5.0), (1, 3.0), (2, 1.0)):
            if ki < len(kps_np):
                kx, ky, kconf = kps_np[ki]
                if kconf >= self._kp_min:
                    points.append((f"kp{ki}", np.array([kx, ky], dtype=float), weight))

        best = None
        for seg_idx, (cls_id, polygon) in enumerate(seg_items):
            if len(polygon) < 3:
                continue
            poly = polygon.astype(np.float32)
            score = 0.0
            evidence = []
            for name, pt, weight in points:
                inside = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
                if inside >= 0:
                    score += weight
                    evidence.append(name)
            if score <= 0.0:
                continue

            centroid = self._polygon_centroid_px(polygon)
            if centroid is not None:
                # Tie-breaker: closer mask centroid to pose center.  Keep small
                # influence so KP containment dominates.
                score -= min(float(np.linalg.norm(pose_center - centroid)) / 250.0, 1.0)

            if best is None or score > best[0]:
                best = (score, cls_id, polygon, seg_idx, ",".join(evidence))

        if best is None:
            return None
        return {
            "score": best[0],
            "cls_id": best[1],
            "polygon": best[2],
            "seg_idx": best[3],
            "evidence": best[4],
        }

    # ── main loop ─────────────────────────────────────────────────────────────
    def _loop(self):
        try:
            frames  = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f or not depth_f:
                return

            self._frame_n += 1
            img  = np.asanyarray(color_f.get_data())
            intr = depth_f.profile.as_video_stream_profile().intrinsics
            vis  = img.copy()

            # Joint state guard
            with self._jlock:
                joint_ok = self.current_joints is not None
            if not joint_ok:
                cv2.putText(vis, "NO JOINT STATE — waiting",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self._show(vis)
                return

            # ── YOLO inference (throttled) — use cached results on skipped frames ──
            run_infer = (self._frame_n % self._infer_n == 0)
            if run_infer:
                seg_res  = self.seg_model(img,  conf=self._conf, verbose=False)[0]
                pose_res = self.pose_model(img, conf=self._conf, verbose=False)[0]

                seg_items = []
                if seg_res.masks is not None and seg_res.boxes is not None:
                    for i, polygon in enumerate(seg_res.masks.xy):
                        cls_id = int(seg_res.boxes.cls[i].item())
                        seg_items.append((cls_id, polygon))

                pose_items = []
                if pose_res.keypoints is not None and pose_res.boxes is not None:
                    for i, kps in enumerate(pose_res.keypoints.data):
                        bbox   = pose_res.boxes.xyxy[i].cpu().numpy()
                        kps_np = kps.cpu().numpy()
                        pose_items.append((bbox, kps_np))

                self._last_seg_items  = seg_items
                self._last_pose_items = pose_items
            else:
                # Reuse last frame's detections so the overlay doesn't flicker
                seg_items  = self._last_seg_items
                pose_items = self._last_pose_items

            # ── draw seg overlays ─────────────────────────────────────────────
            cls_color = {0: COLOR_RIPE, 1: COLOR_UNRIPE, 2: COLOR_SICK}
            overlay   = vis.copy()
            for cls_id, polygon in seg_items:
                if len(polygon) >= 3:
                    color = cls_color.get(cls_id, (200, 200, 200))
                    cv2.fillPoly(overlay, [polygon.astype(np.int32)], color)
            cv2.addWeighted(overlay, SEG_ALPHA, vis, 1 - SEG_ALPHA, 0, vis)
            for cls_id, polygon in seg_items:
                if len(polygon) >= 3:
                    cv2.polylines(vis, [polygon.astype(np.int32)],
                                  True, cls_color.get(cls_id, (200, 200, 200)), 1)

            # ── scene positions (only on fresh inference frames) ─────────────
            if run_infer:
                scene_flat = []
                for cls_id, polygon in seg_items:
                    if cls_id == RIPE_CLASS_ID and len(polygon) >= 3:
                        pt3d = self._polygon_centroid_3d(depth_f, intr, polygon)
                        if pt3d is not None:
                            scene_flat.extend([float(pt3d[0]), float(pt3d[1]), float(pt3d[2])])
                if scene_flat:
                    smsg = Float64MultiArray()
                    smsg.data = scene_flat
                    self.scene_pub.publish(smsg)
                    self.runtime_log.log(
                        "scene_positions_published",
                        topic="/strawberry/detection/scene_positions",
                        positions_m=np.asarray(scene_flat).reshape(-1, 3),
                    )

            # ── fusion: match each pose detection to a seg mask ───────────────
            kp_colors = [COLOR_KP0, COLOR_KP1, COLOR_KP2]
            kp_labels = ["KP0", "KP1", "KP2"]
            ripe_candidates = []

            for pose_bbox, kps_np in pose_items:
                pose_cx = (pose_bbox[0] + pose_bbox[2]) / 2.0
                pose_cy = (pose_bbox[1] + pose_bbox[3]) / 2.0

                # Draw pose bounding box
                cv2.rectangle(vis,
                    (int(pose_bbox[0]), int(pose_bbox[1])),
                    (int(pose_bbox[2]), int(pose_bbox[3])),
                    (200, 200, 0), 1)

                # Draw visible keypoints
                for ki in range(min(3, len(kps_np))):
                    kx, ky, kconf = kps_np[ki]
                    if kconf >= self._kp_min:
                        cv2.circle(vis, (int(kx), int(ky)), 5, kp_colors[ki], -1)
                        cv2.putText(vis, kp_labels[ki],
                                    (int(kx) + 6, int(ky) - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, kp_colors[ki], 1)

                # Find which seg mask belongs to this pose detection.
                match = self._match_seg_for_pose(pose_bbox, kps_np, seg_items)
                if match is None:
                    cv2.putText(vis, "no-seg",
                                (int(pose_cx), int(pose_cy) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
                    continue
                matched_cls = int(match["cls_id"])

                cls_str = {0: "ripe", 1: "unripe", 2: "sick"}.get(matched_cls, str(matched_cls))
                cv2.putText(vis, f"[{cls_str}:{match['evidence']}]",
                            (int(pose_bbox[0]), int(pose_bbox[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            cls_color.get(matched_cls, (200, 200, 200)), 1)

                if matched_cls != RIPE_CLASS_ID:
                    continue  # only harvest ripe

                ripe_metrics = ripe_metrics_in_polygon(img, match["polygon"])
                if not ripe_metrics_pass(ripe_metrics):
                    cv2.putText(
                        vis,
                        "skip ripe-low-red "
                        f"R:{ripe_metrics['red_ratio']:.2f} "
                        f"SR:{ripe_metrics['strong_red_ratio']:.2f} "
                        f"S:{ripe_metrics['sat_mean']:.0f}",
                        (int(pose_bbox[0]), int(pose_bbox[3]) + 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.36,
                        (0, 0, 220),
                        1,
                    )
                    continue

                kp_conf = [
                    float(kps_np[ki][2]) if ki < len(kps_np) else 0.0
                    for ki in range(3)
                ]
                match_evidence = set(match["evidence"].split(","))
                # The seg mask describes the fruit body, while KP0/KP1 are
                # stem-side points and are expected to lie outside that mask.
                # Do not reject center-only matches here. Target safety still
                # depends on ripe metrics, keypoint confidence, 3D stem
                # geometry, and stable multi-frame tracking below.

                required_kps = (0, 1, 2) if self._require_all_stem_kps else (0,)
                low_conf_kps = [ki for ki in required_kps if kp_conf[ki] < self._pick_kp_min]
                if low_conf_kps:
                    if run_infer:
                        self._reject_target(
                            "stem_keypoint_low_confidence",
                            low_conf_keypoints=low_conf_kps,
                            kp_conf=kp_conf,
                            match_evidence=sorted(match_evidence),
                            ripe_metrics=ripe_metrics,
                        )
                    continue

                # ── compute 3D keypoint positions ─────────────────────────────
                # KP0 = stem_base (grasp target — nearest to fruit)
                # KP1 = stem_mid
                # KP2 = stem_tip (direction reference)
                kp3d = {}
                for ki in range(min(3, len(kps_np))):
                    kx, ky, kconf = kps_np[ki]
                    if kconf >= self._pick_kp_min:
                        pt3d = self._px_to_3d(depth_f, intr, kx, ky, radius=6)
                        if pt3d is not None:
                            kp3d[ki] = pt3d

                missing_depth_kps = [ki for ki in required_kps if ki not in kp3d]
                if missing_depth_kps:
                    if run_infer:
                        self._reject_target(
                            "stem_keypoint_depth_invalid",
                            missing_depth_keypoints=missing_depth_kps,
                            kp_conf=kp_conf,
                            match_evidence=sorted(match_evidence),
                        )
                    cv2.putText(vis, "no-depth",
                                (int(pose_cx), int(pose_cy) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 220), 1)
                    continue

                stem_vec = kp3d[2] - kp3d[0] if 2 in kp3d else None

                segment_lengths = {}
                if 0 in kp3d and 1 in kp3d:
                    segment_lengths["kp0_kp1_m"] = float(np.linalg.norm(kp3d[1] - kp3d[0]))
                if 1 in kp3d and 2 in kp3d:
                    segment_lengths["kp1_kp2_m"] = float(np.linalg.norm(kp3d[2] - kp3d[1]))
                if 0 in kp3d and 2 in kp3d:
                    segment_lengths["kp0_kp2_m"] = float(np.linalg.norm(kp3d[2] - kp3d[0]))

                invalid_segments = {
                    name: length for name, length in segment_lengths.items()
                    if name != "kp0_kp2_m"
                    and not self._stem_segment_min_m <= length <= self._stem_segment_max_m
                }
                total_length = segment_lengths.get("kp0_kp2_m")
                if total_length is not None and total_length > self._stem_total_max_m:
                    invalid_segments["kp0_kp2_m"] = total_length
                if invalid_segments:
                    if run_infer:
                        self._reject_target(
                            "stem_geometry_implausible",
                            invalid_segments=invalid_segments,
                            segment_lengths=segment_lengths,
                            kp_conf=kp_conf,
                            kp3d_base_m=kp3d,
                            match_evidence=sorted(match_evidence),
                        )
                    continue

                # The grasp point is close to KP0, so use the local KP0->KP1
                # tangent. KP0->KP2 is only a chord and shifts the target
                # sideways when the stem is bent.
                kp0_to_kp1 = kp3d[1] - kp3d[0]
                kp0_to_kp2 = kp3d[2] - kp3d[0]
                kp0_to_kp1_len = float(np.linalg.norm(kp0_to_kp1))
                kp0_to_kp2_len = float(np.linalg.norm(kp0_to_kp2))
                use_local_direction = (
                    self._stem_grasp_direction_mode == "kp0_to_kp1"
                    and kp0_to_kp1_len > 1e-6
                )
                grasp_vector = kp0_to_kp1 if use_local_direction else kp0_to_kp2
                grasp_vector_len = (
                    kp0_to_kp1_len if use_local_direction else kp0_to_kp2_len
                )
                if grasp_vector_len <= 1e-6:
                    if run_infer:
                        self._reject_target(
                            "stem_grasp_direction_degenerate",
                            mode=self._stem_grasp_direction_mode,
                            kp3d_base_m=kp3d,
                        )
                    continue
                grasp_step_m = min(
                    self._stem_grasp_offset_m,
                    0.8 * grasp_vector_len,
                )
                grasp_direction = grasp_vector / grasp_vector_len
                grasp_pt = kp3d[0] + grasp_step_m * grasp_direction
                grasp_pt[2] += self._grasp_target_base_z_trim_m

                bend_angle_deg = 0.0
                if kp0_to_kp1_len > 1e-6 and kp0_to_kp2_len > 1e-6:
                    bend_cos = float(np.dot(
                        kp0_to_kp1 / kp0_to_kp1_len,
                        kp0_to_kp2 / kp0_to_kp2_len,
                    ))
                    bend_angle_deg = float(np.degrees(np.arccos(np.clip(
                        bend_cos, -1.0, 1.0))))

                target_quality = {
                    "kp_conf": kp_conf,
                    "kp_pixels": [
                        [float(kps_np[ki][0]), float(kps_np[ki][1])]
                        for ki in range(min(3, len(kps_np)))
                    ],
                    "kp3d_base_m": kp3d,
                    "match_evidence": sorted(match_evidence),
                    "match_score": float(match["score"]),
                    "ripe_metrics": ripe_metrics,
                    "stem_segment_lengths_m": segment_lengths,
                    "grasp_target_source": (
                        "kp0_toward_kp1_local_plus_base_z_trim"
                        if use_local_direction
                        else "kp0_toward_kp2_chord_plus_base_z_trim"
                    ),
                    "grasp_offset_from_kp0_m": grasp_step_m,
                    "grasp_direction_base": grasp_direction,
                    "stem_bend_angle_deg": bend_angle_deg,
                    "grasp_target_base_z_trim_m": self._grasp_target_base_z_trim_m,
                    "grasp_target_base_m": grasp_pt,
                }

                local_stem_vec = kp0_to_kp1 if use_local_direction else stem_vec
                quat_xyzw = (stem_vec_to_quat_xyzw(local_stem_vec)
                             if local_stem_vec is not None
                             else np.array([0.0, 0.0, 0.0, 1.0]))

                if run_infer:
                    track_id, track = self._update_track(grasp_pt, quat_xyzw, target_quality)
                else:
                    # Visualization only — find nearest existing track without updating
                    best_id, best_dist = None, float("inf")
                    for tid, t in self._tracks.items():
                        d = float(np.linalg.norm(grasp_pt - t["pos"]))
                        if d < best_dist:
                            best_id, best_dist = tid, d
                    if best_id is None or best_dist > self._track_match_dist:
                        continue
                    track_id, track = best_id, self._tracks[best_id]

                stable = self._track_is_stable(track)
                if stable:
                    # Use image-center priority for the current view, but publish
                    # only one locked target after all candidates are processed.
                    img_cx = img.shape[1] * 0.5
                    img_cy = img.shape[0] * 0.5
                    center_dist_px = float(np.hypot(pose_cx - img_cx, pose_cy - img_cy))
                    ripe_candidates.append({
                        "track_id": track_id,
                        "track": track,
                        "center_dist_px": center_dist_px,
                    })

                gx, gy, gz = track["pos"]
                label = "PICK" if stable else "WAIT"
                cv2.putText(vis,
                    f"{label}#{track_id} h={track['hits']} "
                    f"s={track['position_spread_m']*1000:.0f}mm "
                    f"({gx:.3f},{gy:.3f},{gz:.3f})",
                    (int(pose_bbox[0]), int(pose_bbox[3]) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_MATCH, 1)

                # Draw stem direction arrow (KP0 → KP2 in image)
                if stem_vec is not None and 0 in kp3d and 2 in kp3d:
                    kp0_px = (int(kps_np[0][0]), int(kps_np[0][1]))
                    kp2_px = (int(kps_np[2][0]), int(kps_np[2][1]))
                    cv2.arrowedLine(vis, kp0_px, kp2_px,
                                    (0, 200, 255), 2, tipLength=0.3)

            active_candidate = self._select_active_track(ripe_candidates)
            if run_infer and active_candidate is not None:
                active_track = active_candidate["track"]
                if self._should_publish_track(active_track):
                    self._publish_pick_track(active_track)

            if active_candidate is not None:
                ax, ay, az = active_candidate["track"]["pos"]
                cv2.putText(vis,
                    f"LOCK#{active_candidate['track_id']} ({ax:.3f},{ay:.3f},{az:.3f})",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_MATCH, 1)

            # ── HUD ──────────────────────────────────────────────────────────
            if self._show_cell_grid:
                h, w = vis.shape[:2]
                cx, cy = w // 2, h // 2
                grid_color = (0, 255, 0)
                cv2.line(vis, (cx, 0), (cx, h), grid_color, 1)
                cv2.line(vis, (0, cy), (w, cy), grid_color, 1)
                cv2.putText(vis, "NW", (10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, grid_color, 2)
                cv2.putText(vis, "NE", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, grid_color, 2)
                cv2.putText(vis, "SW", (10, cy + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, grid_color, 2)
                cv2.putText(vis, "SE", (cx + 10, cy + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, grid_color, 2)

            n_ripe = sum(1 for c, _ in seg_items if c == RIPE_CLASS_ID)
            cv2.putText(vis,
                f"seg_ripe={n_ripe}  pose_det={len(pose_items)}  frame={self._frame_n}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(vis, "q: quit",
                        (10, 458), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

            self._show(vis)

        except SystemExit:
            raise
        except Exception as e:
            self.get_logger().error(f"Loop error: {e}")
            import traceback
            traceback.print_exc()

    def _show(self, vis):
        if not self._display:
            return
        cv2.imshow("Fusion Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit


def main():
    rclpy.init()
    node = StrawberryFusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.pipeline.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
