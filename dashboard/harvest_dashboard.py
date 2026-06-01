#!/usr/bin/env python3
"""
harvest_dashboard.py — 딸기 수확 로봇 웹 대시보드
실행: python3 harvest_dashboard.py [--demo] [--no-camera] [--camera-id N] [--port 8765]
CLI: --update start_harvest|harvest_success|harvest_fail|damage|reset
     --msg TEXT [--level info|success|warning|error]
     --status idle|approaching|grasping|returning|error
     --joints J1,J2,J3,J4,J5,J6  --tcp X,Y,Z,RX,RY,RZ
     --gripper POS[,FORCE]  --target N
"""
import argparse, asyncio, json, math, os, random, sys, threading, time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── 상태 파일 ──────────────────────────────────────────────────────────────────
STATE_FILE   = Path(os.environ.get('HARVEST_STATE_FILE', '/tmp/harvest_state.json'))
MAX_MESSAGES = 20

DEFAULT_STATE: dict = {
    "session_start":         None,
    "total_attempts":        0,
    "success_count":         0,
    "damage_count":          0,
    "grasp_times":           [],
    "current_harvest_start": None,
    "messages":              [],
    "status":                "idle",
    "robot_ready":           False,  # teleop_api에서 주기적으로 업데이트됨
    "robot_error":           "",     # 로봇 연결 오류 메시지
    "joint_angles":          [0.0]*6,
    "tcp_pose":              [0.0]*6,
    "target_count":          15,
    "attempt_history":       [],
    "failure_types":         {"ik":0,"obstacle":0,"grasp":0,"detection":0,"other":0},
    "detected_count":        0,
    "skip_reasons":          {"immature":0,"occluded":0,"harvested":0,"other":0},
    "pending_joint_command":  None,
    "pending_tcp_command":    None,
    "last_updated":           None,
    "gripper":                {"position":100.0,"state":"open","force":30.0},
    "planned_duration_hours": 0,
}

def _load() -> dict:
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text())
            for k, v in DEFAULT_STATE.items():
                s.setdefault(k, v)
            return s
        except Exception:
            pass
    return DEFAULT_STATE.copy()

def _save(s: dict) -> None:
    s["last_updated"] = datetime.now().isoformat()
    # DEFAULT_STATE의 모든 키를 포함하도록 병합
    out = DEFAULT_STATE.copy()
    out.update(s)
    tmp = STATE_FILE.with_suffix('.tmp')
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    os.replace(tmp, STATE_FILE)

def _push_msg(s: dict, text: str, level: str = "info") -> None:
    s["messages"].append({"time": datetime.now().strftime("%H:%M:%S"), "level": level, "text": text})
    if len(s["messages"]) > MAX_MESSAGES:
        s["messages"] = s["messages"][-MAX_MESSAGES:]

def _detect_failure_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ['ik','관절 한계','재계획','역기구학']): return 'ik'
    if any(k in t for k in ['장애물','가림','폐색','우회']):        return 'obstacle'
    if any(k in t for k in ['파지 실패','그리퍼','닫힘 실패']):     return 'grasp'
    if any(k in t for k in ['감지','미성숙','탐지']):               return 'detection'
    return 'other'

# ── 카메라 ────────────────────────────────────────────────────────────────────
_cam_locks    = [threading.Lock(), threading.Lock()]
_cam_jpegs:   list = [None, None]          # slot 0 = YOLO cam, slot 1 = overview cam
_cam_infos    = [{"source": "none", "label": "딸기 인식"},
                 {"source": "none", "label": "전경 카메라"}]
_cam_fps_v    = [0.0, 0.0]
_cam_fps_lock = threading.Lock()

def _camera_worker(camera_id: int = 0, slot: int = 0, serial: str = '',
                    url: str = '') -> None:
    """카메라 프레임 수집. url이 있으면 MJPEG URL에서 읽고, 없으면 USB 직접 접근."""
    try: import cv2
    except ImportError: print(f"[Camera{slot}] opencv not installed"); return
    label = _cam_infos[slot]["label"]
    enc   = [cv2.IMWRITE_JPEG_QUALITY, 75]

    # ── URL 소스 (ros2_bridge MJPEG 스트림) ──────────────────────────────────
    # cv2.VideoCapture는 MJPEG URL에서 FATAL 크래시 발생 → urllib 직접 파싱
    if url:
        import urllib.request
        import numpy as np
        print(f"[Camera{slot}] URL 소스 사용: {url}")
        _cam_infos[slot]["source"] = f"bridge:{url}"
        fc, ft = 0, time.time()
        while True:
            try:
                req = urllib.request.urlopen(url, timeout=10)
                buf = b''
                while True:
                    chunk = req.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    # JPEG 프레임 추출 (SOI=0xFFD8, EOI=0xFFD9)
                    while True:
                        a = buf.find(b'\xff\xd8')
                        b = buf.find(b'\xff\xd9', a + 2) if a != -1 else -1
                        if a == -1 or b == -1:
                            break
                        jpg = buf[a:b + 2]
                        buf = buf[b + 2:]
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            _, out = cv2.imencode('.jpg', frame, enc)
                            with _cam_locks[slot]: _cam_jpegs[slot] = out.tobytes()
                            fc += 1
                            now = time.time()
                            if now - ft >= 1.0:
                                with _cam_fps_lock: _cam_fps_v[slot] = fc / (now - ft)
                                fc, ft = 0, now
            except Exception as e:
                print(f"[Camera{slot}] URL 읽기 오류: {e}, 3s 후 재시도")
            time.sleep(3)
        return  # URL 모드에서는 여기까지만

    # ── RealSense SDK 직접 접근 ───────────────────────────────────────────────
    pipeline = None
    try:
        import pyrealsense2 as rs, numpy as np
        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            raise RuntimeError("RealSense 장치 없음")
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline = rs.pipeline()
        pipeline.start(cfg)
        print(f"[Camera{slot}] RealSense 워밍업 중 (serial={serial or 'auto'})...")
        for _ in range(10):
            try: pipeline.wait_for_frames(timeout_ms=300)
            except: pass
        _cam_infos[slot]["source"] = f"RealSense:{serial}" if serial else "RealSense"
        print(f"[Camera{slot}] RealSense SDK 연결 성공 — {label}")
    except Exception as e:
        print(f"[Camera{slot}] pyrealsense2 skip({e}), ffmpeg로 진행")
        pipeline = None

    # ── ffmpeg v4l2 폴백 ──────────────────────────────────────────────────────
    import subprocess, numpy as np
    proc = None
    W, H = 640, 480
    frame_bytes = W * H * 3

    if pipeline is None:
        dev_path = f"/dev/video{camera_id}"
        while True:
            try:
                cmd = [
                    'ffmpeg', '-loglevel', 'quiet',
                    '-f', 'v4l2', '-framerate', '30',
                    '-video_size', f'{W}x{H}',
                    '-i', dev_path,
                    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'
                ]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=frame_bytes*2)
                raw = proc.stdout.read(frame_bytes)
                if len(raw) == frame_bytes:
                    _cam_infos[slot]["source"] = f"webcam:{camera_id}"
                    print(f"[Camera{slot}] ffmpeg /dev/video{camera_id} 연결 — {label}")
                    break
                proc.kill(); proc = None
                print(f"[Camera{slot}] ffmpeg 첫 프레임 실패, 3s 후 재시도")
            except Exception as e:
                print(f"[Camera{slot}] ffmpeg 오류: {e}, 3s 후 재시도")
                if proc: proc.kill(); proc = None
            time.sleep(3)

    # ── 메인 캡처 루프 ────────────────────────────────────────────────────────
    fc, ft = 0, time.time()
    while True:
        frame = None
        try:
            if pipeline:
                fr = pipeline.wait_for_frames(timeout_ms=2000)
                cf = fr.get_color_frame()
                if cf: frame = np.asanyarray(cf.get_data())
            elif proc:
                raw = proc.stdout.read(frame_bytes)
                if len(raw) == frame_bytes:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
                else:
                    proc.kill(); proc = None; time.sleep(1)
        except Exception: pass
        if frame is not None:
            try:
                _, buf = cv2.imencode('.jpg', frame, enc)
                with _cam_locks[slot]: _cam_jpegs[slot] = buf.tobytes()
                fc += 1
                now = time.time()
                if now - ft >= 1.0:
                    with _cam_fps_lock: _cam_fps_v[slot] = fc / (now - ft)
                    fc, ft = 0, now
            except Exception: pass
        elif proc is None and pipeline is None:
            time.sleep(3)
            try:
                proc = subprocess.Popen(
                    ['ffmpeg', '-loglevel', 'quiet', '-f', 'v4l2',
                     '-framerate', '30', '-video_size', f'{W}x{H}',
                     '-i', f'/dev/video{camera_id}',
                     '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=frame_bytes*2)
            except Exception: pass

async def _mjpeg_gen(slot: int = 0):
    hdr = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    while True:
        with _cam_locks[slot]: jpg = _cam_jpegs[slot]
        if jpg: yield hdr + jpg + b'\r\n'
        await asyncio.sleep(1/25)

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>딸기 수확 제어 시스템</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#f1f5f9; --surface:#fff; --card:#fff; --border:#e2e8f0; --border-s:#f1f5f9;
  --green:#16a34a; --green-bg:#f0fdf4; --green-tx:#15803d;
  --blue:#2563eb;  --blue-bg:#eff6ff;  --blue-tx:#1d4ed8;
  --yellow:#d97706;--yellow-bg:#fffbeb;--yellow-tx:#b45309;
  --red:#dc2626;   --red-bg:#fef2f2;   --red-tx:#b91c1c;
  --cyan:#0891b2;  --cyan-bg:#ecfeff;  --cyan-tx:#0e7490;
  --purple:#7c3aed;--purple-bg:#f5f3ff;--purple-tx:#6d28d9;
  --slate:#475569; --slate-bg:#f8fafc; --slate-tx:#334155;
  --t1:#0f172a; --t2:#64748b; --t3:#94a3b8;
  --sh:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
  --sh-sm:0 1px 2px rgba(0,0,0,.05);
  --font:'Inter',system-ui,sans-serif;
  --mono:'JetBrains Mono',monospace; --r:8px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--t1);font-family:var(--font);
  height:100vh;overflow:hidden;display:flex;flex-direction:column;font-size:12px}

/* HEADER */
.hdr{background:var(--surface);border-bottom:1px solid var(--border);
  box-shadow:var(--sh-sm);padding:0 12px;height:40px;flex-shrink:0;
  display:flex;align-items:center;justify-content:space-between;gap:8px;z-index:10}
.hd-brand{display:flex;align-items:center;gap:8px;flex-shrink:0}
.hd-logo{width:26px;height:26px;border-radius:6px;
  background:linear-gradient(135deg,#c0392b,#e74c3c);
  display:flex;align-items:center;justify-content:center;
  font-size:9px;font-weight:800;color:#fff;letter-spacing:.4px;
  box-shadow:0 2px 5px rgba(220,38,38,.3);flex-shrink:0}
.hd-name{font-size:13px;font-weight:700;letter-spacing:-.2px}
.hd-sub{font-size:9.5px;color:var(--t3)}
.hd-live{display:flex;align-items:center;gap:5px;background:var(--green-bg);
  border:1px solid #bbf7d0;border-radius:16px;padding:3px 9px;flex-shrink:0}
.hd-live-dot{width:5px;height:5px;border-radius:50%;background:var(--green);
  animation:livepulse 1.5s ease-in-out infinite}
@keyframes livepulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(22,163,74,.5)}
  50%{opacity:.8;box-shadow:0 0 0 4px rgba(22,163,74,0)}}
.hd-live-txt{font-size:9px;font-weight:700;color:var(--green-tx);letter-spacing:1px}
.conn-row{display:flex;align-items:center;gap:4px;flex-shrink:0}
.cpill{display:flex;align-items:center;gap:3px;padding:2px 6px;border-radius:8px;
  font-size:8.5px;font-weight:700;border:1px solid var(--border);
  background:var(--border-s);color:var(--t3);letter-spacing:.4px;transition:all .3s}
.cpill .cd{width:4px;height:4px;border-radius:50%;background:var(--t3);transition:background .3s}
.cpill.on{background:var(--green-bg);border-color:#bbf7d0;color:var(--green-tx)}
.cpill.on .cd{background:var(--green)}
.cpill.warn{background:var(--yellow-bg);border-color:#fde68a;color:var(--yellow-tx)}
.cpill.warn .cd{background:var(--yellow)}
.cpill.off{background:var(--red-bg);border-color:#fecaca;color:var(--red-tx)}
.cpill.off .cd{background:var(--red)}
.hd-btn{padding:4px 9px;border-radius:5px;border:1px solid var(--border);
  background:var(--surface);color:var(--t2);font-size:10px;font-weight:600;
  cursor:pointer;transition:all .15s;flex-shrink:0}
.hd-btn:hover{background:var(--bg);border-color:var(--blue);color:var(--blue)}
.hd-right{display:flex;align-items:center;gap:12px;flex-shrink:0;text-align:right}
.hd-session{font-size:9px;color:var(--t3);line-height:1.7}
.hd-clock{font-family:var(--mono);font-size:11px;color:var(--blue);font-weight:600}

/* MAIN */
.main{flex:1;min-height:0;display:flex;flex-direction:column;
  padding:5px 10px;gap:5px;overflow:hidden}

/* STATS */
.stats-row{display:grid;grid-template-columns:repeat(8,1fr);gap:4px;flex-shrink:0}
.sc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:5px 8px 4px;position:relative;overflow:hidden;box-shadow:var(--sh);
  transition:box-shadow .2s,transform .15s}
.sc:hover{box-shadow:0 4px 6px rgba(0,0,0,.07);transform:translateY(-1px)}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
  border-radius:var(--r) var(--r) 0 0}
.sg::before{background:var(--green)}.sb::before{background:var(--blue)}
.sy::before{background:var(--yellow)}.sr::before{background:var(--red)}
.sc2::before{background:var(--cyan)}.sp::before{background:var(--purple)}
.ss::before{background:var(--slate)}
.sc-lbl{font-size:11px;font-weight:700;color:var(--t2);text-transform:uppercase;
  letter-spacing:.3px;margin-bottom:2px}
.sc-val{font-family:var(--mono);font-size:21px;font-weight:800;line-height:1;
  font-variant-numeric:tabular-nums;transition:color .3s}
.sc-unit{font-size:11px;font-weight:600;margin-left:2px;font-family:var(--font);color:var(--t3)}
.sg .sc-val{color:var(--green)}.sb .sc-val{color:var(--blue)}
.sy .sc-val{color:var(--yellow)}.sc2 .sc-val{color:var(--cyan)}
.sp .sc-val{color:var(--purple)}.ss .sc-val{color:var(--slate)}
.sc-bar{margin-top:3px;height:2px;border-radius:2px;background:var(--border);overflow:hidden}
.sc-bar-f{height:100%;border-radius:2px;transition:width .6s ease}

/* SESSION INFO BAR */
.sinfo-bar{display:flex;align-items:stretch;background:var(--card);
  border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--sh);
  flex-shrink:0;overflow:hidden;height:30px}
.sinfo-cell{display:flex;align-items:center;gap:6px;padding:0 12px;
  border-right:1px solid var(--border);flex-shrink:0}
.sinfo-cell:last-child{border-right:none}
.sinfo-lbl{font-size:8px;font-weight:700;color:var(--t3);text-transform:uppercase;
  letter-spacing:.5px;white-space:nowrap}
.sinfo-val{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--t1);
  white-space:nowrap;transition:color .4s}
.sinfo-inp{width:44px;padding:1px 4px;border:1px solid var(--border);border-radius:4px;
  font-family:var(--mono);font-size:11px;background:var(--bg);outline:none;text-align:right;color:var(--t1)}
.sinfo-inp:focus{border-color:var(--blue);background:var(--blue-bg)}
.sinfo-btn{padding:2px 8px;border-radius:4px;border:none;background:var(--blue);
  color:#fff;font-size:9px;font-weight:700;cursor:pointer}
.sinfo-btn:hover{background:var(--blue-tx)}
.sinfo-unit{font-size:9px;color:var(--t3);white-space:nowrap}

/* META ROW */
.meta-row{display:grid;grid-template-columns:3fr 2fr 2fr;gap:6px;flex-shrink:0}
.mc{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:5px 12px;box-shadow:var(--sh);display:flex;align-items:center;gap:10px}
.mc-lbl{font-size:8.5px;font-weight:700;color:var(--t3);text-transform:uppercase;
  letter-spacing:.6px;white-space:nowrap}
.tp-bar-w{flex:1;height:5px;background:var(--border);border-radius:2px;overflow:hidden}
.tp-bar-f{height:100%;border-radius:2px;width:0%;background:var(--cyan);transition:width .6s ease,background .4s}
.tp-stat{font-family:var(--mono);font-size:11px;font-weight:700;white-space:nowrap}
.tp-edit{display:flex;align-items:center;gap:3px}
.tp-edit input{width:44px;padding:2px 4px;border:1px solid var(--border);border-radius:4px;
  font-family:var(--mono);font-size:10px;background:var(--bg);outline:none;text-align:right}
.tp-edit input:focus{border-color:var(--blue)}
.tp-edit button{padding:2px 6px;border-radius:4px;border:none;
  background:var(--blue);color:#fff;font-size:9px;font-weight:700;cursor:pointer}
.hist-bars{display:flex;gap:2px;align-items:flex-end;flex:1;height:18px}
.hbar{flex:1;min-width:4px;max-width:16px;border-radius:1px 1px 0 0}
.hbar.s{background:var(--green);height:100%;opacity:.75}
.hbar.f{background:var(--red);height:55%;opacity:.75}
.hist-empty{font-size:10px;color:var(--t3);align-self:center}
.echart-list{display:flex;flex-direction:column;gap:3px;flex:1;justify-content:center}
.echart-row{display:flex;align-items:center;gap:5px}
.echart-lbl{font-size:8.5px;color:var(--t2);width:40px;flex-shrink:0;text-align:right}
.echart-bw{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.echart-bf{height:100%;border-radius:2px;transition:width .5s;min-width:3px}
.echart-cnt{font-family:var(--mono);font-size:8.5px;color:var(--t3);width:18px;text-align:right;flex-shrink:0}
.echart-empty{font-size:10px;color:var(--t3);text-align:center}

/* BOTTOM ROW */
.bot{flex:1;min-height:0;display:grid;
  grid-template-columns:185px 295px 1fr 190px;
  gap:5px;overflow:hidden}

/* GRAPH BANNER (4열 상단) */
.bg-panel{grid-column:1/5;grid-row:1;
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  box-shadow:var(--sh);display:flex;overflow:hidden;flex-shrink:0}
.bg-sec{padding:8px 12px;display:flex;flex-direction:column;justify-content:center;
  gap:3px;overflow:hidden;min-width:0}
.bg-sec-lbl{font-size:9px;font-weight:700;color:var(--t3);text-transform:uppercase;
  letter-spacing:.6px;margin-bottom:3px;flex-shrink:0}
.bg-divider{width:1px;background:var(--border);flex-shrink:0;margin:8px 0}
.bg-bar-w{height:7px;background:var(--border);border-radius:4px;overflow:hidden;margin-bottom:4px;flex-shrink:0}
.bg-bar-f{height:100%;border-radius:4px;background:var(--cyan);width:0%;transition:width .6s,background .4s}
.bg-stat{display:flex;align-items:baseline;gap:3px;flex-shrink:0}
.bg-big{font-family:var(--mono);font-size:26px;font-weight:800;line-height:1;transition:color .3s}
.bg-sub{font-size:12px;color:var(--t3)}
.bg-tgt{display:flex;align-items:center;gap:3px;margin-top:2px;flex-shrink:0}
.bg-tgt span{font-size:9px;color:var(--t3);white-space:nowrap}
.bg-tgt input{width:42px;padding:1px 4px;border:1px solid var(--border);border-radius:3px;
  font-family:var(--mono);font-size:10px;background:var(--bg);outline:none;text-align:right}
.bg-tgt input:focus{border-color:var(--blue)}
.bg-tgt button{padding:2px 6px;border-radius:3px;border:none;
  background:var(--blue);color:#fff;font-size:9px;font-weight:700;cursor:pointer}
.bg-hist{display:flex;gap:2px;align-items:flex-end;height:36px;overflow:hidden;flex:1}
.bg-hbar{flex:1;min-width:4px;max-width:22px;border-radius:2px 2px 0 0}
.bg-hbar.s{background:var(--green);height:100%;opacity:.8}
.bg-hbar.f{background:var(--red);height:55%;opacity:.8}
.bg-hempty{font-size:10px;color:var(--t3);align-self:center}
.bg-errs{display:flex;flex-direction:column;gap:3px;flex:1;justify-content:center;overflow:hidden}
.bg-err-row{display:flex;align-items:center;gap:4px}
.bg-err-lbl{font-size:8px;font-weight:700;width:30px;flex-shrink:0;text-align:right}
.bg-err-bw{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.bg-err-bf{height:100%;border-radius:3px;transition:width .5s;min-width:0}
.bg-err-cnt{font-family:var(--mono);font-size:8.5px;color:var(--t3);width:14px;text-align:right;flex-shrink:0}

/* ETA */
.bg-eta{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:1px;
  white-space:nowrap;transition:color .3s}

/* 파지 분포 히스토그램 */
.bg-grasp-hist{display:flex;gap:3px;align-items:flex-end;flex:1;overflow:hidden}
.bg-gbin{display:flex;flex-direction:column;align-items:center;flex:1;gap:0;min-width:0}
.bg-gbar-w{flex:1;width:100%;display:flex;align-items:flex-end;justify-content:center}
.bg-gbar-f{width:80%;background:var(--cyan);border-radius:2px 2px 0 0;
  transition:height .4s ease;min-height:2px}
.bg-gbin-lbl{font-size:6.5px;color:var(--t3);white-space:nowrap;margin-top:2px;line-height:1}
.bg-gbin-cnt{font-family:var(--mono);font-size:6.5px;color:var(--t2);font-weight:600;line-height:1}

/* 수확 불가 분류 (그래프·조작 패널) */
.sk-row{display:flex;align-items:center;gap:4px;margin-bottom:2px}
.sk-lbl{font-size:8px;font-weight:700;width:32px;flex-shrink:0;text-align:right}
.sk-bw{flex:1;height:4px;background:var(--border);border-radius:2px;overflow:hidden}
.sk-bf{height:100%;border-radius:2px;transition:width .5s;min-width:0}
.sk-cnt{font-family:var(--mono);font-size:8px;color:var(--t3);width:14px;text-align:right;flex-shrink:0}

/* 스냅샷 갤러리 */
.snap-strip{flex-shrink:0;padding:4px 6px;border-top:1px solid var(--border);
  background:var(--bg);height:58px;overflow:hidden}
.snap-gallery{display:flex;gap:3px;height:100%;overflow-x:auto;align-items:center;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.snap-gallery::-webkit-scrollbar{height:2px}
.snap-gallery::-webkit-scrollbar-thumb{background:var(--border)}
.snap-item{flex-shrink:0;width:44px;height:44px;border-radius:4px;
  overflow:hidden;position:relative;cursor:pointer;
  border:2px solid var(--border);transition:border-color .2s}
.snap-item:hover{border-color:var(--blue)}
.snap-item.success{border-color:#86efac}
.snap-item.fail{border-color:#fca5a5}
.snap-img{width:100%;height:100%;object-fit:cover;display:block}
.snap-badge{position:absolute;top:1px;right:1px;font-size:7px;font-weight:800;
  width:11px;height:11px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;line-height:1}
.snap-item.success .snap-badge{background:var(--green);color:#fff}
.snap-item.fail .snap-badge{background:var(--red);color:#fff}
.snap-empty{font-size:9px;color:var(--t3);white-space:nowrap;padding:0 4px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  box-shadow:var(--sh);display:flex;flex-direction:column;overflow:hidden;min-height:0}
.ph{display:flex;align-items:center;gap:5px;padding:5px 10px;
  border-bottom:1px solid var(--border-s);font-size:9.5px;font-weight:700;
  color:var(--t2);text-transform:uppercase;letter-spacing:.7px;
  flex-shrink:0;background:var(--bg)}
.ph-dot{width:4px;height:4px;border-radius:50%;flex-shrink:0}
.ph-badge{margin-left:auto;background:var(--blue-bg);border:1px solid #bfdbfe;
  color:var(--blue-tx);border-radius:8px;padding:1px 6px;font-size:9px;font-weight:700}

/* STATUS PANEL */
.sp-body{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;padding:10px 10px;gap:0}
.sp-icon-wrap{position:relative;margin-bottom:8px}
.sp-ring{position:absolute;inset:-8px;border-radius:50%;border:2px solid transparent}
@keyframes ring-pulse{0%{transform:scale(1);opacity:.7}100%{transform:scale(1.6);opacity:0}}
.sp-icon-bg{width:42px;height:42px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:16px;font-weight:700;font-family:var(--mono);
  border:2px solid var(--border);background:var(--bg);transition:all .4s}
.sp-state{font-size:11px;font-weight:700;color:var(--t2);margin-bottom:3px;transition:color .4s}
.sp-num{font-size:9px;color:var(--t3);margin-bottom:6px}
.sp-timer{font-family:var(--mono);font-size:18px;font-weight:700;
  color:var(--yellow);letter-spacing:2px;min-height:22px}
.sp-div{width:32px;height:1px;background:var(--border);margin:8px 0}
.sp-mini{display:flex;flex-direction:column;gap:3px;width:100%}
.sp-mrow{display:flex;justify-content:space-between;align-items:center;font-size:9px}
.sp-mlbl{color:var(--t3)}
.sp-mval{font-family:var(--mono);font-weight:700;font-size:10px}

/* TABS */
.tab-row{display:flex;border-bottom:1px solid var(--border);background:var(--bg);flex-shrink:0}
.tab-btn{flex:1;padding:5px 2px;font-size:9px;font-weight:700;color:var(--t3);
  border:none;background:transparent;cursor:pointer;
  border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .15s}
.tab-btn.active{color:var(--blue);border-bottom-color:var(--blue);background:var(--surface)}
.tab-btn:hover:not(.active){color:var(--t2);background:var(--border-s)}
.tab-pane{display:none;flex:1;min-height:0;overflow:hidden;flex-direction:column}
.tab-pane.active{display:flex}
.tab-body{flex:1;padding:6px 10px;display:flex;flex-direction:column;overflow:hidden}

/* JOINT / TCP COMMON */
.sec-lbl{font-size:8px;font-weight:700;color:var(--t3);text-transform:uppercase;
  letter-spacing:.6px;margin:4px 0 3px}
.jrow{display:flex;align-items:center;gap:5px;margin-bottom:3px}
.jlbl{font-family:var(--mono);font-size:9.5px;font-weight:700;color:var(--blue);
  width:14px;flex-shrink:0}
.jtrack{flex:1;height:4px;background:var(--border);border-radius:2px;
  position:relative;overflow:visible}
.jfill{position:absolute;height:100%;border-radius:2px;min-width:2px;transition:left .3s,width .3s}
.jcenter{position:absolute;width:1px;height:8px;top:-2px;left:50%;
  transform:translateX(-50%);background:var(--t3);opacity:.45}
.jval{font-family:var(--mono);font-size:9px;color:var(--t1);width:36px;
  text-align:right;flex-shrink:0}
.jdiv{height:1px;background:var(--border);margin:4px 0}
.jgrid{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:3px}
.jcell{display:flex;flex-direction:column;gap:1px}
.jcell label{font-size:8px;color:var(--t3);font-weight:700}
.jcell input,.tcp-inp{width:100%;padding:2px 4px;border:1px solid var(--border);
  border-radius:3px;font-family:var(--mono);font-size:10px;
  background:var(--bg);outline:none;text-align:right;color:var(--t1)}
.jcell input:focus{border-color:var(--blue);background:var(--blue-bg)}
.tcp-inp:focus{border-color:var(--purple);background:var(--purple-bg)}
.jfoot{display:flex;gap:4px;align-items:flex-end;margin-top:3px}
.jvel{display:flex;flex-direction:column;gap:1px}
.jvel label{font-size:8px;color:var(--t3);font-weight:700}
.jvel input{width:44px;padding:2px 4px;border:1px solid var(--border);
  border-radius:3px;font-family:var(--mono);font-size:10px;
  background:var(--bg);outline:none;text-align:right}
.jbtns{display:flex;gap:3px;flex:1}
.jbtn{flex:1;padding:4px 0;border-radius:4px;border:none;cursor:pointer;
  font-size:10px;font-weight:700;transition:all .15s}
.jbtn.p{background:var(--blue);color:#fff}.jbtn.p:hover{background:var(--blue-tx)}
.jbtn.s{background:var(--bg);border:1px solid var(--border);color:var(--t2)}
.jbtn.s:hover{border-color:var(--blue);color:var(--blue);background:var(--blue-bg)}
.jbtn:active{transform:scale(.97)}
.jstatus{font-size:8.5px;color:var(--t3);text-align:center;padding:2px 0;min-height:14px}

/* TCP DISPLAY */
.tcp-dgrid{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:3px}
.tcp-cell{display:flex;flex-direction:column;gap:1px}
.tcp-cell label{font-size:8px;color:var(--t3);font-weight:700}
.tcp-disp{font-family:var(--mono);font-size:10px;color:var(--purple);
  padding:2px 4px;background:var(--purple-bg);border-radius:3px;
  text-align:right;border:1px solid #ddd6fe}

/* GRIPPER */
.g-status-row{display:flex;align-items:center;gap:8px;padding:3px 0}
.g-viz{display:flex;align-items:center;height:24px;flex-shrink:0}
.g-jaw{width:4px;height:22px;border-radius:2px;transition:background .3s;flex-shrink:0}
.g-state{font-size:11px;font-weight:700;transition:color .3s}
.g-pct{font-family:var(--mono);font-size:9px;color:var(--t3);margin-top:1px}
.g-ctrl{display:flex;align-items:center;gap:4px;margin-bottom:4px}
.g-btn{padding:2px 6px;border-radius:3px;border:1px solid var(--border);
  background:var(--bg);font-size:9.5px;font-weight:700;cursor:pointer;
  transition:all .15s;color:var(--t2);white-space:nowrap}
.g-btn.op:hover{background:var(--green-bg);border-color:#bbf7d0;color:var(--green-tx)}
.g-btn.cl:hover{background:var(--blue-bg);border-color:#bfdbfe;color:var(--blue-tx)}
.g-slider{flex:1;height:4px;cursor:pointer;accent-color:var(--cyan);
  -webkit-appearance:none;appearance:none;background:var(--border);
  border-radius:2px;outline:none}
.g-slider::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;
  border-radius:50%;background:var(--cyan);cursor:pointer;
  border:2px solid #fff;box-shadow:0 0 0 1px var(--cyan)}
.g-foot{display:flex;gap:4px;align-items:flex-end}
.g-fld{display:flex;flex-direction:column;gap:1px}
.g-fld label{font-size:8px;color:var(--t3);font-weight:700}
.g-fld input{width:48px;padding:2px 4px;border:1px solid var(--border);
  border-radius:3px;font-family:var(--mono);font-size:10px;
  background:var(--bg);outline:none;text-align:right}
.g-fld input:focus{border-color:var(--cyan);background:var(--cyan-bg)}

/* CAMERA */
.cam-body{flex:1;min-height:0;position:relative;background:#0a0f1e;
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;cursor:pointer}
.cam-img{width:100%;height:100%;object-fit:contain;display:block}
.cam-live{position:absolute;top:7px;right:7px;display:flex;align-items:center;gap:4px;
  background:rgba(220,38,38,.88);border-radius:3px;padding:2px 6px;
  font-size:9px;font-weight:800;letter-spacing:1.4px;color:#fff;
  opacity:0;transition:opacity .4s;pointer-events:none}
.cam-live.on{opacity:1}
.cam-live-dot{width:4px;height:4px;border-radius:50%;background:#fff;animation:blink .8s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
.cam-fps{position:absolute;top:7px;left:7px;background:rgba(0,0,0,.5);
  border-radius:3px;padding:2px 5px;font-family:var(--mono);font-size:9px;
  color:rgba(255,255,255,.7);pointer-events:none;opacity:0;transition:opacity .4s}
.cam-fps.on{opacity:1}
.cam-scan{position:absolute;width:100%;height:40px;pointer-events:none;
  background:linear-gradient(transparent,rgba(255,255,255,.03),transparent);
  animation:scan 5s linear infinite;display:none}
.cam-scan.on{display:block}
@keyframes scan{from{top:-40px}to{top:calc(100% + 40px)}}
.cam-corner{position:absolute;width:10px;height:10px;opacity:.4}
.cam-corner.tl{top:7px;left:7px;border-top:2px solid rgba(255,255,255,.7);border-left:2px solid rgba(255,255,255,.7)}
.cam-corner.tr{top:7px;right:7px;border-top:2px solid rgba(255,255,255,.7);border-right:2px solid rgba(255,255,255,.7)}
.cam-corner.bl{bottom:7px;left:7px;border-bottom:2px solid rgba(255,255,255,.7);border-left:2px solid rgba(255,255,255,.7)}
.cam-corner.br{bottom:7px;right:7px;border-bottom:2px solid rgba(255,255,255,.7);border-right:2px solid rgba(255,255,255,.7)}
.cam-ov{position:absolute;bottom:0;left:0;right:0;padding:4px 8px;
  background:linear-gradient(transparent,rgba(0,0,0,.5));
  display:flex;justify-content:space-between;
  font-family:var(--mono);font-size:9px;color:rgba(255,255,255,.5);
  opacity:0;transition:opacity .4s;pointer-events:none}
.cam-ov.on{opacity:1}
.cam-ph{position:absolute;inset:0;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:7px}
.cam-ph-txt{font-size:10.5px;color:rgba(255,255,255,.25)}
.cam-spin{width:18px;height:18px;border-radius:50%;
  border:2px solid rgba(255,255,255,.1);border-top-color:rgba(255,255,255,.5);
  animation:cspin .8s linear infinite;display:none}
@keyframes cspin{to{transform:rotate(360deg)}}
.cam-fs-hint{position:absolute;bottom:7px;right:7px;
  background:rgba(0,0,0,.4);border-radius:3px;padding:2px 4px;
  font-size:8.5px;color:rgba(255,255,255,.4);pointer-events:none;
  opacity:0;transition:opacity .3s}
.cam-body:hover .cam-fs-hint{opacity:1}

/* MESSAGES */
.msg-list{flex:1;overflow-y:auto;padding:3px 0;display:flex;
  flex-direction:column-reverse;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.msg-list::-webkit-scrollbar{width:3px}
.msg-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.msg-wrap{border-bottom:1px solid var(--border-s)}
.msg-item{display:flex;align-items:flex-start;gap:6px;padding:5px 10px;
  position:relative;animation:msgIn .2s ease}
.msg-item::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px}
@keyframes msgIn{from{opacity:0;transform:translateX(3px)}to{opacity:1;transform:none}}
.msg-ts{font-family:var(--mono);font-size:9px;color:var(--t3);white-space:nowrap;padding-top:1px}
.msg-body{flex:1;min-width:0}
.msg-chip{display:inline-flex;align-items:center;font-size:8px;font-weight:700;
  letter-spacing:.5px;padding:1px 4px;border-radius:3px;margin-bottom:2px;
  text-transform:uppercase;border:1px solid transparent}
.msg-txt{font-size:11px;line-height:1.4;color:var(--t1);word-break:break-word}
.mi-info    .msg-item::before{background:#93c5fd}
.mi-info    .msg-chip{background:var(--blue-bg);color:var(--blue-tx);border-color:#bfdbfe}
.mi-info    .msg-txt{color:var(--t2)}
.mi-success .msg-item::before{background:#86efac}
.mi-success .msg-chip{background:var(--green-bg);color:var(--green-tx);border-color:#bbf7d0}
.mi-success .msg-txt{color:#166534}
.mi-warning .msg-item::before{background:#fcd34d}
.mi-warning .msg-chip{background:var(--yellow-bg);color:var(--yellow-tx);border-color:#fde68a}
.mi-warning .msg-txt{color:#92400e}
.mi-error   .msg-item::before{background:#fca5a5}
.mi-error   .msg-chip{background:var(--red-bg);color:var(--red-tx);border-color:#fecaca}
.mi-error   .msg-txt{color:#991b1b}

/* GRAPH PANEL - error type bars */
.ep-row{display:flex;align-items:center;gap:4px;margin-bottom:3px}
.ep-lbl{font-size:8px;font-weight:700;width:30px;flex-shrink:0;text-align:right}
.ep-bw{flex:1;height:5px;background:var(--border);border-radius:3px;overflow:hidden}
.ep-bf{height:100%;border-radius:3px;transition:width .5s ease;min-width:0}
.ep-cnt{font-family:var(--mono);font-size:8.5px;color:var(--t3);width:16px;
  text-align:right;flex-shrink:0;font-weight:600}

/* DPAD */
.dpad{display:grid;grid-template-columns:repeat(3,1fr);gap:2px;margin-bottom:2px}
.dpbtn{padding:5px 0;border-radius:4px;border:1px solid var(--border);background:var(--bg);
  font-size:14px;cursor:pointer;transition:all .1s;color:var(--t1);font-weight:700;
  user-select:none;-webkit-user-select:none;line-height:1}
.dpbtn:active{background:var(--blue-bg);border-color:var(--blue);color:var(--blue)}
.dpbtn.dstop{background:var(--red-bg);border-color:#fecaca;color:var(--red);font-size:10px}
.dpbtn.dstop:active{background:var(--red);color:#fff}
.dp2row{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin-bottom:4px}
.dpbtn2{padding:3px 0;border-radius:3px;border:1px solid var(--border);background:var(--bg);
  font-size:9px;font-weight:700;cursor:pointer;color:var(--t2);
  transition:all .1s;user-select:none;-webkit-user-select:none}
.dpbtn2:active{background:var(--blue-bg);border-color:var(--blue);color:var(--blue)}
.dp-speed{display:flex;align-items:center;gap:4px;margin-bottom:3px}
.dp-speed label{font-size:8px;color:var(--t3);font-weight:700;white-space:nowrap}
.dp-speed input[type=range]{flex:1;height:3px;accent-color:var(--blue);cursor:pointer}
.dp-speed span{font-family:var(--mono);font-size:8.5px;color:var(--blue);
  white-space:nowrap;width:28px;text-align:right}
.teleop-st{font-size:9px;text-align:center;font-family:var(--mono);
  padding:2px 0;min-height:14px;color:var(--t3);transition:color .2s}

/* 데이터 수집 패널 */
.rec-sep{height:2px;background:linear-gradient(90deg,var(--red),var(--purple));border-radius:1px;margin:6px 10px 5px;opacity:.35}
.rec-section{padding:0 10px 10px}
.rec-status-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:10px;
  font-size:9px;font-weight:700;letter-spacing:.5px;margin-bottom:6px;border:1px solid var(--border);
  background:var(--green-bg);border-color:#86efac;color:var(--green-tx);transition:all .3s}
.rec-status-badge.recording{background:var(--red-bg);border-color:#fca5a5;color:var(--red-tx)}
.rec-status-badge.starting{background:var(--yellow-bg);border-color:#fde68a;color:var(--yellow-tx)}
.rec-status-badge.stopping{background:var(--cyan-bg);border-color:#a5f3fc;color:var(--cyan-tx)}
.rec-dot{width:5px;height:5px;border-radius:50%;background:currentColor}
.rec-dot.blink{animation:blink .7s infinite}
.rec-inp-row{display:flex;flex-direction:column;gap:3px;margin-bottom:5px}
.rec-inp-lbl{font-size:8px;font-weight:700;color:var(--t3);text-transform:uppercase;letter-spacing:.4px}
.rec-inp{width:100%;padding:3px 5px;border:1px solid var(--border);border-radius:4px;
  font-family:var(--mono);font-size:10px;background:var(--bg);outline:none;color:var(--t1)}
.rec-inp:focus{border-color:var(--red);background:var(--red-bg)}
.rec-btns{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-bottom:4px}
.rec-btn{padding:5px 0;border-radius:4px;border:1px solid var(--border);
  background:var(--bg);font-size:9.5px;font-weight:700;cursor:pointer;
  color:var(--t2);transition:all .15s;width:100%}
.rec-btn.start{background:var(--green);color:#fff;border-color:var(--green)}
.rec-btn.start:hover{background:var(--green-tx)}
.rec-btn.stop{background:var(--red);color:#fff;border-color:var(--red)}
.rec-btn.stop:hover{background:var(--red-tx)}
.rec-btn.home{grid-column:1/3}
.rec-btn.convert{grid-column:1/3;background:var(--purple-bg);border-color:#ddd6fe;color:var(--purple-tx)}
.rec-btn.convert:hover{background:var(--purple);color:#fff}
.rec-btn:disabled{opacity:.4;cursor:not-allowed}
.grip-btns{display:grid;grid-template-columns:1fr 1fr 1fr;gap:2px;margin-bottom:4px}
.grip-btn{padding:4px 0;border-radius:3px;border:1px solid var(--border);
  background:var(--bg);font-size:8.5px;font-weight:700;cursor:pointer;color:var(--t2);transition:all .1s}
.grip-btn:active{background:var(--cyan-bg);border-color:var(--cyan);color:var(--cyan-tx)}
.rot-btns{display:grid;grid-template-columns:repeat(4,1fr);gap:2px;margin-bottom:3px}
.rot-btn{padding:3px 0;border-radius:3px;border:1px solid var(--border);
  background:var(--bg);font-size:8px;font-weight:700;cursor:pointer;color:var(--t2);transition:all .1s}
.rot-btn:active{background:var(--purple-bg);border-color:var(--purple);color:var(--purple-tx)}

/* DAMAGE FLASH */
#dmg-flash{position:fixed;inset:0;pointer-events:none;z-index:999;
  opacity:0;border:3px solid var(--red);background:rgba(220,38,38,.06)}
/* WS BADGE */
.ws-badge{position:fixed;bottom:8px;right:12px;z-index:100;
  display:flex;align-items:center;gap:4px;padding:3px 8px;border-radius:14px;
  font-size:9px;font-weight:600;background:var(--surface);
  border:1px solid var(--border);color:var(--t2);box-shadow:var(--sh);transition:all .3s}
.ws-dot{width:4px;height:4px;border-radius:50%;background:var(--green)}
.ws-badge.dis{color:var(--red);border-color:#fecaca}
.ws-badge.dis .ws-dot{background:var(--red)}
</style>
</head>
<body>
<div id="dmg-flash"></div>

<!-- HEADER -->
<div class="hdr">
  <div class="hd-brand">
    <div class="hd-logo">SH</div>
    <div>
      <div class="hd-name">딸기 수확 제어 시스템</div>
      <div class="hd-sub">Strawberry Harvest Control Dashboard</div>
    </div>
  </div>
  <div class="hd-live"><div class="hd-live-dot"></div><div class="hd-live-txt">LIVE</div></div>
  <div class="conn-row">
    <div class="cpill" id="conn-cam"><div class="cd"></div><span>CAM</span></div>
    <div class="cpill" id="conn-robot"><div class="cd"></div><span>ROBOT</span></div>
    <div class="cpill" id="conn-ws"><div class="cd"></div><span>WS</span></div>
  </div>
  <button class="hd-btn" onclick="downloadReport()" style="background:var(--blue);color:#fff;border-color:var(--blue)">📋 보고서</button>
  <button class="hd-btn" onclick="downloadLog()">로그</button>
  <div class="hd-right">
    <div class="hd-session"><div id="session-info" style="color:var(--t3)">세션 시작 전</div></div>
    <div class="hd-clock" id="clock">—</div>
  </div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- 통계 카드 7개 -->
  <div class="stats-row">
    <div class="sc sg">
      <div class="sc-lbl">총 수확량</div>
      <div class="sc-val" id="v-harvest">0<span class="sc-unit">개</span></div>
    </div>
    <div class="sc sb">
      <div class="sc-lbl">수확 성공률</div>
      <div class="sc-val" id="v-rate" style="color:var(--t3)">—</div>
      <div class="sc-bar"><div class="sc-bar-f" id="rate-bar" style="background:var(--blue)"></div></div>
    </div>
    <div class="sc sc2">
      <div class="sc-lbl">평균 파지 시간</div>
      <div class="sc-val" id="v-avg" style="color:var(--t3)">—</div>
    </div>
    <div class="sc sp">
      <div class="sc-lbl">수확 속도</div>
      <div class="sc-val" id="v-speed" style="color:var(--t3)">—</div>
      <div id="v-speed-u" style="font-size:9px;color:var(--t3)"></div>
    </div>
    <div class="sc sy">
      <div class="sc-lbl">현재 수확 시간</div>
      <div class="sc-val" id="v-cur" style="color:var(--t3)">—</div>
    </div>
    <div class="sc sr">
      <div class="sc-lbl">손상률</div>
      <div class="sc-val" id="v-dmg" style="color:var(--t3)">—</div>
      <div class="sc-bar"><div class="sc-bar-f" id="dmg-bar" style="background:var(--red)"></div></div>
    </div>
    <div class="sc ss">
      <div class="sc-lbl">총 시도 횟수</div>
      <div class="sc-val" id="v-att" style="color:var(--slate)">0<span class="sc-unit">회</span></div>
    </div>
    <div class="sc" style="--ac:var(--cyan)">
      <div class="sc::before" style="background:var(--cyan)"></div>
      <div class="sc-lbl" style="color:var(--t3)">감지 딸기</div>
      <div class="sc-val" id="v-det" style="color:var(--cyan)">0<span class="sc-unit">개</span></div>
      <div style="font-size:9px;color:var(--t3);margin-top:2px" id="v-det-sub">미수확 —</div>
    </div>
  </div>

  <!-- 세션 정보 바 -->
  <div class="sinfo-bar">
    <div class="sinfo-cell">
      <span class="sinfo-lbl">세션 시작</span>
      <span class="sinfo-val" id="si-start">—</span>
    </div>
    <div class="sinfo-cell">
      <span class="sinfo-lbl">예약 운영</span>
      <input class="sinfo-inp" type="number" id="si-dur-inp" min="0.1" max="24" step="0.5" value="" placeholder="—">
      <span class="sinfo-unit">시간</span>
      <button class="sinfo-btn" onclick="setPlannedDuration()">설정</button>
    </div>
    <div class="sinfo-cell">
      <span class="sinfo-lbl">종료 예정</span>
      <span class="sinfo-val" id="si-end" style="color:var(--t3)">—</span>
    </div>
    <div class="sinfo-cell" style="flex:1">
      <span class="sinfo-lbl">남은 시간</span>
      <span class="sinfo-val" id="si-remain" style="color:var(--t3)">—</span>
    </div>
  </div>

  <!-- 하단 그리드 (4열) -->
  <div class="bot">

    <!-- ① 로봇 상태 + 방향조작 -->
    <div class="panel">
      <div class="ph">
        <div class="ph-dot" id="sp-hd-dot" style="background:var(--t3)"></div>
        로봇 상태 · 방향조작
      </div>
      <div style="flex:1;overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent">
        <div class="sp-body" style="flex:none;justify-content:flex-start;padding:28px 10px 6px">
          <div class="sp-icon-wrap">
            <div class="sp-ring" id="sp-ring"></div>
            <div class="sp-icon-bg" id="sp-icon-bg">
              <span id="sp-icon" style="color:var(--t3)">○</span>
            </div>
          </div>
          <div class="sp-state" id="sp-state">대기 중</div>
          <div class="sp-num" id="sp-num">수확 대기</div>
          <div class="sp-timer" id="sp-timer"></div>
          <div class="sp-div"></div>
          <div class="sp-mini">
            <div class="sp-mrow"><span class="sp-mlbl">성공</span><span class="sp-mval" id="mini-ok" style="color:var(--green)">—</span></div>
            <div class="sp-mrow"><span class="sp-mlbl">실패</span><span class="sp-mval" id="mini-fail" style="color:var(--red)">—</span></div>
            <div class="sp-mrow"><span class="sp-mlbl">손상</span><span class="sp-mval" id="mini-dmg" style="color:var(--yellow)">—</span></div>
          </div>
        </div>
        <!-- 방향조작 구분선 -->
        <div style="height:2px;background:linear-gradient(90deg,var(--green),var(--cyan));border-radius:1px;margin:2px 10px 5px;opacity:.35"></div>
        <div style="padding:0 10px 8px">
          <div class="sec-lbl">방향 조작</div>
          <div class="dpad">
            <div></div>
            <button class="dpbtn" onmousedown="sendTeleop('forward')" onmouseup="sendTeleop('stop')" ontouchstart="sendTeleop('forward')" ontouchend="sendTeleop('stop')">↑</button>
            <div></div>
            <button class="dpbtn" onmousedown="sendTeleop('left')" onmouseup="sendTeleop('stop')" ontouchstart="sendTeleop('left')" ontouchend="sendTeleop('stop')">←</button>
            <button class="dpbtn dstop" onclick="sendTeleop('stop')">■</button>
            <button class="dpbtn" onmousedown="sendTeleop('right')" onmouseup="sendTeleop('stop')" ontouchstart="sendTeleop('right')" ontouchend="sendTeleop('stop')">→</button>
            <div></div>
            <button class="dpbtn" onmousedown="sendTeleop('backward')" onmouseup="sendTeleop('stop')" ontouchstart="sendTeleop('backward')" ontouchend="sendTeleop('stop')">↓</button>
            <div></div>
          </div>
          <div class="dp2row">
            <button class="dpbtn2" onmousedown="sendTeleop('up')"         onmouseup="sendTeleop('stop')">Z↑</button>
            <button class="dpbtn2" onmousedown="sendTeleop('down')"       onmouseup="sendTeleop('stop')">Z↓</button>
            <button class="dpbtn2" onmousedown="sendTeleop('rotate_ccw')" onmouseup="sendTeleop('stop')">↶</button>
            <button class="dpbtn2" onmousedown="sendTeleop('rotate_cw')"  onmouseup="sendTeleop('stop')">↷</button>
          </div>
          <div class="dp-speed">
            <label>속도</label>
            <input type="range" id="tp-speed" min="0.05" max="1" step="0.05" value="0.2"
              oninput="document.getElementById('tp-spd-val').textContent=Math.round(this.value*100)+'%'">
            <span id="tp-spd-val">20%</span>
          </div>
          <div class="teleop-st" id="teleop-st">정지</div>
          <!-- 추가 회전 버튼 -->
          <div style="margin-top:6px">
            <div style="font-size:9px;color:var(--t3);margin-bottom:2px;font-weight:700">회전 각도</div>
            <input type="range" id="rot-speed" min="1" max="20" step="1" value="5"
              style="width:100%;height:4px;cursor:pointer"
              oninput="document.getElementById('rot-spd-val').textContent=this.value+'°'">
            <span id="rot-spd-val" style="font-size:8px;color:var(--t3)">5°</span>
          </div>
          <div class="rot-btns" style="margin-top:4px">
            <button class="rot-btn" onclick="teleopMove('rx_plus')">Rx+</button>
            <button class="rot-btn" onclick="teleopMove('rx_minus')">Rx-</button>
            <button class="rot-btn" onclick="teleopMove('ry_plus')">Ry+</button>
            <button class="rot-btn" onclick="teleopMove('ry_minus')">Ry-</button>
            <button class="rot-btn" onclick="teleopMove('rz_plus')">Rz+</button>
            <button class="rot-btn" onclick="teleopMove('rz_minus')">Rz-</button>
          </div>
        </div>
        <!-- 데이터 수집 구분선 -->
        <div class="rec-sep"></div>
        <div class="rec-section">
          <div class="sec-lbl" style="color:var(--red-tx)">데이터 수집</div>
          <!-- 녹화 상태 -->
          <div class="rec-status-badge" id="rec-badge">
            <div class="rec-dot" id="rec-dot"></div>
            <span id="rec-badge-txt">대기</span>
          </div>
          <!-- 메타 입력 -->
          <div class="rec-inp-row">
            <span class="rec-inp-lbl">에피소드 (비우면 자동)</span>
            <input class="rec-inp" id="rec-ep" type="text" placeholder="자동 부여">
          </div>
          <div class="rec-inp-row">
            <span class="rec-inp-lbl">태스크</span>
            <input class="rec-inp" id="rec-task" type="text"
              value="Grasp the strawberry stem and pick it.">
          </div>
          <div class="rec-inp-row">
            <span class="rec-inp-lbl">카테고리</span>
            <input class="rec-inp" id="rec-cat" type="text" placeholder="no_occlusion">
          </div>
          <!-- 홈 포즈 -->
          <div class="rec-inp-row">
            <span class="rec-inp-lbl">홈 포즈</span>
            <select class="rec-inp" id="rec-home">
              <option value="top_left">top_left</option>
              <option value="top_right">top_right</option>
              <option value="bottom_left">bottom_left</option>
              <option value="bottom_right">bottom_right</option>
            </select>
          </div>
          <!-- 데이터 저장 경로 -->
          <div class="rec-inp-row">
            <span class="rec-inp-lbl">데이터 경로</span>
            <input class="rec-inp" id="rec-rawdir" type="text"
              placeholder="/vla_ws/data/raw/final_project/vla_dataset_v0.3.0"
              value="/vla_ws/data/raw/final_project/vla_dataset_v0.3.0"
              style="font-size:9px;padding:4px">
          </div>
          <!-- 제어 버튼 -->
          <div class="rec-btns">
            <button class="rec-btn start" id="btn-rec-start" onclick="startRecording()">▶ 녹화 시작</button>
            <button class="rec-btn stop"  id="btn-rec-stop"  onclick="stopRecording()" disabled>■ 녹화 종료</button>
            <button class="rec-btn home"  onclick="moveHome()">⌂ 홈 이동</button>
            <button class="rec-btn convert" onclick="convertData()">⚙ 데이터 변환</button>
            <!-- 변환 진행률 바 -->
            <div id="convert-progress-container" style="display:none;margin-top:6px">
              <div style="font-size:9px;color:var(--t3);margin-bottom:2px;font-weight:700">변환 중...</div>
              <div style="width:100%;height:12px;background:var(--bg);border:1px solid var(--border);border-radius:2px;overflow:hidden">
                <div id="convert-progress-bar" style="height:100%;background:var(--purple);width:0%;transition:width 0.3s;display:flex;align-items:center;justify-content:center">
                  <span id="convert-progress-text" style="font-size:8px;color:#fff;font-weight:700"></span>
                </div>
              </div>
            </div>
          </div>
          <!-- 그리퍼 -->
          <div class="sec-lbl" style="color:var(--cyan-tx);margin-top:4px">그리퍼 즉시 제어</div>
          <div class="grip-btns">
            <button class="grip-btn" onclick="teleopGripper(0)">열기</button>
            <button class="grip-btn" onclick="teleopGripper(740)">파지</button>
            <button class="grip-btn" onclick="teleopGripper(600)">홈(600)</button>
          </div>
        </div>
      </div>
    </div>

    <!-- ② 관절 + TCP + 그리퍼 (세로 배치) -->
    <div class="panel">
      <div class="ph">
        <div class="ph-dot" style="background:var(--blue)"></div>
        관절 · TCP · 그리퍼
        <span class="ph-badge" id="j-badge">대기</span>
        <span class="ph-badge" id="tg-badge" style="background:var(--purple-bg);border-color:#ddd6fe;color:var(--purple-tx)">대기</span>
        <span class="ph-badge" id="gr-badge" style="background:var(--cyan-bg);border-color:#a5f3fc;color:var(--cyan-tx)">열림</span>
      </div>
      <div class="tab-body" style="overflow-y:auto;scrollbar-width:thin;scrollbar-color:var(--border) transparent">
        <!-- 관절 -->
        <div class="sec-lbl">현재 각도</div>
        <div class="jrow"><span class="jlbl">J1</span><div class="jtrack"><div class="jfill" id="jf1"></div><div class="jcenter"></div></div><span class="jval" id="jv1">0.0°</span></div>
        <div class="jrow"><span class="jlbl">J2</span><div class="jtrack"><div class="jfill" id="jf2"></div><div class="jcenter"></div></div><span class="jval" id="jv2">0.0°</span></div>
        <div class="jrow"><span class="jlbl">J3</span><div class="jtrack"><div class="jfill" id="jf3"></div><div class="jcenter"></div></div><span class="jval" id="jv3">0.0°</span></div>
        <div class="jrow"><span class="jlbl">J4</span><div class="jtrack"><div class="jfill" id="jf4"></div><div class="jcenter"></div></div><span class="jval" id="jv4">0.0°</span></div>
        <div class="jrow"><span class="jlbl">J5</span><div class="jtrack"><div class="jfill" id="jf5"></div><div class="jcenter"></div></div><span class="jval" id="jv5">0.0°</span></div>
        <div class="jrow"><span class="jlbl">J6</span><div class="jtrack"><div class="jfill" id="jf6"></div><div class="jcenter"></div></div><span class="jval" id="jv6">0.0°</span></div>
        <div class="jdiv"></div>
        <div class="sec-lbl">관절 이동 명령</div>
        <div class="jgrid">
          <div class="jcell"><label>J1 (°)</label><input type="number" id="ji1" step="1" min="-360" max="360" value="0"></div>
          <div class="jcell"><label>J2 (°)</label><input type="number" id="ji2" step="1" min="-360" max="360" value="0"></div>
          <div class="jcell"><label>J3 (°)</label><input type="number" id="ji3" step="1" min="-150" max="150" value="0"></div>
          <div class="jcell"><label>J4 (°)</label><input type="number" id="ji4" step="1" min="-360" max="360" value="0"></div>
          <div class="jcell"><label>J5 (°)</label><input type="number" id="ji5" step="1" min="-360" max="360" value="0"></div>
          <div class="jcell"><label>J6 (°)</label><input type="number" id="ji6" step="1" min="-360" max="360" value="0"></div>
        </div>
        <div class="jfoot">
          <div class="jvel"><label>속도(%)</label><input type="number" id="jvel" value="10" min="1" max="100"></div>
          <div class="jbtns">
            <button class="jbtn s" onclick="loadCurAngles()">현재값</button>
            <button class="jbtn p" onclick="sendJointCmd().catch(()=>{})">이동</button>
          </div>
        </div>
        <div class="jstatus" id="jstatus"></div>
        <!-- TCP 구분선 -->
        <div style="height:2px;background:linear-gradient(90deg,var(--blue),var(--purple));border-radius:1px;margin:7px 0 6px;opacity:.35"></div>
        <!-- TCP 현재 좌표 -->
        <div class="sec-lbl" style="color:var(--purple-tx)">TCP 현재 좌표</div>
        <div class="tcp-dgrid">
          <div class="tcp-cell"><label>X (mm)</label><div class="tcp-disp" id="td0">0.0</div></div>
          <div class="tcp-cell"><label>Y (mm)</label><div class="tcp-disp" id="td1">0.0</div></div>
          <div class="tcp-cell"><label>Z (mm)</label><div class="tcp-disp" id="td2">0.0</div></div>
          <div class="tcp-cell"><label>RX (°)</label><div class="tcp-disp" id="td3">0.0</div></div>
          <div class="tcp-cell"><label>RY (°)</label><div class="tcp-disp" id="td4">0.0</div></div>
          <div class="tcp-cell"><label>RZ (°)</label><div class="tcp-disp" id="td5">0.0</div></div>
        </div>
        <div class="jdiv"></div>
        <div class="sec-lbl" style="color:var(--purple-tx)">TCP 이동 명령</div>
        <div class="tcp-dgrid">
          <div class="tcp-cell"><label>X</label><input class="tcp-inp" type="number" id="ti0" step="1" value="0"></div>
          <div class="tcp-cell"><label>Y</label><input class="tcp-inp" type="number" id="ti1" step="1" value="0"></div>
          <div class="tcp-cell"><label>Z</label><input class="tcp-inp" type="number" id="ti2" step="1" value="0"></div>
          <div class="tcp-cell"><label>RX</label><input class="tcp-inp" type="number" id="ti3" step="0.1" value="0"></div>
          <div class="tcp-cell"><label>RY</label><input class="tcp-inp" type="number" id="ti4" step="0.1" value="0"></div>
          <div class="tcp-cell"><label>RZ</label><input class="tcp-inp" type="number" id="ti5" step="0.1" value="0"></div>
        </div>
        <div class="jfoot">
          <div class="jvel"><label>속도(%)</label><input type="number" id="tvel" value="10" min="1" max="100"></div>
          <div class="jbtns">
            <button class="jbtn s" onclick="loadCurTcp()">현재값</button>
            <button class="jbtn p" onclick="sendTcpCmd().catch(()=>{})">이동</button>
          </div>
        </div>
        <!-- 그리퍼 구분선 -->
        <div style="height:2px;background:linear-gradient(90deg,var(--purple),var(--cyan));border-radius:1px;margin:7px 0 6px;opacity:.35"></div>
        <!-- 그리퍼 상태 -->
        <div class="sec-lbl" style="color:var(--cyan-tx)">그리퍼 상태</div>
        <div class="g-status-row">
          <div class="g-viz">
            <div class="g-jaw" id="gjl"></div>
            <div id="gjgap" style="flex-shrink:0;transition:width .3s"></div>
            <div class="g-jaw" id="gjr"></div>
          </div>
          <div>
            <div class="g-state" id="g-state">열림</div>
            <div class="g-pct" id="g-pct">100% | 30 N</div>
          </div>
        </div>
        <div class="g-ctrl" style="margin-top:4px">
          <button class="g-btn op" onclick="sendGripCmd(100)">열기</button>
          <input type="range" class="g-slider" id="g-slider" min="0" max="100" value="100"
            oninput="document.getElementById('g-inp').value=this.value">
          <button class="g-btn cl" onclick="sendGripCmd(0)">닫기</button>
        </div>
        <div class="g-foot">
          <div class="g-fld">
            <label>개도(%)</label>
            <input type="number" id="g-inp" value="100" min="0" max="100"
              oninput="document.getElementById('g-slider').value=this.value">
          </div>
          <div class="g-fld"><label>힘(N)</label><input type="number" id="g-force" value="30" min="0" max="150"></div>
          <div style="display:flex;flex-direction:column;gap:1px;flex:1">
            <label style="font-size:8px;color:transparent">-</label>
            <button class="jbtn p" onclick="sendGripCmd()">이동</button>
          </div>
        </div>
      </div>
    </div>

    <!-- ③ 카메라 (듀얼) -->
    <div class="panel">
      <div class="ph">
        <div class="ph-dot" id="cam-hd-dot" style="background:var(--t3)"></div>
        카메라 피드
        <span class="ph-badge" id="cam0-src" style="background:var(--border-s);border-color:var(--border);color:var(--t3)">YOLO · 연결 중</span>
        <span class="ph-badge" id="cam1-src" style="background:var(--border-s);border-color:var(--border);color:var(--t3)">전경 · 연결 중</span>
      </div>
      <div style="display:flex;flex:1;min-height:0;gap:4px;padding:4px">

        <!-- 슬롯 0: 딸기 인식 카메라 -->
        <div style="flex:1;display:flex;flex-direction:column;min-width:0">
          <div style="font-size:8.5px;font-weight:700;color:var(--red-tx);letter-spacing:.5px;
                      text-align:center;padding:2px 0;background:var(--red-bg);
                      border-radius:4px 4px 0 0;border:1px solid #fecaca;border-bottom:none;flex-shrink:0">
            딸기 인식
          </div>
          <div class="cam-body" id="cam-body-0" onclick="toggleCamFS(0)" style="border-radius:0 0 4px 4px;flex:1">
            <img id="cam-img-0" class="cam-img" alt="" style="display:none">
            <div id="cam-live-0" class="cam-live"><div class="cam-live-dot"></div>LIVE</div>
            <div id="cam-fps-0" class="cam-fps"></div>
            <div id="cam-scan-0" class="cam-scan"></div>
            <div class="cam-corner tl"></div><div class="cam-corner tr"></div>
            <div class="cam-corner bl"></div><div class="cam-corner br"></div>
            <div class="cam-ph" id="cam-ph-0">
              <div class="cam-spin" id="cam-spin-0"></div>
              <div class="cam-ph-txt" id="cam-ph-txt-0">카메라 연결 없음</div>
            </div>
            <div class="cam-ov" id="cam-ov-0">
              <span id="cam-ov-ts"></span>
              <span id="cam-ov-src-0"></span>
            </div>
            <div class="cam-fs-hint">클릭: 전체화면</div>
          </div>
        </div>

        <!-- 슬롯 1: 전경 카메라 -->
        <div style="flex:1;display:flex;flex-direction:column;min-width:0">
          <div style="font-size:8.5px;font-weight:700;color:var(--blue-tx);letter-spacing:.5px;
                      text-align:center;padding:2px 0;background:var(--blue-bg);
                      border-radius:4px 4px 0 0;border:1px solid #bfdbfe;border-bottom:none;flex-shrink:0">
            전경 카메라
          </div>
          <div class="cam-body" id="cam-body-1" onclick="toggleCamFS(1)" style="border-radius:0 0 4px 4px;flex:1">
            <img id="cam-img-1" class="cam-img" alt="" style="display:none">
            <div id="cam-live-1" class="cam-live"><div class="cam-live-dot"></div>LIVE</div>
            <div id="cam-fps-1" class="cam-fps"></div>
            <div id="cam-scan-1" class="cam-scan"></div>
            <div class="cam-corner tl"></div><div class="cam-corner tr"></div>
            <div class="cam-corner bl"></div><div class="cam-corner br"></div>
            <div class="cam-ph" id="cam-ph-1">
              <div class="cam-spin" id="cam-spin-1"></div>
              <div class="cam-ph-txt" id="cam-ph-txt-1">카메라 연결 없음</div>
            </div>
            <div class="cam-fs-hint">클릭: 전체화면</div>
          </div>
        </div>

      </div>
      <div class="snap-strip">
        <div class="snap-gallery" id="snap-gallery">
          <div class="snap-empty">수확 시 자동 캡처됩니다</div>
        </div>
      </div>
    </div>

    <!-- ④ 메시지 -->
    <div class="panel">
      <div class="ph">
        <div class="ph-dot" style="background:var(--green);animation:blink 1.5s infinite"></div>
        실시간 메시지
        <span class="ph-badge" id="msg-cnt">0</span>
      </div>
      <div class="msg-list" id="msg-list">
        <div class="mi-info"><div class="msg-wrap"><div class="msg-item">
          <span class="msg-ts">—</span>
          <div class="msg-body">
            <div class="msg-chip">INFO</div>
            <div class="msg-txt">시스템 연결 대기 중...</div>
          </div>
        </div></div></div>
      </div>
    </div>

  </div><!-- bot -->
</div><!-- main -->

<div class="ws-badge dis" id="ws-badge">
  <div class="ws-dot"></div><span id="ws-txt">연결 중...</span>
</div>

<script>
/* 상수 */
const STATUS = {
  idle:       {label:'대기 중', icon:'●', color:'var(--green)',  ring:false},  // 로봇 연결됨
  approaching:{label:'접근 중', icon:'→', color:'var(--cyan)',   ring:true},
  grasping:   {label:'파지 중', icon:'●', color:'var(--yellow)', ring:true},
  returning:  {label:'복귀 중', icon:'←', color:'var(--green)',  ring:true},
  error:      {label:'오류',    icon:'✕', color:'var(--red)',    ring:false},
};
const CBGMAP = {'var(--cyan)':'var(--cyan-bg)','var(--yellow)':'var(--yellow-bg)',
  'var(--green)':'var(--green-bg)','var(--red)':'var(--red-bg)','var(--t3)':'var(--bg)'};
const LVL_CLS  = {info:'mi-info',success:'mi-success',warning:'mi-warning',error:'mi-error'};
const LVL_CHIP = {info:'INFO',success:'SUCCESS',warning:'WARNING',error:'ERROR'};
const J_LIM    = [[-360,360],[-360,360],[-150,150],[-360,360],[-360,360],[-360,360]];
const ECHART_T = [
  {key:'ik',       label:'IK 실패',  color:'var(--purple)'},
  {key:'obstacle', label:'장애물',   color:'var(--yellow)'},
  {key:'grasp',    label:'파지 실패',color:'var(--red)'},
  {key:'detection',label:'감지',     color:'var(--cyan)'},
  {key:'other',    label:'기타',     color:'var(--slate)'},
];
const GSTATES = {
  open:    {label:'열림',   color:'var(--green)'},
  closed:  {label:'닫힘',   color:'var(--blue)'},
  grasping:{label:'파지 중',color:'var(--yellow)'},
  error:   {label:'오류',   color:'var(--red)'},
};

let lastState={}, prevDmg=0, prevMsgCnt=0, harvestStart=null;

function fmt(s){return String(Math.floor(s/60)).padStart(2,'0')+':'+String(Math.floor(s%60)).padStart(2,'0')}

/* 시계 */
setInterval(()=>{
  document.getElementById('clock').textContent=
    new Date().toLocaleString('ko-KR',{year:'numeric',month:'2-digit',day:'2-digit',
      hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});
  document.getElementById('cam-ov-ts').textContent=
    new Date().toLocaleTimeString('ko-KR',{hour12:false});
},500);

/* 수확 타이머 */
setInterval(()=>{
  const ce=document.getElementById('v-cur'), te=document.getElementById('sp-timer');
  if(!harvestStart){ce.textContent='—';ce.style.color='var(--t3)';te.textContent='';return;}
  const s=fmt((Date.now()-harvestStart)/1000);
  ce.textContent=s;ce.style.color='var(--yellow)';te.textContent=s;
},200);

/* 남은 시간 카운트다운 */
let _siSessionStart=null, _siPlannedHours=0;
setInterval(()=>{
  const remEl=document.getElementById('si-remain');
  if(!remEl)return;
  if(!_siSessionStart||!_siPlannedHours){remEl.textContent='—';remEl.style.color='var(--t3)';return;}
  const endMs=_siSessionStart.getTime()+_siPlannedHours*3600000;
  const remMs=endMs-Date.now();
  if(remMs<=0){remEl.textContent='종료';remEl.style.color='var(--t3)';return;}
  const rs=Math.floor(remMs/1000);
  const hh=Math.floor(rs/3600),mm=Math.floor((rs%3600)/60),ss=rs%60;
  remEl.textContent=hh>0
    ?`${hh}:${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`
    :`${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
  remEl.style.color=remMs<1800000?'var(--red)':remMs<3600000?'var(--yellow)':'var(--green)';
},1000);

/* 데미지 플래시 */
function triggerDmgFlash(){
  const el=document.getElementById('dmg-flash');
  el.style.transition='none';el.style.opacity='1';
  requestAnimationFrame(()=>requestAnimationFrame(()=>{
    el.style.transition='opacity 1s ease';el.style.opacity='0';
  }));
}

/* 로그 저장 */
function downloadLog(){
  const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),...lastState},null,2)],
    {type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='harvest_log_'+new Date().toISOString().slice(0,19).replace(/[T:]/g,'-')+'.json';
  document.body.appendChild(a);a.click();document.body.removeChild(a);
}

/* 보고서 다운로드 */
async function downloadReport(){
  try{
    const r=await fetch('/api/report');
    if(!r.ok)throw new Error();
    const data=await r.json();
    const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
    const a=document.createElement('a');
    a.href=URL.createObjectURL(blob);
    a.download='harvest_report_'+new Date().toISOString().slice(0,19).replace(/[T:]/g,'-')+'.json';
    document.body.appendChild(a);a.click();document.body.removeChild(a);
  }catch(e){alert('보고서 생성 실패');}
}

/* 스냅샷 */
let snapshots=[],prevSuccSnap=0,prevAttSnap=0;
async function captureSnapshot(event){
  try{
    const img=document.getElementById('cam-img-0');
    if(!img||!cams[0].live)return;
    const c=document.createElement('canvas');
    c.width=img.naturalWidth||320;c.height=img.naturalHeight||240;
    c.getContext('2d').drawImage(img,0,0);
    const dataUrl=c.toDataURL('image/jpeg',0.75);
    const time=new Date().toLocaleTimeString('ko-KR',{hour12:false});
    snapshots.unshift({dataUrl,event,time});
    if(snapshots.length>12)snapshots.pop();
    renderSnapshots();
  }catch(e){}
}
function renderSnapshots(){
  const el=document.getElementById('snap-gallery');if(!el)return;
  if(!snapshots.length){el.innerHTML='<div class="snap-empty">수확 시 자동 캡처됩니다</div>';return;}
  el.innerHTML=snapshots.map((s,i)=>
    `<div class="snap-item ${s.event}" onclick="viewSnap(${i})" title="${s.time} ${s.event==='success'?'✓성공':'✕실패'}">
      <img class="snap-img" src="${s.dataUrl}">
      <div class="snap-badge">${s.event==='success'?'✓':'✕'}</div>
    </div>`).join('');
}
function viewSnap(i){
  const s=snapshots[i];if(!s)return;
  const w=window.open('','_blank','width=720,height=560');
  w.document.write(`<html><body style="margin:0;background:#111">
    <img src="${s.dataUrl}" style="max-width:100%;display:block">
    <p style="color:#fff;font:12px monospace;padding:8px">${s.time} — ${s.event==='success'?'수확 성공':'수확 실패'}</p>
  </body></html>`);
}
function checkSnapshots(s){
  const succ=s.success_count||0,att=s.total_attempts||0;
  if(succ>prevSuccSnap){captureSnapshot('success');prevSuccSnap=succ;}
  else if(att>prevAttSnap&&succ===prevSuccSnap){captureSnapshot('fail');}
  prevAttSnap=att;
}

/* 예약 운영 시간 설정 */
async function setPlannedDuration(){
  const h=parseFloat(document.getElementById('si-dur-inp').value);
  if(!h||h<=0)return;
  await fetch('/api/set-planned-duration',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({hours:h})}).catch(()=>{});
}

/* 목표 설정 */
async function setTarget(){
  const v=parseInt(document.getElementById('bg-inp').value);
  if(!v||v<1)return;
  await fetch('/api/set-target',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({target:v})}).catch(()=>{});
}

/* 관절 */
function loadCurAngles(){
  (lastState.joint_angles||[0,0,0,0,0,0]).forEach((v,i)=>{
    const el=document.getElementById('ji'+(i+1));if(el)el.value=v.toFixed(1);});
}
async function sendJointCmd(){
  const angles=[];
  for(let i=1;i<=6;i++){const e=document.getElementById('ji'+i);angles.push(parseFloat(e?.value)||0);}
  const vel=parseFloat(document.getElementById('jvel').value)||10;
  const badge=document.getElementById('j-badge');
  const stat=document.getElementById('jstatus');
  try{
    const r=await fetch(TELEOP_API+'/move',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'joint',angles,velocity:vel})});
    if(r.ok){
      badge.textContent='전송됨';
      badge.style.cssText='background:var(--green-bg);border-color:#bbf7d0;color:var(--green-tx)';
      stat.style.color='var(--green-tx)';stat.textContent='명령 전송됨';
      setTimeout(()=>{badge.textContent='대기';badge.style.cssText='';stat.textContent='';},3500);
    }else{
      stat.style.color='var(--red-tx)';stat.textContent='로봇 제어 오류';
    }
  }catch(e){stat.style.color='var(--red-tx)';stat.textContent='전송 실패';}
}

/* TCP */
function loadCurTcp(){
  (lastState.tcp_pose||[0,0,0,0,0,0]).forEach((v,i)=>{
    const e=document.getElementById('ti'+i);if(e)e.value=v.toFixed(i<3?1:2);});
}
async function sendTcpCmd(){
  const pose=[];
  for(let i=0;i<6;i++){const e=document.getElementById('ti'+i);pose.push(parseFloat(e?.value)||0);}
  const vel=parseFloat(document.getElementById('tvel').value)||10;
  const badge=document.getElementById('tg-badge');
  try{
    const r=await fetch(TELEOP_API+'/move',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({command:'tcp',pose,velocity:vel})});
    if(r.ok){
      badge.textContent='TCP 전송';
      badge.style.cssText='background:var(--purple-bg);border-color:#ddd6fe;color:var(--purple-tx)';
      setTimeout(()=>{badge.textContent='대기';badge.style.cssText='';},3500);
    }
  }catch(e){}
}

/* 텔레오프 */
const TELEOP_LBL={forward:'전진 ↑',backward:'후진 ↓',left:'좌 ←',right:'우 →',
  up:'Z↑',down:'Z↓',rotate_cw:'CW ↷',rotate_ccw:'CCW ↶',stop:'정지'};
async function sendTeleop(cmd){
  const sp=parseFloat(document.getElementById('tp-speed')?.value||0.2);
  const ts=document.getElementById('teleop-st');
  if(ts){ts.textContent=TELEOP_LBL[cmd]||cmd;
    ts.style.color=cmd==='stop'?'var(--t3)':'var(--blue)';}
  await fetch('/api/teleop',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({command:cmd,speed:sp})}).catch(()=>{});
}

/* 그리퍼 */
async function sendGripCmd(pos){
  const position=pos!==undefined?pos:parseFloat(document.getElementById('g-inp').value??100);
  const force=parseFloat(document.getElementById('g-force').value)||30;
  let state='open';
  if(position<=5)state='closed';else if(position<95)state='grasping';
  const badge=document.getElementById('tg-badge');
  try{
    const r=await fetch(TELEOP_API+'/gripper',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({position,force,state})});
    if(r.ok){
      badge.textContent='그리퍼 전송';
      badge.style.cssText='background:var(--cyan-bg);border-color:#a5f3fc;color:var(--cyan-tx)';
      setTimeout(()=>{badge.textContent='대기';badge.style.cssText='';},3000);
    }
  }catch(e){}
  const sl=document.getElementById('g-slider');if(sl)sl.value=position;
  const ip=document.getElementById('g-inp');if(ip)ip.value=position.toFixed(0);
}

/* 렌더: 관절 */
function renderJoints(s){
  (s.joint_angles||[0,0,0,0,0,0]).forEach((a,i)=>{
    const[mn,mx]=J_LIM[i];
    const ve=document.getElementById('jv'+(i+1));
    const ie=document.getElementById('ji'+(i+1));  // 관절 입력 필드도 함께 업데이트
    const fe=document.getElementById('jf'+(i+1));
    if(ve)ve.textContent=a.toFixed(1)+'°';
    // 입력 필드가 포커스 중이 아닐 때만 현재 각도로 업데이트 (사용자 입력 보호)
    if(ie && ie !== document.activeElement)ie.value=a.toFixed(1);
    if(fe){
      if(a>=0){const w=Math.min(a/mx*50,50);fe.style.left='50%';fe.style.width=w+'%';fe.style.background='var(--blue)';}
      else{const w=Math.min(Math.abs(a)/Math.abs(mn)*50,50);fe.style.left=(50-w)+'%';fe.style.width=w+'%';fe.style.background='var(--purple)';}
    }
  });
}

/* 렌더: TCP */
function renderTcp(s){
  const p=s.tcp_pose||[0,0,0,0,0,0];
  const u=['mm','mm','mm','°','°','°'];
  p.forEach((v,i)=>{
    const de=document.getElementById('td'+i);if(de)de.textContent=v.toFixed(i<3?1:2)+' '+u[i];
    const ie=document.getElementById('ti'+i);
    if(ie&&document.activeElement!==ie)ie.value=v.toFixed(i<3?1:2);
  });
}

/* 렌더: 그리퍼 */
function renderGripper(s){
  const g=s.gripper||{position:100,state:'open',force:30};
  const pos=g.position??100;
  const st=GSTATES[g.state]||{label:g.state||'?',color:'var(--t3)'};
  const ge=document.getElementById('gjgap');if(ge)ge.style.width=((pos/100)*28).toFixed(1)+'px';
  ['gjl','gjr'].forEach(id=>{const e=document.getElementById(id);if(e)e.style.background=st.color;});
  const se=document.getElementById('g-state');if(se){se.textContent=st.label;se.style.color=st.color;}
  const pe=document.getElementById('g-pct');
  if(pe){
    const raw=(g.raw_pos!=null)?'  |  '+g.raw_pos+'/740':'';
    pe.textContent=pos.toFixed(0)+'%'+raw+'  |  '+(g.force??30).toFixed(0)+' N';
  }
  const sl=document.getElementById('g-slider');if(sl&&document.activeElement!==sl)sl.value=pos;
  const ip=document.getElementById('g-inp');if(ip&&document.activeElement!==ip)ip.value=pos.toFixed(0);
  const gb=document.getElementById('gr-badge');if(gb)gb.textContent=st.label;
}

/* ETA 계산 */
function calcETA(s){
  const t=s.target_count||15,c=s.success_count||0;
  if(c>=t)return'완료!';
  if(!s.session_start||c===0)return'—';
  const elapsed=(Date.now()-new Date(s.session_start))/1000;
  if(elapsed<5)return'—';
  const rate=c/elapsed;
  const eta=Math.round((t-c)/rate);
  if(eta>7200)return'2h+';
  const m=Math.floor(eta/60),sec=eta%60;
  return m>0?`약 ${m}분 후`:`약 ${sec}초 후`;
}

/* 렌더: 목표 + ETA (그래프 배너) */
function renderTarget(s){
  const t=s.target_count||15,c=s.success_count||0;
  const p=Math.min(c/t*100,100);
  const col=p>=100?'var(--green)':p>=70?'var(--blue)':'var(--cyan)';
  const bar=document.getElementById('bg-bar');
  if(bar){bar.style.width=p+'%';bar.style.background=col;}
  const ce=document.getElementById('bg-cur');
  if(ce){ce.textContent=c;ce.style.color=col;}
  const me=document.getElementById('bg-max');if(me)me.textContent=t;
  const ie=document.getElementById('bg-inp');if(ie&&document.activeElement!==ie)ie.value=t;
  const eta=document.getElementById('bg-eta');
  if(eta){const v=calcETA(s);eta.textContent='ETA: '+v;
    eta.style.color=v==='완료!'?'var(--green)':v==='—'?'var(--t3)':'var(--blue)';}
}

/* 렌더: 파지 분포 히스토그램 */
function renderGraspHistogram(s){
  const times=s.grasp_times||[];
  const el=document.getElementById('bg-grasp-hist');if(!el)return;
  if(!times.length){el.innerHTML='<div class="bg-hempty">데이터 없음</div>';return;}
  const bins=[0,0,0,0,0];
  times.forEach(t=>{
    if(t<=5)bins[0]++;else if(t<=10)bins[1]++;else if(t<=15)bins[2]++;
    else if(t<=20)bins[3]++;else bins[4]++;
  });
  const mx=Math.max(1,...bins);
  const lbls=['≤5s','≤10','≤15','≤20','>20'];
  const cols=['var(--green)','var(--cyan)','var(--yellow)','var(--red)','var(--purple)'];
  el.innerHTML=bins.map((b,i)=>`
    <div class="bg-gbin">
      <div class="bg-gbar-w">
        <div class="bg-gbar-f" style="height:${(b/mx*100).toFixed(0)}%;background:${cols[i]}"></div>
      </div>
      <div class="bg-gbin-lbl">${lbls[i]}</div>
      <div class="bg-gbin-cnt">${b}</div>
    </div>`).join('');
}

/* 렌더: 수확 불가 분류 */
function renderSkipReasons(s){
  const sr=s.skip_reasons||{};
  const total=Math.max(1,Object.values(sr).reduce((a,b)=>a+b,0));
  [['immature','미성숙'],['occluded','가림'],['harvested','수확됨'],['other','기타']].forEach(([k])=>{
    const c=sr[k]||0,pct=(c/total*100).toFixed(0)+'%';
    const b=document.getElementById('sk-'+k);if(b)b.style.width=c?pct:'0%';
    const n=document.getElementById('skc-'+k);if(n)n.textContent=c;
  });
}

/* 렌더: 감지 딸기 */
function renderDetected(s){
  const det=s.detected_count||0;
  const att=s.total_attempts||0;
  const el=document.getElementById('v-det');if(el)el.textContent=det;
  const sub=document.getElementById('v-det-sub');
  if(sub){
    const missed=Math.max(0,det-att);
    sub.textContent=att>0?`미수확 ${missed}개 (${(missed/Math.max(1,det)*100).toFixed(0)}%)`:'미수확 —';
  }
}

/* 렌더: 히스토리 (그래프 배너) */
function renderHistory(s){
  const h=s.attempt_history||[];
  const el=document.getElementById('bg-hist');if(!el)return;
  if(!h.length){el.innerHTML='<div class="bg-hempty">기록 없음</div>';return;}
  el.innerHTML=h.map(v=>`<div class="bg-hbar ${v?'s':'f'}" title="${v?'성공':'실패'}"></div>`).join('');
}

/* 렌더: 오류 유형 */
function renderFailureChart(s){
  const ft=s.failure_types||{};
  const total=Math.max(1,Object.values(ft).reduce((a,b)=>a+b,0));
  /* 그래프 배너 */
  ECHART_T.forEach(t=>{
    const c=ft[t.key]||0;
    const pct=(c/total*100).toFixed(0)+'%';
    const be=document.getElementById('bge-'+t.key);if(be)be.style.width=c?pct:'0%';
    const ce=document.getElementById('bgc-'+t.key);if(ce)ce.textContent=c;
  });
  /* 그래프·조작 패널 */
  ECHART_T.forEach(t=>{
    const c=ft[t.key]||0;
    const pct=(c/total*100).toFixed(0)+'%';
    const bar=document.getElementById('ep-'+t.key);if(bar)bar.style.width=c?pct:'0%';
    const cnt=document.getElementById('ec-'+t.key);if(cnt)cnt.textContent=c;
  });
}

/* 렌더: 연결 상태 */
function renderConns(s){
  const cam=document.getElementById('conn-cam');
  const anyLive=cams.some(c=>c.live)||((s.camera_fps_0||0)>1)||((s.camera_fps_1||0)>1);
  cam.className='cpill '+(anyLive?'on':'off');
  const rob=document.getElementById('conn-robot');
  if(s.last_updated){
    const age=(Date.now()-new Date(s.last_updated))/1000;
    rob.className='cpill '+(age<5?'on':age<30?'warn':'off');
  }else rob.className='cpill off';
}

/* MAIN RENDER */
function render(s){
  lastState=s;
  if((s.damage_count||0)>prevDmg)triggerDmgFlash();
  prevDmg=s.damage_count||0;

  if(s.session_start){
    const st=new Date(s.session_start);
    document.getElementById('session-info').textContent=
      st.toLocaleTimeString('ko-KR',{hour12:false})+' 시작 / '+fmt((Date.now()-st)/1000)+' 경과';
    _siSessionStart=st;
    document.getElementById('si-start').textContent=st.toLocaleTimeString('ko-KR',{hour12:false});
  } else {
    _siSessionStart=null;
    document.getElementById('si-start').textContent='—';
  }
  _siPlannedHours=s.planned_duration_hours||0;
  if(_siPlannedHours>0){
    const inp=document.getElementById('si-dur-inp');
    if(inp&&inp!==document.activeElement)inp.value=_siPlannedHours;
    if(_siSessionStart){
      const endTime=new Date(_siSessionStart.getTime()+_siPlannedHours*3600000);
      document.getElementById('si-end').textContent=endTime.toLocaleTimeString('ko-KR',{hour12:false});
      document.getElementById('si-end').style.color='var(--t1)';
    }
  } else {
    document.getElementById('si-end').textContent='—';
    document.getElementById('si-end').style.color='var(--t3)';
  }

  document.getElementById('v-harvest').innerHTML=(s.success_count||0)+'<span class="sc-unit">개</span>';
  document.getElementById('v-att').innerHTML=(s.total_attempts||0)+'<span class="sc-unit">회</span>';

  /* 성공률: 익은 딸기(target_count) 기준 */
  const rEl=document.getElementById('v-rate');
  const ripe=s.target_count||1;
  if((s.success_count||0)>0){
    const r=Math.min((s.success_count/ripe)*100,100);
    const rc=r>=80?'var(--green)':r>=50?'var(--yellow)':'var(--red)';
    rEl.textContent=r.toFixed(1)+'%';rEl.style.color=rc;
    const bar=document.getElementById('rate-bar');
    bar.style.width=r+'%';bar.style.background=rc;
  }else{rEl.textContent='—';rEl.style.color='var(--t3)';}

  /* 평균 파지 */
  const aEl=document.getElementById('v-avg');
  if(s.grasp_times&&s.grasp_times.length){
    const avg=s.grasp_times.reduce((a,b)=>a+b,0)/s.grasp_times.length;
    aEl.innerHTML=avg.toFixed(1)+'<span class="sc-unit">초</span>';aEl.style.color='var(--cyan)';
  }else{aEl.textContent='—';aEl.style.color='var(--t3)';}

  /* 수확 속도 */
  const spE=document.getElementById('v-speed'),spU=document.getElementById('v-speed-u');
  if(s.session_start&&s.success_count>0){
    const sec=(Date.now()-new Date(s.session_start))/1000;
    spE.innerHTML=(sec/s.success_count).toFixed(1)+'<span class="sc-unit">초</span>';
    spE.style.color='var(--purple)';spU.textContent='/ 개';
  }else{spE.textContent='—';spE.style.color='var(--t3)';spU.textContent='';}

  /* 손상률 */
  const dEl=document.getElementById('v-dmg');
  if(s.success_count>0){
    const d=s.damage_count/s.success_count*100;
    const dc=d<=5?'var(--green)':d<=15?'var(--yellow)':'var(--red)';
    dEl.textContent=d.toFixed(1)+'%';dEl.style.color=dc;
    const bar=document.getElementById('dmg-bar');bar.style.width=Math.min(d,100)+'%';bar.style.background=dc;
  }else{dEl.textContent='—';dEl.style.color='var(--t3)';}

  harvestStart=s.current_harvest_start?new Date(s.current_harvest_start):null;

  /* 로봇 상태 */
  let si = STATUS.idle;  // 기본값
  if(!s.robot_ready) {
    // 로봇 미연결: 회색으로 표시
    si = {label:'연결 대기', icon:'○', color:'var(--t3)', ring:false};
  } else {
    // 로봇 연결됨: status에 따라 표시
    si = STATUS[s.status||'idle'] || STATUS.idle;
  }
  document.getElementById('sp-icon').textContent=si.icon;
  document.getElementById('sp-icon').style.color=si.color;
  document.getElementById('sp-icon-bg').style.borderColor=si.color;
  document.getElementById('sp-icon-bg').style.background=CBGMAP[si.color]||'var(--bg)';
  document.getElementById('sp-state').textContent=si.label;
  document.getElementById('sp-state').style.color=si.color;
  document.getElementById('sp-hd-dot').style.background=si.color;
  const ring=document.getElementById('sp-ring');
  ring.style.borderColor=si.color;ring.style.animation=si.ring?'ring-pulse 1.3s ease-out infinite':'none';
  const ok=s.success_count||0,fail=(s.total_attempts||0)-ok;
  document.getElementById('sp-num').textContent=ok>0?'수확 #'+ok:'수확 대기';
  document.getElementById('mini-ok').textContent=ok+' 회';
  document.getElementById('mini-fail').textContent=fail+' 회';
  document.getElementById('mini-dmg').textContent=(s.damage_count||0)+' 회';

  /* 메시지 */
  const msgs=s.messages||[];
  if(msgs.length!==prevMsgCnt){
    prevMsgCnt=msgs.length;
    document.getElementById('msg-cnt').textContent=msgs.length;
    document.getElementById('msg-list').innerHTML=[...msgs].reverse().map(m=>{
      const lv=m.level||'info',cls=LVL_CLS[lv]||'mi-info',chip=LVL_CHIP[lv]||'INFO';
      return `<div class="${cls}"><div class="msg-wrap"><div class="msg-item">
        <span class="msg-ts">${m.time}</span>
        <div class="msg-body"><div class="msg-chip">${chip}</div>
        <div class="msg-txt">${m.text}</div></div></div></div></div>`;
    }).join('');
  }

  /* FPS */
  [0,1].forEach(i=>{
    const fe=document.getElementById('cam-fps-'+i);
    const fps=i===0?(s.camera_fps_0||0):(s.camera_fps_1||0);
    if(fe){if(fps>0&&cams[i].live){fe.textContent=fps.toFixed(0)+' fps';fe.classList.add('on');}
      else fe.classList.remove('on');}
  });

  renderJoints(s);renderTcp(s);renderGripper(s);
  renderConns(s);renderDetected(s);checkSnapshots(s);
}

/* WebSocket */
let ws,wsRetry=0;
function connectWS(){
  ws=new WebSocket(`ws://${location.host}/ws`);
  const bg=document.getElementById('ws-badge'),tx=document.getElementById('ws-txt');
  ws.onopen=()=>{wsRetry=0;bg.className='ws-badge';tx.textContent='연결됨';
    document.getElementById('conn-ws').className='cpill on';};
  ws.onmessage=e=>{try{render(JSON.parse(e.data));}catch{}};
  ws.onclose=()=>{bg.className='ws-badge dis';tx.textContent='재연결 중...';
    document.getElementById('conn-ws').className='cpill off';
    wsRetry++;setTimeout(connectWS,Math.min(1000*wsRetry,5000));};
  ws.onerror=()=>ws.close();
}
connectWS();

/* ── 텔레오퍼레이션 API (포트 8767) ────────────────────────────────────────── */
// 대시보드와 teleop-api 모두 host 네트워크를 사용하므로 localhost로 접근 가능
const TELEOP_API = `http://${window.location.hostname}:8767`;

async function teleopMove(cmd) {
  // 회전 명령어이면 rot-speed 값을 속도로 전달
  let body = {command:cmd};
  if(['rx_plus','rx_minus','ry_plus','ry_minus','rz_plus','rz_minus'].includes(cmd)) {
    const rotSpeed = parseFloat(document.getElementById('rot-speed')?.value || 5);
    body.angle_scale = rotSpeed / 5;  // 기본값(5°)대비 배율
  }
  await fetch(TELEOP_API+'/move', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify(body)
  }).catch(()=>{});
}

async function teleopGripper(pos) {
  await fetch(TELEOP_API+'/gripper', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({position:pos})
  }).catch(()=>{});
}

async function moveHome() {
  const home = document.getElementById('rec-home')?.value || '';
  await fetch(TELEOP_API+'/home', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({home_pose:home})
  }).catch(()=>{});
}

async function startRecording() {
  const btn = document.getElementById('btn-rec-start');
  btn.disabled = true;  // 중복 방지

  const ep   = document.getElementById('rec-ep')?.value.trim() || '';
  const task = document.getElementById('rec-task')?.value.trim() || '';
  const cat  = document.getElementById('rec-cat')?.value.trim() || '';
  const raw  = document.getElementById('rec-rawdir')?.value.trim() || '';
  const r    = await fetch(TELEOP_API+'/record/start', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({episode:ep||undefined, task, category:cat, raw_dir:raw||undefined})
  }).catch(()=>null);
  if(r&&r.ok){
    const d=await r.json();
    if(d.ok && d.episode) document.getElementById('rec-ep').value = d.episode;
  } else {
    btn.disabled = false;  // 실패 시 다시 활성화
  }
}

async function stopRecording() {
  const btn = document.getElementById('btn-rec-stop');
  btn.disabled = true;  // 중복 방지

  await fetch(TELEOP_API+'/record/stop', {method:'POST'}).catch(()=>{
    btn.disabled = false;  // 실패 시 다시 활성화
  });
}

async function convertData() {
  const r = await fetch(TELEOP_API+'/convert', {method:'POST'}).catch(()=>null);
  if(!r||!r.ok) {alert('변환 실패'); return;}

  // 진행률 바 표시
  document.getElementById('convert-progress-container').style.display = 'block';
  let lastProgress = 0;

  // 진행률 폴링 (1초마다)
  const interval = setInterval(async ()=>{
    const statusR = await fetch(TELEOP_API+'/status').catch(()=>null);
    if(!statusR||!statusR.ok) return;
    const d = await statusR.json();

    const bar = document.getElementById('convert-progress-bar');
    const text = document.getElementById('convert-progress-text');
    const pct = Math.max(lastProgress, d.convert_progress || 0);
    lastProgress = pct;

    bar.style.width = pct + '%';
    text.textContent = pct + '%';

    console.log(`변환: ${pct}% (converting=${d.converting})`);

    // 완료
    if(!d.converting && pct >= 100) {
      clearInterval(interval);
      alert('✅ 데이터 변환 완료!');
      setTimeout(()=>{
        document.getElementById('convert-progress-container').style.display = 'none';
        bar.style.width = '0%';
        text.textContent = '';
        lastProgress = 0;
      }, 2000);
    }
  }, 1000);
}

/* 녹화 상태 폴링 (1 초 간격) */
function updateRecBadge(phase, robotReady, robotError) {
  const badge  = document.getElementById('rec-badge');
  const dot    = document.getElementById('rec-dot');
  const txt    = document.getElementById('rec-badge-txt');
  const btnS   = document.getElementById('btn-rec-start');
  const btnE   = document.getElementById('btn-rec-stop');
  if(!badge) return;

  if(!robotReady) {
    badge.className = 'rec-status-badge';
    dot.className   = 'rec-dot';
    txt.textContent = robotError ? `로봇 미연결: ${robotError}` : '로봇 API 대기 중...';
    if(btnS) btnS.disabled = true;
    if(btnE) btnE.disabled = true;
    return;
  }

  badge.className = 'rec-status-badge ' + (phase==='recording'?'recording':phase==='starting'?'starting':phase==='stopping'?'stopping':'');
  dot.className   = 'rec-dot' + (phase==='recording'||phase==='starting'?' blink':'');
  const labels = {idle:'대기', starting:'시작 중...', recording:'● 녹화 중', stopping:'종료 중...'};
  txt.textContent = labels[phase] || phase;
  if(btnS) btnS.disabled = (phase !== 'idle');
  if(btnE) btnE.disabled = (phase === 'idle' || phase === 'stopping');
}

setInterval(async()=>{
  const r = await fetch(TELEOP_API+'/status').catch(()=>null);
  if(!r||!r.ok){updateRecBadge('idle', false, 'API 서버 미응답');return;}
  const d = await r.json();
  updateRecBadge(d.phase||'idle', d.robot_ready, d.robot_error);
}, 1000);

/* 기존 sendTeleop: D-pad 버튼 → 로봇 실제 이동도 호출 */
const _origSendTeleop = sendTeleop;
sendTeleop = async function(cmd) {
  const sp = parseFloat(document.getElementById('tp-speed')?.value||0.2);
  const ts = document.getElementById('teleop-st');
  if(ts){ts.textContent=TELEOP_LBL[cmd]||cmd;
    ts.style.color=cmd==='stop'?'var(--t3)':'var(--blue)';}
  await fetch('/api/teleop', {method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({command:cmd, speed:sp})}).catch(()=>{});
  if(cmd!=='stop') await teleopMove(cmd);
};

/* 듀얼 카메라 */
const CAM_SRCS=['/camera/0','/camera/1'];
const CAM_LABELS=['딸기 인식','전경 카메라'];
const CAM_BADGE_IDS=['cam0-src','cam1-src'];
const cams=[{live:false,retryT:null},{live:false,retryT:null}];

function camSetLive(i){
  cams[i].live=true;
  const img=document.getElementById('cam-img-'+i);
  const ph=document.getElementById('cam-ph-'+i);
  const live=document.getElementById('cam-live-'+i);
  const scan=document.getElementById('cam-scan-'+i);
  const spin=document.getElementById('cam-spin-'+i);
  if(img)img.style.display='block';
  if(ph)ph.style.display='none';
  if(live)live.classList.add('on');
  if(scan)scan.classList.add('on');
  if(spin)spin.style.display='none';
  const badge=document.getElementById(CAM_BADGE_IDS[i]);
  if(badge){badge.textContent=CAM_LABELS[i]+' · LIVE';
    badge.style.cssText='background:var(--green-bg);border-color:#bbf7d0;color:var(--green-tx)';}
  const dot=document.getElementById('cam-hd-dot');
  if(dot&&cams.some(c=>c.live))dot.style.background='var(--red)';
  if(cams[i].retryT){clearTimeout(cams[i].retryT);cams[i].retryT=null;}
}
function camSetError(i){
  cams[i].live=false;
  const img=document.getElementById('cam-img-'+i);
  const ph=document.getElementById('cam-ph-'+i);
  const live=document.getElementById('cam-live-'+i);
  const scan=document.getElementById('cam-scan-'+i);
  const spin=document.getElementById('cam-spin-'+i);
  const ptx=document.getElementById('cam-ph-txt-'+i);
  const fe=document.getElementById('cam-fps-'+i);
  if(img)img.style.display='none';
  if(ph)ph.style.display='flex';
  if(live)live.classList.remove('on');
  if(scan)scan.classList.remove('on');
  if(spin)spin.style.display='none';
  if(ptx)ptx.textContent='카메라 연결 없음';
  if(fe)fe.classList.remove('on');
  const badge=document.getElementById(CAM_BADGE_IDS[i]);
  if(badge){badge.textContent=CAM_LABELS[i]+' · 연결 없음';badge.style.cssText='';}
  const dot=document.getElementById('cam-hd-dot');
  if(dot&&!cams.some(c=>c.live))dot.style.background='var(--t3)';
  cams[i].retryT=setTimeout(()=>camRetry(i),5000);
}
function camRetry(i){
  const spin=document.getElementById('cam-spin-'+i);
  const ptx=document.getElementById('cam-ph-txt-'+i);
  const badge=document.getElementById(CAM_BADGE_IDS[i]);
  if(spin)spin.style.display='block';
  if(ptx)ptx.textContent='재연결 중...';
  if(badge)badge.textContent=CAM_LABELS[i]+' · 재연결 중...';
  const img=document.getElementById('cam-img-'+i);
  if(img)img.src=CAM_SRCS[i]+'?'+Date.now();
}
/* MJPEG 스트림 대신 단일 JPEG 폴링 (브라우저 호환성 확실) */
const CAM_SNAP=['/api/snapshot/0','/api/snapshot/1'];
function initCam(i){
  const img=document.getElementById('cam-img-'+i);
  const spin=document.getElementById('cam-spin-'+i);
  const ptx=document.getElementById('cam-ph-txt-'+i);
  if(!img)return;
  img.onload=()=>camSetLive(i);
  img.onerror=()=>{}; // 폴링 중 일시적 204는 무시
  if(spin)spin.style.display='block';
  if(ptx)ptx.textContent='카메라 연결 중...';

  let failCount=0;
  async function poll(){
    try{
      const r=await fetch(CAM_SNAP[i]+'?t='+Date.now());
      if(r.ok && r.status===200){
        const blob=await r.blob();
        if(blob.size>0){
          if(img._url)URL.revokeObjectURL(img._url);
          img._url=URL.createObjectURL(blob);
          img.src=img._url;
          failCount=0;
        }
      }else{
        failCount++;
        if(failCount>10)camSetError(i);
      }
    }catch(e){
      failCount++;
      if(failCount>10)camSetError(i);
    }
    setTimeout(poll, 100);  // ~10fps
  }
  poll();
}
[0,1].forEach(initCam);

setInterval(async()=>{
  try{const r=await fetch('/camera-info');
    if(r.ok){const d=await r.json();
      const camList=d.cams||[];
      camList.forEach((c,i)=>{
        if(c.source&&c.source!=='none'&&cams[i].live){
          const b=document.getElementById(CAM_BADGE_IDS[i]);
          if(b)b.textContent=CAM_LABELS[i]+' · '+c.source;
        }
      });
    }}catch{}
},3000);

function toggleCamFS(i){
  const b=document.getElementById('cam-body-'+(i||0));
  if(!document.fullscreenElement)b&&b.requestFullscreen&&b.requestFullscreen();
  else document.exitFullscreen&&document.exitFullscreen();
}
</script>
</body>
</html>
"""

# ── FastAPI ───────────────────────────────────────────────────────────────────
def make_app(demo=False, camera_id=0, camera_id_1=-1, no_camera=False,
             serial_cam0='', serial_cam1='',
             camera_url_0='', camera_url_1=''):
    try:
        from fastapi import FastAPI, WebSocket, Request
        from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
    except ImportError:
        print("pip install fastapi 'uvicorn[standard]' websockets"); sys.exit(1)

    app = FastAPI()

    @app.get("/")
    async def index():
        # 브라우저가 옛 HTML/JS를 캐싱하지 못하도록 강제
        return HTMLResponse(HTML, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        })

    @app.get("/camera-info")
    async def cam_info():
        return JSONResponse({"cams": _cam_infos, "fps": _cam_fps_v})

    @app.get("/camera")           # backward-compat alias for slot 0
    @app.get("/camera/0")
    async def cam_ep0():
        if _cam_jpegs[0] is None: return Response(status_code=503)
        return StreamingResponse(_mjpeg_gen(0), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/camera/1")
    async def cam_ep1():
        if _cam_jpegs[1] is None: return Response(status_code=503)
        return StreamingResponse(_mjpeg_gen(1), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/joint-command")
    async def joint_cmd(request: Request):
        b = await request.json()
        s = _load()
        s["pending_joint_command"] = {"angles": b.get("angles",[0]*6),
          "velocity": float(b.get("velocity",10)), "sent_at": datetime.now().isoformat()}
        _save(s); return JSONResponse({"ok": True})

    @app.post("/api/tcp-command")
    async def tcp_cmd(request: Request):
        b = await request.json()
        s = _load()
        s["pending_tcp_command"] = {"pose": b.get("pose",[0]*6),
          "velocity": float(b.get("velocity",10)), "sent_at": datetime.now().isoformat()}
        _save(s); return JSONResponse({"ok": True})

    @app.post("/api/gripper-command")
    async def grip_cmd(request: Request):
        b = await request.json()
        s = _load()
        s["gripper"] = {"position": max(0.0,min(100.0,float(b.get("position",100)))),
          "state": b.get("state","open"), "force": float(b.get("force",30))}
        _save(s); return JSONResponse({"ok": True})

    @app.post("/api/set-target")
    async def set_target(request: Request):
        b = await request.json()
        s = _load(); s["target_count"] = max(1, int(b.get("target",15)))
        _save(s); return JSONResponse({"ok": True})

    @app.post("/api/set-planned-duration")
    async def set_planned_duration(request: Request):
        b = await request.json()
        s = _load(); s["planned_duration_hours"] = max(0.0, float(b.get("hours", 0)))
        _save(s); return JSONResponse({"ok": True})

    @app.post("/api/teleop")
    async def teleop_cmd(request: Request):
        b = await request.json()
        s = _load()
        s["pending_teleop_command"] = {
            "command": b.get("command", "stop"),
            "speed":   float(b.get("speed", 0.2)),
            "sent_at": datetime.now().isoformat()
        }
        _save(s); return JSONResponse({"ok": True})

    @app.get("/api/report")
    async def get_report():
        s = _load()
        now = datetime.now()
        st = datetime.fromisoformat(s["session_start"]) if s.get("session_start") else now
        elapsed = max(1, (now - st).total_seconds())
        succ = s.get("success_count",0); att = s.get("total_attempts",0)
        tgt = s.get("target_count",15); gt = s.get("grasp_times",[])
        h = elapsed / 3600
        report = {
            "generated_at": now.isoformat(),
            "session": {
                "start": s.get("session_start"),
                "duration_s": round(elapsed),
                "duration": f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s"
            },
            "target": {"ripe_berries": tgt, "harvested": succ,
                        "completion_pct": round(succ/max(1,tgt)*100,1)},
            "harvest": {
                "total_attempts": att, "success": succ,
                "fail": att-succ, "damage": s.get("damage_count",0),
                "success_rate_pct": round(succ/max(1,att)*100,1),
                "target_rate_pct": round(succ/max(1,tgt)*100,1),
                "avg_grasp_s": round(sum(gt)/max(1,len(gt)),2) if gt else 0,
                "rate_per_hour": round(succ/max(0.001,h),1)
            },
            "detected": {"count": s.get("detected_count",0),
                          "skip_reasons": s.get("skip_reasons",{})},
            "failures": s.get("failure_types",{}),
            "grasp_times": gt
        }
        return JSONResponse(report)

    @app.get("/api/snapshot")
    @app.get("/api/snapshot/0")
    async def get_snapshot0():
        with _cam_locks[0]: frame = _cam_jpegs[0]
        if frame:
            return Response(content=frame, media_type="image/jpeg",
                            headers={"Cache-Control":"no-store"})
        return Response(status_code=204)

    @app.get("/api/snapshot/1")
    async def get_snapshot1():
        with _cam_locks[1]: frame = _cam_jpegs[1]
        if frame:
            return Response(content=frame, media_type="image/jpeg",
                            headers={"Cache-Control":"no-store"})
        return Response(status_code=204)

    @app.websocket("/ws")
    async def ws_ep(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                data = _load()
                with _cam_fps_lock:
                    data["camera_fps_0"] = _cam_fps_v[0]
                    data["camera_fps_1"] = _cam_fps_v[1]
                await websocket.send_json(data)
                await asyncio.sleep(0.4)
        except Exception: pass

    # 로봇 상태 주기적 업데이트
    def _sync_robot_status_worker():
        import urllib.request
        while True:
            try:
                import time; time.sleep(2)
                req = urllib.request.Request('http://localhost:8767/status')
                with urllib.request.urlopen(req, timeout=2) as resp:
                    api_status = json.loads(resp.read())
                    s = _load()
                    s['robot_ready'] = api_status.get('robot_ready', False)
                    s['robot_error'] = api_status.get('robot_error', '')
                    if 'tcp_pose' in api_status:
                        s['tcp_pose'] = api_status['tcp_pose']
                    if 'joint_angles' in api_status:
                        s['joint_angles'] = api_status['joint_angles']
                    if 'gripper' in api_status:
                        s['gripper'] = api_status['gripper']
                    _save(s)
            except Exception as e:
                print(f'[WARN] 로봇 상태 동기화 실패: {e}')

    threading.Thread(target=_sync_robot_status_worker, daemon=True).start()

    if not no_camera:
        threading.Thread(target=_camera_worker,
            kwargs=dict(camera_id=camera_id, slot=0, serial=serial_cam0, url=camera_url_0),
            daemon=True).start()
        if camera_id_1 >= 0 or camera_url_1:
            threading.Thread(target=_camera_worker,
                kwargs=dict(camera_id=camera_id_1, slot=1, serial=serial_cam1, url=camera_url_1),
                daemon=True).start()
    if demo:
        threading.Thread(target=_run_demo, daemon=True).start()
    return app

# ── 데모 ──────────────────────────────────────────────────────────────────────
_DEMO_MSGS = [
    ("딸기 감지됨 — 파지 접근 시작", "info"),
    ("꼭지 가림 — 접근 방향 변경", "warning"),
    ("잎 장애물 감지 — 우회 경로 계획", "warning"),
    ("미성숙 딸기 — 건너뜀", "warning"),
    ("그리퍼 작동 — 줄기 파지 시도", "info"),
    ("파지 완료", "success"),
    ("홈 포즈 복귀 중", "info"),
    ("폐색률 68% — Heavy 카테고리", "warning"),
    ("IK 재계획 — 관절 한계 회피", "warning"),
]
_FAIL_TYPES = ['ik','obstacle','grasp','detection','other']

def _run_demo():
    s = DEFAULT_STATE.copy()
    s["session_start"] = datetime.now().isoformat()
    s["target_count"] = 15
    _push_msg(s, "대시보드 시작 — 딸기 수확 로봇 연결됨", "info")
    _save(s); step = 0
    while True:
        phase = step % 28; t = step * 0.3
        s["joint_angles"] = [round(math.sin(t*.12)*45,1), round(math.cos(t*.10)*30-20,1),
          round(math.sin(t*.15)*40+80,1), round(math.cos(t*.18)*25,1),
          round(math.sin(t*.09)*20,1),   round(math.cos(t*.22)*50,1)]
        s["tcp_pose"] = [round(300+math.sin(t*.1)*50,1), round(100+math.cos(t*.1)*50,1),
          round(400+math.sin(t*.05)*30,1), round(math.sin(t*.2)*10,1),
          round(math.cos(t*.2)*10,1),     round(math.sin(t*.1)*10,1)]
        if phase==0:
            s["status"]="approaching"; s["current_harvest_start"]=datetime.now().isoformat()
            s["gripper"]={"position":100.0,"state":"open","force":30.0}
            s["detected_count"] = s.get("detected_count",0) + random.randint(1,3)
            _push_msg(s, random.choice(_DEMO_MSGS[:4])[0], random.choice(_DEMO_MSGS[:4])[1])
        elif phase==8:  s["gripper"]={"position":50.0,"state":"grasping","force":40.0}
        elif phase==10: s["status"]="grasping"; s["gripper"]={"position":10.0,"state":"grasping","force":50.0}; _push_msg(s,"그리퍼 작동 — 파지 시도","info")
        elif phase==15: s["gripper"]={"position":5.0,"state":"closed","force":55.0}
        elif phase==20:
            ok = random.random() > 0.15
            gt = round(random.uniform(2.8, 18.5), 1)
            s["total_attempts"] += 1
            s["attempt_history"] = (s.get("attempt_history",[]) + [1 if ok else 0])[-30:]
            if ok:
                s["success_count"] += 1; s["grasp_times"].append(gt)
                if random.random() < 0.09:
                    s["damage_count"] += 1; _push_msg(s, f"수확 완료 — 손상 ({gt}초)", "warning")
                else: _push_msg(s, f"수확 성공 ({gt}초)", "success")
            else:
                ft = random.choice(_FAIL_TYPES)
                s["failure_types"][ft] = s["failure_types"].get(ft,0) + 1
                sk = random.choice(["immature","occluded","harvested","other"])
                s["skip_reasons"][sk] = s["skip_reasons"].get(sk,0) + 1
                _push_msg(s, "파지 실패 — 다음으로 이동", "error")
            s["status"]="returning"; s["current_harvest_start"]=None
            s["gripper"]={"position":100.0,"state":"open","force":30.0}
        elif phase==24:
            s["status"]="idle"
            if random.random()<.35: _push_msg(s, random.choice(_DEMO_MSGS[4:])[0], random.choice(_DEMO_MSGS[4:])[1])
        _save(s); time.sleep(0.3); step += 1

# ── CLI ───────────────────────────────────────────────────────────────────────
def cmd_update(action: str) -> None:
    s = _load()
    if not s.get("session_start"): s["session_start"] = datetime.now().isoformat()
    if action == "start_harvest":
        s["current_harvest_start"] = datetime.now().isoformat(); s["status"] = "approaching"
        _push_msg(s, "수확 시작 — 딸기 접근 중", "info")
    elif action == "harvest_success":
        gt = None
        if s.get("current_harvest_start"):
            gt = round((datetime.now()-datetime.fromisoformat(s["current_harvest_start"])).total_seconds(),2)
            s["grasp_times"].append(gt)
        s["total_attempts"]+=1; s["success_count"]+=1
        s["attempt_history"]=(s.get("attempt_history",[])+[1])[-30:]
        s["current_harvest_start"]=None; s["status"]="returning"
        _push_msg(s, f"수확 성공{f' ({gt}초)' if gt else ''}", "success")
    elif action == "harvest_fail":
        s["total_attempts"]+=1
        s["attempt_history"]=(s.get("attempt_history",[])+[0])[-30:]
        last_msg = s["messages"][-1]["text"] if s.get("messages") else ""
        ft = _detect_failure_type(last_msg)
        s["failure_types"][ft] = s["failure_types"].get(ft,0) + 1
        s["current_harvest_start"]=None; s["status"]="idle"
        _push_msg(s, "파지 실패", "error")
    elif action == "damage":
        s["damage_count"]+=1; _push_msg(s, "손상 감지", "warning")
    elif action == "reset":
        s = DEFAULT_STATE.copy(); s["session_start"]=datetime.now().isoformat()
        _push_msg(s, "대시보드 리셋", "info")
    _save(s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo',      action='store_true')
    p.add_argument('--no-camera', action='store_true')
    p.add_argument('--camera-id',   type=int, default=int(os.environ.get('CAMERA_ID',   6)))
    p.add_argument('--camera-id-1', type=int, default=int(os.environ.get('CAMERA_ID_1', 0)))
    p.add_argument('--serial-cam0', default=os.environ.get('REALSENSE_SERIAL_0', ''))
    p.add_argument('--serial-cam1', default=os.environ.get('REALSENSE_SERIAL_1', ''))
    p.add_argument('--camera-url-0', default=os.environ.get('CAMERA_URL_0', ''))
    p.add_argument('--camera-url-1', default=os.environ.get('CAMERA_URL_1', ''))
    p.add_argument('--port',      type=int, default=8765)
    p.add_argument('--host',      default='0.0.0.0')
    p.add_argument('--update', choices=['start_harvest','harvest_success','harvest_fail','damage','reset'])
    p.add_argument('--msg');   p.add_argument('--level', default='info', choices=['info','success','warning','error'])
    p.add_argument('--status', choices=['idle','approaching','grasping','returning','error'])
    p.add_argument('--joints', metavar='J1..J6')
    p.add_argument('--tcp',    metavar='X,Y,Z,RX,RY,RZ')
    p.add_argument('--target',   type=int)
    p.add_argument('--gripper',  metavar='POS[,FORCE]')
    p.add_argument('--detected', type=int, help='감지된 딸기 수 설정')
    p.add_argument('--skip',     choices=['immature','occluded','harvested','other'], help='건너뛴 이유 기록')
    args = p.parse_args()

    if args.update: cmd_update(args.update); return
    if args.msg:
        s=_load(); s.setdefault("session_start",datetime.now().isoformat())
        _push_msg(s,args.msg,args.level); _save(s); return
    if args.status: s=_load(); s["status"]=args.status; _save(s); return
    if args.joints:
        s=_load(); s["joint_angles"]=[float(x) for x in args.joints.split(',')][:6]; _save(s); return
    if args.tcp:
        s=_load(); s["tcp_pose"]=[float(x) for x in args.tcp.split(',')][:6]; _save(s); return
    if args.target: s=_load(); s["target_count"]=max(1,args.target); _save(s); return
    if args.gripper:
        parts=[float(x) for x in args.gripper.split(',')]
        pos=max(0.,min(100.,parts[0])); force=parts[1] if len(parts)>1 else 30.
        state="open" if pos>=95 else ("closed" if pos<=5 else "grasping")
        s=_load(); s["gripper"]={"position":pos,"state":state,"force":force}; _save(s); return
    if args.detected is not None:
        s=_load(); s["detected_count"]=max(0,args.detected); _save(s); return
    if args.skip:
        s=_load(); sr=s.setdefault("skip_reasons",{"immature":0,"occluded":0,"harvested":0,"other":0})
        sr[args.skip]=sr.get(args.skip,0)+1; _save(s); return

    try: import uvicorn
    except ImportError: print("pip install 'uvicorn[standard]'"); sys.exit(1)
    app = make_app(demo=args.demo, camera_id=args.camera_id,
                   camera_id_1=args.camera_id_1, no_camera=args.no_camera,
                   serial_cam0=args.serial_cam0, serial_cam1=args.serial_cam1,
                   camera_url_0=args.camera_url_0, camera_url_1=args.camera_url_1)
    print(f"\n  딸기 수확 대시보드  →  http://localhost:{args.port}\n")
    if args.demo: print("  [데모 모드] 시뮬레이션 진행 중\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == '__main__':
    main()
