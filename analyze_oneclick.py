#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click ROS1 bag analyzer (Ubuntu) - NO score threshold.
Rule for detections:
  - If /object_detections has messages => "detecting" (detected=1)
  - If no messages => "not detecting" (detected=0)
  - detect_start / detect_lost inferred from message time gaps

Usage:
  python3 analyze_oneclick.py                # auto-pick newest bag in cwd
  python3 analyze_oneclick.py <bag_or_active>

Requirements:
  - ROS1 python env (rosbag, genpy). Recommend:
      source /opt/ros/noetic/setup.bash
      /usr/bin/python3 analyze_oneclick.py ...

Outputs:
  analysis_out_<bagname>/
    - traj_xy_ego_target.png
    - xy_speed_vs_time.png
    - altitude_vs_time.png
    - relative_xy_distance.png
    - detections_presence_fulltime.png
    - object_detections_score_fulltime.png
    - detection_events_metrics.csv
    - CSV exports + topic_summary.csv + analysis_report.txt
"""

import os
import re
import sys
import glob
import shutil
import subprocess
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROS imports
try:
    import rosbag
    from rosbag.bag import ROSBagException, ROSBagUnindexedException
    from genpy.dynamic import generate_dynamic
except ModuleNotFoundError as e:
    print("\n[ERROR] ROS Python deps missing. Use system ROS python.")
    print("Try:")
    print("  source /opt/ros/noetic/setup.bash")
    print("  /usr/bin/python3 analyze_oneclick.py <bag>\n")
    if "Cryptodome" in str(e):
        print("Install:")
        print("  sudo apt update && sudo apt install python3-pycryptodomex\n")
    raise


# ----------------- utils -----------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

def to_sec(t):
    return float(t.to_sec())

def write_csv(path: str, header: List[str], rows: List[List]):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

def interp_series(t_src, y_src, t_dst):
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)
    if len(t_src) < 2:
        return np.full_like(t_dst, np.nan, dtype=float)
    keep = np.r_[True, np.diff(t_src) > 0]
    t_src = t_src[keep]
    y_src = y_src[keep]
    if len(t_src) < 2:
        return np.full_like(t_dst, y_src[0] if len(y_src) else np.nan, dtype=float)
    return np.interp(t_dst, t_src, y_src, left=y_src[0], right=y_src[-1])


# ----------------- MAVROS extraction -----------------

def _has_chain(obj, chain):
    cur = obj
    for name in chain:
        if not hasattr(cur, name):
            return False
        cur = getattr(cur, name)
    return True

def extract_xyz_from_pose_like(msg):
    # PoseStamped: msg.pose.position
    # Odometry: msg.pose.pose.position
    candidates = [
        ("pose", "position"),
        ("pose", "pose", "position"),
        ("position",),
    ]
    for chain in candidates:
        if _has_chain(msg, chain):
            cur = msg
            for n in chain:
                cur = getattr(cur, n)
            if all(hasattr(cur, k) for k in ("x", "y", "z")):
                return float(cur.x), float(cur.y), float(cur.z)
    return None

def extract_twist_linear_angular(msg):
    # TwistStamped: msg.twist.linear / angular
    # Odometry: msg.twist.twist.linear / angular
    candidates = [
        ("twist",),
        ("twist", "twist"),
    ]
    for chain in candidates:
        if _has_chain(msg, chain):
            cur = msg
            for n in chain:
                cur = getattr(cur, n)
            if hasattr(cur, "linear") and hasattr(cur, "angular"):
                lin, ang = cur.linear, cur.angular
                if all(hasattr(lin, k) for k in ("x","y","z")) and all(hasattr(ang, k) for k in ("x","y","z")):
                    return (float(lin.x), float(lin.y), float(lin.z),
                            float(ang.x), float(ang.y), float(ang.z))
    return None

def read_pose_series(bag, topic: str, t0: float):
    ts, xs, ys, zs = [], [], [], []
    for _, msg, t in bag.read_messages(topics=[topic]):
        p = extract_xyz_from_pose_like(msg)
        if p is None:
            continue
        ts.append(to_sec(t) - t0)
        xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
    if len(ts) < 2:
        return None
    return np.asarray(ts, float), np.stack([xs,ys,zs], axis=1).astype(float)

def read_vel_series(bag, topic: str, t0: float):
    ts, v = [], []
    for _, msg, t in bag.read_messages(topics=[topic]):
        tw = extract_twist_linear_angular(msg)
        if tw is None:
            continue
        ts.append(to_sec(t) - t0)
        v.append(tw)
    if len(ts) < 2:
        return None
    return np.asarray(ts, float), np.asarray(v, float)

def xy_speed_from_vel(v6: np.ndarray):
    return np.sqrt(v6[:,0]**2 + v6[:,1]**2)


# ----------------- topic discovery -----------------

POSE_SUFFIX = "mavros/local_position/pose"
VEL_SUFFIX  = "mavros/local_position/velocity_local"

def bag_topic_summary(bag):
    info = bag.get_type_and_topic_info()
    rows = []
    for topic, tt in info.topics.items():
        hz = float(tt.frequency) if tt.frequency is not None else 0.0
        rows.append((topic, tt.msg_type, int(tt.message_count), hz))
    rows.sort(key=lambda x: (-x[2], x[0]))
    try:
        start = bag.get_start_time()
        end = bag.get_end_time()
        dur = max(1e-9, end-start)
    except ROSBagException:
        start, end, dur = None, None, 0.0
    if dur > 0:
        fixed = []
        for topic, msg_type, count, hz in rows:
            if hz == 0.0 and count > 0:
                hz = count / dur
            fixed.append((topic, msg_type, count, hz))
        rows = fixed
    return start, end, dur, rows

def find_mavros_topics(all_topics: List[str]):
    poses = [t for t in all_topics if POSE_SUFFIX in t]
    vels  = [t for t in all_topics if VEL_SUFFIX in t]
    def is_target(t): return "target" in t.lower()
    ego_pose = next((t for t in poses if not is_target(t)), None)
    ego_vel  = next((t for t in vels  if not is_target(t)), None)
    tgt_pose = next((t for t in poses if is_target(t)), None)
    tgt_vel  = next((t for t in vels  if is_target(t)), None)
    return ego_pose, ego_vel, tgt_pose, tgt_vel

def find_detections_topic(all_topics: List[str]):
    if "/object_detections" in all_topics:
        return "/object_detections"
    for t in all_topics:
        if "object_detections" in t.lower():
            return t
    for t in all_topics:
        if "detections" in t.lower():
            return t
    return None


# ----------------- detections: dynamic msg + raw deserialize -----------------

def find_detection_msg_path(bag_path: str) -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    bag_dir = os.path.dirname(os.path.abspath(bag_path))
    candidates = [
        os.path.join(cwd, "Detection.msg"),
        os.path.join(here, "Detection.msg"),
        os.path.join(bag_dir, "Detection.msg"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None

def load_msg_def_text(msg_path: str) -> str:
    with open(msg_path, "r", encoding="utf-8") as f:
        return f.read().strip() + "\n"

def build_dynamic_msg_class(full_type: str, msg_def_text: str):
    classes = generate_dynamic(full_type, msg_def_text)
    return classes.get(full_type, list(classes.values())[0])

def extract_raw_bytes(msg_obj) -> Optional[bytes]:
    if isinstance(msg_obj, (bytes, bytearray)):
        return bytes(msg_obj)
    if isinstance(msg_obj, (tuple, list)):
        for x in msg_obj:
            if isinstance(x, (bytes, bytearray)):
                return bytes(x)
    return None

def read_detections_series_raw(bag, topic: str, t0: float, DetectionCls):
    """
    No threshold, no filtering:
      - If a message exists -> detected=1 at that timestamp
      - We still record score/class_name for reference
    """
    ts, scores, cls = [], [], []
    for _, msg_obj, t in bag.read_messages(topics=[topic], raw=True):
        raw = extract_raw_bytes(msg_obj)
        if raw is None:
            continue
        det = DetectionCls()
        det.deserialize(raw)

        tt = to_sec(t) - t0
        s = float(getattr(det, "score", float("nan")))
        cn = str(getattr(det, "class_name", ""))

        ts.append(tt)
        scores.append(s)
        cls.append(cn)

    if len(ts) < 2:
        return None
    return (np.asarray(ts, float),
            np.asarray(scores, float),
            np.asarray(cls, object))

def infer_intervals_from_message_times(det_t: np.ndarray, gap_factor=5.0, min_gap_s=0.5):
    """
    Infer detect intervals purely from message time continuity.
    """
    t = np.asarray(det_t, float)
    if len(t) == 0:
        return [], [], []
    t = np.sort(t)

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 1e-6)]
    if len(dt) == 0:
        intervals = [(float(t[0]), float(t[-1]), len(t))]
        return intervals, [float(t[0])], [float(t[-1])]

    median_dt = float(np.median(dt))
    gap_th = max(min_gap_s, gap_factor * median_dt)

    intervals = []
    s = float(t[0]); n = 1
    for i in range(len(t)-1):
        gap = float(t[i+1]-t[i])
        if gap > gap_th:
            intervals.append((s, float(t[i]), n))
            s = float(t[i+1]); n = 1
        else:
            n += 1
    intervals.append((s, float(t[-1]), n))
    starts = [it[0] for it in intervals]
    ends   = [it[1] for it in intervals]
    return intervals, starts, ends


# ----------------- plots -----------------

def plot_traj(out_dir, ego_xy, tgt_xy):
    plt.figure()
    plt.plot(ego_xy[:,0], ego_xy[:,1], label="ego")
    plt.plot(tgt_xy[:,0], tgt_xy[:,1], label="target")
    plt.scatter([ego_xy[0,0]],[ego_xy[0,1]], marker="o")
    plt.scatter([ego_xy[-1,0]],[ego_xy[-1,1]], marker="x")
    plt.scatter([tgt_xy[0,0]],[tgt_xy[0,1]], marker="o")
    plt.scatter([tgt_xy[-1,0]],[tgt_xy[-1,1]], marker="x")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.title("XY trajectory (ego & target)")
    plt.axis("equal"); plt.legend()
    save_fig(os.path.join(out_dir, "traj_xy_ego_target.png"))

def plot_alt(out_dir, ego_t, ego_z, tgt_t, tgt_z, duration):
    plt.figure()
    plt.plot(ego_t, ego_z, label="ego_z")
    plt.plot(tgt_t, tgt_z, label="target_z")
    plt.xlim(0, duration)
    plt.xlabel("time (s)"); plt.ylabel("z (m)")
    plt.title("Altitude vs time")
    plt.legend()
    save_fig(os.path.join(out_dir, "altitude_vs_time.png"))

def plot_speed(out_dir, ego_t, ego_v6, ego_vxy, tgt_t, tgt_v6, tgt_vxy, duration):
    plt.figure()
    if ego_v6 is not None:
        plt.plot(ego_t, ego_v6[:,0], label="ego_vx")
        plt.plot(ego_t, ego_v6[:,1], label="ego_vy")
        plt.plot(ego_t, ego_vxy, label="ego_vxy")
    if tgt_v6 is not None:
        plt.plot(tgt_t, tgt_v6[:,0], label="target_vx")
        plt.plot(tgt_t, tgt_v6[:,1], label="target_vy")
        plt.plot(tgt_t, tgt_vxy, label="target_vxy")
    plt.xlim(0, duration)
    plt.xlabel("time (s)"); plt.ylabel("m/s")
    plt.title("Velocity and XY resultant speed")
    plt.legend()
    save_fig(os.path.join(out_dir, "xy_speed_vs_time.png"))

def plot_rel_dist(out_dir, t, dist, starts, ends, duration):
    plt.figure()
    plt.plot(t, dist, label="rel_xy_dist (m)")
    for x in starts:
        plt.axvline(x, linestyle="--", linewidth=1)
    for x in ends:
        plt.axvline(x, linestyle=":", linewidth=1)
    plt.xlim(0, duration)
    plt.xlabel("time (s)"); plt.ylabel("m")
    plt.title("Relative XY distance with detection events")
    plt.legend()
    save_fig(os.path.join(out_dir, "relative_xy_distance.png"))

def plot_detections_presence_fulltime(out_dir, duration, intervals):
    """
    Show 0~duration, with detected=1 during inferred intervals.
    """
    plt.figure()
    # step-like visualization using spans
    for (s, e, _) in intervals:
        plt.axvspan(s, e, alpha=0.25)
    plt.hlines(1.0, 0, duration, colors="none")  # keep y range
    plt.ylim(-0.1, 1.2)
    plt.xlim(0, duration)
    plt.yticks([0,1], ["0","1"])
    plt.xlabel("time (s)")
    plt.ylabel("detected (by message presence)")
    plt.title("Detections presence over full bag time (shaded = messages exist)")
    save_fig(os.path.join(out_dir, "detections_presence_fulltime.png"))

def plot_det_score_fulltime(out_dir, duration, det_t, score, intervals):
    plt.figure()
    if det_t is not None and len(det_t) > 0:
        plt.plot(det_t, score, label="score")
    for (s, e, _) in intervals:
        plt.axvspan(s, e, alpha=0.15)
    plt.xlim(0, duration)
    plt.xlabel("time (s)")
    plt.ylabel("score")
    plt.title("Detections score over full bag time (shaded = message intervals)")
    plt.legend()
    save_fig(os.path.join(out_dir, "object_detections_score_fulltime.png"))


# ----------------- event metrics -----------------

def compute_event_metrics(event_times, ego_pose_t, ego_xyz, tgt_pose_t, tgt_xyz, tgt_vel_t, tgt_vxy):
    if not event_times:
        return []
    et = np.asarray(event_times, float)
    ex = interp_series(ego_pose_t, ego_xyz[:,0], et)
    ey = interp_series(ego_pose_t, ego_xyz[:,1], et)
    ez = interp_series(ego_pose_t, ego_xyz[:,2], et)
    tx = interp_series(tgt_pose_t, tgt_xyz[:,0], et)
    ty = interp_series(tgt_pose_t, tgt_xyz[:,1], et)
    tz = interp_series(tgt_pose_t, tgt_xyz[:,2], et)
    rel = np.sqrt((tx-ex)**2 + (ty-ey)**2)
    dz = tz - ez
    if tgt_vel_t is not None and tgt_vxy is not None and len(tgt_vel_t) >= 2:
        tsp = interp_series(tgt_vel_t, tgt_vxy, et)
    else:
        tsp = np.full_like(et, np.nan, float)
    rows = []
    for i in range(len(et)):
        rows.append([f"{et[i]:.6f}", f"{rel[i]:.6f}", f"{dz[i]:.6f}", f"{tsp[i]:.6f}"])
    return rows


# ----------------- bag helper -----------------

def pick_newest_bag_in_cwd() -> Optional[str]:
    candidates = sorted(
        glob.glob("*.bag") + glob.glob("*.bag.active"),
        key=lambda p: os.path.getmtime(p),
        reverse=True
    )
    return candidates[0] if candidates else None

def maybe_reindex_active(bag_path: str) -> str:
    if not bag_path.endswith(".active"):
        return bag_path
    base = bag_path[:-7]  # strip ".active"
    recovered = base + ".recovered.bag"
    if not os.path.exists(recovered):
        print(f"[INFO] Creating recovered bag: {recovered}")
        shutil.copy2(bag_path, recovered)
    print("[INFO] Trying rosbag reindex on recovered bag...")
    try:
        subprocess.run(["rosbag", "reindex", recovered],
                       check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        pass
    return recovered


# ----------------- main -----------------

def main():
    # One-click: if no args, pick newest bag in cwd
    if len(sys.argv) >= 2:
        bag_path = sys.argv[1]
    else:
        bag_path = pick_newest_bag_in_cwd()
        if bag_path is None:
            print("[ERROR] No .bag or .bag.active found in current directory.")
            print("Usage: python3 analyze_oneclick.py <bag_or_active>")
            sys.exit(2)
        print(f"[INFO] No bag arg provided. Using newest: {bag_path}")

    if not os.path.isfile(bag_path):
        print("[ERROR] bag not found:", bag_path)
        sys.exit(2)

    bag_try = bag_path
    if bag_path.endswith(".active"):
        bag_try = maybe_reindex_active(bag_path)

    out_dir = "analysis_out_" + os.path.basename(bag_path).replace("/", "_")
    ensure_dir(out_dir)

    det_msg = find_detection_msg_path(bag_path) or find_detection_msg_path(bag_try)
    if det_msg is None:
        print("[ERROR] Detection.msg not found.")
        print("Put Detection.msg in one of these places:")
        print("  - current directory")
        print("  - same directory as this script")
        print("  - same directory as the bag file")
        sys.exit(3)

    print("[INFO] Detection.msg:", det_msg)

    try:
        bag = rosbag.Bag(bag_try, "r", allow_unindexed=True)
    except ROSBagUnindexedException:
        print("[ERROR] Bag unindexed and cannot be read. Try:")
        print("  cp your.bag.active recovered.bag && rosbag reindex recovered.bag")
        sys.exit(4)
    except Exception as e:
        print("[ERROR] Failed to open bag:", e)
        sys.exit(4)

    with bag:
        start, end, dur, rows = bag_topic_summary(bag)
        write_csv(os.path.join(out_dir, "topic_summary.csv"),
                  ["topic","msg_type","message_count","estimated_hz"], rows)

        if start is None:
            print("[ERROR] Empty bag (no messages).")
            sys.exit(5)

        all_topics = [r[0] for r in rows if r[2] > 0]
        ego_pose_topic, ego_vel_topic, tgt_pose_topic, tgt_vel_topic = find_mavros_topics(all_topics)
        det_topic = find_detections_topic(all_topics)

        print(f"[INFO] Duration: {dur:.3f}s")
        print("[INFO] Selected topics:")
        print("  ego_pose:   ", ego_pose_topic)
        print("  ego_vel:    ", ego_vel_topic)
        print("  target_pose:", tgt_pose_topic)
        print("  target_vel: ", tgt_vel_topic)
        print("  detections: ", det_topic)

        if ego_pose_topic is None or tgt_pose_topic is None:
            print("[ERROR] Missing ego/target pose topics.")
            sys.exit(6)

        t0 = start

        ego_pose = read_pose_series(bag, ego_pose_topic, t0)
        tgt_pose = read_pose_series(bag, tgt_pose_topic, t0)
        if ego_pose is None or tgt_pose is None:
            print("[ERROR] Cannot parse enough pose messages.")
            sys.exit(7)

        ego_pose_t, ego_xyz = ego_pose
        tgt_pose_t, tgt_xyz = tgt_pose
        ego_xy = ego_xyz[:, :2]
        tgt_xy = tgt_xyz[:, :2]

        ego_vel = read_vel_series(bag, ego_vel_topic, t0) if ego_vel_topic else None
        tgt_vel = read_vel_series(bag, tgt_vel_topic, t0) if tgt_vel_topic else None

        ego_vel_t = ego_v6 = ego_vxy = None
        tgt_vel_t = tgt_v6 = tgt_vxy = None
        if ego_vel is not None:
            ego_vel_t, ego_v6 = ego_vel
            ego_vxy = xy_speed_from_vel(ego_v6)
        if tgt_vel is not None:
            tgt_vel_t, tgt_v6 = tgt_vel
            tgt_vxy = xy_speed_from_vel(tgt_v6)

        # plots (force full time range on x-axis)
        plot_traj(out_dir, ego_xy, tgt_xy)
        plot_alt(out_dir, ego_pose_t, ego_xyz[:,2], tgt_pose_t, tgt_xyz[:,2], dur)
        plot_speed(out_dir,
                   ego_vel_t if ego_vel_t is not None else np.array([]), ego_v6, ego_vxy,
                   tgt_vel_t if tgt_vel_t is not None else np.array([]), tgt_v6, tgt_vxy,
                   dur)

        # relative distance on ego timeline
        tx = interp_series(tgt_pose_t, tgt_xy[:,0], ego_pose_t)
        ty = interp_series(tgt_pose_t, tgt_xy[:,1], ego_pose_t)
        dx = tx - ego_xy[:,0]
        dy = ty - ego_xy[:,1]
        rel_dist = np.sqrt(dx*dx + dy*dy)
        write_csv(os.path.join(out_dir, "relative_xy_distance.csv"),
                  ["t","dist_xy_m","dx","dy"],
                  [[f"{ego_pose_t[i]:.6f}", f"{rel_dist[i]:.6f}", f"{dx[i]:.6f}", f"{dy[i]:.6f}"]
                   for i in range(len(ego_pose_t))])

        # detections: NO threshold, just message existence
        starts, ends, intervals = [], [], []
        det_t = det_score = det_class = None

        det_def = load_msg_def_text(det_msg)
        DetectionCls = build_dynamic_msg_class("detection_msgs/Detection", det_def)

        if det_topic is not None:
            det_series = read_detections_series_raw(bag, det_topic, t0, DetectionCls)
            if det_series is not None:
                det_t, det_score, det_class = det_series

                # infer intervals purely from message times
                intervals, starts, ends = infer_intervals_from_message_times(
                    det_t, gap_factor=5.0, min_gap_s=0.5
                )

                # export detections series
                write_csv(os.path.join(out_dir, "object_detections_series.csv"),
                          ["t","score","class_name"],
                          [[f"{det_t[i]:.6f}",
                            (f"{det_score[i]:.6f}" if np.isfinite(det_score[i]) else "nan"),
                            str(det_class[i])]
                           for i in range(len(det_t))])

                plot_detections_presence_fulltime(out_dir, dur, intervals)
                plot_det_score_fulltime(out_dir, dur, det_t, det_score, intervals)
            else:
                print("[WARN] detections topic exists but cannot parse enough messages.")
        else:
            print("[WARN] detections topic not found.")

        # relative distance plot with event markers (full duration)
        plot_rel_dist(out_dir, ego_pose_t, rel_dist, starts, ends, dur)

        # event metrics
        event_rows = []
        if starts or ends:
            sm = compute_event_metrics(starts, ego_pose_t, ego_xyz, tgt_pose_t, tgt_xyz, tgt_vel_t, tgt_vxy)
            for row in sm:
                event_rows.append(["detect_start"] + row)
            em = compute_event_metrics(ends, ego_pose_t, ego_xyz, tgt_pose_t, tgt_xyz, tgt_vel_t, tgt_vxy)
            for row in em:
                event_rows.append(["detect_lost"] + row)
            event_rows.sort(key=lambda r: float(r[1]))
            write_csv(os.path.join(out_dir, "detection_events_metrics.csv"),
                      ["event","t","rel_xy_dist_m","dz_target_minus_ego_m","target_xy_speed_mps"],
                      event_rows)

        # exports
        write_csv(os.path.join(out_dir, "ego_pose.csv"),
                  ["t","x","y","z"],
                  [[f"{ego_pose_t[i]:.6f}", f"{ego_xyz[i,0]:.6f}", f"{ego_xyz[i,1]:.6f}", f"{ego_xyz[i,2]:.6f}"]
                   for i in range(len(ego_pose_t))])
        write_csv(os.path.join(out_dir, "target_pose.csv"),
                  ["t","x","y","z"],
                  [[f"{tgt_pose_t[i]:.6f}", f"{tgt_xyz[i,0]:.6f}", f"{tgt_xyz[i,1]:.6f}", f"{tgt_xyz[i,2]:.6f}"]
                   for i in range(len(tgt_pose_t))])

        # report
        with open(os.path.join(out_dir, "analysis_report.txt"), "w", encoding="utf-8") as f:
            f.write(f"bag: {bag_path}\n")
            f.write(f"bag_opened_as: {bag_try}\n")
            f.write(f"duration_s: {dur:.6f}\n\n")
            f.write("selected_topics:\n")
            f.write(f"  ego_pose: {ego_pose_topic}\n")
            f.write(f"  ego_vel: {ego_vel_topic}\n")
            f.write(f"  target_pose: {tgt_pose_topic}\n")
            f.write(f"  target_vel: {tgt_vel_topic}\n")
            f.write(f"  detections: {det_topic}\n\n")
            f.write(f"Detection.msg: {det_msg}\n")
            f.write("detections_logic: message_presence_only (no score threshold)\n")
            f.write(f"intervals: {len(intervals)}\n")
            f.write(f"starts: {len(starts)}\n")
            f.write(f"ends: {len(ends)}\n")

        print("[DONE] Output:", out_dir)
        print("  - traj_xy_ego_target.png")
        print("  - altitude_vs_time.png")
        print("  - xy_speed_vs_time.png")
        print("  - relative_xy_distance.png")
        print("  - detections_presence_fulltime.png")
        print("  - object_detections_score_fulltime.png")
        print("  - detection_events_metrics.csv (if any intervals)")
        print("  - CSV exports + topic_summary.csv + analysis_report.txt")


if __name__ == "__main__":
    main()
