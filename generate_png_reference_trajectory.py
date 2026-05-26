#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a horizontal PNG reference trajectory from target position in a ROS1 bag,
then compare it with the real test-aircraft trajectory during the selected work
mode segment.

Usage:
  python3 generate_png_reference_trajectory.py recording_xxx.bag

Main outputs:
  reference_out_<bagname>/
    reference_trajectory.csv
    trajectory_xy_compare.png
    reference_tracking_error.png
    target_convergence.png
    work_intervals.csv
    state_mode_changes.csv
    reference_summary.txt

Notes:
  - Only XY motion is used. Height is ignored.
  - The compared real trajectory is clipped to the selected work-mode interval.
  - By default the script prefers GUIDED, then falls back to AUTO. In the
    recorded bags for this project, the real interception/control segment can
    be reported by MAVROS as GUIDED even though the operation is described as
    starting from AUTO.
  - Default PNG parameters are chosen from the project context: N=5, V=5 m/s.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


POSE_SUFFIX = "mavros/local_position/pose"
VEL_SUFFIX = "mavros/local_position/velocity_local"
STATE_SUFFIX = "mavros/state"
REF_COLLISION_THRESHOLD_M = 0.15


def fail(msg: str, code: int = 1):
    print("[ERROR] " + msg, file=sys.stderr)
    sys.exit(code)


def warn(msg: str):
    print("[WARN] " + msg, file=sys.stderr)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_sec(t) -> float:
    return float(t.to_sec())


def wrap_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_pi_scalar(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def interp_series(t_src, y_src, t_dst):
    t_src = np.asarray(t_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)
    if len(t_src) == 0:
        return np.full_like(t_dst, np.nan, dtype=float)
    if len(t_src) == 1:
        return np.full_like(t_dst, y_src[0], dtype=float)
    keep = np.r_[True, np.diff(t_src) > 0]
    t_src = t_src[keep]
    y_src = y_src[keep]
    if len(t_src) == 1:
        return np.full_like(t_dst, y_src[0], dtype=float)
    return np.interp(t_dst, t_src, y_src)


def interp_xy(t_src, xy_src, t_dst):
    return np.column_stack([
        interp_series(t_src, xy_src[:, 0], t_dst),
        interp_series(t_src, xy_src[:, 1], t_dst),
    ])


def _has_chain(obj, chain):
    cur = obj
    for name in chain:
        if not hasattr(cur, name):
            return False
        cur = getattr(cur, name)
    return True


def extract_xyz_from_pose_like(msg):
    candidates = [
        ("pose", "position"),
        ("pose", "pose", "position"),
        ("position",),
    ]
    for chain in candidates:
        if _has_chain(msg, chain):
            cur = msg
            for name in chain:
                cur = getattr(cur, name)
            if all(hasattr(cur, k) for k in ("x", "y", "z")):
                return float(cur.x), float(cur.y), float(cur.z)
    return None


def extract_twist_xy(msg):
    candidates = [
        ("twist",),
        ("twist", "twist"),
    ]
    for chain in candidates:
        if _has_chain(msg, chain):
            cur = msg
            for name in chain:
                cur = getattr(cur, name)
            if hasattr(cur, "linear") and all(hasattr(cur.linear, k) for k in ("x", "y")):
                return float(cur.linear.x), float(cur.linear.y)
    return None


def topic_summary(bag):
    info = bag.get_type_and_topic_info()
    rows = []
    for topic, tt in info.topics.items():
        hz = float(tt.frequency) if tt.frequency is not None else 0.0
        rows.append((topic, tt.msg_type, int(tt.message_count), hz))
    rows.sort(key=lambda r: (-r[2], r[0]))
    start = bag.get_start_time()
    end = bag.get_end_time()
    dur = max(0.0, end - start)
    if dur > 0:
        fixed = []
        for topic, msg_type, count, hz in rows:
            if hz == 0.0 and count > 0:
                hz = count / dur
            fixed.append((topic, msg_type, count, hz))
        rows = fixed
    return start, end, dur, rows


def find_pose_topics(all_topics: Sequence[str]):
    poses = [t for t in all_topics if POSE_SUFFIX in t]

    def is_target(topic: str) -> bool:
        return "target" in topic.lower()

    ego_pose = next((t for t in poses if not is_target(t)), None)
    target_pose = next((t for t in poses if is_target(t)), None)
    return ego_pose, target_pose


def find_velocity_topic(all_topics: Sequence[str]):
    vels = [t for t in all_topics if VEL_SUFFIX in t]
    return next((t for t in vels if "target" not in t.lower()), None)


def find_state_topic(all_topics: Sequence[str]):
    if "/mavros/state" in all_topics:
        return "/mavros/state"
    states = [t for t in all_topics if t.endswith(STATE_SUFFIX) and "target" not in t.lower()]
    if states:
        return states[0]
    for t in all_topics:
        low = t.lower()
        if "mavros/state" in low and "target" not in low:
            return t
    return None


def read_pose_series(bag, topic: str, t0: float):
    ts, xyz = [], []
    for _, msg, t in bag.read_messages(topics=[topic]):
        p = extract_xyz_from_pose_like(msg)
        if p is None:
            continue
        ts.append(to_sec(t) - t0)
        xyz.append(p)
    if len(ts) < 2:
        return None
    return np.asarray(ts, dtype=float), np.asarray(xyz, dtype=float)


def read_velocity_series(bag, topic: str, t0: float):
    ts, vxy = [], []
    for _, msg, t in bag.read_messages(topics=[topic]):
        v = extract_twist_xy(msg)
        if v is None:
            continue
        ts.append(to_sec(t) - t0)
        vxy.append(v)
    if len(ts) < 2:
        return None
    return np.asarray(ts, dtype=float), np.asarray(vxy, dtype=float)


def read_state_series(bag, topic: str, t0: float):
    ts, modes = [], []
    for _, msg, t in bag.read_messages(topics=[topic]):
        mode = getattr(msg, "mode", "")
        ts.append(to_sec(t) - t0)
        modes.append(str(mode))
    if len(ts) < 1:
        return None
    return np.asarray(ts, dtype=float), modes


def mode_matches(mode: str, pattern: str, exact: bool) -> bool:
    mode_u = str(mode).upper()
    pat_u = str(pattern).upper()
    return mode_u == pat_u if exact else pat_u in mode_u


def build_mode_intervals(state_t, modes, bag_duration, pattern: str, exact: bool):
    intervals = []
    active_start = None
    active_mode = None

    for i, (t, mode) in enumerate(zip(state_t, modes)):
        is_active = mode_matches(mode, pattern, exact)
        next_t = state_t[i + 1] if i + 1 < len(state_t) else bag_duration

        if is_active and active_start is None:
            active_start = float(t)
            active_mode = mode
        if active_start is not None and (not is_active):
            end_t = float(t)
            if end_t > active_start:
                intervals.append((active_start, end_t, active_mode))
            active_start = None
            active_mode = None

        if active_start is not None and i == len(state_t) - 1:
            end_t = float(next_t)
            if end_t > active_start:
                intervals.append((active_start, end_t, active_mode))

    merged = []
    for st, en, mode in intervals:
        if not merged or st - merged[-1][1] > 1e-6:
            merged.append([st, en, mode])
        else:
            merged[-1][1] = max(merged[-1][1], en)
    return [(float(st), float(en), str(mode)) for st, en, mode in merged]


def mode_change_rows(state_t, modes):
    rows = []
    prev = object()
    for t, mode in zip(state_t, modes):
        if mode != prev:
            rows.append((float(t), str(mode)))
            prev = mode
    return rows


def resolve_mode_patterns(mode_spec: str):
    spec = str(mode_spec or "auto").strip()
    if spec.lower() == "auto":
        return ["GUIDED", "AUTO"]
    return [item.strip() for item in spec.split(",") if item.strip()]


def select_work_intervals(state_t, modes, bag_duration, mode_spec: str, exact: bool):
    patterns = resolve_mode_patterns(mode_spec)
    if not patterns:
        patterns = ["GUIDED", "AUTO"]

    all_rows = []
    for pattern in patterns:
        intervals = build_mode_intervals(state_t, modes, bag_duration, pattern, exact)
        for st, en, mode in intervals:
            all_rows.append((pattern, st, en, mode))
        if intervals:
            return pattern, intervals, all_rows
    return None, [], all_rows


def estimate_heading_from_positions(t, xy, at_t, window_s=0.8, min_move=0.15):
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float)
    idx0 = int(np.searchsorted(t, at_t, side="left"))
    idx1 = int(np.searchsorted(t, at_t + window_s, side="right")) - 1
    idx0 = max(0, min(idx0, len(t) - 1))
    idx1 = max(idx0, min(idx1, len(t) - 1))
    d = xy[idx1] - xy[idx0]
    if float(np.linalg.norm(d)) < min_move and idx0 + 1 < len(t):
        idx1 = min(len(t) - 1, idx0 + 3)
        d = xy[idx1] - xy[idx0]
    if float(np.linalg.norm(d)) < min_move:
        return None
    return math.atan2(float(d[1]), float(d[0]))


def estimate_speed_from_positions(t, xy, start_t, end_t):
    mask = (t >= start_t) & (t <= end_t)
    tt = t[mask]
    pp = xy[mask]
    if len(tt) < 3:
        return None
    dt = np.diff(tt)
    dp = np.linalg.norm(np.diff(pp, axis=0), axis=1)
    good = dt > 1e-3
    if not np.any(good):
        return None
    speeds = dp[good] / dt[good]
    speeds = speeds[np.isfinite(speeds)]
    if len(speeds) == 0:
        return None
    return float(np.median(speeds))


def heading_from_xy(t, xy, min_speed=0.05):
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if len(t) < 2:
        return np.full(len(t), np.nan), np.zeros(len(t))
    dx = np.gradient(xy[:, 0], t)
    dy = np.gradient(xy[:, 1], t)
    speed = np.sqrt(dx * dx + dy * dy)
    heading = np.arctan2(dy, dx)
    heading[speed < min_speed] = np.nan
    return heading, speed


def simulate_png_reference(
    grid_t,
    target_xy,
    start_xy,
    initial_heading,
    speed,
    n_nav,
    k_yaw,
    k_yaw_d,
    max_yaw_rate,
    lambda_alpha,
    max_side_step_rad,
):
    n = len(grid_t)
    ref_xy = np.zeros((n, 2), dtype=float)
    ref_heading = np.zeros(n, dtype=float)
    lambda_raw = np.zeros(n, dtype=float)
    lambda_filt = np.zeros(n, dtype=float)
    lambda_dot = np.zeros(n, dtype=float)
    yaw_rate_cmd = np.zeros(n, dtype=float)

    ref_xy[0] = np.asarray(start_xy, dtype=float)
    ref_heading[0] = float(initial_heading)

    los0 = math.atan2(target_xy[0, 1] - ref_xy[0, 1], target_xy[0, 0] - ref_xy[0, 0])
    lambda_raw[0] = wrap_pi_scalar(los0 - ref_heading[0])
    lambda_filt[0] = lambda_raw[0]
    prev_filt = lambda_filt[0]

    for i in range(1, n):
        dt = max(1e-3, float(grid_t[i] - grid_t[i - 1]))
        prev_pos = ref_xy[i - 1]
        prev_heading = ref_heading[i - 1]

        los = math.atan2(target_xy[i - 1, 1] - prev_pos[1], target_xy[i - 1, 0] - prev_pos[0])
        raw = wrap_pi_scalar(los - prev_heading)
        lambda_raw[i - 1] = raw

        delta = wrap_pi_scalar(raw - prev_filt)
        filt = wrap_pi_scalar(prev_filt + lambda_alpha * delta)
        lambda_filt[i - 1] = filt
        rate = wrap_pi_scalar(filt - prev_filt) / dt
        prev_filt = filt
        lambda_dot[i - 1] = rate

        side_step = clamp(n_nav * rate * dt, -max_side_step_rad, max_side_step_rad)
        vx_body = speed * math.cos(side_step)
        vy_body = speed * math.sin(side_step)

        yaw_rate = 0.0
        if abs(filt) > math.radians(0.1):
            yaw_rate = clamp(k_yaw * filt + k_yaw_d * rate, -max_yaw_rate, max_yaw_rate)
        yaw_rate_cmd[i - 1] = yaw_rate

        c, s = math.cos(prev_heading), math.sin(prev_heading)
        vx_world = vx_body * c - vy_body * s
        vy_world = vx_body * s + vy_body * c

        ref_xy[i] = prev_pos + np.array([vx_world, vy_world]) * dt
        ref_heading[i] = wrap_pi_scalar(prev_heading + yaw_rate * dt)

    if n > 1:
        los = np.arctan2(target_xy[-1, 1] - ref_xy[-1, 1], target_xy[-1, 0] - ref_xy[-1, 0])
        lambda_raw[-1] = wrap_pi_scalar(float(los - ref_heading[-1]))
        lambda_filt[-1] = lambda_filt[-2]
        lambda_dot[-1] = lambda_dot[-2]
        yaw_rate_cmd[-1] = yaw_rate_cmd[-2]

    return {
        "xy": ref_xy,
        "heading": ref_heading,
        "lambda_raw": lambda_raw,
        "lambda_filt": lambda_filt,
        "lambda_dot": lambda_dot,
        "yaw_rate_cmd": yaw_rate_cmd,
    }


def compute_component_errors(t, real_xy, ref_xy, ref_heading):
    delta = real_xy - ref_xy
    total = np.linalg.norm(delta, axis=1)
    tx = np.cos(ref_heading)
    ty = np.sin(ref_heading)
    along = delta[:, 0] * tx + delta[:, 1] * ty
    cross = tx * delta[:, 1] - ty * delta[:, 0]
    return total, along, cross


def rmse(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    return float(math.sqrt(np.mean(x * x)))


def finite_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def finite_max_abs(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.max(np.abs(x))) if len(x) else float("nan")


def make_plots(out_dir, rel_t, target_xy, real_xy, ref_xy, ref_heading,
               total_err, along_err, cross_err, real_dist, ref_dist,
               lambda_real, lambda_ref, eval_end_rel,
               xy_target_full=None, xy_real_full=None, xy_real_dist_full=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        warn(f"matplotlib unavailable, skip plots: {e}")
        return []

    made = []
    xy_target = target_xy if xy_target_full is None else xy_target_full
    xy_real = real_xy if xy_real_full is None else xy_real_full
    xy_real_dist = real_dist if xy_real_dist_full is None else xy_real_dist_full

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    ax.plot(xy_target[:, 0], xy_target[:, 1], label="target real XY", linewidth=2, color="tab:green")
    ax.plot(xy_real[:, 0], xy_real[:, 1], label="test real work-segment XY", linewidth=2, color="tab:blue")
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], label="PNG reference XY", linewidth=2, linestyle="--", color="tab:orange")
    ax.scatter([xy_target[0, 0]], [xy_target[0, 1]], marker="o", s=40, label="target start")
    ax.scatter([xy_real[0, 0]], [xy_real[0, 1]], marker="s", s=40, label="test start")
    i_real_min = int(np.nanargmin(xy_real_dist))
    i_ref_min = int(np.nanargmin(ref_dist))
    ax.scatter([xy_real[i_real_min, 0]], [xy_real[i_real_min, 1]], marker="x", s=70,
               color="tab:blue", label="real closest (test)")
    ax.scatter([xy_target[i_real_min, 0]], [xy_target[i_real_min, 1]], marker="x", s=70,
               color="tab:green", label="real closest (target)")
    ax.scatter([ref_xy[i_ref_min, 0]], [ref_xy[i_ref_min, 1]], marker="^", s=70,
               color="tab:orange", label="ref closest (ref)")
    ax.scatter([target_xy[i_ref_min, 0]], [target_xy[i_ref_min, 1]], marker="^", s=70,
               facecolors="none", edgecolors="tab:green", linewidths=1.4, label="ref closest (target)")
    ax.set_title("XY trajectory comparison (selected work segment)")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.axis("equal")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "trajectory_xy_compare.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].plot(rel_t, total_err, color="black")
    axes[0].set_ylabel("real-ref / m")
    axes[0].grid(True, alpha=0.35)
    axes[0].set_title("Reference tracking error")
    axes[1].plot(rel_t, along_err, label="along-track")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("along / m")
    axes[1].grid(True, alpha=0.35)
    axes[2].plot(rel_t, cross_err, label="cross-track", color="tab:red")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("cross / m")
    axes[2].set_xlabel("time since work-segment start / s")
    axes[2].grid(True, alpha=0.35)
    for ax in axes:
        ax.axvline(eval_end_rel, color="gray", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    path = out_dir / "reference_tracking_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
    axes[0].plot(rel_t, real_dist, label="real test-target distance")
    axes[0].plot(rel_t, ref_dist, label="ref-target distance", linestyle="--")
    axes[0].set_ylabel("distance / m")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()
    axes[0].set_title("Horizontal convergence to target")
    axes[1].plot(rel_t, np.degrees(lambda_real), label="real LOS bearing error")
    axes[1].plot(rel_t, np.degrees(lambda_ref), label="ref LOS bearing error", linestyle="--")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("bearing error / deg")
    axes[1].set_xlabel("time since work-segment start / s")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend()
    for ax in axes:
        ax.axvline(eval_end_rel, color="gray", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    path = out_dir / "target_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    made.append(path)

    return made


def format_metric(v, unit="m"):
    if not np.isfinite(v):
        return "nan"
    return f"{v:.3f} {unit}" if unit else f"{v:.3f}"


def write_summary(
    path: Path,
    bag_path: str,
    selected_topics,
    mode_changes,
    selected_pattern,
    intervals,
    selected_interval,
    params,
    metrics,
    output_files,
):
    st, en, mode = selected_interval
    cross_mean = metrics["approach_cross_mean"]
    if abs(cross_mean) < 0.2:
        lateral_text = "真实轨迹整体横向偏差不明显。"
    elif cross_mean > 0:
        lateral_text = "真实轨迹整体偏在参考轨迹左侧。"
    else:
        lateral_text = "真实轨迹整体偏在参考轨迹右侧。"

    along_mean = metrics["approach_along_mean"]
    if abs(along_mean) < 0.3:
        along_text = "沿参考方向的提前/滞后不明显。"
    elif along_mean > 0:
        along_text = "真实轨迹相对参考轨迹更靠前，可能存在速度偏大或切入更早的情况。"
    else:
        along_text = "真实轨迹相对参考轨迹偏滞后，可能存在速度不足、切入较晚或控制响应偏慢的情况。"

    real_min = metrics["real_min_dist"]
    ref_min = metrics["ref_min_dist"]
    if np.isfinite(real_min) and np.isfinite(ref_min):
        diff = real_min - ref_min
        if diff <= 0.5:
            convergence_text = "真实轨迹最近距离接近参考轨迹，横向收敛效果总体可以。"
        elif diff <= 2.0:
            convergence_text = "真实轨迹比参考轨迹最近距离更大，横向收敛有一定差距。"
        else:
            convergence_text = "真实轨迹与参考轨迹最近距离差距较大，需要重点检查检测稳定性、切入时机和速度/偏航响应。"
    else:
        convergence_text = "最近距离指标无法稳定计算，需要先检查所选作业段数据是否完整。"

    with path.open("w", encoding="utf-8") as f:
        f.write("PNG horizontal reference trajectory analysis\n")
        f.write("=" * 52 + "\n\n")
        f.write(f"bag: {bag_path}\n")
        f.write("selected_topics:\n")
        f.write(f"  ego_pose: {selected_topics['ego_pose']}\n")
        f.write(f"  target_pose: {selected_topics['target_pose']}\n")
        f.write(f"  ego_velocity: {selected_topics.get('ego_velocity')}\n")
        f.write(f"  ego_state: {selected_topics['state']}\n\n")

        f.write("state mode changes:\n")
        for t, m in mode_changes:
            f.write(f"  t={t:.3f}s mode={m}\n")
        f.write("\n")

        f.write(f"selected mode pattern: {selected_pattern}\n")
        f.write("work intervals detected:\n")
        for i, (a, b, m) in enumerate(intervals):
            f.write(f"  [{i}] start={a:.3f}s end={b:.3f}s duration={b-a:.3f}s mode={m}\n")
        f.write(f"\nselected_work_interval: start={st:.3f}s end={en:.3f}s duration={en-st:.3f}s mode={mode}\n\n")

        f.write("reference_model:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("metrics, approach window only:\n")
        f.write(f"  evaluation_end_since_work_start_s: {metrics['eval_end_rel']:.3f}\n")
        f.write(f"  real_vs_reference_RMSE_xy: {format_metric(metrics['approach_rmse'])}\n")
        f.write(f"  real_vs_reference_mean_xy: {format_metric(metrics['approach_mean_error'])}\n")
        f.write(f"  real_vs_reference_max_xy: {format_metric(metrics['approach_max_error'])}\n")
        f.write(f"  mean_abs_cross_track_error: {format_metric(metrics['approach_cross_abs_mean'])}\n")
        f.write(f"  max_abs_cross_track_error: {format_metric(metrics['approach_cross_abs_max'])}\n")
        f.write(f"  signed_mean_cross_track_error: {format_metric(metrics['approach_cross_mean'])}\n")
        f.write(f"  signed_mean_along_track_error: {format_metric(metrics['approach_along_mean'])}\n\n")

        f.write("metrics, whole selected work interval:\n")
        f.write(f"  real_min_distance_to_target: {format_metric(metrics['real_min_dist'])} at t={metrics['real_min_time']:.3f}s since work start\n")
        f.write(f"  ref_min_distance_to_target: {format_metric(metrics['ref_min_dist'])} at t={metrics['ref_min_time']:.3f}s since work start\n")
        f.write(f"  final_real_distance_to_target: {format_metric(metrics['real_final_dist'])}\n")
        f.write(f"  final_ref_distance_to_target: {format_metric(metrics['ref_final_dist'])}\n")
        f.write(f"  whole_work_interval_real_vs_reference_RMSE_xy: {format_metric(metrics['whole_rmse'])}\n\n")

        f.write("conclusion:\n")
        f.write(f"  {convergence_text}\n")
        f.write(f"  {lateral_text}\n")
        f.write(f"  {along_text}\n\n")

        f.write("outputs:\n")
        for item in output_files:
            f.write(f"  {item}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a horizontal PNG reference trajectory and compare it with the real work-segment trajectory."
    )
    parser.add_argument("bag", help="ROS1 bag path")
    parser.add_argument("--ego-pose-topic", default=None, help="default: auto-detect /mavros/local_position/pose")
    parser.add_argument("--target-pose-topic", default=None, help="default: auto-detect /target/mavros/local_position/pose")
    parser.add_argument("--state-topic", default=None, help="default: auto-detect /mavros/state")
    parser.add_argument("--ego-velocity-topic", default=None, help="optional, used for initial heading if available")
    parser.add_argument(
        "--work-mode",
        default="auto",
        help="work mode selector. default 'auto' means prefer GUIDED, then AUTO. "
             "Can also be a substring or comma list, e.g. GUIDED or GUIDED,AUTO.",
    )
    parser.add_argument("--work-exact", action="store_true", help="require exact work-mode match instead of substring match")
    parser.add_argument("--work-index", type=int, default=None, help="which matched work interval to analyze, default 0")
    parser.add_argument("--auto-mode", default=None, help="deprecated alias of --work-mode")
    parser.add_argument("--auto-exact", action="store_true", help="deprecated alias of --work-exact")
    parser.add_argument("--auto-index", type=int, default=None, help="deprecated alias of --work-index")
    parser.add_argument("--dt", type=float, default=0.05, help="reference time step, default 0.05s")
    parser.add_argument("--speed", default="10.0", help="reference speed in m/s, or 'auto' to use median real work-segment speed")
    parser.add_argument("--N", type=float, default=5.0, help="PNG navigation constant, default 5")
    parser.add_argument("--k-yaw", type=float, default=1.5, help="yaw proportional gain used in reference heading update")
    parser.add_argument("--k-yaw-d", type=float, default=0.3, help="yaw rate damping gain used in reference heading update")
    parser.add_argument("--max-yaw-rate", type=float, default=1.0, help="max yaw rate in rad/s")
    parser.add_argument("--lambda-alpha", type=float, default=0.2, help="LOS bearing error smoothing alpha")
    parser.add_argument("--max-side-step-deg", type=float, default=30.0, help="limit of one-step PNG side-slip angle")
    parser.add_argument("--post-ref-closest-window", type=float, default=0.5, help="approach metrics end this many seconds after ref closest point")
    parser.add_argument("--out-dir", default=None, help="output directory, default reference_out_<bagstem>")
    return parser.parse_args()


def main():
    args = parse_args()
    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        fail(f"bag not found: {bag_path}", 2)
    if args.dt <= 0:
        fail("--dt must be positive", 2)

    try:
        import rosbag
        from rosbag.bag import ROSBagUnindexedException
    except ModuleNotFoundError:
        print("\n[ERROR] ROS Python deps missing. Run this script with ROS1 Python, for example:")
        print("  source /opt/ros/noetic/setup.bash")
        print(f"  python3 {Path(__file__).name} {bag_path.name}\n")
        sys.exit(3)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path.cwd() / f"reference_out_{bag_path.stem}"
    ensure_dir(out_dir)

    try:
        bag = rosbag.Bag(str(bag_path), "r", allow_unindexed=True)
    except ROSBagUnindexedException:
        fail("bag is unindexed. Try: rosbag reindex <bag>", 4)
    except Exception as e:
        fail(f"failed to open bag: {e}", 4)

    with bag:
        start_abs, end_abs, bag_duration, rows = topic_summary(bag)
        write_csv(out_dir / "topic_summary.csv",
                  ["topic", "msg_type", "message_count", "estimated_hz"], rows)
        all_topics = [r[0] for r in rows if r[2] > 0]

        auto_ego_pose, auto_target_pose = find_pose_topics(all_topics)
        ego_pose_topic = args.ego_pose_topic or auto_ego_pose
        target_pose_topic = args.target_pose_topic or auto_target_pose
        state_topic = args.state_topic or find_state_topic(all_topics)
        ego_vel_topic = args.ego_velocity_topic or find_velocity_topic(all_topics)

        if not ego_pose_topic or not target_pose_topic:
            fail("missing ego or target pose topic. Use --ego-pose-topic / --target-pose-topic.", 5)
        if not state_topic:
            fail("missing ego state topic. Use --state-topic, or record /mavros/state.", 5)

        ego_pose = read_pose_series(bag, ego_pose_topic, start_abs)
        target_pose = read_pose_series(bag, target_pose_topic, start_abs)
        state_series = read_state_series(bag, state_topic, start_abs)
        ego_vel = read_velocity_series(bag, ego_vel_topic, start_abs) if ego_vel_topic else None

    if ego_pose is None or target_pose is None:
        fail("cannot parse enough ego/target pose messages.", 6)
    if state_series is None:
        fail("cannot parse state messages.", 6)

    ego_t, ego_xyz = ego_pose
    target_t, target_xyz = target_pose
    ego_xy = ego_xyz[:, :2]
    target_xy_raw = target_xyz[:, :2]
    state_t, modes = state_series

    mode_spec = args.auto_mode if args.auto_mode is not None else args.work_mode
    mode_exact = bool(args.work_exact or args.auto_exact)
    work_index = args.auto_index if args.auto_index is not None else args.work_index
    if work_index is None:
        work_index = 0

    mode_changes = mode_change_rows(state_t, modes)
    selected_pattern, intervals, all_mode_rows = select_work_intervals(
        state_t, modes, bag_duration, mode_spec, mode_exact
    )

    write_csv(out_dir / "state_mode_changes.csv",
              ["t_s", "mode"],
              [[f"{t:.6f}", m] for t, m in mode_changes])
    interval_rows = [[i, f"{a:.6f}", f"{b:.6f}", f"{b-a:.6f}", m]
                     for i, (a, b, m) in enumerate(intervals)]
    write_csv(out_dir / "work_intervals.csv",
              ["index", "start_s", "end_s", "duration_s", "mode"],
              interval_rows)
    # Keep the old filename as a diagnostic of true AUTO matches. The selected
    # analysis segment is written to work_intervals.csv above.
    legacy_auto_intervals = build_mode_intervals(state_t, modes, bag_duration, "AUTO", False)
    write_csv(out_dir / "auto_intervals.csv",
              ["index", "start_s", "end_s", "duration_s", "mode"],
              [[i, f"{a:.6f}", f"{b:.6f}", f"{b-a:.6f}", m]
               for i, (a, b, m) in enumerate(legacy_auto_intervals)])
    write_csv(out_dir / "mode_candidate_intervals.csv",
              ["pattern", "start_s", "end_s", "duration_s", "mode"],
              [[pat, f"{a:.6f}", f"{b:.6f}", f"{b-a:.6f}", m]
               for pat, a, b, m in all_mode_rows])

    if not intervals:
        fail(f"no work interval matched mode selector {mode_spec!r}", 7)
    if work_index < 0 or work_index >= len(intervals):
        fail(f"--work-index/--auto-index {work_index} out of range, detected {len(intervals)} intervals", 7)

    work_start, work_end, work_mode = intervals[work_index]
    grid_start = max(work_start, float(ego_t[0]), float(target_t[0]))
    grid_end = min(work_end, float(ego_t[-1]), float(target_t[-1]))
    if grid_end - grid_start < max(1.0, args.dt * 5):
        fail("selected work interval has too little overlapping ego/target pose data.", 8)

    grid_t = np.arange(grid_start, grid_end + args.dt * 0.5, args.dt)
    if grid_t[-1] > grid_end:
        grid_t[-1] = grid_end
    rel_t = grid_t - grid_start

    real_xy = interp_xy(ego_t, ego_xy, grid_t)
    target_xy = interp_xy(target_t, target_xy_raw, grid_t)

    if str(args.speed).lower() == "auto":
        spd = estimate_speed_from_positions(ego_t, ego_xy, grid_start, grid_end)
        if spd is None or not np.isfinite(spd) or spd <= 0:
            warn("cannot estimate real work-segment speed; fallback to 5.0 m/s")
            spd = 5.0
    else:
        try:
            spd = float(args.speed)
        except ValueError:
            fail("--speed must be a number or 'auto'", 2)
    if spd <= 0:
        fail("reference speed must be positive", 2)

    initial_heading = None
    if ego_vel is not None:
        vel_t, vel_xy = ego_vel
        vx0 = float(interp_series(vel_t, vel_xy[:, 0], [grid_start])[0])
        vy0 = float(interp_series(vel_t, vel_xy[:, 1], [grid_start])[0])
        if math.hypot(vx0, vy0) > 0.2:
            initial_heading = math.atan2(vy0, vx0)

    if initial_heading is None:
        initial_heading = estimate_heading_from_positions(ego_t, ego_xy, grid_start)
    if initial_heading is None:
        initial_heading = math.atan2(target_xy[0, 1] - real_xy[0, 1], target_xy[0, 0] - real_xy[0, 0])

    ref = simulate_png_reference(
        grid_t=grid_t,
        target_xy=target_xy,
        start_xy=real_xy[0],
        initial_heading=initial_heading,
        speed=spd,
        n_nav=args.N,
        k_yaw=args.k_yaw,
        k_yaw_d=args.k_yaw_d,
        max_yaw_rate=args.max_yaw_rate,
        lambda_alpha=args.lambda_alpha,
        max_side_step_rad=math.radians(args.max_side_step_deg),
    )

    ref_xy = ref["xy"]
    ref_heading = ref["heading"]
    ref_dist = np.linalg.norm(target_xy - ref_xy, axis=1)
    xy_target_full = target_xy.copy()
    xy_real_full = real_xy.copy()
    xy_real_dist_full = np.linalg.norm(xy_target_full - xy_real_full, axis=1)
    full_ref_min_idx = int(np.nanargmin(ref_dist))
    ref_collision_reached = float(ref_dist[full_ref_min_idx]) <= REF_COLLISION_THRESHOLD_M
    ref_collision_rel_t = None
    if ref_collision_reached:
        stop_n = full_ref_min_idx + 1
        grid_t = grid_t[:stop_n]
        rel_t = rel_t[:stop_n]
        target_xy = target_xy[:stop_n]
        real_xy = real_xy[:stop_n]
        ref_xy = ref_xy[:stop_n]
        ref_heading = ref_heading[:stop_n]
        for key in ("lambda_raw", "lambda_filt", "lambda_dot", "yaw_rate_cmd"):
            ref[key] = ref[key][:stop_n]
        ref_collision_rel_t = float(rel_t[-1])

    total_err, along_err, cross_err = compute_component_errors(grid_t, real_xy, ref_xy, ref_heading)
    real_dist = np.linalg.norm(target_xy - real_xy, axis=1)
    ref_dist = np.linalg.norm(target_xy - ref_xy, axis=1)

    real_heading, real_speed = heading_from_xy(grid_t, real_xy)
    real_los = np.arctan2(target_xy[:, 1] - real_xy[:, 1], target_xy[:, 0] - real_xy[:, 0])
    ref_los = np.arctan2(target_xy[:, 1] - ref_xy[:, 1], target_xy[:, 0] - ref_xy[:, 0])
    lambda_real = wrap_pi(real_los - real_heading)
    lambda_ref = wrap_pi(ref_los - ref_heading)

    ref_min_idx = int(np.nanargmin(ref_dist))
    real_min_idx = int(np.nanargmin(real_dist))
    eval_end_rel = min(float(rel_t[-1]), float(rel_t[ref_min_idx] + max(0.0, args.post_ref_closest_window)))
    eval_mask = rel_t <= eval_end_rel
    if np.count_nonzero(eval_mask) < 2:
        eval_mask[:] = True

    metrics = {
        "eval_end_rel": eval_end_rel,
        "approach_rmse": rmse(total_err[eval_mask]),
        "approach_mean_error": finite_mean(total_err[eval_mask]),
        "approach_max_error": finite_max_abs(total_err[eval_mask]),
        "approach_cross_abs_mean": finite_mean(np.abs(cross_err[eval_mask])),
        "approach_cross_abs_max": finite_max_abs(cross_err[eval_mask]),
        "approach_cross_mean": finite_mean(cross_err[eval_mask]),
        "approach_along_mean": finite_mean(along_err[eval_mask]),
        "real_min_dist": float(real_dist[real_min_idx]),
        "real_min_time": float(rel_t[real_min_idx]),
        "ref_min_dist": float(ref_dist[ref_min_idx]),
        "ref_min_time": float(rel_t[ref_min_idx]),
        "real_final_dist": float(real_dist[-1]),
        "ref_final_dist": float(ref_dist[-1]),
        "whole_rmse": rmse(total_err),
        "ref_collision_reached": ref_collision_reached,
        "ref_collision_threshold_m": REF_COLLISION_THRESHOLD_M,
        "ref_collision_time": ref_collision_rel_t,
    }

    rows = []
    for i in range(len(grid_t)):
        rows.append([
            f"{grid_t[i]:.6f}",
            f"{rel_t[i]:.6f}",
            f"{target_xy[i,0]:.6f}", f"{target_xy[i,1]:.6f}",
            f"{real_xy[i,0]:.6f}", f"{real_xy[i,1]:.6f}",
            f"{ref_xy[i,0]:.6f}", f"{ref_xy[i,1]:.6f}",
            f"{math.degrees(ref_heading[i]):.6f}",
            f"{total_err[i]:.6f}", f"{along_err[i]:.6f}", f"{cross_err[i]:.6f}",
            f"{real_dist[i]:.6f}", f"{ref_dist[i]:.6f}",
            f"{math.degrees(lambda_real[i]) if np.isfinite(lambda_real[i]) else float('nan'):.6f}",
            f"{math.degrees(lambda_ref[i]):.6f}",
            f"{ref['lambda_dot'][i]:.6f}",
            f"{ref['yaw_rate_cmd'][i]:.6f}",
        ])
    write_csv(
        out_dir / "reference_trajectory.csv",
        [
            "t_bag_s", "t_since_work_start_s",
            "target_x", "target_y",
            "real_x", "real_y",
            "ref_x", "ref_y",
            "ref_heading_deg",
            "real_ref_error_xy_m", "along_track_error_m", "cross_track_error_m",
            "real_target_dist_m", "ref_target_dist_m",
            "real_los_bearing_error_deg", "ref_los_bearing_error_deg",
            "ref_lambda_dot_rad_s", "ref_yaw_rate_cmd_rad_s",
        ],
        rows,
    )

    output_files = [
        "reference_trajectory.csv",
        "work_intervals.csv",
        "auto_intervals.csv",
        "state_mode_changes.csv",
        "mode_candidate_intervals.csv",
        "topic_summary.csv",
    ]
    plots = make_plots(
        out_dir=out_dir,
        rel_t=rel_t,
        target_xy=target_xy,
        real_xy=real_xy,
        ref_xy=ref_xy,
        ref_heading=ref_heading,
        total_err=total_err,
        along_err=along_err,
        cross_err=cross_err,
        real_dist=real_dist,
        ref_dist=ref_dist,
        lambda_real=lambda_real,
        lambda_ref=lambda_ref,
        eval_end_rel=eval_end_rel,
        xy_target_full=xy_target_full,
        xy_real_full=xy_real_full,
        xy_real_dist_full=xy_real_dist_full,
    )
    output_files.extend([p.name for p in plots])

    params = {
        "horizontal_only": True,
        "N_nav": args.N,
        "reference_speed_mps": f"{spd:.6f}",
        "dt_s": args.dt,
        "k_yaw": args.k_yaw,
        "k_yaw_d": args.k_yaw_d,
        "max_yaw_rate_rad_s": args.max_yaw_rate,
        "lambda_alpha": args.lambda_alpha,
        "max_side_step_deg": args.max_side_step_deg,
        "initial_heading_deg": f"{math.degrees(initial_heading):.6f}",
        "work_mode_selector": mode_spec,
        "selected_mode_pattern": selected_pattern,
        "work_exact": mode_exact,
        "work_index": work_index,
        "ref_collision_threshold_m": REF_COLLISION_THRESHOLD_M,
        "ref_collision_reached": ref_collision_reached,
        "ref_collision_time_since_work_start_s": "none" if ref_collision_rel_t is None else f"{ref_collision_rel_t:.6f}",
    }
    selected_topics = {
        "ego_pose": ego_pose_topic,
        "target_pose": target_pose_topic,
        "ego_velocity": ego_vel_topic,
        "state": state_topic,
    }
    write_summary(
        path=out_dir / "reference_summary.txt",
        bag_path=str(bag_path),
        selected_topics=selected_topics,
        mode_changes=mode_changes,
        selected_pattern=selected_pattern,
        intervals=intervals,
        selected_interval=(work_start, work_end, work_mode),
        params=params,
        metrics=metrics,
        output_files=output_files + ["reference_summary.txt"],
    )

    print("[DONE] Output:", out_dir)
    for name in output_files + ["reference_summary.txt"]:
        print("  -", name)


if __name__ == "__main__":
    main()
