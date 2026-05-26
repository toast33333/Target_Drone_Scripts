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
  - The reference generator follows the horizontal logic in rpy_png_no_alt.cpp:
    LOS rate Kalman filter -> PNG lateral acceleration -> filtered lateral
    velocity -> world-frame velocity command.
  - Default PNG parameters follow the project launch defaults: N=3, V=5 m/s.
"""

import argparse
import configparser
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

DEFAULT_ARGS = {
    "ego_pose_topic": None,
    "target_pose_topic": None,
    "state_topic": None,
    "ego_velocity_topic": None,
    "work_mode": "auto",
    "work_exact": False,
    "work_index": None,
    "auto_mode": None,
    "auto_exact": False,
    "auto_index": None,
    "dt": 0.05,
    "speed": "5.0",
    "N": 3.0,
    "v_lat_max": 3.0,
    "tau_lat": 0.5,
    "lambda_alpha": 0.2,
    "lambda_window_size": 5,
    "lambda_kalman_q_position": None,
    "lambda_kalman_q_rate": None,
    "lambda_kalman_r": None,
    "heading_source": "auto",
    "k_yaw": 1.5,
    "k_yaw_d": 0.3,
    "max_yaw_rate": 1.0,
    "max_side_step_deg": 30.0,
    "post_ref_closest_window": 0.5,
    "ref_collision_threshold": REF_COLLISION_THRESHOLD_M,
    "out_dir": None,
}

CONFIG_TYPES = {
    "ego_pose_topic": str,
    "target_pose_topic": str,
    "state_topic": str,
    "ego_velocity_topic": str,
    "work_mode": str,
    "work_exact": bool,
    "work_index": int,
    "auto_mode": str,
    "auto_exact": bool,
    "auto_index": int,
    "dt": float,
    "speed": str,
    "N": float,
    "v_lat_max": float,
    "tau_lat": float,
    "lambda_alpha": float,
    "lambda_window_size": int,
    "lambda_kalman_q_position": float,
    "lambda_kalman_q_rate": float,
    "lambda_kalman_r": float,
    "heading_source": str,
    "k_yaw": float,
    "k_yaw_d": float,
    "max_yaw_rate": float,
    "max_side_step_deg": float,
    "post_ref_closest_window": float,
    "ref_collision_threshold": float,
    "out_dir": str,
}

CONFIG_TEMPLATE = """# PNG reference trajectory analysis config.
# Usage:
#   python3 generate_png_reference_trajectory.py 03.bag --config png_reference_flight_config.ini
#
# Command-line arguments override values in this file.
# Empty values mean auto/default where supported.

[topics]
# Leave empty to auto-detect.
ego_pose_topic =
target_pose_topic =
state_topic =
ego_velocity_topic =

[work_segment]
# auto means prefer GUIDED, then AUTO. Use AUTO or GUIDED to force one mode.
work_mode = auto
work_exact = false
# Empty means 0. Set 1, 2, ... when one mode appears in multiple intervals.
work_index =

[reference_model]
# Match these values to the parameters used in the real flight.
dt = 0.05
speed = 5.0
N = 3.0
v_lat_max = 3.0
tau_lat = 0.5

# These two legacy values are mapped to Kalman noise by the same rule as the
# real project. Usually keep them unless the flight launch file changed them.
lambda_alpha = 0.2
lambda_window_size = 5

# Leave empty to use the mapped defaults above. Fill numbers only if the flight
# explicitly set lambda_kalman_* parameters.
lambda_kalman_q_position =
lambda_kalman_q_rate =
lambda_kalman_r =

# auto: use bag pose yaw if available, otherwise use reference course.
# bag-yaw: require pose yaw.
# course: always use generated trajectory course.
heading_source = auto

[legacy_compatibility]
# Kept only so older command lines/configs do not break. The current
# rpy_png_no_alt-style model does not use these values.
k_yaw = 1.5
k_yaw_d = 0.3
max_yaw_rate = 1.0
max_side_step_deg = 30.0

[output]
post_ref_closest_window = 0.5
ref_collision_threshold = 0.15
out_dir =
"""


def fail(msg: str, code: int = 1):
    print("[ERROR] " + msg, file=sys.stderr)
    sys.exit(code)


def warn(msg: str):
    print("[WARN] " + msg, file=sys.stderr)


def parse_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def coerce_config_value(key: str, value: str):
    text = str(value).strip()
    if text == "":
        return None if DEFAULT_ARGS.get(key) is None else DEFAULT_ARGS[key]
    typ = CONFIG_TYPES[key]
    if typ is bool:
        return parse_bool(text)
    if typ is str:
        return text
    return typ(text)


def read_config_defaults(config_path: Path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        fail(f"config file not found: {config_path}", 2)

    cp = configparser.ConfigParser()
    cp.optionxform = str
    cp.read(config_path, encoding="utf-8-sig")

    defaults = {}
    known = set(DEFAULT_ARGS)
    for section in cp.sections():
        for key, value in cp.items(section):
            if key not in known:
                warn(f"unknown config key ignored: [{section}] {key}")
                continue
            try:
                defaults[key] = coerce_config_value(key, value)
            except ValueError as e:
                fail(f"bad config value for [{section}] {key}: {e}", 2)
    return defaults


def write_config_template(path: Path):
    path = Path(path).expanduser().resolve()
    ensure_dir(path.parent)
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return path


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


class ScalarRateKalmanFilter:
    """Python equivalent of png_planner/src/los_rate_kalman.h."""

    def __init__(self, q_position: float = 1e-4, q_rate: float = 5e-1, r_measurement: float = 2e-4):
        self.set_noise(q_position, q_rate, r_measurement)
        self.reset()

    def set_noise(self, q_position: float, q_rate: float, r_measurement: float):
        self.process_noise_position = max(float(q_position), 1e-9)
        self.process_noise_rate = max(float(q_rate), 1e-9)
        self.measurement_noise = max(float(r_measurement), 1e-9)

    def reset(self):
        self.initialized = False
        self._value = 0.0
        self._rate = 0.0
        self.p00 = 1.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 1.0

    def update(self, measurement: float, dt: float):
        dt = max(float(dt), 1e-4)
        measurement = float(measurement)

        if not self.initialized:
            self.initialized = True
            self._value = measurement
            self._rate = 0.0
            self.p00 = 1.0
            self.p01 = 0.0
            self.p10 = 0.0
            self.p11 = 1.0
            return

        self._value += dt * self._rate

        predicted_p00 = (
            self.p00
            + dt * (self.p10 + self.p01)
            + dt * dt * self.p11
            + self.process_noise_position * dt * dt
        )
        predicted_p01 = self.p01 + dt * self.p11
        predicted_p10 = self.p10 + dt * self.p11
        predicted_p11 = self.p11 + self.process_noise_rate * dt

        innovation = measurement - self._value
        innovation_cov = predicted_p00 + self.measurement_noise
        k0 = predicted_p00 / innovation_cov
        k1 = predicted_p10 / innovation_cov

        self._value += k0 * innovation
        self._rate += k1 * innovation

        self.p00 = (1.0 - k0) * predicted_p00
        self.p01 = (1.0 - k0) * predicted_p01
        self.p10 = self.p01
        self.p11 = predicted_p11 - k1 * predicted_p01

    def value(self) -> float:
        return self._value

    def rate(self) -> float:
        return self._rate


def map_legacy_los_chain_to_kalman(alpha: float, window_size: int):
    """Match mapLegacyLosChainToKalman() from the real project."""
    alpha = clamp(float(alpha), 0.05, 0.95)
    window = max(2, int(window_size))
    alpha_scale = alpha / 0.2
    window_scale = 3.0 / float(window)
    q_position = max(1e-6, 1e-4 * alpha_scale * window_scale)
    q_rate = max(1e-3, 5e-1 * alpha_scale * window_scale)
    r_measurement = max(1e-6, 2e-4 * (0.2 / alpha) * (0.2 / alpha) / window_scale)
    return q_position, q_rate, r_measurement


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


def interp_angle_series(t_src, angle_src, t_dst):
    angle_src = np.asarray(angle_src, dtype=float)
    if len(angle_src) == 0:
        return np.full_like(np.asarray(t_dst, dtype=float), np.nan, dtype=float)
    return wrap_pi(interp_series(t_src, np.unwrap(angle_src), t_dst))


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


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_yaw_from_pose_like(msg):
    candidates = [
        ("pose", "orientation"),
        ("pose", "pose", "orientation"),
        ("orientation",),
    ]
    for chain in candidates:
        if _has_chain(msg, chain):
            cur = msg
            for name in chain:
                cur = getattr(cur, name)
            if all(hasattr(cur, k) for k in ("x", "y", "z", "w")):
                return quaternion_to_yaw(float(cur.x), float(cur.y), float(cur.z), float(cur.w))
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


def read_yaw_series(bag, topic: str, t0: float):
    ts, yaws = [], []
    for _, msg, t in bag.read_messages(topics=[topic]):
        yaw = extract_yaw_from_pose_like(msg)
        if yaw is None:
            continue
        ts.append(to_sec(t) - t0)
        yaws.append(yaw)
    if len(ts) < 2:
        return None
    return np.asarray(ts, dtype=float), np.asarray(yaws, dtype=float)


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
    v_lat_max,
    tau_lat,
    lambda_kalman_q_position,
    lambda_kalman_q_rate,
    lambda_kalman_r,
    heading_profile=None,
):
    n = len(grid_t)
    ref_xy = np.zeros((n, 2), dtype=float)
    ref_heading = np.zeros(n, dtype=float)
    lambda_raw = np.zeros(n, dtype=float)
    lambda_filt = np.zeros(n, dtype=float)
    lambda_dot = np.zeros(n, dtype=float)
    yaw_rate_cmd = np.zeros(n, dtype=float)
    a_lat_cmd = np.zeros(n, dtype=float)
    v_lat_cmd = np.zeros(n, dtype=float)

    ref_xy[0] = np.asarray(start_xy, dtype=float)
    ref_heading[0] = float(initial_heading)

    if heading_profile is not None:
        heading_profile = np.asarray(heading_profile, dtype=float)
        if len(heading_profile) != n:
            raise ValueError("heading_profile length must match grid_t")

    lambda_filter = ScalarRateKalmanFilter(
        lambda_kalman_q_position,
        lambda_kalman_q_rate,
        lambda_kalman_r,
    )
    v_lat = 0.0
    course_heading = float(initial_heading)

    def body_yaw_at(idx: int, fallback: float) -> float:
        if heading_profile is not None and np.isfinite(heading_profile[idx]):
            return float(heading_profile[idx])
        return float(fallback)

    for i in range(1, n):
        dt = max(1e-3, float(grid_t[i] - grid_t[i - 1]))
        prev_pos = ref_xy[i - 1]
        yaw_b = body_yaw_at(i - 1, course_heading)
        ref_heading[i - 1] = yaw_b

        los = math.atan2(target_xy[i - 1, 1] - prev_pos[1], target_xy[i - 1, 0] - prev_pos[0])
        # The online node receives lambda_meas = -gimbal_yaw_.  Offline, the
        # target/reference geometry gives the same horizontal LOS bearing error.
        raw = wrap_pi_scalar(los - yaw_b)
        lambda_raw[i - 1] = raw

        if lambda_filter.initialized:
            measurement = lambda_filter.value() + wrap_pi_scalar(raw - lambda_filter.value())
        else:
            measurement = raw
        lambda_filter.update(measurement, dt)
        filt = wrap_pi_scalar(lambda_filter.value())
        rate = lambda_filter.rate()
        lambda_filt[i - 1] = filt
        lambda_dot[i - 1] = rate

        a_lat = n_nav * speed * rate
        v_lat_target = clamp(v_lat + a_lat * dt, -abs(v_lat_max), abs(v_lat_max))
        alpha = dt / max(float(tau_lat), dt)
        v_lat = (1.0 - alpha) * v_lat + alpha * v_lat_target
        a_lat_cmd[i - 1] = a_lat
        v_lat_cmd[i - 1] = v_lat

        vx_body = speed
        vy_body = v_lat

        c, s = math.cos(yaw_b), math.sin(yaw_b)
        vx_world = vx_body * c - vy_body * s
        vy_world = vx_body * s + vy_body * c

        ref_xy[i] = prev_pos + np.array([vx_world, vy_world]) * dt

        if heading_profile is None:
            if math.hypot(vx_world, vy_world) > 1e-6:
                course_heading = math.atan2(vy_world, vx_world)
            ref_heading[i] = course_heading
        else:
            ref_heading[i] = body_yaw_at(i, yaw_b)

    if n > 1:
        final_yaw = body_yaw_at(n - 1, course_heading)
        ref_heading[-1] = final_yaw
        los = math.atan2(target_xy[-1, 1] - ref_xy[-1, 1], target_xy[-1, 0] - ref_xy[-1, 0])
        lambda_raw[-1] = wrap_pi_scalar(los - final_yaw)
        lambda_filt[-1] = wrap_pi_scalar(lambda_filter.value()) if lambda_filter.initialized else lambda_raw[-1]
        lambda_dot[-1] = lambda_filter.rate() if lambda_filter.initialized else 0.0
        yaw_rate_cmd[-1] = 0.0
        a_lat_cmd[-1] = a_lat_cmd[-2]
        v_lat_cmd[-1] = v_lat_cmd[-2]

    return {
        "xy": ref_xy,
        "heading": ref_heading,
        "lambda_raw": lambda_raw,
        "lambda_filt": lambda_filt,
        "lambda_dot": lambda_dot,
        "yaw_rate_cmd": yaw_rate_cmd,
        "a_lat_cmd": a_lat_cmd,
        "v_lat_cmd": v_lat_cmd,
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


def first_threshold_crossing(ref_xy, target_xy, threshold_m: float):
    """Return (segment_start_index, alpha) for first ref-target threshold crossing."""
    ref_xy = np.asarray(ref_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    threshold2 = float(threshold_m) * float(threshold_m)
    if len(ref_xy) == 0:
        return None

    rel0 = ref_xy[0] - target_xy[0]
    if float(np.dot(rel0, rel0)) <= threshold2:
        return 0, 0.0

    for i in range(1, len(ref_xy)):
        r0 = ref_xy[i - 1] - target_xy[i - 1]
        r1 = ref_xy[i] - target_xy[i]
        dr = r1 - r0
        a = float(np.dot(dr, dr))
        b = float(2.0 * np.dot(r0, dr))
        c = float(np.dot(r0, r0) - threshold2)

        if c <= 0.0:
            return i - 1, 0.0
        if a < 1e-12:
            continue

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            continue

        sqrt_disc = math.sqrt(max(0.0, disc))
        roots = sorted(((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)))
        for root in roots:
            if -1e-9 <= root <= 1.0 + 1e-9:
                return i - 1, clamp(root, 0.0, 1.0)

    return None


def lerp_angle(a0: float, a1: float, alpha: float) -> float:
    return wrap_pi_scalar(float(a0) + float(alpha) * wrap_pi_scalar(float(a1) - float(a0)))


def trim_reference_at_collision(grid_t, rel_t, target_xy, real_xy, ref_xy, ref_heading, ref, threshold_m: float):
    crossing = first_threshold_crossing(ref_xy, target_xy, threshold_m)
    if crossing is None:
        return grid_t, rel_t, target_xy, real_xy, ref_xy, ref_heading, ref, False, None

    seg_start, alpha = crossing
    if alpha <= 1e-9:
        stop_n = seg_start + 1
        grid_t = grid_t[:stop_n]
        rel_t = rel_t[:stop_n]
        target_xy = target_xy[:stop_n]
        real_xy = real_xy[:stop_n]
        ref_xy = ref_xy[:stop_n]
        ref_heading = ref_heading[:stop_n]
        for key in ("lambda_raw", "lambda_filt", "lambda_dot", "yaw_rate_cmd", "a_lat_cmd", "v_lat_cmd"):
            ref[key] = ref[key][:stop_n]
        return grid_t, rel_t, target_xy, real_xy, ref_xy, ref_heading, ref, True, float(rel_t[-1])

    if alpha >= 1.0 - 1e-9:
        stop_n = seg_start + 2
        grid_t = grid_t[:stop_n]
        rel_t = rel_t[:stop_n]
        target_xy = target_xy[:stop_n]
        real_xy = real_xy[:stop_n]
        ref_xy = ref_xy[:stop_n]
        ref_heading = ref_heading[:stop_n]
        for key in ("lambda_raw", "lambda_filt", "lambda_dot", "yaw_rate_cmd", "a_lat_cmd", "v_lat_cmd"):
            ref[key] = ref[key][:stop_n]
        return grid_t, rel_t, target_xy, real_xy, ref_xy, ref_heading, ref, True, float(rel_t[-1])

    j = seg_start
    collision_t = (1.0 - alpha) * grid_t[j] + alpha * grid_t[j + 1]
    collision_rel_t = (1.0 - alpha) * rel_t[j] + alpha * rel_t[j + 1]
    collision_target = (1.0 - alpha) * target_xy[j] + alpha * target_xy[j + 1]
    collision_real = (1.0 - alpha) * real_xy[j] + alpha * real_xy[j + 1]
    collision_ref = (1.0 - alpha) * ref_xy[j] + alpha * ref_xy[j + 1]
    collision_heading = lerp_angle(ref_heading[j], ref_heading[j + 1], alpha)

    grid_t = np.concatenate([grid_t[:j + 1], np.asarray([collision_t])])
    rel_t = np.concatenate([rel_t[:j + 1], np.asarray([collision_rel_t])])
    target_xy = np.vstack([target_xy[:j + 1], collision_target])
    real_xy = np.vstack([real_xy[:j + 1], collision_real])
    ref_xy = np.vstack([ref_xy[:j + 1], collision_ref])
    ref_heading = np.concatenate([ref_heading[:j + 1], np.asarray([collision_heading])])

    for key in ("lambda_raw", "lambda_filt", "lambda_dot", "yaw_rate_cmd", "a_lat_cmd", "v_lat_cmd"):
        collision_value = (1.0 - alpha) * ref[key][j] + alpha * ref[key][j + 1]
        ref[key] = np.concatenate([ref[key][:j + 1], np.asarray([collision_value])])

    return grid_t, rel_t, target_xy, real_xy, ref_xy, ref_heading, ref, True, float(collision_rel_t)


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
    target_color = "tab:blue"
    test_color = "tab:purple"
    ax.plot(xy_target[:, 0], xy_target[:, 1], label="target real XY", linewidth=2, color=target_color)
    ax.plot(xy_real[:, 0], xy_real[:, 1], label="test real work-segment XY", linewidth=2, color=test_color)
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], label="PNG reference XY", linewidth=2, linestyle="--", color="tab:orange")
    ax.scatter([xy_target[0, 0]], [xy_target[0, 1]], marker="o", s=40, color=target_color, label="target start")
    ax.scatter([xy_real[0, 0]], [xy_real[0, 1]], marker="s", s=40, color=test_color, label="test start")
    i_real_min = int(np.nanargmin(xy_real_dist))
    i_ref_min = int(np.nanargmin(ref_dist))
    ax.scatter([xy_real[i_real_min, 0]], [xy_real[i_real_min, 1]], marker="x", s=70,
               color=test_color, label="real closest (test)")
    ax.scatter([xy_target[i_real_min, 0]], [xy_target[i_real_min, 1]], marker="x", s=70,
               color=target_color, label="real closest (target)")
    ax.scatter([ref_xy[i_ref_min, 0]], [ref_xy[i_ref_min, 1]], marker="^", s=70,
               color="tab:orange", label="ref closest (ref)")
    ax.scatter([target_xy[i_ref_min, 0]], [target_xy[i_ref_min, 1]], marker="^", s=70,
               facecolors="none", edgecolors=target_color, linewidths=1.4, label="ref closest (target)")
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


def build_arg_parser(defaults):
    parser = argparse.ArgumentParser(
        description="Generate a horizontal PNG reference trajectory and compare it with the real work-segment trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bag", nargs="?", help="ROS1 bag path")
    parser.add_argument("--config", default=None, help="optional .ini config file for one flight/test run")
    parser.add_argument(
        "--write-config-template",
        nargs="?",
        const="png_reference_flight_config.ini",
        default=None,
        help="write a tunable .ini template and exit. Optional path can be supplied.",
    )
    parser.add_argument("--ego-pose-topic", default=defaults["ego_pose_topic"], help="default: auto-detect /mavros/local_position/pose")
    parser.add_argument("--target-pose-topic", default=defaults["target_pose_topic"], help="default: auto-detect /target/mavros/local_position/pose")
    parser.add_argument("--state-topic", default=defaults["state_topic"], help="default: auto-detect /mavros/state")
    parser.add_argument("--ego-velocity-topic", default=defaults["ego_velocity_topic"], help="optional, used for initial heading if available")
    parser.add_argument(
        "--work-mode",
        default=defaults["work_mode"],
        help="work mode selector. default 'auto' means prefer GUIDED, then AUTO. "
             "Can also be a substring or comma list, e.g. GUIDED or GUIDED,AUTO.",
    )
    parser.add_argument("--work-exact", action="store_true", default=defaults["work_exact"], help="require exact work-mode match instead of substring match")
    parser.add_argument("--work-index", type=int, default=defaults["work_index"], help="which matched work interval to analyze, default 0")
    parser.add_argument("--auto-mode", default=defaults["auto_mode"], help="deprecated alias of --work-mode")
    parser.add_argument("--auto-exact", action="store_true", default=defaults["auto_exact"], help="deprecated alias of --work-exact")
    parser.add_argument("--auto-index", type=int, default=defaults["auto_index"], help="deprecated alias of --work-index")
    parser.add_argument("--dt", type=float, default=defaults["dt"], help="reference time step")
    parser.add_argument("--speed", default=defaults["speed"], help="V_forward in m/s, or 'auto' to use median real work-segment speed")
    parser.add_argument("--N", type=float, default=defaults["N"], help="N_nav PNG navigation constant")
    parser.add_argument("--v-lat-max", type=float, default=defaults["v_lat_max"], help="v_lat_max lateral speed limit in m/s")
    parser.add_argument("--tau-lat", type=float, default=defaults["tau_lat"], help="tau_lat lateral-speed first-order time constant in seconds")
    parser.add_argument("--lambda-alpha", type=float, default=defaults["lambda_alpha"], help="legacy lambda_filt_alpha used only to derive Kalman noise defaults")
    parser.add_argument("--lambda-window-size", type=int, default=defaults["lambda_window_size"], help="legacy lambda_window_size used only to derive Kalman noise defaults")
    parser.add_argument("--lambda-kalman-q-position", type=float, default=defaults["lambda_kalman_q_position"], help="override lambda_kalman_q_position; default is mapped from alpha/window like the project")
    parser.add_argument("--lambda-kalman-q-rate", type=float, default=defaults["lambda_kalman_q_rate"], help="override lambda_kalman_q_rate; default is mapped from alpha/window like the project")
    parser.add_argument("--lambda-kalman-r", type=float, default=defaults["lambda_kalman_r"], help="override lambda_kalman_r; default is mapped from alpha/window like the project")
    parser.add_argument("--heading-source", choices=["auto", "bag-yaw", "course"], default=defaults["heading_source"],
                        help="body yaw used when rotating body velocity to world frame. auto uses bag pose yaw if available, else course")
    parser.add_argument("--k-yaw", type=float, default=defaults["k_yaw"], help="legacy option kept for CLI compatibility; not used by rpy_png_no_alt model")
    parser.add_argument("--k-yaw-d", type=float, default=defaults["k_yaw_d"], help="legacy option kept for CLI compatibility; not used by rpy_png_no_alt model")
    parser.add_argument("--max-yaw-rate", type=float, default=defaults["max_yaw_rate"], help="legacy option kept for CLI compatibility; not used by rpy_png_no_alt model")
    parser.add_argument("--max-side-step-deg", type=float, default=defaults["max_side_step_deg"], help="legacy option kept for CLI compatibility; not used by rpy_png_no_alt model")
    parser.add_argument("--post-ref-closest-window", type=float, default=defaults["post_ref_closest_window"], help="approach metrics end this many seconds after ref closest point")
    parser.add_argument("--ref-collision-threshold", type=float, default=defaults["ref_collision_threshold"], help="stop reference trajectory once ref-target distance is within this threshold in meters")
    parser.add_argument("--out-dir", default=defaults["out_dir"], help="output directory, default reference_out_<bagstem>")
    return parser


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_parser.add_argument("--write-config-template", nargs="?", const="png_reference_flight_config.ini", default=None)
    pre_args, _ = pre_parser.parse_known_args()

    defaults = dict(DEFAULT_ARGS)
    if pre_args.config:
        defaults.update(read_config_defaults(Path(pre_args.config)))

    parser = build_arg_parser(defaults)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.write_config_template:
        config_path = write_config_template(Path(args.write_config_template))
        print("[DONE] Config template:", config_path)
        return

    if not args.bag:
        fail("bag path is required. Use --write-config-template to create a config file without analyzing a bag.", 2)

    bag_path = Path(args.bag).expanduser().resolve()
    if not bag_path.exists():
        fail(f"bag not found: {bag_path}", 2)
    if args.dt <= 0:
        fail("--dt must be positive", 2)
    if args.v_lat_max <= 0:
        fail("--v-lat-max must be positive", 2)
    if args.tau_lat <= 0:
        fail("--tau-lat must be positive", 2)
    if args.lambda_window_size < 2:
        fail("--lambda-window-size must be at least 2", 2)
    if args.ref_collision_threshold <= 0:
        fail("--ref-collision-threshold must be positive", 2)

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
        ego_yaw = read_yaw_series(bag, ego_pose_topic, start_abs)
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

    heading_profile = None
    heading_source_used = "reference_course"
    if args.heading_source in ("auto", "bag-yaw"):
        if ego_yaw is not None:
            yaw_t, yaw_values = ego_yaw
            heading_profile = interp_angle_series(yaw_t, yaw_values, grid_t)
            if np.any(np.isfinite(heading_profile)):
                heading_source_used = "bag_pose_yaw"
                if np.isfinite(heading_profile[0]):
                    initial_heading = float(heading_profile[0])
            elif args.heading_source == "bag-yaw":
                fail("bag pose yaw could not be interpolated for the selected work interval", 6)
            else:
                heading_profile = None
        elif args.heading_source == "bag-yaw":
            fail("ego pose topic does not contain usable orientation yaw", 6)

    lambda_q_position_default, lambda_q_rate_default, lambda_r_default = map_legacy_los_chain_to_kalman(
        args.lambda_alpha,
        args.lambda_window_size,
    )
    lambda_q_position = (
        lambda_q_position_default
        if args.lambda_kalman_q_position is None
        else args.lambda_kalman_q_position
    )
    lambda_q_rate = (
        lambda_q_rate_default
        if args.lambda_kalman_q_rate is None
        else args.lambda_kalman_q_rate
    )
    lambda_r = lambda_r_default if args.lambda_kalman_r is None else args.lambda_kalman_r

    ref = simulate_png_reference(
        grid_t=grid_t,
        target_xy=target_xy,
        start_xy=real_xy[0],
        initial_heading=initial_heading,
        speed=spd,
        n_nav=args.N,
        v_lat_max=args.v_lat_max,
        tau_lat=args.tau_lat,
        lambda_kalman_q_position=lambda_q_position,
        lambda_kalman_q_rate=lambda_q_rate,
        lambda_kalman_r=lambda_r,
        heading_profile=heading_profile,
    )

    ref_xy = ref["xy"]
    ref_heading = ref["heading"]
    ref_dist = np.linalg.norm(target_xy - ref_xy, axis=1)
    xy_target_full = target_xy.copy()
    xy_real_full = real_xy.copy()
    xy_real_dist_full = np.linalg.norm(xy_target_full - xy_real_full, axis=1)
    ref_collision_threshold = float(args.ref_collision_threshold)
    grid_t, rel_t, target_xy, real_xy, ref_xy, ref_heading, ref, ref_collision_reached, ref_collision_rel_t = (
        trim_reference_at_collision(
            grid_t=grid_t,
            rel_t=rel_t,
            target_xy=target_xy,
            real_xy=real_xy,
            ref_xy=ref_xy,
            ref_heading=ref_heading,
            ref=ref,
            threshold_m=ref_collision_threshold,
        )
    )

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
        "ref_collision_threshold_m": ref_collision_threshold,
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
            f"{ref['a_lat_cmd'][i]:.6f}",
            f"{ref['v_lat_cmd'][i]:.6f}",
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
            "ref_lambda_dot_rad_s", "ref_a_lat_cmd_m_s2", "ref_v_lat_cmd_m_s",
            "ref_yaw_rate_cmd_rad_s",
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
        "reference_model": "rpy_png_no_alt.cpp lateral-velocity PNG",
        "config_file": "none" if not args.config else str(Path(args.config).expanduser().resolve()),
        "N_nav": args.N,
        "reference_speed_mps": f"{spd:.6f}",
        "dt_s": args.dt,
        "v_lat_max_mps": args.v_lat_max,
        "tau_lat_s": args.tau_lat,
        "lambda_alpha_legacy": args.lambda_alpha,
        "lambda_window_size_legacy": args.lambda_window_size,
        "lambda_kalman_q_position": f"{lambda_q_position:.9g}",
        "lambda_kalman_q_rate": f"{lambda_q_rate:.9g}",
        "lambda_kalman_r": f"{lambda_r:.9g}",
        "heading_source": heading_source_used,
        "initial_heading_deg": f"{math.degrees(initial_heading):.6f}",
        "legacy_k_yaw_not_used": args.k_yaw,
        "legacy_k_yaw_d_not_used": args.k_yaw_d,
        "legacy_max_yaw_rate_not_used": args.max_yaw_rate,
        "legacy_max_side_step_deg_not_used": args.max_side_step_deg,
        "work_mode_selector": mode_spec,
        "selected_mode_pattern": selected_pattern,
        "work_exact": mode_exact,
        "work_index": work_index,
        "ref_collision_threshold_m": ref_collision_threshold,
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
