#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实时计算两机在 xy 平面内的相对距离，并在结束时输出：
1) 仅在测试机进入 AUTO/GUIDED 模式期间统计的最小距离与碰撞判定
2) 测试机在 GUIDED 模式下每次目标丢失时：
   - 靶机到测试机速度方向延长线的垂直距离
   - 此刻两飞机的平面相对距离

默认话题：
- 测试机位置: /mavros/local_position/pose              (PoseStamped)
- 靶机位置:   /target/mavros/local_position/pose       (PoseStamped)
- 测试机速度: /mavros/local_position/velocity_local    (TwistStamped)
- 测试机状态: /mavros/state                            (mavros_msgs/State)
- 检测话题:   /object_detections                       (任意消息类型，只要有消息就算“检测到目标”)

用法：
  python3 realtime_xy_distance.py

可选：
  python3 realtime_xy_distance.py \
      --self_topic /mavros/local_position/pose \
      --target_topic /target/mavros/local_position/pose \
      --self_vel_topic /mavros/local_position/velocity_local \
      --state_topic /mavros/state \
      --detect_topic /object_detections \
      --threshold 0.3 \
      --print_hz 10 \
      --lost_timeout 0.5 \
      --min_speed 0.05
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class XY:
    x: float
    y: float
    stamp: float  # seconds (ROS time)


@dataclass
class VXY:
    vx: float
    vy: float
    stamp: float  # seconds (ROS time)


@dataclass
class LostEvent:
    event_id: int
    t_wall: float
    t_ros: Optional[float]
    guided_mode: bool
    self_x: float
    self_y: float
    target_x: float
    target_y: float
    vx: float
    vy: float
    perp_dist: float
    rel_dist_xy: float


class DistanceMonitor:
    def __init__(
        self,
        self_topic: str,
        target_topic: str,
        self_vel_topic: str,
        state_topic: str,
        detect_topic: str,
        threshold: float,
        print_hz: float,
        lost_timeout: float,
        min_speed: float,
    ):
        self.self_topic = self_topic
        self.target_topic = target_topic
        self.self_vel_topic = self_vel_topic
        self.state_topic = state_topic
        self.detect_topic = detect_topic
        self.threshold = threshold
        self.print_hz = max(0.1, float(print_hz))
        self._print_period = 1.0 / self.print_hz
        self.lost_timeout = max(0.01, float(lost_timeout))
        self.min_speed = max(0.0, float(min_speed))

        self.self_xy: Optional[XY] = None
        self.target_xy: Optional[XY] = None
        self.self_vxy: Optional[VXY] = None

        self.start_wall = time.time()
        self.start_ros: Optional[float] = None

        # 仅在 AUTO/GUIDED 模式内统计的最小距离
        self.min_dist: float = float("inf")
        self.min_at: Optional[float] = None
        self.min_dx: Optional[float] = None
        self.min_dy: Optional[float] = None
        self.min_mode: Optional[str] = None
        self._computed_count: int = 0

        self._last_print_wall: float = 0.0

        # 飞行模式
        self.current_mode: Optional[str] = None
        self.is_guided: bool = False
        self.is_auto_or_guided: bool = False

        # 检测/丢失
        self.last_detect_wall: Optional[float] = None
        self.last_detect_ros: Optional[float] = None
        self._lost_active: bool = False
        self.lost_events: List[LostEvent] = []

    def _now_rel(self) -> float:
        return time.time() - self.start_wall

    def _valid_xy(self, x: float, y: float) -> bool:
        return (
            x is not None and y is not None
            and not math.isnan(x) and not math.isnan(y)
            and math.isfinite(x) and math.isfinite(y)
        )

    def _valid_vxy(self, vx: float, vy: float) -> bool:
        return (
            vx is not None and vy is not None
            and not math.isnan(vx) and not math.isnan(vy)
            and math.isfinite(vx) and math.isfinite(vy)
        )

    def _ros_rel_time(self, stamp_a: float, stamp_b: float) -> Optional[float]:
        newest_stamp = max(stamp_a, stamp_b)
        if not math.isfinite(newest_stamp):
            return None
        if self.start_ros is None:
            self.start_ros = newest_stamp
        return newest_stamp - self.start_ros

    def _compute_current_dx_dy_dist(self) -> Optional[tuple]:
        if self.self_xy is None or self.target_xy is None:
            return None

        dx = self.self_xy.x - self.target_xy.x
        dy = self.self_xy.y - self.target_xy.y
        dist = math.hypot(dx, dy)

        if not math.isfinite(dist):
            return None
        return dx, dy, dist

    def _maybe_compute_and_print(self):
        cur = self._compute_current_dx_dy_dist()
        if cur is None:
            return

        dx, dy, dist = cur

        # 仅在 AUTO/GUIDED 模式内更新最小距离
        if self.is_auto_or_guided:
            self._computed_count += 1
            if dist < self.min_dist:
                self.min_dist = dist
                self.min_at = self._now_rel()
                self.min_dx = dx
                self.min_dy = dy
                self.min_mode = self.current_mode

        noww = time.time()
        if (noww - self._last_print_wall) >= self._print_period:
            self._last_print_wall = noww
            t_rel = self._ros_rel_time(self.self_xy.stamp, self.target_xy.stamp)
            if t_rel is None:
                t_rel_str = "unknown"
            else:
                t_rel_str = f"{t_rel:8.3f}s"

            print(
                f"[t={t_rel_str}] "
                f"dx={dx:+.3f} dy={dy:+.3f} dist_xy={dist:.3f} m "
                f"mode={self.current_mode}",
                flush=True
            )

    def _compute_perpendicular_distance(self) -> Optional[float]:
        """
        靶机点到“测试机当前位置 + 测试机速度方向”的二维直线的垂直距离。
        | (P_target - P_self) x v | / |v|
        """
        if self.self_xy is None or self.target_xy is None or self.self_vxy is None:
            return None

        vx = self.self_vxy.vx
        vy = self.self_vxy.vy
        vnorm = math.hypot(vx, vy)
        if not math.isfinite(vnorm) or vnorm < self.min_speed:
            return None

        dx = self.target_xy.x - self.self_xy.x
        dy = self.target_xy.y - self.self_xy.y

        perp = abs(dx * vy - dy * vx) / vnorm
        if not math.isfinite(perp):
            return None
        return perp

    def _maybe_mark_lost(self):
        """
        只在 GUIDED 模式下检测丢失：
        - 已经有过检测
        - 距离上次检测超过 lost_timeout
        - 当前不是已丢失态
        """
        if not self.is_guided:
            return

        if self.last_detect_wall is None:
            return

        now_wall = time.time()
        if (now_wall - self.last_detect_wall) < self.lost_timeout:
            return

        if self._lost_active:
            return

        perp = self._compute_perpendicular_distance()
        cur = self._compute_current_dx_dy_dist()
        if perp is None or cur is None:
            return

        _, _, rel_dist_xy = cur

        t_ros = None
        if self.self_xy is not None and self.target_xy is not None:
            t_ros = self._ros_rel_time(self.self_xy.stamp, self.target_xy.stamp)

        event = LostEvent(
            event_id=len(self.lost_events) + 1,
            t_wall=self._now_rel(),
            t_ros=t_ros,
            guided_mode=self.is_guided,
            self_x=self.self_xy.x,
            self_y=self.self_xy.y,
            target_x=self.target_xy.x,
            target_y=self.target_xy.y,
            vx=self.self_vxy.vx,
            vy=self.self_vxy.vy,
            perp_dist=perp,
            rel_dist_xy=rel_dist_xy,
        )
        self.lost_events.append(event)
        self._lost_active = True

        t_show = f"{t_ros:.3f}s(ros)" if t_ros is not None else f"{event.t_wall:.3f}s(wall)"
        print(
            f"[LOST #{event.event_id} @ {t_show}] "
            f"guided=1 perp_dist={event.perp_dist:.3f} m "
            f"rel_dist_xy={event.rel_dist_xy:.3f} m "
            f"self=({event.self_x:.3f},{event.self_y:.3f}) "
            f"target=({event.target_x:.3f},{event.target_y:.3f}) "
            f"vel=({event.vx:+.3f},{event.vy:+.3f})",
            flush=True
        )

    def cb_self(self, msg):
        try:
            stamp = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else float("nan")
            x = float(msg.pose.position.x)
            y = float(msg.pose.position.y)
        except Exception:
            return

        if not self._valid_xy(x, y):
            return

        self.self_xy = XY(x=x, y=y, stamp=stamp)
        self._maybe_compute_and_print()
        self._maybe_mark_lost()

    def cb_target(self, msg):
        try:
            stamp = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else float("nan")
            x = float(msg.pose.position.x)
            y = float(msg.pose.position.y)
        except Exception:
            return

        if not self._valid_xy(x, y):
            return

        self.target_xy = XY(x=x, y=y, stamp=stamp)
        self._maybe_compute_and_print()
        self._maybe_mark_lost()

    def cb_self_vel(self, msg):
        try:
            stamp = msg.header.stamp.to_sec() if msg.header and msg.header.stamp else float("nan")
            vx = float(msg.twist.linear.x)
            vy = float(msg.twist.linear.y)
        except Exception:
            return

        if not self._valid_vxy(vx, vy):
            return

        self.self_vxy = VXY(vx=vx, vy=vy, stamp=stamp)
        self._maybe_mark_lost()

    def cb_state(self, msg):
        try:
            mode = str(msg.mode)
        except Exception:
            return

        self.current_mode = mode
        self.is_guided = (mode == "GUIDED")
        self.is_auto_or_guided = self.is_guided or ("AUTO" in mode if mode else False)

        # 退出 GUIDED 时清理连续丢失状态
        if not self.is_guided:
            self._lost_active = False

    def cb_detect(self, msg):
        self.last_detect_wall = time.time()

        try:
            if hasattr(msg, "header") and msg.header and msg.header.stamp:
                self.last_detect_ros = msg.header.stamp.to_sec()
            else:
                self.last_detect_ros = None
        except Exception:
            self.last_detect_ros = None

        # 检测恢复，退出丢失态
        self._lost_active = False

    def on_shutdown(self):
        print("\n=== Run Summary ===", flush=True)

        if self._computed_count == 0 or not math.isfinite(self.min_dist):
            print(
                "No valid min-distance computed in AUTO/GUIDED mode.\n"
                "Possible reasons:\n"
                "  - never entered AUTO or GUIDED mode\n"
                "  - missing pose data during AUTO/GUIDED",
                flush=True
            )
        else:
            when = f"{self.min_at:.3f}s (wall)" if self.min_at is not None else "unknown"
            print(
                f"Min XY distance (AUTO/GUIDED only): {self.min_dist:.3f} m  "
                f"@ {when}, mode={self.min_mode}",
                flush=True
            )

            if self.min_dx is not None and self.min_dy is not None:
                print(
                    f"At min-distance moment: |dx|={abs(self.min_dx):.3f} m, |dy|={abs(self.min_dy):.3f} m "
                    f"(dx={self.min_dx:+.3f}, dy={self.min_dy:+.3f})",
                    flush=True
                )

            print(f"Threshold: {self.threshold:.3f} m", flush=True)
            if self.min_dist <= self.threshold:
                print("碰撞成功", flush=True)

        print("\n=== Guided-mode Target Lost Events ===", flush=True)
        if not self.lost_events:
            print(
                "No valid lost events recorded.\n"
                "Possible reasons:\n"
                "  - never entered GUIDED mode\n"
                "  - detect topic kept updating continuously\n"
                "  - missing pose/velocity data at loss moment\n"
                "  - self speed too small to define direction line",
                flush=True
            )
            return

        for ev in self.lost_events:
            t_str = f"{ev.t_ros:.3f}s(ros)" if ev.t_ros is not None else f"{ev.t_wall:.3f}s(wall)"
            print(
                f"[{ev.event_id:02d}] lost_at={t_str}, "
                f"perp_dist={ev.perp_dist:.3f} m, "
                f"rel_dist_xy={ev.rel_dist_xy:.3f} m, "
                f"self=({ev.self_x:.3f},{ev.self_y:.3f}), "
                f"target=({ev.target_x:.3f},{ev.target_y:.3f}), "
                f"vel=({ev.vx:+.3f},{ev.vy:+.3f})",
                flush=True
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self_topic", default="/mavros/local_position/pose", help="测试机 PoseStamped 话题")
    parser.add_argument("--target_topic", default="/target/mavros/local_position/pose", help="靶机 PoseStamped 话题")
    parser.add_argument("--self_vel_topic", default="/mavros/local_position/velocity_local", help="测试机速度 TwistStamped 话题")
    parser.add_argument("--state_topic", default="/mavros/state", help="测试机状态话题（需含 mode 字段）")
    parser.add_argument("--detect_topic", default="/object_detections", help="目标检测话题；超过 lost_timeout 秒无新消息视为丢失")
    parser.add_argument("--threshold", type=float, default=0.3, help="碰撞阈值(米)，默认 0.3")
    parser.add_argument("--print_hz", type=float, default=10.0, help="实时输出频率(Hz)，默认 10")
    parser.add_argument("--lost_timeout", type=float, default=0.5, help="目标丢失判定超时(秒)，默认 0.5")
    parser.add_argument("--min_speed", type=float, default=0.05, help="定义速度方向线的最小速度阈值(m/s)，默认 0.05")
    args = parser.parse_args()

    if args.threshold <= 0:
        print("ERROR: --threshold 必须为正数", file=sys.stderr)
        sys.exit(2)
    if args.lost_timeout <= 0:
        print("ERROR: --lost_timeout 必须为正数", file=sys.stderr)
        sys.exit(2)
    if args.min_speed < 0:
        print("ERROR: --min_speed 不能为负数", file=sys.stderr)
        sys.exit(2)

    try:
        import rospy
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from mavros_msgs.msg import State
        from rospy.msg import AnyMsg
    except Exception as e:
        print("ERROR: 无法导入 rospy / 消息类型。请在 ROS1 环境中运行并 source setup.bash。", file=sys.stderr)
        print(f"Import error: {e}", file=sys.stderr)
        sys.exit(1)

    rospy.init_node("realtime_xy_distance_monitor", anonymous=True, disable_signals=True)

    mon = DistanceMonitor(
        self_topic=args.self_topic,
        target_topic=args.target_topic,
        self_vel_topic=args.self_vel_topic,
        state_topic=args.state_topic,
        detect_topic=args.detect_topic,
        threshold=args.threshold,
        print_hz=args.print_hz,
        lost_timeout=args.lost_timeout,
        min_speed=args.min_speed,
    )

    rospy.Subscriber(args.self_topic, PoseStamped, mon.cb_self, queue_size=50)
    rospy.Subscriber(args.target_topic, PoseStamped, mon.cb_target, queue_size=50)
    rospy.Subscriber(args.self_vel_topic, TwistStamped, mon.cb_self_vel, queue_size=50)
    rospy.Subscriber(args.state_topic, State, mon.cb_state, queue_size=20)
    rospy.Subscriber(args.detect_topic, AnyMsg, mon.cb_detect, queue_size=100)

    rospy.on_shutdown(mon.on_shutdown)

    print("=== Realtime XY Distance Monitor ===", flush=True)
    print(f"self_topic:      {args.self_topic}", flush=True)
    print(f"target_topic:    {args.target_topic}", flush=True)
    print(f"self_vel_topic:  {args.self_vel_topic}", flush=True)
    print(f"state_topic:     {args.state_topic}", flush=True)
    print(f"detect_topic:    {args.detect_topic}", flush=True)
    print("min-distance mode filter: AUTO or GUIDED", flush=True)
    print("lost-event mode filter:   GUIDED only", flush=True)
    print(f"threshold:       {args.threshold:.3f} m", flush=True)
    print(f"print_hz:        {args.print_hz:.1f} Hz", flush=True)
    print(f"lost_timeout:    {args.lost_timeout:.3f} s", flush=True)
    print(f"min_speed:       {args.min_speed:.3f} m/s", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    try:
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            mon._maybe_mark_lost()
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            rospy.signal_shutdown("user interrupt")
        except Exception:
            pass


if __name__ == "__main__":
    main()