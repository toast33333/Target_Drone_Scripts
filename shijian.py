#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析 rosbag 中每个话题的“有消息的时间段”(intervals)。

输出：
- 每个 topic：从录制第几秒开始有消息、到第几秒结束
- 如果断断续续：每一段起止都列出来
- 只 print，不写文件

用法：
  python3 bag_topic_intervals.py your.bag

可选：
  python3 bag_topic_intervals.py your.bag --gap 1.0
    - 强制把“消息间隔 > 1.0 秒”视作断开，拆成多段（不指定则每个 topic 自动估计阈值）

说明：
- 使用 bag 中记录的时间戳 t（rosbag.read_messages 返回的 time），
  相对 bag start_time 转成“录制第几秒”。
- 需要在 ROS1 环境中运行（可 import rosbag）。
"""

import argparse
import math
import statistics
from typing import Dict, List, Tuple


# （可选）期望话题列表：来自你强制录制脚本 TOPICS
EXPECTED_TOPICS = [
    "/mavros/local_position/pose",
    "/mavros/local_position/velocity_local",
    "/mavros/global_position/global",
    "/target/mavros/local_position/pose",
    "/target/mavros/local_position/velocity_local",
    "/target/mavros/global_position/global",
    "/mavros/setpoint_position/local",
    "/object_detections",
    "/object_kcf",
    "/mavros/setpoint_velocity/cmd_vel",
]


def format_s(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.3f}s"


def build_intervals(times: List[float], gap_override: float = None) -> Tuple[List[Tuple[float, float]], float]:
    """
    给定某 topic 的消息时间点列表(相对录制起点的秒数，已排序)，切分成若干 intervals。

    interval 规则：
      - 若 gap_override 指定，则 gap > gap_override 断开
      - 否则根据该 topic 的消息间隔分布自适应估计：
          median_dt = median(diffs)
          gap_threshold = max(3*median_dt, 0.5)
        再用 gap > gap_threshold 断开

    返回：intervals, gap_threshold
    """
    if not times:
        return [], float("nan")

    if len(times) == 1:
        thr = gap_override if gap_override is not None else float("nan")
        return [(times[0], times[0])], thr

    diffs = []
    for t1, t2 in zip(times[:-1], times[1:]):
        if t2 >= t1:
            diffs.append(t2 - t1)

    if not diffs:
        thr = gap_override if gap_override is not None else float("nan")
        return [(times[0], times[-1])], thr

    if gap_override is not None:
        gap_thr = float(gap_override)
    else:
        median_dt = statistics.median(diffs)
        gap_thr = max(3.0 * median_dt, 0.5)

    intervals: List[Tuple[float, float]] = []
    start = times[0]
    prev = times[0]

    for t in times[1:]:
        if (t - prev) > gap_thr:
            intervals.append((start, prev))
            start = t
        prev = t

    intervals.append((start, prev))
    return intervals, gap_thr


def main():
    parser = argparse.ArgumentParser(description="Analyze rosbag topic activity intervals.")
    parser.add_argument("bag", help="Path to .bag file")
    parser.add_argument(
        "--gap",
        type=float,
        default=None,
        help="Gap threshold in seconds. If set, gap > threshold splits intervals. If omitted, auto-estimate per topic.",
    )
    parser.add_argument(
        "--no-expected",
        action="store_true",
        help="Do not print missing topics from EXPECTED_TOPICS list.",
    )
    args = parser.parse_args()

    if args.gap is not None and args.gap <= 0:
        print("ERROR: --gap 必须是正数（单位秒），例如 --gap 1.0")
        return

    try:
        import rosbag  # type: ignore
    except Exception as e:
        print("ERROR: 无法导入 rosbag。请确认在 ROS1 环境下运行（source 过 setup.bash），并且装了 rosbag Python 包。")
        print(f"Import error: {e}")
        return

    bag_path = args.bag
    times_by_topic: Dict[str, List[float]] = {}

    # 打开 bag + 读消息（这里最容易因路径/损坏报错）
    try:
        with rosbag.Bag(bag_path, "r") as bag:
            start_time = bag.get_start_time()
            end_time = bag.get_end_time()
            duration = end_time - start_time

            # 遍历所有消息，按 topic 收集相对时间
            for topic, msg, t in bag.read_messages():
                # t 一般是 rospy.Time/roslib.rostime.Time，有 to_sec()
                rel = float(t.to_sec() - start_time)
                times_by_topic.setdefault(topic, []).append(rel)

    except Exception as e:
        print(f"ERROR: 打开或读取 bag 失败：{bag_path}")
        print(f"Exception: {e}")
        return

    # 输出概览
    print("=== rosbag topic intervals analyzer ===")
    print(f"bag: {bag_path}")
    print(f"duration: {duration:.3f}s")
    print("")

    if not times_by_topic:
        print("bag 内没有任何消息（空 bag 或者没有录到数据）。")
        if not args.no_expected and EXPECTED_TOPICS:
            print("\n=== Expected topics missing in bag ===")
            for t in EXPECTED_TOPICS:
                print(f"  - {t}")
        return

    # 可选：输出期望话题缺失情况（强制订阅但没数据时，bag 里不会出现该 topic）
    if not args.no_expected:
        present = set(times_by_topic.keys())
        missing = [t for t in EXPECTED_TOPICS if t not in present]
        if missing:
            print("=== Expected topics missing in bag (subscribed but never produced messages) ===")
            for t in missing:
                print(f"  - {t}")
            print("")

    # 输出每个 topic 的 intervals
    print("=== Per-topic intervals ===")
    for topic in sorted(times_by_topic.keys()):
        times = sorted(times_by_topic[topic])
        intervals, gap_thr = build_intervals(times, gap_override=args.gap)

        count = len(times)
        first_t = times[0]
        last_t = times[-1]

        if args.gap is not None:
            thr_note = f"gap_threshold={format_s(args.gap)} (manual)"
        else:
            thr_note = f"gap_threshold={format_s(gap_thr)} (auto per-topic)"

        print(f"\n[{topic}]")
        print(f"  messages: {count}")
        print(f"  first_msg: {format_s(first_t)}   last_msg: {format_s(last_t)}   {thr_note}")

        print(f"  intervals: {len(intervals)} segment(s)")
        for i, (s, e) in enumerate(intervals, 1):
            dur = e - s
            print(f"    {i:02d}. {format_s(s)}  ->  {format_s(e)}   (len={format_s(dur)})")


if __name__ == "__main__":
    main()
