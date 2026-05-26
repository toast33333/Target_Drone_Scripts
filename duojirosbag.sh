#!/bin/bash

# ROS话题录制脚本（强制录制版）
# - 不再因为“启动时没检测到话题”就跳过。
# - 即使话题暂时不存在/未发布，也会强制订阅并等待数据到来。
#
# 用法：
#   ./duojirosbag.sh          # 强制录制下方 TOPICS 列表（推荐）
#   ./duojirosbag.sh --all    # 录制所有话题（等价 rosbag record -a）

set -e

echo "=== ROS话题录制脚本（强制录制版）==="

# 获取当前时间作为文件名时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BAG_NAME="recording_$TIMESTAMP"

echo "创建录制文件: $BAG_NAME.bag"

# 定义要录制的话题列表
TOPICS=(
    "/mavros/local_position/pose"
    "/mavros/local_position/velocity_local" 
    "/mavros/global_position/global"
    "/target/mavros/local_position/pose"
    "/target/mavros/local_position/velocity_local" 
    "/target/mavros/global_position/global"
    "/mavros/setpoint_position/local"
    "/object_detections"
    "/object_kcf"
    "/mavros/setpoint_velocity/cmd_vel"
    "/mavros/state"
    "/target/mavros/state"
    "/mavros/global_position/rel_alt"
    "/target/mavros/global_position/rel_alt"
    "/mavros/altitude"
    "/target/mavros/altitude"
    "/mavros/vfr_hud"
    "/target/mavros/vfr_hud"
)

echo ""

# 检查ROS环境
if ! rostopic list &>/dev/null; then
    echo "错误：无法连接到ROS master！"
    echo "请先启动ROS: source devel/setup.bash && roscore"
    exit 1
fi

# 参数：--all 则录制所有话题
if [[ "${1:-}" == "--all" ]]; then
    echo "模式：录制所有话题 (-a)。"
    echo "开始录制...（Ctrl+C 停止）"
    exec rosbag record -O "$BAG_NAME" -a
fi

# 仅做提示：显示当前时刻是否在 rostopic list 里（不影响录制逻辑）
echo "提示：下面仅做'当前时刻'话题存在性提示（不会跳过任何话题）。"
echo "================================"
CURRENT_LIST=$(rostopic list 2>/dev/null || true)
for topic in "${TOPICS[@]}"; do
    if echo "$CURRENT_LIST" | grep -q "^${topic}$"; then
        echo "✓ 话题当前可见: $topic"
    else
        echo "… 话题当前不可见/未发布: $topic（仍将强制录制，等它出现）"
    fi
done
echo "================================"

echo "开始强制录制指定话题...（如果一开始没数据，bag 会先空着，直到话题开始发布）"
echo "停止录制请按 Ctrl+C"

# 关键变化：直接把 TOPICS 全部交给 rosbag record（不做过滤/不做兜底切 -a）
exec rosbag record -O "$BAG_NAME" "${TOPICS[@]}"
