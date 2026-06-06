# Target Drone Scripts

本仓库用于存放靶机/测试机实验中常用的辅助脚本，主要覆盖四类工作：

- ROS bag 录制；
- rosbag 离线分析；
- 实验过程中的实时距离和检测丢失监控；
- PNG 制导参考轨迹生成与对比。

这些脚本本身不是 YOLO 或 KCF 的实时检测主程序。实时视觉检测/跟踪代码仍在原项目工作空间中，例如：

- YOLO 检测节点：`png_planner_ws/src/detection_msgs/scripts/ros1Main.py`、`png_planner_ws/src/detection_msgs/scripts/ros1Func.py`
- KCF 跟踪节点：`png_planner_ws/src/kcf_msgs/scripts/KCF_limited*.py`
- 控制端视觉订阅：`png_planner_ws/src/png_planner/src/rp_png.cpp`、`rp_png_bbox_vz.cpp`、`rp_los_bbox_vz.cpp`、`rp_z_only_bbox_yaw_test.cpp`、`FSM.cpp`
- 控制端视觉配置：`png_planner_ws/src/png_planner/config/rp_common.yaml`

其中 `/object_detections` 通常对应 YOLO 输出，`/object_kcf` 通常对应 KCF 输出。

## 文件说明

| 文件 | 作用 | 对应项目部分 | 对应代码/话题 |
| --- | --- | --- | --- |
| `duojirosbag.sh` | 一键录制实验所需 ROS 话题，生成 `recording_时间戳.bag` | 实验数据采集 | 录制 MAVROS 位置/速度/状态、目标机位置/速度/状态、`/object_detections`、`/object_kcf` 等 |
| `analyze_oneclick.py` | 离线分析 rosbag，输出轨迹、速度、高度、相对距离、检测存在情况和检测事件 CSV | 实验结果分析，偏 YOLO 检测结果分析 | 默认查找并分析 `/object_detections`；使用 `Detection.msg` 动态解析检测消息 |
| `realtime_xy_distance.py` | 实验过程中实时计算测试机与靶机 XY 平面距离，并记录检测丢失事件 | 在线实验监控 | 默认监听 `/object_detections`，也可以通过 `--detect_topic /object_kcf` 改为监听 KCF |
| `shijian.py` | 分析 rosbag 中每个 topic 有消息的时间段 | bag 完整性检查 | 可检查 `/object_detections`、`/object_kcf` 是否在实验过程中实际产生消息 |
| `generate_png_reference_trajectory.py` | 根据 rosbag 中目标机轨迹生成 PNG 水平参考轨迹，并和真实测试机轨迹对比 | PNG 制导/控制算法离线分析 | 对应 `rpy_png_no_alt.cpp` 的水平制导逻辑，不直接分析视觉检测框 |
| `png_reference_flight_config_template.ini` | `generate_png_reference_trajectory.py` 的配置模板 | PNG 参考轨迹分析参数配置 | 配置 pose/state topic、工作模式段、`N`、速度、横向速度限制、滤波参数等 |
| `Detection.msg` | 检测消息格式定义 | YOLO 检测消息解析 | 对应原项目 `detection_msgs/msg/Detection.msg`，供 `analyze_oneclick.py` 动态解析 rosbag 使用 |
| `for_test.bag` | 测试用 rosbag 文件 | 脚本测试数据 | 可用于验证分析脚本是否能正常读取 bag |

## 视觉相关链路

项目里的视觉输出主要有两个 topic：

1. `/object_detections`

   该话题通常来自 YOLO 检测节点。对应原项目代码为：

   - `png_planner_ws/src/detection_msgs/scripts/ros1Main.py`
   - `png_planner_ws/src/detection_msgs/scripts/ros1Func.py`

   `ros1Main.py` 负责打开相机、设置采集分辨率、加载 RKNN 模型并创建 ROS 发布器；`ros1Func.py` 负责 YOLO 前处理、模型推理后的后处理、检测框坐标还原，以及向 `/object_detections` 发布 `Detection` 消息。

2. `/object_kcf`

   该话题通常来自 KCF 跟踪节点。对应原项目代码为：

   - `png_planner_ws/src/kcf_msgs/scripts/KCF_limited.py`
   - `png_planner_ws/src/kcf_msgs/scripts/KCF_limited_First.py`
   - `png_planner_ws/src/kcf_msgs/scripts/KCF_limited_First1.py`
   - `png_planner_ws/src/kcf_msgs/scripts/KCF_limited_First2.py`

   这些脚本负责打开相机、初始化或更新 KCF 跟踪框，并将跟踪框发布到 `/object_kcf`。

控制端会根据 `png_planner_ws/src/png_planner/config/rp_common.yaml` 中的 `detection_mode` 决定使用哪个视觉 topic：

- `detection_mode: 0`：使用 KCF，订阅 `/object_kcf`
- `detection_mode: 1`：使用 YOLO，订阅 `/object_detections`

因此，做视觉距离实验时需要区分“YOLO 能否检测到”和“KCF 能否持续框住”。`analyze_oneclick.py` 默认主要分析 `/object_detections`，如果实验最终依赖 KCF，则还需要结合 `/object_kcf` 的消息情况一起判断。

## 脚本详细说明

### `duojirosbag.sh`

该脚本用于录制实验数据，默认强制录制一组固定 topic。即使某些 topic 在启动录制时暂时还没有发布，脚本也会把 topic 交给 `rosbag record` 等待数据出现。

默认录制的关键 topic 包括：

- 测试机位置/速度/状态：`/mavros/local_position/pose`、`/mavros/local_position/velocity_local`、`/mavros/state`
- 靶机位置/速度/状态：`/target/mavros/local_position/pose`、`/target/mavros/local_position/velocity_local`、`/target/mavros/state`
- YOLO 检测输出：`/object_detections`
- KCF 跟踪输出：`/object_kcf`
- 控制指令与高度相关话题：`/mavros/setpoint_velocity/cmd_vel`、`/mavros/altitude`、`/target/mavros/altitude` 等

常用命令：

```bash
./duojirosbag.sh
```

如需录制所有 ROS topic：

```bash
./duojirosbag.sh --all
```

该脚本对应整个项目中的“实验数据采集”部分。后续 `analyze_oneclick.py`、`shijian.py`、`generate_png_reference_trajectory.py` 都可以使用它录制出来的 bag 文件。

### `analyze_oneclick.py`

该脚本用于对 rosbag 做离线一键分析。它会自动读取测试机/靶机轨迹、速度、高度、相对距离和 YOLO 检测消息，并输出图片、CSV 和文字报告。

主要输出目录格式为：

```text
analysis_out_<bagname>/
```

主要输出内容包括：

- `traj_xy_ego_target.png`：测试机和靶机 XY 轨迹；
- `xy_speed_vs_time.png`：XY 平面速度；
- `altitude_vs_time.png`：高度变化；
- `relative_xy_distance.png`：XY 相对距离；
- `detections_presence_fulltime.png`：检测消息是否存在；
- `object_detections_score_fulltime.png`：检测分数随时间变化；
- `detection_events_metrics.csv`：检测开始/丢失事件统计；
- `object_detections_series.csv`：检测消息时间序列；
- `topic_summary.csv`、`analysis_report.txt`：topic 和分析摘要。

常用命令：

```bash
python3 analyze_oneclick.py recording_xxx.bag
```

如果当前目录只有一个或最新的 bag，也可以直接运行：

```bash
python3 analyze_oneclick.py
```

该脚本对应项目中的“YOLO 检测结果离线分析”。它默认寻找 `/object_detections`，并通过 `Detection.msg` 动态解析消息。

注意：该脚本目前的检测判定逻辑偏宽松，基本是“只要 `/object_detections` 有消息，就认为有检测”。它不等同于严格判断检测框是否稳定、是否框准目标、是否满足远距离可靠识别。如果要评估 KCF，需要额外分析 `/object_kcf`。

### `realtime_xy_distance.py`

该脚本用于实验过程中实时显示测试机和靶机在 XY 平面内的相对距离，同时在检测消息长时间没有更新时记录“检测丢失”事件。

默认监听：

- `/mavros/local_position/pose`
- `/target/mavros/local_position/pose`
- `/mavros/local_position/velocity_local`
- `/mavros/state`
- `/object_detections`

常用命令：

```bash
python3 realtime_xy_distance.py
```

如果要监控 KCF 输出而不是 YOLO 输出，可以改检测 topic：

```bash
python3 realtime_xy_distance.py --detect_topic /object_kcf
```

常用参数：

- `--threshold`：碰撞/接近距离阈值；
- `--print_hz`：打印频率；
- `--lost_timeout`：超过多少秒没有检测消息就认为目标丢失；
- `--min_speed`：计算速度方向相关距离时使用的最小速度阈值。

该脚本对应项目中的“在线实验监控”部分。它不参与控制，也不做图像识别，只根据 ROS topic 判断距离和检测消息是否持续存在。

### `shijian.py`

该脚本用于检查 rosbag 中各 topic 的消息活跃时间段。它会输出每个 topic 第一次出现、最后一次出现，以及中间是否存在较长断档。

常用命令：

```bash
python3 shijian.py recording_xxx.bag
```

指定断档阈值：

```bash
python3 shijian.py recording_xxx.bag --gap 1.0
```

该脚本对应项目中的“bag 完整性检查”。对于视觉实验，它可以回答：

- `/object_detections` 有没有被录进去；
- `/object_kcf` 有没有被录进去；
- 它们分别在哪些时间段有消息；
- 实验中是否存在长时间无检测/无跟踪消息的情况。

但它只检查消息时间，不判断检测框位置是否正确，也不判断目标是否真的被准确框住。

### `generate_png_reference_trajectory.py`

该脚本用于从 rosbag 中读取靶机轨迹，并按项目中的 PNG 水平制导逻辑生成一条参考轨迹，再与真实测试机轨迹进行对比。

它主要对应原项目中的：

```text
png_planner_ws/src/png_planner/src/rpy_png_no_alt.cpp
```

脚本中采用的参考模型为：

```text
LOS 角速度估计/滤波 -> PNG 横向加速度 -> 横向速度滤波 -> 世界坐标系速度指令
```

常用命令：

```bash
python3 generate_png_reference_trajectory.py recording_xxx.bag
```

使用配置文件：

```bash
python3 generate_png_reference_trajectory.py recording_xxx.bag --config png_reference_flight_config_template.ini
```

主要输出目录格式为：

```text
reference_out_<bagname>/
```

主要输出内容包括：

- `reference_trajectory.csv`：生成的参考轨迹；
- `trajectory_xy_compare.png`：真实轨迹与参考轨迹 XY 对比；
- `reference_tracking_error.png`：参考轨迹跟踪误差；
- `target_convergence.png`：相对目标的收敛情况；
- `work_intervals.csv`：工作模式时间段；
- `state_mode_changes.csv`：MAVROS 模式切换；
- `reference_summary.txt`：分析摘要。

该脚本不直接读取 YOLO/KCF 的检测框，也不判断目标是否被视觉框住。它主要用于验证飞行轨迹与 PNG 制导逻辑是否一致。

### `png_reference_flight_config_template.ini`

这是 `generate_png_reference_trajectory.py` 的配置模板。建议每次实飞分析时复制一份，再根据该次实验实际参数修改。

主要配置部分：

- `[topics]`：测试机/靶机 pose、状态、速度 topic；
- `[work_segment]`：选择分析 AUTO 或 GUIDED 等工作模式段；
- `[reference_model]`：PNG 参考模型参数，例如 `dt`、`speed`、`N`、`v_lat_max`、`tau_lat`、LOS 滤波参数；
- `[output]`：输出目录和接近距离判定参数。

这些参数应尽量和实际运行时 launch 文件或参数服务器中的配置保持一致。

### `Detection.msg`

该文件定义检测消息格式：

```text
float32 x1
float32 y1
float32 x2
float32 y2
float32 score
string class_name
int32 track_id
bool is_tracking
```

`analyze_oneclick.py` 会使用它动态解析 rosbag 中的 `/object_detections` 消息。它应与原项目中的 `detection_msgs/msg/Detection.msg` 保持一致。

字段含义：

- `x1, y1, x2, y2`：检测框坐标；
- `score`：检测置信度；
- `class_name`：检测类别；
- `track_id`：跟踪编号；
- `is_tracking`：是否处于跟踪状态。

### `for_test.bag`

这是一个测试用 rosbag 文件，可用于验证脚本能否正常读取 ROS bag、解析 topic，并生成分析结果。

示例：

```bash
python3 shijian.py for_test.bag
python3 analyze_oneclick.py for_test.bag
python3 generate_png_reference_trajectory.py for_test.bag
```

## 推荐实验流程

1. 启动测试机、靶机、视觉节点和控制节点。
2. 使用 `duojirosbag.sh` 录制实验数据。
3. 实验过程中可运行 `realtime_xy_distance.py` 观察距离和检测丢失。
4. 实验后先用 `shijian.py` 检查 bag 中关键 topic 是否完整。
5. 使用 `analyze_oneclick.py` 分析 YOLO 检测输出和轨迹数据。
6. 如需分析制导轨迹，再使用 `generate_png_reference_trajectory.py`。

如果实验目标是验证远距离视觉能力，需要同时关注：

- `/object_detections`：YOLO 是否持续输出；
- `/object_kcf`：KCF 是否持续输出；
- 检测框是否稳定落在靶机上；
- 对应时刻的测试机与靶机实际距离。

仅有检测 topic 的消息存在，并不能单独证明目标已经被稳定、准确地框住。
