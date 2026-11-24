# NVIDIA Single Aisle Checkpoint Runbook

Pipeline checkpoint: `962d105` (`feat: checkpoint NVIDIA single-aisle pipeline`)

This checkpoint contains the code and configuration used for the completed
single-aisle experiment. ROS bags and generated datasets are stored outside
Git and must remain at the paths shown below.

## Restore the checkpoint in a separate worktree

```bash
cd /home/orc/SAM3
git worktree add /home/orc/SAM3_checkpoint_962d105 962d105

cd /home/orc/SAM3_checkpoint_962d105/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select conceptgraphs_bringup
source install/setup.bash
```

When running from the normal `main` working tree, use `/home/orc/SAM3/ros2_ws`
instead and source its `install/setup.bash`.

## Visualize the completed single aisle

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch conceptgraphs_bringup visualization.launch.py \
  bag_path:=/home/orc/SAM3/ros2_ws/dataset/nvidia_data_20261029_144639_humble \
  lidar_map_topic:=/cloud_registered \
  start_sam3:=true \
  start_crate_markers:=false \
  sam3_device:=0 \
  sam3_confidence:=0.7 \
  crate_min_confidence:=0.6 \
  crate_min_observations:=2 \
  crate_cluster_radius:=1.5 \
  marker_scale:=1.2 \
  marker_z_offset:=0.5 \
  rate:=1.0 \
  z_min:=-1.3 \
  z_max:=1.5
```

Expected input bag:

```text
/home/orc/SAM3/ros2_ws/dataset/nvidia_data_20261029_144639_humble
```

## Query the generated HMSG dataset

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run conceptgraphs_bringup query_hmsg \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20261029_aisle01 \
  --query "Find the ladder" \
  --top-k 5
```

Expected generated dataset:

```text
/home/orc/SAM3/ros2_ws/datasets/greenhouse_20261029_aisle01
```

## Remove the temporary checkpoint worktree

```bash
cd /home/orc/SAM3
git worktree remove /home/orc/SAM3_checkpoint_962d105
```
