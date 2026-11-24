# SAM3 ROS 2 – Khôi phục baseline đang chạy ổn

Tài liệu này dùng để quay lại trạng thái đã kiểm tra hoạt động của pipeline SAM3,
FAST-LIO, point-cloud filtering và RViz vào ngày 2026-09-29.

## 1. Các checkpoint

Project chính:

```text
Repository: /home/orc/SAM3
Commit:     01b3aff
Tag:        sam3-semantic-mapping-baseline-v2
```

FAST-LIO là một repository Git riêng nằm bên trong `livox_ws`:

```text
Repository: /home/orc/SAM3/livox_ws/src/FAST_LIO_ROS2
Commit:     9687b4e
Tag:        sam3-rear-fov-baseline-v1
```

Tag FAST-LIO chứa bộ lọc FOV ngang dùng để loại vùng quét phía sau robot.

## 2. Kiểm tra checkpoint còn tồn tại

```bash
cd /home/orc/SAM3
git show --stat sam3-semantic-mapping-baseline-v2

cd /home/orc/SAM3/livox_ws/src/FAST_LIO_ROS2
git show --stat sam3-rear-fov-baseline-v1
```

## 3. Khôi phục an toàn bằng nhánh mới

Cách này không xóa nhánh đang phát triển và không cần dùng `git reset --hard`.

### 3.1. Project chính

```bash
cd /home/orc/SAM3
git switch -c restore-semantic-v2 sam3-semantic-mapping-baseline-v2
```

Nếu nhánh `restore-semantic-v2` đã tồn tại:

```bash
cd /home/orc/SAM3
git switch restore-semantic-v2
```

### 3.2. FAST-LIO

Mở terminal khác hoặc thực hiện tiếp:

```bash
cd /home/orc/SAM3/livox_ws/src/FAST_LIO_ROS2
git switch -c restore-fov-v1 sam3-rear-fov-baseline-v1
```

Nếu nhánh `restore-fov-v1` đã tồn tại:

```bash
cd /home/orc/SAM3/livox_ws/src/FAST_LIO_ROS2
git switch restore-fov-v1
```

## 4. Build lại FAST-LIO

```bash
cd /home/orc/SAM3/livox_ws
source /opt/ros/humble/setup.bash

colcon build \
  --symlink-install \
  --packages-select fast_lio
```

## 5. Build lại package SAM3 ROS 2

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source /home/orc/SAM3/livox_ws/install/setup.bash

colcon build \
  --merge-install \
  --packages-select conceptgraphs_bringup \
  --install-base install_ros
```

## 6. Chạy baseline visualization

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source /home/orc/SAM3/livox_ws/install/setup.bash
source install_ros/setup.bash

ros2 launch conceptgraphs_bringup visualization.launch.py \
  bag_path:=/home/orc/SAM3/ros2_ws/dataset/mid360_camera_20260910_150208_processed_fov \
  lidar_map_topic:=/cloud_registered \
  start_sam3:=true \
  start_crate_markers:=false \
  sam3_device:=0 \
  rate:=1.0 \
  sam3_confidence:=0.7 \
  crate_min_confidence:=0.6 \
  crate_min_observations:=2 \
  crate_cluster_radius:=1.5 \
  marker_scale:=1.2 \
  marker_z_offset:=0.5 \
  z_min:=-1.3 \
  z_max:=1.5
```

## 7. Dấu hiệu pipeline hoạt động đúng

Terminal phải cho thấy:

- Bag player không báo `Bag path ... does not exist`.
- `z_passthrough_filter` đọc `/cloud_registered` và xuất `/cloud_filtered`.
- Log `Z filter cloud=...` tiếp tục tăng.
- SAM3 chạy với `sam3_device:=0`.
- Log hiển thị các prompt semantic đã cấu hình.
- RViz dùng fixed frame `camera_init`.
- Display point cloud đọc topic `/cloud_filtered`.
- Marker semantic xuất hiện trên `/visualization_marker_array` khi object đủ số lần quan sát.

Các lệnh kiểm tra nhanh:

```bash
ros2 node list
ros2 topic list | grep -E 'cloud_filtered|cloud_registered|visualization_marker_array'
ros2 topic hz /cloud_filtered
ros2 topic echo /visualization_marker_array --once
```

## 8. Dữ liệu không nằm trong Git

Các checkpoint không chứa dữ liệu lớn hoặc file sinh ra trong lúc chạy:

```text
/home/orc/SAM3/ros2_ws/dataset/
/home/orc/SAM3/ros2_ws/build/
/home/orc/SAM3/ros2_ws/install/
/home/orc/SAM3/ros2_ws/log/
/home/orc/SAM3/livox_ws/build/
/home/orc/SAM3/livox_ws/install/
/home/orc/SAM3/livox_ws/log/
/home/orc/SAM3/sam3.pt
```

Do đó trước khi chạy lại cần bảo đảm bag và model vẫn tồn tại:

```bash
test -d /home/orc/SAM3/ros2_ws/dataset/mid360_camera_20260910_150208_processed_fov
test -f /home/orc/SAM3/sam3.pt
```

## 9. Lưu ý về GitHub

Checkpoint đã được commit và gắn tag trong Git cục bộ. Repository chính hiện chưa
cấu hình GitHub remote, vì vậy tag chưa được sao lưu lên GitHub. Không xóa thư mục
`.git` cho đến khi đã cấu hình remote và push cả commit lẫn tag.

