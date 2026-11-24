# Project Context — SAM3 Greenhouse Semantic Mapping

**Cập nhật:** 2026-10-30
**ROS:** ROS 2 Humble
**Baseline project:** `sam3-semantic-mapping-baseline-v2`
**Baseline FAST-LIO:** `sam3-rear-fov-baseline-v1`

## 1. Mục tiêu hiện tại

Phát lại dữ liệu greenhouse đã mapping, hiển thị point cloud trong RViz và dùng
SAM3 để phát hiện các loại plastic container. Mask 2D được kết hợp với aligned
depth, camera calibration và odometry để suy ra vị trí object trong bản đồ 3D.

## 2. Sensor và dữ liệu

- LiDAR 3D: Livox Mid360.
- RGB-D: Intel RealSense D455.
- Mapping/odometry: FAST-LIO.
- Bag đang dùng:
  `ros2_ws/dataset/mid360_camera_20260910_150208_processed_fov`.
- Frame bản đồ: `camera_init`.

Các topic chính trong bag:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/cloud_registered
/Laser_map
/Odometry
/tf
/tf_static
```

## 3. Lệnh baseline hiện tại

```bash
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

## 4. Khi chạy lệnh trên

### Bag playback

- Phát bag MCAP ở tốc độ thời gian thực `rate=1.0`.
- Phát `/clock`; các node sử dụng simulation time.
- Bag đã chứa kết quả FAST-LIO, vì vậy lệnh này **không chạy FAST-LIO lại**.

### Point cloud

```text
/cloud_registered
    → z_passthrough_filter
    → giữ -1.3 m <= Z <= 1.5 m
    → /cloud_filtered
    → RViz tích lũy và hiển thị
```

`/cloud_registered` là các scan đã nằm trong world frame. Dùng topic này giúp
bộ lọc Z chạy liên tục; không lọc trực tiếp `/Laser_map` vì cloud tích lũy đó rất
lớn và từng làm node Python không theo kịp.

### SAM3 semantic mapper

`start_sam3:=true` khởi động `sam3_crate_mapper` và một SAM3 worker dùng GPU 0.
Prompt hiện được khai báo trong source:

```text
yellow plastic container on the ground
red plastic container on the ground
```

Input của mapper:

```text
RGB + aligned depth + CameraInfo + Odometry + tf_static
```

Luồng xử lý:

```text
RGB → SAM3 mask
mask + aligned depth → robust 3D center trong camera frame
camera/odometry transforms → tọa độ trong camera_init
same label + khoảng cách XY → object cluster
confirmed cluster → RViz sphere marker
```

Chỉ detection cách camera tối đa khoảng `3 m` được dùng. SAM3 confidence tối
thiểu là `0.7`. Worker chỉ giữ một job chờ, nên có thể bỏ qua frame nếu GPU chưa
xử lý xong frame trước; bag player không chờ SAM3.

### Object confirmation và marker

- Observation cùng label trong bán kính XY `1.5 m` được gộp chung.
- Cluster cần ít nhất `2` observations.
- Cluster phải từng có confidence tối thiểu `0.6`.
- Vị trí cluster là median của các observation 3D.
- Sphere có đường kính `1.2 m`.
- Sphere được nâng `0.5 m` theo trục Z.
- Marker được publish trên `/yellow_crate_markers` trong frame `camera_init`.

`start_crate_markers:=false` chỉ tắt node đọc marker cũ từ CSV. Nó **không tắt**
marker real-time do SAM3 mapper tạo.

Hiện logic màu xanh dương chỉ dành cho label chính xác
`yellow leaves on the ground`. Hai prompt plastic container hiện tại đều tạo
sphere màu xanh lá.

### Các node khác

Launch còn tự động chạy:

- `map_server`: publish occupancy grid `/map`.
- `costmap_server`: publish `/costmap`.
- Nav2 lifecycle manager cho hai map server.
- RViz với config `conceptgraphs_map.rviz`.

Khi đóng RViz, toàn bộ launch được shutdown.

## 5. Output quan trọng

```text
/cloud_filtered          Point cloud đã cắt theo Z
/sam3/debug_image        Ảnh RGB có kết quả SAM3 overlay
/yellow_crate_markers    MarkerArray object 3D đã xác nhận
/map                     Occupancy grid
/costmap                 Costmap 2D
```

## 6. Trạng thái đã hoàn thành

- Chuyển bag sang định dạng dùng được với ROS 2 Humble.
- Mapping bằng FAST-LIO và ghi `/cloud_registered`, `/Laser_map`, odometry/TF.
- Lọc vùng LiDAR phía sau trước mapping bằng FOV filter trong FAST-LIO.
- Lọc point cloud theo world Z và tích lũy trong RViz.
- Chạy SAM3 bằng GPU với nhiều text prompt.
- Chiếu mask qua aligned depth sang tọa độ 3D.
- Gộp observation và hiển thị sphere marker trong map.
- Có launch riêng trích mỗi N frame RGB thành JPEG.
- Có Git tag và hướng dẫn phục hồi trong `RECOVERY_README.md`.

## 7. Hạn chế hiện tại

- Prompt SAM3 vẫn khai báo cứng trong `sam3_crate_mapper.py`.
- Ghép object chỉ dựa trên label và khoảng cách XY; chưa có visual embedding.
- Object cluster chỉ tồn tại trong RAM, chưa lưu thành semantic database.
- Chưa có View node, CLIP embedding, caption, object-view graph hoặc truy vấn
  ngôn ngữ tự nhiên.
- Bag playback không đồng bộ theo tốc độ inference SAM3, nên không bảo đảm xử lý
  mọi RGB frame.
- Map hiện giả định môi trường tĩnh.

## 8. Hướng đang thảo luận, chưa triển khai

Xây semantic database offline theo ba bước:

1. Chọn keyframe khi robot đi khoảng `1 m` hoặc quay đủ một góc; lưu RGB, aligned
   depth, timestamp và pose.
2. Chạy SAM3, CLIP và Gemma/Ollama để tạo mask, embedding và caption cho View
   node.
3. Association observations thành Object node ổn định, lưu tọa độ 3D và quan hệ
   view-object để phân biệt nhiều plastic container cùng loại.

Mục tiêu truy vấn sau này: từ câu như “plastic container ở cuối luống”, hệ thống
trả về đúng object instance, ảnh đại diện và tọa độ 3D của nó.

## 9. Source chính

```text
ros2_ws/src/conceptgraphs_bringup/launch/visualization.launch.py
ros2_ws/src/conceptgraphs_bringup/conceptgraphs_bringup/sam3_crate_mapper.py
ros2_ws/src/conceptgraphs_bringup/scripts/sam3_inference_worker.py
ros2_ws/src/conceptgraphs_bringup/conceptgraphs_bringup/z_passthrough_filter.py
ros2_ws/src/conceptgraphs_bringup/config/conceptgraphs_map.rviz
```

## 10. Milestone M1 — Aisle layout

M1 đã triển khai cho dataset pilot một aisle:

```text
greenhouse_01
└── aisle_01
```

- Layout nằm tại `config/greenhouse_layout.yaml`, dùng frame `camera_init` và
  `mode: single_aisle`.
- `assign_aisle(x, y)` luôn trả `aisle_01` trong chế độ pilot; loader đã có
  nhánh `polygon` để mở rộng sau này.
- Node `greenhouse_layout_markers` publish MarkerArray trên
  `/greenhouse_layout_markers` với transient-local QoS.
- Launch argument `start_greenhouse_layout` bật/tắt node độc lập; mặc định bật.
- RViz đã có display `Greenhouse aisle layout`.
- Layout pilot hiện khai báo polygon từ X `1.0..30.0`, Y `-1.4..2.0`,
  centerline từ `[1.0, 0.3]` tới `[30.0, 0.3]`, với Head ở đầu X nhỏ và Tail ở
  đầu X lớn. Đây là tọa độ tuyệt đối trong frame `camera_init`, không phải tọa
  độ tương đối tính từ đầu aisle.
- RViz cho phép bật/tắt riêng các namespace `aisle_polygon`,
  `aisle_centerline`, `aisle_endpoints` và `layout_labels`. Geometry có trường
  YAML rỗng sẽ không được publish.
- Unit test bao phủ đọc/validate YAML, single-aisle assignment và đường mở rộng
  polygon. Build và 6 test đều pass ngày 2026-10-01.
- Baseline đã được smoke-test với `start_greenhouse_layout:=false`: map server,
  costmap server, lifecycle manager, Z filter và bag playback vẫn hoạt động.

Các tọa độ pilot hiện tại đã được điền sau khi kiểm tra sơ bộ trong RViz, nhưng
vẫn cần đo/duyệt chính xác trên LiDAR map và xác nhận trajectory nằm trong aisle
trước khi bắt đầu M2. Trường tùy chọn `name: Aisle 1` hiện chưa được loader dùng;
label RViz được suy ra từ `id: aisle_01`.

### Lệnh chạy lại M1

Đóng hoàn toàn phiên launch/RViz cũ trước khi chạy để xóa point-cloud buffer của
dataset trước. Sau đó chạy:

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch conceptgraphs_bringup visualization.launch.py \
  bag_path:=/home/orc/SAM3/ros2_ws/dataset/mid360_camera_20260910_150208_processed_fov \
  start_static_maps:=false \
  start_crate_markers:=false \
  start_greenhouse_layout:=true \
  start_sam3:=true \
  sam3_device:=0 \
  sam3_confidence:=0.7
```

Lệnh này phát bag pilot, hiển thị layout M1 và chạy SAM3 baseline; không nạp
occupancy map, costmap hoặc marker CSV của dataset cũ.

## 11. Cách ly dữ liệu giữa các bag

`occupancy_grid.yaml`, `costmap_2d.yaml` và `crate_clusters.csv` là artifact của
dataset cũ, không được sinh lại tự động khi đổi `bag_path`. Để tránh chồng dữ
liệu, launch hiện mặc định:

- `start_static_maps:=false`: không chạy map server/costmap server/lifecycle
  manager cho map tĩnh.
- `start_crate_markers:=false`: không publish marker từ CSV cũ.

Chỉ bật lại khi truyền đúng artifact của cùng dataset:

```bash
start_static_maps:=true \
occupancy_map_yaml:=/absolute/path/to/new_occupancy.yaml \
costmap_yaml:=/absolute/path/to/new_costmap.yaml \
start_crate_markers:=true \
crate_csv:=/absolute/path/to/new_crate_clusters.csv
```

RViz tích lũy `/cloud_filtered` trong 600 giây để tạo map từ các scan. Khi đổi
bag trong cùng một phiên, phải đóng phiên launch/RViz cũ và mở lại để xóa buffer
point cloud trong bộ nhớ.


## 12. Milestone M2 — Keyframe Extractor

Executable `extract_keyframes` đã xuất dataset độc lập tại
`/home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01/`: 108 View, gồm
54 forward và 54 return. Mỗi View có RGB, depth PNG 16-bit, CameraInfo, timestamp,
pose camera trong `camera_init`, greenhouse/aisle ID và travel pass. Khoảng cách
trung bình là 0.491 m; 4/4057 RGB bị bỏ vì không có pose trong 100 ms.

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run conceptgraphs_bringup extract_keyframes
```

## 13. Milestone M3 — SAM3 Offline Observation Dataset

Executable `run_sam3_offline` đọc 108 View mà không phát rosbag. Kết quả có 63
View processed, 45 no-detection, 0 failed và 83 observations: 48 red, 35 yellow.
Prompt và threshold nằm trong dataset `pipeline_config.yaml`. Output gồm mask,
crop, context crop, overlay, `observations.jsonl` và `sam3_report.json`.

```bash
ros2 run conceptgraphs_bringup run_sam3_offline
```

M3 chỉ tạo proposal 2D với `projection_status: pending`; không projection,
Object association, CLIP hoặc Gemma.

## 14. Milestone M4 — Observation 3D

Executable `project_observations_3d` đọc mask M3, aligned depth, CameraInfo và
camera pose M2. Point được dựng trong `camera_color_optical_frame`, rồi chuyển
bằng pose camera sang `camera_init`; không giả định camera frame trùng LiDAR hoặc
body frame. Logic lọc depth được đối chiếu với `sam3_crate_mapper.py`.

Output mới, không sửa dữ liệu M2–M3:

```text
observations/observations_3d.jsonl
reports/projection_report.json
```

Đủ 83 observations: 16 valid, 67 rejected; 11 `insufficient_valid_depth`, 56
`object_beyond_maximum_distance`. Nhóm valid có median depth trung vị 1.682 m và
valid-depth ratio trung bình 0.705. Resume đủ 83 record không đổi checksum.

Diagnostic nearest cùng label giữa hai lượt: 7 cặp, khoảng cách trung bình 0.623
m (0.574..0.698 m), tất cả dưới 1 m. Mean delta `[-0.615, 0.058, 0.008]` m có
lệch hệ thống chủ yếu theo X; phải kiểm tra trực quan trên LiDAR map trước M5.
Đây chỉ là QA TF/timestamp, không phải Object association.

### Lệnh chạy lại M4

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run conceptgraphs_bringup project_observations_3d --limit 5 --force
ros2 run conceptgraphs_bringup project_observations_3d
```

Kiểm tra marker với LiDAR map:

```bash
ros2 launch conceptgraphs_bringup visualization.launch.py \
  bag_path:=/home/orc/SAM3/ros2_ws/dataset/mid360_camera_20260910_150208_processed_fov \
  lidar_map_topic:=/cloud_registered \
  start_sam3:=false \
  start_static_maps:=false \
  start_crate_markers:=false \
  start_observation_3d_markers:=true \
  observations_3d_path:=/home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01/observations/observations_3d.jsonl
```

Trong RViz bật `M4 observation 3D` và `/cloud_filtered`, kiểm tra centroid sphere,
wireframe bbox và label/ID cùng nằm trên object. Marker publisher đã smoke-test
48 markers cho 16 observations trong frame `camera_init`. Kết quả này là input
cho M5 và không bị M5 sửa đổi.


## 15. Milestone M5 — Persistent Object Association

M5 dùng `tracklet + Hungarian`, không dùng DBSCAN. Pipeline `associate_objects`
chỉ đọc 16 Observation M4 valid; file M4 và 67 rejected không bị sửa. Tracklet
được tách theo label, aisle và forward/return, rồi ghép hai lượt one-to-one bằng
Hungarian. Gate mặc định nằm trong `pipeline_config.yaml`: 0.5 m trong pass,
đường kính tracklet 0.75 m, cross-pass radius 1.0 m, size difference 0.6 và
maximum cost 1.1. Unmatched tracklet được giữ dưới dạng Object `single_sided`,
`ambiguous`, không bị ép ghép.

Kết quả dataset: 2 forward + 2 return tracklet, 2 cross-pass matches, 2 Object
multi-pass, 0 single-sided, 0 ambiguous. Red Object có 8 observations và đường
kính 0.762 m; yellow Object có 8 observations và đường kính 0.874 m. Có 16 edge
`View OBSERVES Object`; validation xác nhận unique ID và reference integrity.
Chạy lại giữ nguyên checksum ba JSONL.

Kiểm tra thủ công đủ 16 crop và representative overlay `view_000071` (red),
`view_000088` (yellow) cho thấy các thành viên của mỗi Object là cùng loại và
cùng physical container trong sample valid hiện tại; chưa thấy false merge hay
false split. Sai lệch forward/return vẫn khoảng 0.64–0.67 m nên cần bật link
marker trên LiDAR map khi QA.

Output:

```text
objects/objects.jsonl
graph/nodes.jsonl
graph/edges.jsonl
reports/association_report.json
```

### Lệnh chạy lại M5

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run conceptgraphs_bringup associate_objects
```

Kiểm tra Object association trong RViz:

```bash
ros2 launch conceptgraphs_bringup visualization.launch.py \
  bag_path:=/home/orc/SAM3/ros2_ws/dataset/mid360_camera_20260910_150208_processed_fov \
  lidar_map_topic:=/cloud_registered \
  start_sam3:=false \
  start_static_maps:=false \
  start_crate_markers:=false \
  start_observation_3d_markers:=true \
  start_object_markers:=true
```

Bật display `M5 Object associations`. Namespace `object_observation_links` giúp
nhìn Observation nào bị merge vào Object nào; tắt/bật riêng centroid, bbox,
label, observation và links để kiểm tra. Output này là input đã khóa cho M6.


## 16. Milestone M6 — Multimodal Enrichment

M6 đã hoàn thành bằng executable `enrich_semantics`; chưa tạo database hoặc
retrieval. Pipeline giữ nguyên M1–M5 và tạo embedding/caption ngoài các artifact
cũ.

CLIP dùng OpenAI `ViT-B-16` (`ViT-B/16` trong library), 512 chiều, weights
SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.
`view_embeddings.npy` có shape `[108, 512]`; `object_embeddings.npy` có shape
`[4, 512]`, gồm crop-mean và representative-View vector cho mỗi Object. Tất cả
vector đã normalize L2; `index_metadata.json` ánh xạ chính xác vector index với
View/Object và input paths.

Ollama local chạy tại `http://127.0.0.1:11434`. Cấu hình mặc định vẫn là
`gemma4:12b-it-q4_K_M`, nhưng tag này chưa được cài. Run dataset dùng fallback
local `gemma4:e4b`, Gemma 4 vision 8.0B Q4_K_M, digest đầy đủ
`c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb`.
GPU là NVIDIA RTX PRO 5000 Blackwell 48,935 MiB; Ollama dùng khoảng 7,606 MiB.
Backend lấy model tag từ config và không có field schema phụ thuộc Gemma, nên có
thể đổi sang `qwen3-vl` sau này.

Kết quả hiện tại dùng `greenhouse_semantics_v2`: 2 Object captions, 16
linked-View captions và 16 traversability assessments. Schema nested triển khai
đủ mục 8.3 gồm `scene_summary`, `physical_objects`, `greenhouse_context`,
`traversability_cues`, physical properties và object traversability. Có 18
complete, 0 failed; report cuối ghi 7 validation retries/25 attempts, 12 record có
uncertain attributes và tổng model time 153.69 s. Mọi record bắt buộc
`requires_geometric_verification: true`; VLM không được sửa Object ID, aisle hoặc
geometry M4/M5. JSON Schema gửi Ollama ràng buộc động số `physical_objects` bằng
số linked Object để tránh model thêm pathway, plant hoặc greenhouse structure.

Prompt không còn hard-code trong Python. Cấu hình đang dùng nằm tại
`config/semantic_prompts_v2.yaml`, được trỏ bởi
`semantic_enrichment.prompt_config` trong dataset `pipeline_config.yaml`; snapshot
prompt/schema thực tế nằm trong `semantics/prompts/`. Pydantic schema nằm tại
`scripts/semantic_schema_worker.py`.

Output:

```text
embeddings/view_embeddings.npy
embeddings/object_embeddings.npy
embeddings/index_metadata.json
semantics/object_captions.jsonl
semantics/view_captions.jsonl
semantics/traversability_assessments.jsonl
semantics/raw_responses/
semantics/prompts/
previews/semantic_review.html
reports/semantic_enrichment_report.json
```

### Lệnh chạy lại M6

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Pilot Gemma v2, không chạy lại CLIP
ros2 run conceptgraphs_bringup enrich_semantics \
  --object-limit 1 --view-limit 1 --force --skip-embeddings

# Chạy lại toàn bộ Gemma v2, không chạy lại CLIP
ros2 run conceptgraphs_bringup enrich_semantics --force --skip-embeddings

# Resume các record còn thiếu/failed
ros2 run conceptgraphs_bringup enrich_semantics --skip-embeddings
```

Muốn bắt buộc model khác đã có trong Ollama:

```bash
ros2 run conceptgraphs_bringup enrich_semantics --model qwen3-vl
```

Pipeline validate Pydantic schema, path/ID references, disjoint blocking và
traversable Object IDs, consistency giữa passability/`can_robot_pass`, quan hệ với
`visible_obstacles` và số linked Object; lưu raw Ollama response và retry tối đa
2 lần. Resume full run không tạo record trùng. M7 chưa bắt đầu.

## 17. Milestone M7 — HMSG SQLite Database

M7 đã hoàn thành ngày 2026-10-14. Importer `build_hmsg_database` đọc artifact
M1–M6 nhưng không sửa hoặc chạy lại các stage đó. Schema v1 lưu metadata,
geometry, provenance và path tương đối; ảnh, depth, mask và vector NumPy vẫn nằm
trên filesystem. Database production nằm tại:

```text
datasets/greenhouse_20260910_aisle01/database/greenhouse_hmsg.sqlite
```

Row count: 1 dataset, 1 greenhouse, 1 aisle, 108 View, 83 Observation, 2 Object,
16 Object–Observation link, 16 `View OBSERVES Object` edge, 18 caption, 16
traversability assessment và 112 embedding-index entry. Database size 704,512
bytes. Import lần hai: 0 inserted, 0 updated, 375 unchanged. Integrity check,
foreign keys, filesystem paths, caption owners và embedding vector indexes đều
hợp lệ.

Lệnh build/import/validate:

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select conceptgraphs_bringup
source install/setup.bash

ros2 run conceptgraphs_bringup build_hmsg_database \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01

ros2 run conceptgraphs_bringup build_hmsg_database \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01 \
  --validate-only
```

Structured query examples:

```bash
ros2 run conceptgraphs_bringup query_hmsg_database \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01 \
  --list-objects --aisle-id aisle_01

ros2 run conceptgraphs_bringup query_hmsg_database \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01 \
  --get-object object_000001 --format json

ros2 run conceptgraphs_bringup query_hmsg_database \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01 \
  --views-for-object object_000001
```

M8 chưa bắt đầu. CLIP similarity retrieval, natural-language query parsing và
Gemma top-k verification không thuộc M7.

## 18. Milestone M8 — Fast-to-Slow Retrieval

M8 đã hoàn thành ngày 2026-10-14 bằng executable `query_hmsg`. Pipeline nhận
natural-language text, dùng rule parser trước, Gemma local cho intent phức tạp,
lọc hierarchy/metadata trong SQLite, rồi mã hóa text bằng đúng CLIP ViT-B-16 của
M6 để rank Object hoặc View. Image embeddings không được tính lại.

Ambiguity gate dùng parser confidence, minimum similarity và top-1 margin. Query
rõ đi fast path; query mơ hồ chỉ gửi tối đa 3 candidate cho Gemma verification.
Gemma không được sửa ID/geometry và navigation query luôn giữ
`requires_geometric_verification: true`.

Pilot production gồm 4 query: 3 fast path, 1 Gemma verification, 0 not-found.
`Find the red plastic container` trả `object_000001` với similarity 0.26384.
`Find a container` có margin 0.00131 nên Gemma `gemma4:e4b`, digest
`c6eb396dbd59...`, xác minh cả `object_000001` và `object_000002`. Text CLIP chạy
CPU để không phụ thuộc GPU; vector ảnh M6 vẫn giữ nguyên. Average latency của bốn
record hiện tại là 19.42 s, bao gồm query Gemma 55.05 s.

Output:

```text
retrieval/query_results.jsonl
retrieval/previews/index.html
reports/retrieval_report.json
```

Lệnh chạy:

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run conceptgraphs_bringup query_hmsg \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01 \
  --query "Find the red plastic container" --top-k 5
```

M9 chưa bắt đầu. M8 không tạo local path, collision check hoặc navigation command.

## 19. Milestone M8.1 — Interactive Semantic Query Interface

M8.1 hoàn thành ngày 2026-10-14 bằng executable `query_hmsg_web`. FastAPI gọi
trực tiếp hàm `retrieve()` của M8 trong backend; frontend không gọi Ollama/Gemma
và không chứa bản sao retrieval logic. Dữ liệu định danh, aisle, centroid, bbox và
quan hệ graph được đọc từ HMSG SQLite. Caption, physical properties và
traversability evidence lấy từ các record M6 đã import ở M7.

UI hiển thị object crop, representative View và tối đa ba observing View bổ sung.
Scene graph chỉ chứa Object/View node và các edge `OBSERVES` đang tồn tại trong
database; không tạo `CONTAINS` hay `PARTIALLY_OBSTRUCTS` giả. Object traversability
và path passability được trình bày thành hai trường riêng, kèm cảnh báo geometric
verification trước mọi robot motion.

Lệnh cài web dependency và chạy:

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
python3 -m pip install --user -r src/conceptgraphs_bringup/requirements-web.txt
colcon build --packages-select conceptgraphs_bringup
source install/setup.bash

ros2 run conceptgraphs_bringup query_hmsg_web \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20260910_aisle01 \
  --host 127.0.0.1 --port 8080
```

Mở `http://127.0.0.1:8080`. Query production đã kiểm tra: red container trả
`object_000001`, yellow container trả `object_000002`, blue container trả
`not_found`, và follow-up “can the robot pass this object?” giữ Object context.
M8.1 không thêm voice, Nav2, robot movement, perception, association hoặc M9.

## 20. Milestone M8.2 — Intent-aware Semantic Dialogue

M8.2 hoàn thành ngày 2026-10-14 trên cùng executable `query_hmsg_web`. Lớp mới
`semantic_dialogue.py` đứng trên M8 retrieval, không thay thế hoặc duplicate CLIP,
SQLite filtering hay Gemma verification. Router nhẹ nhận diện `find_object`,
`list_objects`, `count_objects`, `locate_object` và `assess_traversability`; query
ngoài rule tiếp tục đi qua structured Gemma intent parser của M8.

Spatial reasoner chiếu Object centroid lên aisle centerline M1 và tạo verified
facts gồm `section`, `tendency`, khoảng cách từ head/tail và centerline offset.
Red container production nằm ở section `middle`, nghiêng về tail, cách head
17.83 m và tail 11.17 m. Các facts này do Python tính trong `camera_init`, không
do Gemma suy đoán.

Gemma response composer dùng prompt ngoài source tại
`config/dialogue_prompts_v1.yaml`. Model chỉ nhận user query và verified facts;
không truy cập SQLite/dataset. Backend chặn ID ngoài candidate, thay đổi geometry
safety requirement và response phủ nhận một retrieval đã xác minh. Khi Ollama
không sẵn sàng hoặc output không nhất quán, intent-aware template được dùng.

Query production đã kiểm tra:

- `What objects are in the map?` → `object_000001`, `object_000002`.
- `Where is the red container along the aisle?` → middle/toward-tail, 17.83 m
  từ head và 11.17 m tới tail.
- `Find a container the color of watermelon flesh` → `object_000001`.
- `Tìm container có màu giống quả dưa hấu` → hỏi lại green rind hay red flesh.

Dialogue records có stable ID và được upsert tại
`retrieval/dialogue_results.jsonl`. Cách build/chạy và URL giữ nguyên M8.1. M8.2
không thêm voice, navigation execution, perception, association hoặc M9.

## 21. Milestone M8.2.1 — General Chat Routing

M8.2.1 hoàn thành ngày 2026-10-14 trên `query_hmsg_web`. Web UI có ba chế độ:
`Auto`, `Chat` và `Explore Map`. Chat mode gọi Gemma local bằng prompt
`config/general_chat_prompts_v1.yaml`, chỉ nhận tối đa 10 message browser history
và không gọi database, CLIP hay M8 retrieval. Explore Map luôn giữ pipeline M8.2.
Auto dùng rule router cho câu rõ ràng và Gemma JSON classifier cho câu chưa rõ.

General chat không được nhận hoặc tự tạo current-map facts; khi cần Object/View/
aisle/geometry thật, model hướng người dùng sang Explore Map. UI hiển thị chat
bubbles cùng provenance `manual`, `rule`, `gemma_classifier`,
`gemma_general_chat` hoặc map retrieval. Session history chỉ nằm trong browser
memory và mất khi reload trang.

Ollama generation cho classifier/general chat được giới hạn token và model được
giữ warm 30 phút. Kiểm tra production xác nhận `Bạn có thể làm được gì?` ở Chat
và Auto có `retrieval: null`; `Where is the red container?` ở Explore Map trả
`object_000001`. Cách chạy và URL giữ nguyên M8.1/M8.2. M9 chưa bắt đầu.
