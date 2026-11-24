# Kế hoạch xây dựng Greenhouse HMSG Dataset và VLM Retrieval

**Phiên bản:** Draft v0.1
**Ngày cập nhật:** 2026-10-01
**Phạm vi pilot:** Một aisle, robot chạy từ đầu đến cuối rồi quay về
**ROS:** ROS 2 Humble
**Map frame:** `camera_init`

## 1. Mục tiêu

Mở rộng pipeline SAM3 semantic mapping hiện tại thành pipeline offline xây dựng
Hierarchical Multi-modal Scene Graph (HMSG) cho greenhouse:

```text
Greenhouse
└── Aisle
    ├── View
    └── Object

View ──OBSERVES──> Object
```

Pipeline đích:

```text
Rosbag đã có FAST-LIO pose và registered point cloud
        ↓
Keyframe Extractor (mỗi khoảng 0.5 m hoặc khi quay đủ góc)
        ↓
View Dataset: RGB + aligned depth + calibration + pose
        ↓
SAM3 + 3D Projection
        ↓
Observation và View/Object Association
        ↓
CLIP/SigLIP embedding + Gemma 4 structured caption
        ↓
Greenhouse HMSG Database
        ↓
Fast retrieval → Gemma 4 verification
        ↓
Object ID + representative image + tọa độ 3D
```

MVP phải trả lời được các truy vấn như:

- “Tìm yellow plastic container trong aisle 1.”
- “Tìm container ở cuối aisle.”
- “Cho tôi ảnh nhìn rõ nhất của red plastic container.”

## 2. Trạng thái baseline hiện tại

Pipeline hiện có:

- Phát rosbag MCAP với simulation time.
- Point cloud `/cloud_registered` đã nằm trong world frame.
- FAST-LIO pose/odometry và TF đã có trong bag.
- SAM3 nhận RGB và tạo mask theo text prompt.
- Mask được kết hợp với aligned depth và CameraInfo để tính robust 3D center.
- Detection được chuyển về frame `camera_init`.
- Observation cùng label và gần nhau theo XY được gom thành cluster trong RAM.
- Cluster được hiển thị bằng RViz marker.

Các phần còn thiếu cho HMSG:

- Keyframe RGB-D có pose và ID ổn định.
- Định nghĩa Greenhouse/Aisle region.
- View node và Object node lưu bền vững.
- Quan hệ `View OBSERVES Object`.
- Visual embedding và structured caption.
- HMSG database và query pipeline.

## 3. Quyết định thiết kế cho pilot

### 3.1 Cấu trúc node

```text
greenhouse_01
└── aisle_01
    ├── view_000001
    ├── view_000002
    ├── ...
    ├── object_000001
    └── object_000002
```

Các edge:

```text
Greenhouse CONTAINS Aisle
Aisle CONTAINS View
Aisle CONTAINS Object
View OBSERVES Object
View ADJACENT_TO View
```

### 3.2 Định nghĩa Aisle

Trong pilot chỉ có một aisle nhưng vẫn khai báo layout để không phải đổi schema
khi mở rộng:

```yaml
greenhouse:
  id: greenhouse_01
  frame_id: camera_init

aisles:
  - id: aisle_01
    polygon_xy: []
    centerline: []
    head_point: []
    tail_point: []
```

Polygon được vẽ trên LiDAR map nhìn từ trên xuống và lưu bằng mét trong
`camera_init`. Centerline xác định tọa độ chuẩn hóa dọc aisle:

```text
head:   0.00 <= s < 0.33
middle: 0.33 <= s < 0.67
tail:   0.67 <= s <= 1.00
```

`head` và `tail` là thuộc tính cố định của aisle, không thay đổi theo hướng robot
đang chạy.

## 4. Keyframe và View node

### 4.1 Nguồn pose

Khoảng cách 0.5 m được tính từ pose của robot/camera đã có trong `/Odometry` và
TF, sau khi chuyển về frame `camera_init`.

Không đo khoảng cách bằng cách so trực tiếp hai point cloud. Point cloud dùng để
tạo bản đồ và kiểm tra trực quan; pose là nguồn phù hợp để chọn keyframe.

Khoảng cách giữa hai pose:

```text
d = sqrt((x_t - x_last)^2 + (y_t - y_last)^2 + (z_t - z_last)^2)
```

Điều kiện chọn keyframe:

```text
d >= 0.50 m
OR abs(yaw_t - yaw_last) >= 15 degrees
```

Có thể thêm `minimum_time_interval` khoảng 0.3-0.5 s để tránh tạo quá nhiều
frame khi TF hoặc odometry nhiễu.

Điều kiện góc quay là cần thiết vì robot quay đầu có thể thay đổi góc nhìn nhiều
nhưng gần như không thay đổi vị trí.

### 4.2 Đồng bộ dữ liệu

Tại timestamp của RGB keyframe, lấy:

- RGB image gần nhất.
- Aligned depth gần nhất.
- CameraInfo tương ứng.
- Camera pose được nội suy hoặc lấy gần nhất.
- TF static cần thiết để chuyển camera pose sang `camera_init`.

Mỗi keyframe phải có trạng thái đồng bộ. Không tạo View nếu thiếu depth, pose
hoặc calibration.

### 4.3 View node format

```json
{
  "view_id": "view_000123",
  "dataset_id": "greenhouse_20260910_aisle01",
  "greenhouse_id": "greenhouse_01",
  "aisle_id": "aisle_01",
  "timestamp_ns": 0,
  "frame_id": "camera_init",
  "rgb_path": "keyframes/rgb/view_000123.jpg",
  "depth_path": "keyframes/depth/view_000123.png",
  "camera_info_path": "keyframes/camera_info/view_000123.json",
  "pose": {
    "translation_xyz": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "aisle_coordinate_s": 0.0,
  "aisle_section": "head",
  "travel_pass": "forward",
  "processing_status": "extracted"
}
```

`travel_pass` chỉ dùng để phân tích/debug. Object association không được xem hai
lượt chạy là hai aisle khác nhau.

## 5. SAM3 Observation Dataset

SAM3 phải chạy offline trên toàn bộ View dataset để không bỏ frame khi GPU xử lý
chậm hơn rosbag playback.

Prompt chuyển khỏi source code và đưa vào cấu hình:

```yaml
sam3:
  prompts:
    - yellow plastic container on the ground
    - red plastic container on the ground
  confidence_threshold: 0.7
  maximum_depth_m: 3.0
```

Mỗi detection tạo một Observation độc lập:

```json
{
  "observation_id": "obs_000456",
  "view_id": "view_000123",
  "aisle_id": "aisle_01",
  "prompt": "yellow plastic container on the ground",
  "canonical_label": "yellow_plastic_container",
  "sam_confidence": 0.88,
  "bbox_2d_xyxy": [0, 0, 0, 0],
  "mask_path": "observations/masks/obs_000456.png",
  "crop_path": "observations/crops/obs_000456.jpg",
  "centroid_3d": [0.0, 0.0, 0.0],
  "bbox_3d_min": [0.0, 0.0, 0.0],
  "bbox_3d_max": [0.0, 0.0, 0.0],
  "mean_depth_m": 0.0,
  "valid_depth_ratio": 0.0,
  "point_count": 0,
  "projection_status": "valid"
}
```

Nên lưu cả mask, crop sát object và crop có thêm context xung quanh.

## 6. Projection 3D

Tái sử dụng logic đã có trong `sam3_crate_mapper.py`:

```text
SAM3 mask
+ aligned depth
+ camera intrinsics
        ↓
3D points trong camera frame
        ↓
camera/odometry transforms
        ↓
3D points trong camera_init
```

Yêu cầu bổ sung:

- Lọc depth invalid và outlier.
- Lưu `valid_depth_ratio` và số điểm hợp lệ.
- Lưu centroid, median và 3D bounding box nếu đủ dữ liệu.
- Observation không đạt chất lượng vẫn được ghi lại với trạng thái rejected để
  có thể debug.

## 7. View/Object Association

Một Object có thể xuất hiện trong nhiều View. Ví dụ:

```text
view_000010 ──OBSERVES──┐
view_000011 ──OBSERVES──┼── object_000003
view_000042 ──OBSERVES──┘
```

Trong đó `view_000010` và `view_000011` có thể thuộc lượt đi, còn
`view_000042` thuộc lượt về.

Association gồm hai tầng:

1. Geometric gating:
   - Cùng canonical label.
   - Cùng aisle.
   - Centroid 3D nằm trong bán kính cho phép.
   - Kích thước 3D không khác biệt bất thường.
2. Visual verification:
   - So sánh CLIP/SigLIP embedding của object crop.
   - Kiểm tra màu, hình dạng và context.

Điểm association dự kiến:

```text
association_score =
    w_position * spatial_similarity
  + w_visual   * visual_similarity
  + w_size     * size_similarity
  + w_region   * aisle_consistency
```

Nếu không candidate nào vượt threshold thì tạo Object mới.

Object node format:

```json
{
  "object_id": "object_000003",
  "greenhouse_id": "greenhouse_01",
  "aisle_id": "aisle_01",
  "canonical_label": "yellow_plastic_container",
  "centroid_3d": [0.0, 0.0, 0.0],
  "bbox_3d_min": [0.0, 0.0, 0.0],
  "bbox_3d_max": [0.0, 0.0, 0.0],
  "aisle_coordinate_s": 0.82,
  "aisle_section": "tail",
  "observation_count": 4,
  "representative_view_id": "view_000042",
  "first_seen_timestamp_ns": 0,
  "last_seen_timestamp_ns": 0,
  "dynamic_state": "unknown"
}
```

Representative View được chọn dựa trên confidence, object pixel area, valid
depth ratio, blur, occlusion và khoảng cách tới camera.

## 8. Embedding và Gemma 4

### 8.1 Phân vai

```text
SAM3                 → mask và object proposal
Depth/LiDAR/pose     → geometry, metric size và position
CLIP hoặc SigLIP     → embedding và fast retrieval
Gemma 4              → caption, semantic attributes và candidate verification
```

Gemma 4 không phải nguồn sự thật cho tọa độ, kích thước metric hoặc
traversability.

### 8.2 Input cho captioning

Đối với View:

- Ảnh RGB đầy đủ.
- Danh sách object crop/overlay nếu có.
- Không cần gửi toàn bộ chuỗi video.

Đối với Object:

- Representative crop.
- Một crop có context.
- Có thể bổ sung 2-3 View tốt nhất nếu cần multi-view verification.

### 8.3 Structured caption schema

Gemma 4 nên trả JSON theo schema cố định thay vì chỉ lưu caption tự do:

```json
{
  "scene_summary": "",
  "physical_objects": [
    {
      "category": "plastic container",
      "color": "yellow",
      "material": "plastic",
      "shape": "rectangular",
      "state": "on the ground",
      "relative_position": "right side of the aisle",
      "occlusion": "low",
      "physical_properties": {
        "rigidity": "rigid",
        "deformability": "low",
        "compressibility": "low",
        "fragility": "medium",
        "estimated_mass_class": "unknown",
        "anchored_to_ground": false,
        "confidence": 0.82,
        "evidence": "Visual appearance is consistent with a rigid plastic container"
      },
      "object_traversability": {
        "semantic_class": "non_traversable_obstacle",
        "pass_over_recommendation": "forbidden",
        "contact_recommendation": "avoid",
        "navigation_cost": "lethal",
        "potential_risks": [
          "collision",
          "robot_instability",
          "object_damage"
        ],
        "requires_geometric_verification": true,
        "confidence": 0.88
      }
    }
  ],
  "greenhouse_context": {
    "crop_type_visible": "unknown",
    "irrigation_equipment_visible": false,
    "ground_condition": "unknown"
  },
  "traversability_cues": {
    "vlm_path_assessment": "blocked",
    "can_robot_pass": false,
    "blocking_object_ids": ["object_000003"],
    "traversable_object_ids": [],
    "visible_obstacles": [
      {
        "object_id": "object_000003",
        "category": "plastic container",
        "position_relative_to_path": "center",
        "occupies_drivable_space": true,
        "assessment": "non_traversable",
        "reason": "A rigid container is located in the middle of the driving path"
      }
    ],
    "reasoning_summary": "The aisle is blocked by a rigid container in the robot path",
    "requires_geometric_verification": true,
    "confidence": 0.91
  },
  "uncertain_attributes": []
}
```

`physical_properties` mô tả tính chất vật lý được suy luận từ hình ảnh:

- `rigidity`: `rigid`, `semi_rigid`, `soft` hoặc `unknown`.
- `deformability`: `low`, `medium`, `high` hoặc `unknown`.
- `compressibility`: `low`, `medium`, `high` hoặc `unknown`.
- `fragility`: nguy cơ object bị hỏng khi robot tiếp xúc hoặc đi qua.
- `estimated_mass_class`: `light`, `medium`, `heavy` hoặc `unknown`.
- `anchored_to_ground`: object có vẻ được cố định vào nền hay không.
- `evidence`: mô tả ngắn căn cứ thị giác của VLM.

`object_traversability` biểu diễn khả năng robot đi xuyên qua hoặc đi đè lên
chính object được phát hiện:

- `semantic_class`: `non_traversable_obstacle`, `potentially_traversable`,
  `traversable` hoặc `unknown`.
- `pass_over_recommendation`: `forbidden`, `avoid`, `allowed` hoặc `unknown`.
- `contact_recommendation`: `avoid`, `cautious_contact`, `allowed` hoặc
  `unknown`.
- `navigation_cost`: `free`, `low`, `high`, `lethal` hoặc `unknown`.
- `potential_risks`: collision, entanglement, slipping, wheel blockage, object
  damage hoặc crop damage.
- `requires_geometric_verification`: luôn là `true` đối với kết quả do VLM suy
  luận trước khi dùng cho navigation.

`traversability_cues` là kết luận theo ngữ cảnh của toàn View, không chỉ theo
vật liệu của từng object. Gemma 4 phải xem xét đồng thời:

- Object là vật gì và có tính chất vật lý nào.
- Object nằm giữa, bên cạnh hay ngoài vùng robot dự kiến di chuyển.
- Object chiếm bao nhiêu phần chiều rộng lối đi.
- Có khoảng trống rõ ràng để đi vòng hay không.
- Có nhiều object kết hợp tạo thành vùng cản hay không.
- Điều kiện nền có dấu hiệu trơn, lún hoặc bị che khuất hay không.

Giá trị `vlm_path_assessment` chỉ nhận một trong:

```text
passable   → VLM đánh giá robot có thể đi qua
blocked    → VLM đánh giá đường bị chặn
uncertain  → ảnh không đủ bằng chứng hoặc có rủi ro chưa rõ
```

Ví dụ chậu nhựa cứng nằm giữa đường:

```json
{
  "vlm_path_assessment": "blocked",
  "can_robot_pass": false,
  "blocking_object_ids": ["object_flower_pot_01"],
  "reasoning_summary": "A rigid flower pot occupies the center of the driving path",
  "confidence": 0.94
}
```

Ví dụ lá rụng thưa trên đường:

```json
{
  "vlm_path_assessment": "passable",
  "can_robot_pass": true,
  "blocking_object_ids": [],
  "traversable_object_ids": ["object_fallen_leaves_01"],
  "reasoning_summary": "Sparse soft leaves are present but do not appear to block the driving path",
  "confidence": 0.86
}
```

Ví dụ tấm vải che một phần đường:

```json
{
  "vlm_path_assessment": "uncertain",
  "can_robot_pass": null,
  "blocking_object_ids": [],
  "reasoning_summary": "The cloth is deformable but may entangle with the robot wheels",
  "confidence": 0.72
}
```

Ví dụ semantic prior theo loại object:

| Object | Tính chất dự kiến | Traversability prior |
|---|---|---|
| Plastic container cứng | Rigid, khó biến dạng | `non_traversable_obstacle`, `forbidden` |
| Cành cây lớn | Semi-rigid hoặc rigid | `non_traversable_obstacle` hoặc `avoid` |
| Lá rơi rời rạc | Mềm, nhẹ, dễ nén | `potentially_traversable` |
| Tấm vải | Mềm và biến dạng | `unknown` hoặc `avoid` do nguy cơ quấn bánh xe |
| Ống tưới | Semi-rigid | `avoid` do nguy cơ kẹt bánh hoặc làm hỏng hệ thống |

Không gán `traversable` chỉ vì object mềm. Ví dụ tấm vải có thể bị nén nhưng
cũng có thể quấn vào bánh xe; lá rơi có thể trơn hoặc che khuất bề mặt. Semantic
prior cần được kết hợp với geometry và policy của robot:

```text
final_traversability =
    semantic physical prior từ VLM
  + object height/width từ depth hoặc LiDAR
  + ground clearance và wheel geometry của robot
  + costmap/free-space
  + class-specific safety policy
```

Các thuộc tính đo được nên lưu tách biệt:

```json
{
  "traversability_geometry": {
    "object_height_m": 0.0,
    "object_width_m": 0.0,
    "object_length_m": 0.0,
    "minimum_clearance_m": 0.0,
    "free_space_around_object_m": 0.0,
    "measurement_source": "aligned_depth_and_lidar",
    "measurement_confidence": 0.0
  },
  "traversability_decision": {
    "decision": "blocked",
    "decision_source": "rule_fusion",
    "reason": "Rigid object exceeds robot ground clearance",
    "policy_version": "greenhouse_traversability_v1"
  }
}
```

Nếu “tranversal” trong KPI NVIDIA có nghĩa là `traversability`, cần tách:

- `physical_properties` và `object_traversability`: semantic prior từ VLM về
  bản thân object.
- `traversability_cues`: kết luận `passable`, `blocked` hoặc `uncertain` của VLM
  dựa trên toàn cảnh, vị trí object và phần lối đi bị chiếm dụng.
- `traversability_geometry`: giá trị đo từ costmap/LiDAR, ví dụ free-space,
  kích thước object, clearance và obstacle distance.
- `traversability_decision`: kết quả cuối của rule/policy fusion.

`traversability_cues` là decision semantic cuối của VLM. Chỉ
`traversability_decision` sau khi hợp nhất decision này với geometry và robot
policy mới được dùng trực tiếp cho navigation safety.

### 8.4 Provenance

Mọi caption và embedding phải lưu:

- Model name/tag.
- Model version hoặc digest.
- Prompt version.
- Input image IDs.
- Timestamp xử lý.
- Raw response.
- Parsed response và trạng thái validation.

Danh sách KPI/attribute chính thức của dự án NVIDIA cần được đưa vào một file
schema cấu hình riêng khi được xác nhận. Không hard-code thuộc tính KPI trong
logic database.

## 9. Dataset format đề xuất

```text
greenhouse_20260910_aisle01/
├── manifest.json
├── pipeline_config.yaml
├── map/
│   ├── lidar_map.ply
│   ├── greenhouse_layout.yaml
│   └── aisles.geojson
├── keyframes/
│   ├── rgb/
│   ├── depth/
│   ├── camera_info/
│   └── views.jsonl
├── observations/
│   ├── masks/
│   ├── crops/
│   └── observations.jsonl
├── objects/
│   ├── pointclouds/
│   └── objects.jsonl
├── semantics/
│   ├── view_captions.jsonl
│   ├── object_captions.jsonl
│   └── prompts/
├── embeddings/
│   ├── view_embeddings.npy
│   ├── object_embeddings.npy
│   └── index_metadata.json
├── graph/
│   ├── nodes.jsonl
│   └── edges.jsonl
├── database/
│   └── greenhouse_hmsg.sqlite
├── reports/
│   ├── extraction_report.json
│   ├── projection_report.json
│   ├── association_report.json
│   └── caption_report.json
└── previews/
```

Quy ước lưu trữ:

- Ảnh, depth, mask và point cloud nằm trên filesystem.
- JSONL là artifact trung gian dễ kiểm tra và chạy lại.
- SQLite lưu node, edge, metadata và đường dẫn file.
- Vector index phục vụ fast retrieval.
- Không lưu binary image trực tiếp trong SQLite.

## 10. HMSG database

Các bảng tối thiểu:

```text
datasets
processing_runs
greenhouses
aisles
views
objects
observations
view_object_edges
view_view_edges
captions
embeddings
```

Mỗi stage phải idempotent: chạy lại không tạo record trùng và có thể tiếp tục từ
record chưa hoàn thành.

## 11. Retrieval và VLM verification

Query flow:

```text
Natural-language query
        ↓
Parse greenhouse/aisle/object/spatial attributes
        ↓
Filter theo hierarchy và geometry
        ↓
CLIP/SigLIP top-k fast retrieval
        ↓
Nếu kết quả rõ ràng: trả về
Nếu mơ hồ: Gemma 4 kiểm tra top-k representative Views
        ↓
Object ID + View ID + image + centroid 3D + confidence
```

Không gửi toàn bộ View dataset cho Gemma. Chỉ gửi top-k candidate sau bước fast
retrieval.

## 12. Các milestone triển khai

### M1 - Aisle layout

- [x] Tạo schema/layout loader cho `greenhouse_01` và `aisle_01` trong frame
  `camera_init`.
- [x] Triển khai chế độ pilot `single_aisle`; mọi điểm XY được gán vào
  `aisle_01`.
- [x] Publish `/greenhouse_layout_markers` và tích hợp tùy chọn
  `start_greenhouse_layout` vào launch/RViz.
- [x] Validate YAML, chuẩn bị chế độ gán theo polygon và thêm test.
- [x] Khai báo tọa độ pilot ban đầu cho polygon, centerline, head và tail của
  `aisle_01` trong frame `camera_init`.
- [ ] Sau khi có tọa độ, kiểm tra trực quan cùng trajectory và xác nhận toàn bộ
  đường chạy thuộc vùng hợp lệ.

Trạng thái ngày 2026-10-01: phần mềm M1 cho pilot một aisle đã hoàn thành và đã
có bộ tọa độ pilot ban đầu. Cần QA lại biên polygon và hướng Head/Tail trên LiDAR
map trước khi khóa M1. Chưa bắt đầu M2.

Các artifact tĩnh theo dataset (occupancy map, costmap và marker CSV) mặc định
được tắt trong launch để không chồng dữ liệu cũ khi đổi bag. Chỉ bật chúng với
đường dẫn artifact thuộc đúng dataset đang chạy.

### M2 - Keyframe extractor

- [x] Đọc MCAP offline, chọn View mỗi 0.5 m hoặc 15 độ.
- [x] Lưu đồng bộ RGB, aligned depth, CameraInfo và camera pose.
- [x] Gán `greenhouse_01`, `aisle_01`, tọa độ s, section và travel pass.
- [x] ID deterministic, report sync/skip, marker pose và resume an toàn.

Kết quả full bag ngày 2026-10-01: 108 View (54 forward, 54 return); 4/4057 RGB
bị bỏ vì thiếu pose trong tolerance 100 ms. Khoảng cách View trung bình 0.491 m.

### M3 - SAM3 offline dataset

- [x] Prompt/canonical label/threshold nằm trong `pipeline_config.yaml`.
- [x] SAM3 worker `offline_2d` chạy tuần tự đủ 108 View, không drop do GPU bận.
- [x] Lưu mask, crop, context crop, overlay và metadata/provenance.
- [x] Resume theo fingerprint, View status và validate mọi record/path.

Kết quả: 63 processed, 45 no-detection, 0 failed, 83 observations; 48 red và
35 yellow. M3 giữ `projection_status: pending` và không làm projection.

### M4 - Observation 3D

- [x] Kết hợp mask, aligned depth, CameraInfo và camera pose của từng View.
- [x] Đối chiếu baseline: erode 5 px, depth 0.15..15 m, median outlier band
  `max(0.20 m, 10%)`, tối thiểu 10 điểm, object cách camera tối đa 3 m.
- [x] Dựng point trong `camera_color_optical_frame`, rồi dùng pose camera M2 để
  chuyển sang `camera_init`; không giả định camera trùng LiDAR/body frame.
- [x] Lưu đủ 83 record valid/rejected, checkpoint/resume và không sửa M2–M3.
- [x] Publish centroid, bbox, label/ID trên `/observation_3d_markers`; node mặc
  định tắt và bật độc lập trong launch.

Kết quả: 16 valid, 67 rejected; 11 thiếu depth hợp lệ và 56 xa hơn 3 m. Nhóm
valid có median depth trung vị 1.682 m, valid-depth ratio trung bình 0.705.
Diagnostic 7 nearest pairs cùng label giữa hai lượt có khoảng cách trung bình
0.623 m, đều dưới 1 m, nhưng mean delta `[-0.615, 0.058, 0.008]` m cho thấy lệch
có hệ thống cần QA marker trên LiDAR map; M5 đã giữ sai lệch này trong report
và dùng midpoint hai lượt cho Object center.

### M5 - Persistent Object association

- [x] Chỉ association 16 Observation M4 valid; 67 rejected không bị sửa.
- [x] Tạo tracklet riêng theo label, aisle và travel pass bằng Hungarian theo
  từng View; gate khoảng cách 0.5 m, kích thước và đường kính chống transitive
  chaining.
- [x] Ghép tracklet forward/return one-to-one bằng Hungarian trong radius 1.0 m,
  có size/cost gate và không ép ghép.
- [x] Object multi-pass dùng midpoint của hai robust pass median; unmatched
  tracklet thành `single_sided` và `ambiguous`.
- [x] Object/edge ID deterministic, resume theo fingerprint, tạo 16 edge
  `View OBSERVES Object` và chọn representative View bằng quality score.
- [x] Publish centroid, bbox, label và liên kết tới Observation trên
  `/object_association_markers` để QA false merge/false split.

Kết quả ngày 2026-10-01: 4 tracklet (2 forward, 2 return), 2 cross-pass matches
và 2 Object multi-pass; không có single-sided/ambiguous. Mỗi Object có 8
Observation. Cluster diameter red 0.762 m, yellow 0.874 m. Kiểm tra thủ công đủ
16 crop và hai representative overlay cho thấy mỗi Object giữ đúng một container
theo màu; chưa thấy false merge/split trong sample valid nhỏ này. Output M5 được giữ nguyên khi chạy M6.

### M6 - Multimodal enrichment

- [x] Tạo normalized CLIP ViT-B-16 embeddings cho đủ 108 View RGB và 4 Object
  vectors: crop-mean + representative View cho mỗi Object; mapping vector/ID,
  model version và SHA-256 nằm trong `index_metadata.json`.
- [x] Preflight Ollama local, vision capability và GPU/VRAM; backend chỉ phụ
  thuộc Ollama model tag nên có thể đổi sang qwen3-vl mà không đổi schema.
- [x] Caption 2 Object bằng representative crop, context crop và tối đa 3 View;
  caption/đánh giá traversability đủ 16 View có `OBSERVES` edge.
- [x] Pydantic JSON Schema, temperature 0, retry có giới hạn, raw response,
  prompt/model provenance và record lỗi bền vững.
- [x] `greenhouse_semantics_v2` triển khai đầy đủ schema nested mục 8.3:
  `scene_summary`, `physical_objects`, `greenhouse_context`,
  `traversability_cues`, physical properties và object traversability. Prompt nằm
  ngoài source Python tại `config/semantic_prompts_v2.yaml`; JSON Schema động ép
  số `physical_objects` bằng số linked Object của View.
- [x] Schema không chứa object ID/aisle/geometry có thể bị VLM ghi đè; mọi record
  bắt buộc `requires_geometric_verification: true`.
- [x] Checkpoint/resume, stable IDs, path/reference validation và semantic HTML
  preview; không sửa artifact M1–M5.

Kết quả ngày 2026-10-01: CLIP 512-D tạo 108 View vectors và 4 Object vectors,
L2 norm trong 0.9999999..1.0000001. Tag mặc định cấu hình là
`gemma4:12b-it-q4_K_M` nhưng chưa có local, nên run dùng fallback local
`gemma4:e4b` 8B Q4_K_M, digest `c6eb396dbd59...`. Có 18/18 semantic records
complete, 0 failed theo schema v2; report cuối ghi 7 validation retries trên 25
attempts và 12 record có thuộc tính chưa chắc chắn. Mọi record đều yêu cầu geometry
verification. Tổng model time của các record cuối là 153.69 s. GPU dùng khoảng
7.6 GiB VRAM. M7 chưa bắt đầu.

### M7 - HMSG database

- [x] Import JSONL vào SQLite, giữ stable ID, provenance và geometry.
- [x] Truy vấn được Greenhouse → Aisle → View/Object.
- [x] Truy xuất được tất cả View/Observation quan sát một Object.
- [x] Validate foreign key, filesystem path, embedding index và idempotency.

Kết quả ngày 2026-10-14: schema v1 tạo
`database/greenhouse_hmsg.sqlite` (704,512 bytes). Database chứa 108 View, 83
Observation, 2 Object, 16 quan hệ `OBSERVES`, 18 caption, 16 traversability
assessment và 112 embedding-index entry. Import lần hai tạo 0 row mới, cập nhật
0 row và giữ nguyên 375 row. `PRAGMA integrity_check`, foreign-key check và toàn
bộ path/reference validation đều pass. M8 chưa bắt đầu.

### M8 - Fast-to-slow query

- [x] Rule-first/Gemma intent parsing và hierarchical SQLite filtering.
- [x] Fast retrieval bằng CLIP text embedding trên vector M6.
- [x] Ambiguity gates và Gemma 4 verification chỉ cho top-k candidate.
- [x] Trả Object/View, ảnh đại diện, caption, geometry và traversability evidence.

Kết quả ngày 2026-10-14: `greenhouse_retrieval_v1` chạy 4 query production;
3 query đi fast path và 1 query mơ hồ được Gemma `gemma4:e4b` xác minh. Query
`Find a container` phát hiện top-1 margin nhỏ và trả đúng cả hai Object. Query
navigation giữ `requires_geometric_verification: true`. Kết quả idempotent nằm
trong `retrieval/query_results.jsonl`; M9 chưa bắt đầu.

### M8.1 - Interactive Semantic Query Interface

- [x] FastAPI backend tái sử dụng trực tiếp M8 retrieval/Gemma qua `retrieve()`.
- [x] Web UI nhập natural-language query và hiển thị response, ảnh, geometry,
  physical properties, object traversability và path assessment tách biệt.
- [x] Scene graph chỉ dựng các Object/View node và `OBSERVES` edge thật từ HMSG.
- [x] Query không đủ confidence trả `not_found`; follow-up “this object” giữ context
  của kết quả hiện tại mà không cho Gemma sửa ID hoặc geometry.

Kết quả ngày 2026-10-14: interface local tại `http://127.0.0.1:8080` trả
`object_000001` cho red container, `object_000002` cho yellow container và không
trả Object cho blue container. Mỗi Object production hiển thị crop, representative
View, tối đa ba related View, centroid trong `camera_init` và neighborhood gồm 8
`OBSERVES` edge thật. Đây chỉ là semantic inspection; mọi kết luận navigation vẫn
bắt buộc geometric verification. M9 chưa bắt đầu.

### M8.2 - Intent-aware Semantic Dialogue

- [x] Rule-first router cho `find`, `list`, `count`, `locate` và
  `assess_traversability`; query phức tạp tiếp tục dùng Gemma intent parser M8.
- [x] Query mô tả gián tiếp được parse/rank/verify bằng M8; mô tả thật sự mơ hồ
  như “màu quả dưa hấu” phải hỏi lại vỏ xanh hay ruột đỏ.
- [x] Spatial facts được tính bằng Python từ Object centroid và centerline/head/
  tail M1: section, tendency, distance from head/tail và centerline offset.
- [x] Gemma response composer chỉ nhận verified facts, không truy cập dataset và
  không được sửa ID, aisle, geometry, relation hoặc safety requirement.
- [x] Validation chặn Object ID lạ, navigation safety sai và câu `not found` mâu
  thuẫn với retrieval; Ollama lỗi hoặc output sai sẽ dùng intent-aware template.
- [x] Web UI hiển thị nguồn câu trả lời, intent và danh sách nhiều Object; dialogue
  record idempotent lưu trong `retrieval/dialogue_results.jsonl`.

Kết quả ngày 2026-10-14: `What objects are in the map?` trả cả
`object_000001` và `object_000002`; query vị trí red container tính section
`middle`, nghiêng về tail, cách head 17.83 m và cách tail 11.17 m. Query
`watermelon flesh` chọn đúng `object_000001`; query chỉ nói `watermelon` yêu cầu
làm rõ. M8.2 không thay đổi artifact M1–M8 và chưa bắt đầu M9.

### M8.2.1 - General Chat Routing

- [x] UI có ba mode độc lập: `Auto`, `Chat`, `Explore Map`.
- [x] `Chat` gọi Gemma general-chat đúng một lần và không thể gọi SQLite, CLIP,
  retrieval hoặc top-k verification.
- [x] `Explore Map` luôn giữ pipeline M8.2; `Auto` dùng rule router trước và chỉ
  dùng Gemma classifier cho câu chưa rõ channel.
- [x] General chat có prompt riêng, lịch sử tối đa 10 message trong browser và
  không nhận dữ liệu map; model hướng người dùng sang Explore Map khi cần facts.
- [x] Web UI dùng chat bubbles và hiển thị provenance manual/rule/classifier,
  general chat hoặc map retrieval.
- [x] Ollama giữ model warm 30 phút và giới hạn general-chat generation để giảm
  thời gian chờ; lỗi Ollama có fallback không bịa map facts.

Kết quả ngày 2026-10-14: `Bạn có thể làm được gì?` ở Chat và Auto đều đi
`general_chat`, trả `retrieval: null`; `Where is the red container?` ở Explore
Map đi M8.2 và trả `object_000001`. M8.2.1 không lưu conversation vào database,
không thay đổi artifact M1–M8 và chưa bắt đầu M9.

### M9 - Mở rộng toàn greenhouse

- Thêm nhiều Aisle polygon.
- Xử lý head/tail cross-aisles và junction.
- Association không được merge object xuyên Aisle.
- Đánh giá hierarchical retrieval theo Aisle.

## 13. Tiêu chí pilot thành công

1. Mọi View hợp lệ có RGB, aligned depth, calibration và pose trong
   `camera_init`.
2. Khoảng cách giữa các View liên tiếp xấp xỉ 0.5 m, trừ keyframe do quay góc.
3. View lượt đi và lượt về được giữ như các node riêng.
4. SAM3 xử lý toàn bộ View và không bỏ frame do inference backlog.
5. Observation 3D nằm đúng vị trí object trên LiDAR map.
6. Cùng một object xuất hiện ở hai lượt chạy được hợp nhất thành một Object.
7. Hai object khác nhau không bị merge chỉ vì cùng label.
8. Object được phân loại đúng `head`, `middle` hoặc `tail`.
9. Gemma 4 tạo JSON caption hợp lệ và có provenance.
10. Query trả đúng Object ID, representative View và centroid 3D.

Các metric chính:

```text
View extraction coverage
RGB-depth-pose synchronization failure rate
SAM3 detection precision/recall trên sample được gán nhãn
3D localization error
Object false-merge rate
Object false-split rate
Object retrieval Top-1/Top-5
VLM verification accuracy
End-to-end query latency
```

## 14. Rủi ro và biện pháp

### Odometry drift

Hai lượt đi/về có thể tạo centroid lệch nhau. Trước khi tăng association radius,
cần kiểm tra trajectory loop consistency trên map.

### Keyframe quá dày

0.5 m có thể tạo nhiều View và tăng chi phí SAM3/VLM. Giữ toàn bộ View cho
geometry nhưng chỉ caption representative/candidate Views nếu cần giảm chi phí.

### Keyframe bỏ mất góc quan trọng

Luôn kết hợp ngưỡng khoảng cách với ngưỡng góc quay.

### SAM3 prompt bias

Lưu prompt/version và ground-truth một sample nhỏ để đo precision/recall.

### VLM hallucination

Không dùng VLM làm nguồn tọa độ, kích thước metric, Aisle ID hoặc quyết định
navigation safety. Lưu trường `uncertain_attributes` và bắt buộc validate JSON.

### Object false merge/split

Không chỉ association bằng label và XY. Kết hợp geometry, visual embedding,
size và multi-view consistency.

## 15. Việc cần xác nhận trước khi code

- Topic pose chính thức dùng `/Odometry` hay transform cụ thể trong `/tf`.
- Transform chính xác từ camera optical frame sang `camera_init`.
- Depth encoding và đơn vị của aligned depth.
- Polygon/centerline/head/tail của `aisle_01`.
- GPU/VRAM dùng chung giữa SAM3 và Gemma 4.
- Gemma 4 tag/quantization cụ thể trong Ollama.
- Model embedding: CLIP hay SigLIP và phiên bản cụ thể.
- Danh sách attribute/KPI chính thức của dự án NVIDIA, đặc biệt định nghĩa
  `physical object` và `traversability`.

## 16. Bước triển khai ngay tiếp theo

Ưu tiên theo thứ tự:

1. Xác nhận pose và transform chain trong bag.
2. Tạo `greenhouse_layout.yaml` cho một aisle.
3. Viết Keyframe Extractor 0.5 m + 15 độ.
4. Xuất `views.jsonl` và kiểm tra trực quan trên RViz.
5. Chuyển SAM3 hiện tại sang chế độ xử lý offline theo View dataset.

Không bắt đầu Gemma 4 trước khi M2-M5 ổn định, vì caption tốt không sửa được lỗi
đồng bộ pose, projection hoặc object association.
