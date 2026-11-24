# NVIDIA Three-Aisle HMSG Checkpoint Runbook

Current pipeline result: three aisles, six associated objects, Gemma 4 12B
semantic enrichment, HMSG database, CLI retrieval, and local Web UI.

## Paths

```text
Bag:
/home/orc/SAM3/ros2_ws/dataset/nvidia_data_20261029_145306_3_aisle_multi_object_humble

Dataset:
/home/orc/SAM3/ros2_ws/datasets/greenhouse_20261029_three_aisles

HMSG database:
/home/orc/SAM3/ros2_ws/datasets/greenhouse_20261029_three_aisles/database/greenhouse_hmsg.sqlite
```

## Start the HMSG Web UI

The Web command checks `http://127.0.0.1:11435` automatically. If Ollama is
not running, it starts a CUDA Ollama server, waits for it to become ready, and
then starts FastAPI. If Ollama was already running, Web reuses it. An Ollama
process started by Web is stopped when Web exits.

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run conceptgraphs_bringup query_hmsg_web \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20261029_three_aisles \
  --host 0.0.0.0 \
  --port 8080
```

Open locally:

```text
http://localhost:8080
```

From another computer on the same network:

```text
http://<being2-IP>:8080
```

Keep the server terminal open while using the Web UI. Stop it with `Ctrl+C`.

## CLI retrieval smoke test

```bash
cd /home/orc/SAM3/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run conceptgraphs_bringup query_hmsg \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20261029_three_aisles \
  --query "Find the wheelbarrow in aisle 3" \
  --top-k 5
```

Expected result: `object_000005`, `wheelbarrow`, `aisle_03`.

## Recreate the reasoning demo video

```bash
cd /home/orc/SAM3/ros2_ws

python3 src/conceptgraphs_bringup/scripts/create_reasoning_demo_video.py \
  --dataset /home/orc/SAM3/ros2_ws/datasets/greenhouse_20261029_three_aisles \
  --sam-video /home/orc/SAM3/ros2_ws/demo_videos/sam3_segmentation_rebuilt_raw.mp4 \
  --depth-video /home/orc/SAM3/ros2_ws/demo_videos/depth_camera.mp4 \
  --output /home/orc/SAM3/ros2_ws/demo_videos/greenhouse_reasoning_demo_4x.mp4 \
  --speed 4 \
  --pause-sec 5 \
  --fps 30
```

