# Runbook — Build, Test, Query & Visualize

All standard commands can be run from `/home/duc/VLM` via `Makefile`.

## 1. Build & Test

```bash
make build          # colcon build --symlink-install --packages-select conceptgraphs_bringup
make test           # Run pytest suites (skips test_query_web.py if fastapi/uvicorn absent)
make test-all       # Run all 13 pytest suites (requires requirements-web.txt)
```

To install Web UI dependencies:
```bash
python3 -m pip install --user -r ros2_ws/src/conceptgraphs_bringup/requirements-web.txt
```

## 2. Database Validation & Structured Queries

```bash
# Validate active 3-aisle dataset
make validate-db

# Validate another dataset
make validate-db DATASET=/home/duc/VLM/data/datasets/greenhouse_20260910_aisle01

# List mapped objects
make query-db
```

Direct ROS 2 CLI examples (after `source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash`):
```bash
ros2 run conceptgraphs_bringup query_hmsg_database \
  --dataset /home/duc/VLM/data/datasets/greenhouse_20261029_three_aisles \
  --get-object object_000005 --format json

ros2 run conceptgraphs_bringup query_hmsg \
  --dataset /home/duc/VLM/data/datasets/greenhouse_20261029_three_aisles \
  --query "Find the wheelbarrow in aisle 3" --top-k 5
```

## 3. Local Interactive Web UI (M8.2.1)

Requires `requirements-web.txt` and Ollama (auto-started on `http://127.0.0.1:11435` if configured and installed):
```bash
make web
# Or bind all interfaces:
make web HOST=0.0.0.0 PORT=8080
```
Open `http://127.0.0.1:8080`.

## 4. Offline Pipeline Stages (For New Datasets)

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
export VLM_ROOT=/home/duc/VLM
export VLM_WORKER_PYTHON=/path/to/env/bin/python   # if SAM3/CLIP live in a dedicated venv/conda

ros2 run conceptgraphs_bringup extract_keyframes --config ros2_ws/src/conceptgraphs_bringup/config/datasets/greenhouse_20261029_three_aisles.yaml
ros2 run conceptgraphs_bringup run_sam3_offline --dataset /home/duc/VLM/data/datasets/<dataset_id>
ros2 run conceptgraphs_bringup project_observations_3d --dataset /home/duc/VLM/data/datasets/<dataset_id>
ros2 run conceptgraphs_bringup associate_objects --dataset /home/duc/VLM/data/datasets/<dataset_id>
ros2 run conceptgraphs_bringup enrich_semantics --dataset /home/duc/VLM/data/datasets/<dataset_id>
ros2 run conceptgraphs_bringup build_hmsg_database --dataset /home/duc/VLM/data/datasets/<dataset_id>
```

## 5. RViz Dataset Visualization (No Bag Required)

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch conceptgraphs_bringup view_dataset.launch.py \
  views_path:=/home/duc/VLM/data/datasets/greenhouse_20261029_three_aisles/keyframes/views.jsonl \
  layout_path:=/home/duc/VLM/ros2_ws/src/conceptgraphs_bringup/config/layouts/greenhouse_20261029_three_aisles.yaml
```
