# Playbook: Onboard a New Greenhouse Dataset

1. **Place the processed MCAP bag** under `data/bags/<bag_name>/` (or set `VLM_BAGS_DIR`).
2. **Create an aisle layout YAML** in `ros2_ws/src/conceptgraphs_bringup/config/layouts/<dataset_name>.yaml` defining `greenhouse`, `frame_id: camera_init`, and aisle polygons/centerlines.
3. **Create a dataset override config** in `ros2_ws/src/conceptgraphs_bringup/config/datasets/<dataset_name>.yaml` extending `../keyframe_extractor.yaml` with unique `dataset_id`, `bag_path`, `output_dir` (`data/datasets/<dataset_id>`), and `layout_path`.
4. **Add an isolation assertion** in `ros2_ws/src/conceptgraphs_bringup/test/test_dataset_configs.py` verifying inputs/outputs do not collide with existing datasets.
5. **Rebuild & run stages M2 → M7** (see `docs/RUNBOOK.md`), then verify with:
   ```bash
   make test
   make validate-db DATASET=/home/duc/VLM/data/datasets/<dataset_id>
   ```
6. **Update `docs/HANDOFF.md`** with the new dataset stats and validation outcome.
