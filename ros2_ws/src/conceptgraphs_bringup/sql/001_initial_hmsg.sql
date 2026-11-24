PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_versions (
  version INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
  dataset_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL,
  frame_id TEXT NOT NULL,
  source_bag TEXT,
  manifest_json TEXT NOT NULL,
  dataset_root TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS greenhouses (
  greenhouse_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS aisles (
  aisle_id TEXT PRIMARY KEY,
  greenhouse_id TEXT NOT NULL REFERENCES greenhouses(greenhouse_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS views (
  view_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  greenhouse_id TEXT NOT NULL REFERENCES greenhouses(greenhouse_id),
  aisle_id TEXT NOT NULL REFERENCES aisles(aisle_id),
  timestamp_ns INTEGER NOT NULL,
  frame_id TEXT NOT NULL,
  camera_frame_id TEXT,
  rgb_path TEXT NOT NULL,
  depth_path TEXT NOT NULL,
  camera_info_path TEXT NOT NULL,
  pose_json TEXT NOT NULL,
  body_pose_json TEXT,
  aisle_coordinate_s REAL,
  aisle_section TEXT,
  travel_pass TEXT,
  processing_status TEXT,
  source_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
  observation_id TEXT PRIMARY KEY,
  view_id TEXT NOT NULL REFERENCES views(view_id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  aisle_id TEXT NOT NULL REFERENCES aisles(aisle_id),
  timestamp_ns INTEGER,
  prompt TEXT,
  canonical_label TEXT NOT NULL,
  sam_confidence REAL,
  bbox_2d_json TEXT,
  mask_path TEXT,
  crop_path TEXT,
  context_crop_path TEXT,
  overlay_path TEXT,
  projection_status TEXT NOT NULL,
  frame_id TEXT,
  centroid_3d_json TEXT,
  bbox_3d_min_json TEXT,
  bbox_3d_max_json TEXT,
  median_depth_m REAL,
  valid_depth_ratio REAL,
  point_count INTEGER,
  rejection_reason TEXT,
  model_version TEXT,
  prompt_version TEXT,
  source_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
  object_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  greenhouse_id TEXT NOT NULL REFERENCES greenhouses(greenhouse_id),
  aisle_id TEXT NOT NULL REFERENCES aisles(aisle_id),
  canonical_label TEXT NOT NULL,
  frame_id TEXT NOT NULL,
  centroid_3d_json TEXT NOT NULL,
  bbox_3d_min_json TEXT,
  bbox_3d_max_json TEXT,
  position_quality TEXT,
  association_status TEXT,
  representative_view_id TEXT REFERENCES views(view_id),
  representative_observation_id TEXT REFERENCES observations(observation_id),
  observation_count INTEGER,
  cluster_diameter_m REAL,
  source_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS object_observations (
  object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
  observation_id TEXT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
  PRIMARY KEY (object_id, observation_id)
);

CREATE TABLE IF NOT EXISTS view_object_edges (
  edge_id TEXT PRIMARY KEY,
  view_id TEXT NOT NULL REFERENCES views(view_id) ON DELETE CASCADE,
  object_id TEXT NOT NULL REFERENCES objects(object_id) ON DELETE CASCADE,
  observation_id TEXT NOT NULL REFERENCES observations(observation_id),
  edge_type TEXT NOT NULL CHECK(edge_type = 'OBSERVES'),
  source_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS captions (
  caption_id TEXT PRIMARY KEY,
  owner_type TEXT NOT NULL CHECK(owner_type IN ('object','view')),
  owner_id TEXT NOT NULL,
  status TEXT NOT NULL,
  model_tag TEXT,
  model_digest TEXT,
  prompt_version TEXT,
  category TEXT,
  color TEXT,
  material TEXT,
  semantic_class TEXT,
  path_assessment TEXT,
  confidence REAL,
  scene_summary TEXT,
  parsed_json TEXT,
  raw_response_path TEXT,
  input_image_paths_json TEXT,
  input_ids_json TEXT,
  source_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traversability_assessments (
  view_id TEXT PRIMARY KEY REFERENCES views(view_id) ON DELETE CASCADE,
  status TEXT NOT NULL,
  passability TEXT,
  can_robot_pass INTEGER,
  blocking_object_ids_json TEXT NOT NULL,
  traversable_object_ids_json TEXT NOT NULL,
  confidence REAL,
  model_tag TEXT,
  model_digest TEXT,
  prompt_version TEXT,
  source_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_indexes (
  embedding_id TEXT PRIMARY KEY,
  owner_type TEXT NOT NULL CHECK(owner_type IN ('view','object')),
  owner_id TEXT NOT NULL,
  entry_type TEXT NOT NULL,
  npy_path TEXT NOT NULL,
  vector_index INTEGER NOT NULL,
  dimension INTEGER NOT NULL,
  dtype TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  model_sha256 TEXT,
  input_paths_json TEXT NOT NULL,
  UNIQUE(npy_path, vector_index)
);

CREATE TABLE IF NOT EXISTS processing_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
  stage TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  details_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_views_aisle ON views(aisle_id);
CREATE INDEX IF NOT EXISTS idx_observations_view ON observations(view_id);
CREATE INDEX IF NOT EXISTS idx_objects_aisle ON objects(aisle_id);
CREATE INDEX IF NOT EXISTS idx_edges_object ON view_object_edges(object_id);
CREATE INDEX IF NOT EXISTS idx_edges_view ON view_object_edges(view_id);
CREATE INDEX IF NOT EXISTS idx_captions_owner ON captions(owner_type, owner_id);
