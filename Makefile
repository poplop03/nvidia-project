SHELL := /bin/bash
ROOT := $(CURDIR)
ROS_SETUP := /opt/ros/humble/setup.bash
PKG_DIR := $(ROOT)/ros2_ws/src/conceptgraphs_bringup
DATASET ?= $(ROOT)/data/datasets/greenhouse_20261029_three_aisles
HOST ?= 127.0.0.1
PORT ?= 8080

.PHONY: help build test test-all validate-db query-db web clean

help:
	@echo "Greenhouse HMSG targets:"
	@echo "  make build        - Build conceptgraphs_bringup with colcon (--symlink-install)"
	@echo "  make test         - Run pytest suites (skips web test if fastapi/uvicorn absent)"
	@echo "  make test-all     - Run all 13 pytest suites including FastAPI web tests"
	@echo "  make validate-db  - Validate HMSG SQLite DB integrity & paths for DATASET"
	@echo "  make query-db     - List objects in DATASET via query_hmsg_database"
	@echo "  make web          - Launch local HMSG FastAPI Web UI on HOST:PORT"
	@echo "  make clean        - Remove colcon build/install/log and Python caches"

build:
	@set -e; \
	source $(ROS_SETUP); \
	cd $(ROOT)/ros2_ws && colcon build --symlink-install --packages-select conceptgraphs_bringup

test:
	@set -e; \
	source $(ROS_SETUP); \
	export VLM_ROOT="$(ROOT)"; \
	export PYTHONPATH="$(PKG_DIR):$${PYTHONPATH}"; \
	EXTRA_ARGS=""; \
	if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then \
		echo "[make test] fastapi/uvicorn not installed in system python3; skipping test_query_web.py"; \
		EXTRA_ARGS="--ignore=$(PKG_DIR)/test/test_query_web.py"; \
	fi; \
	python3 -m pytest -p no:anyio $$EXTRA_ARGS $(PKG_DIR)/test -q

test-all:
	@set -e; \
	source $(ROS_SETUP); \
	export VLM_ROOT="$(ROOT)"; \
	export PYTHONPATH="$(PKG_DIR):$${PYTHONPATH}"; \
	python3 -m pytest -p no:anyio $(PKG_DIR)/test -q

validate-db:
	@set -e; \
	source $(ROS_SETUP); \
	export VLM_ROOT="$(ROOT)"; \
	export PYTHONPATH="$(PKG_DIR):$${PYTHONPATH}"; \
	python3 -m conceptgraphs_bringup.hmsg_database --dataset "$(DATASET)" --validate-only

query-db:
	@set -e; \
	source $(ROS_SETUP); \
	export VLM_ROOT="$(ROOT)"; \
	export PYTHONPATH="$(PKG_DIR):$${PYTHONPATH}"; \
	python3 -m conceptgraphs_bringup.hmsg_query --dataset "$(DATASET)" --list-objects

web:
	@set -e; \
	source $(ROS_SETUP); \
	source $(ROOT)/ros2_ws/install/setup.bash; \
	export VLM_ROOT="$(ROOT)"; \
	ros2 run conceptgraphs_bringup query_hmsg_web --dataset "$(DATASET)" --host "$(HOST)" --port "$(PORT)"

clean:
	rm -rf $(ROOT)/ros2_ws/build $(ROOT)/ros2_ws/install $(ROOT)/ros2_ws/log $(ROOT)/.pytest_cache
	find $(ROOT)/ros2_ws/src -type d -name '__pycache__' -exec rm -rf {} +
