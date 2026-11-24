# ROS 2 Workspace (`ros2_ws`)

This workspace contains the `conceptgraphs_bringup` ROS 2 Humble package.

See the repository root [README.md](../README.md), [AGENTS.md](../AGENTS.md), and [docs/RUNBOOK.md](../docs/RUNBOOK.md) for current instructions.

## Quick Commands (run from repository root)

```bash
make build          # colcon build --symlink-install --packages-select conceptgraphs_bringup
make test           # Run unit tests
make validate-db    # Validate HMSG SQLite DB
make web            # Start the local HMSG Web UI
```

Never edit files under `build/` or `install/` directly; edit `src/conceptgraphs_bringup/` and rebuild with `make build`.
