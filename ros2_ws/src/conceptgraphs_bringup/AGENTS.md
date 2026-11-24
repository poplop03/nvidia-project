# Package Instructions — `conceptgraphs_bringup`

Read root `AGENTS.md` first. Package-specific rules:

- **Entry points:** All ROS 2 console scripts are declared in `setup.py` (`console_scripts`). Do not rename or remove existing entry points without updating `docs/RUNBOOK.md` and `Makefile`.
- **Package data:** Any new files under `launch/`, `config/`, `maps/`, `scripts/`, `sql/`, or `web/` must match the globs in `setup.py` `data_files` so `colcon build` installs them cleanly.
- **Path portability:** Always import helpers from `conceptgraphs_bringup.paths` rather than hardcoding absolute paths or `Path.home() / "SAM3"`.
- **Testing:** Add or update unit tests in `test/` for every behavioral change and run `make test` from the repository root.
