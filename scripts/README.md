# Command-line scripts

Scripts are grouped by responsibility:

- `data/` prepares CrowdPose data and generates model-input caches.
- `training/` launches canonical model training and sweep jobs.
- `evaluation/` runs benchmarks and generates prediction caches for evaluation.
- `investigation/` contains probes and exploratory inspection tools.
- `utils/` contains environment and command helpers.
- `legacy/` contains retained examples that are not part of the active
  architecture.

Run commands from the repository root. Training scripts intentionally still
contain duplicated setup and lifecycle code; extracting shared training
infrastructure is a separate follow-up refactor.
