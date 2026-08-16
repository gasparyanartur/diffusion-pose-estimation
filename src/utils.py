from __future__ import annotations

from pathlib import Path


def find_last_run(sweep_dir: Path) -> dict[int, Path]:
    """Return the most recent run directory for each task in a sweep.

    Expects the layout::

        sweep_dir/
          task_1/
            20260501_120000/   ← run directories (YYYYMMDD_HHMMSS)
            20260502_093000/
          task_2/
            ...

    Returns a dict mapping task_id (int) to the most recent run Path.
    Tasks with no run subdirectories are omitted.
    """
    result: dict[int, Path] = {}
    for task_dir in sorted(sweep_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        try:
            task_id = int(task_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        run_dirs = sorted(d for d in task_dir.iterdir() if d.is_dir())
        if run_dirs:
            result[task_id] = run_dirs[-1]
    return result


def resolve_checkpoint(raw: str) -> str:
    """Resolve a checkpoint path that may contain ``...`` as a wildcard segment.

    ``...`` matches exactly one directory level, just like ``*`` in a glob.
    The most recently modified match is returned.  Raises ``FileNotFoundError``
    when nothing matches.

    Example::

        results/backbone_benchmark/backbone_benchmark/.../checkpoints/last.ckpt
        # expands to e.g.
        results/backbone_benchmark/backbone_benchmark/20260507_105549/checkpoints/last.ckpt
    """
    if "..." not in raw:
        return raw
    # Replace every occurrence of /.../ with /*/ for pathlib glob
    glob_pattern = raw.replace(".../", "*/")
    # Handle trailing ... (unlikely but tidy)
    glob_pattern = glob_pattern.replace("...", "*")
    import glob as _glob

    matches = sorted(
        (Path(p) for p in _glob.glob(glob_pattern)),
        key=lambda p: p.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found matching pattern: {glob_pattern!r}\n"
            f"  (original path: {raw!r})"
        )
    resolved = matches[-1]  # most recently modified
    return str(resolved)
