from pathlib import Path


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