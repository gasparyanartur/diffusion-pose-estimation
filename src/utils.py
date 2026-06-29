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


def load_backbone_probe_module(
    run_dir: Path,
    ckpt_name: str = "last.ckpt",
    device: str = "cpu",
):
    """Rebuild a BackboneProbeModule from a benchmark_backbone run directory.

    Loads the Hydra config from ``run_dir/.hydra/config.yaml``, rebuilds the
    backbone and probe module, then loads the checkpoint weights.

    Args:
        run_dir:   Path to the timestamped run directory (contains ``.hydra/``
                   and ``checkpoints/``).
        ckpt_name: Filename of the checkpoint to load (default: ``last.ckpt``).
        device:    torch device string to map the module to.

    Returns:
        ``(module, cfg)`` where *module* is a :class:`BackboneProbeModule` in
        eval mode on *device* and *cfg* is the raw ``OmegaConf`` DictConfig.
    """
    import torch
    from omegaconf import OmegaConf

    from src.base_config import BaseConfig
    from src.data import DataConfig
    from src.features import BaseFeatureExtractorConfig

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    backbone_cfg = BaseFeatureExtractorConfig.from_hydra_config(cfg.backbone)
    backbone = backbone_cfg.build()

    data_cfg = DataConfig.from_hydra_config(cfg.data)
    data = data_cfg.build()

    probe_cfg = BaseConfig.from_hydra_config(cfg.training_module)
    module = probe_cfg.build(backbone, data=data)

    ckpt_path = run_dir / "checkpoints" / ckpt_name
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state["state_dict"]
    # Backward compat: probe_head.stem_proj was renamed to probe_head.proj
    sd = {k.replace("probe_head.stem_proj.", "probe_head.proj."): v for k, v in sd.items()}
    # Backward compat: probe_head.proj was a Conv2d(1×1); now it is nn.Linear.
    # Conv2d weight shape [D, C, 1, 1] → Linear weight shape [D, C].
    if "probe_head.proj.weight" in sd and sd["probe_head.proj.weight"].ndim == 4:
        sd["probe_head.proj.weight"] = sd["probe_head.proj.weight"].squeeze(-1).squeeze(-1)
    # Backward compat: concat backbone used to emit "tokens"; now emits "spatial".
    # Patch the probe's feature_key so inference works with the updated backbone.
    if hasattr(module, "probe_head") and getattr(module.probe_head, "feature_key", None) == "tokens":
        module.probe_head.feature_key = "spatial"

    # Backbone weights are not saved in checkpoints (frozen pretrained model).
    # Inject them from the freshly built module so strict loading works.
    # Use nn.Module.state_dict directly to bypass BackboneProbeModule's override.
    import torch.nn as nn
    backbone_sd = {k: v for k, v in nn.Module.state_dict(module).items() if k.startswith("backbone.")}
    sd = {**backbone_sd, **sd}
    module.load_state_dict(sd, strict=True)
    module.to(device)
    module.eval()

    return module, cfg


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