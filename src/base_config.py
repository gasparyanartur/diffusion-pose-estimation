# Artur Gasparyan, 2026

from __future__ import annotations

from abc import ABC
from functools import lru_cache
import logging
import os
from pathlib import Path
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from typing import Any, Generic, Mapping, TypeVar, cast
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError
from hydra.utils import instantiate
import torch

DEFAULT_PROJECT_NAME = "posegraphdiff"


def flatten_configs(configs: Mapping[str, Any] | BaseModel, prefix="") -> dict[str, Any]:
    flat_configs = {}

    if isinstance(configs, BaseModel):
        configs = configs.model_dump(mode="json")

    for key, value in configs.items():
        if isinstance(value, Mapping):
            value_dict = value
            flat_configs[prefix + key] = type(value)
            flattened_dict = flatten_configs(value_dict, prefix=f"{prefix}{key}.")
            flat_configs.update(flattened_dict)
        else:
            flat_configs[prefix + key] = value

    return flat_configs

C = TypeVar("C", bound=BaseModel)


def _instantiate_targets(obj: Any) -> Any:
    """Recursively instantiate Hydra nodes that declare a target class.

    Supports both Hydra's `_target_` and the legacy `__target__` key.
    """
    if isinstance(obj, dict):
        if "_target_" in obj or "__target__" in obj:
            payload = dict(obj)
            if "__target__" in payload and "_target_" not in payload:
                payload["_target_"] = payload.pop("__target__")
            return instantiate(OmegaConf.create(payload), _convert_="all")

        return {k: _instantiate_targets(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_instantiate_targets(v) for v in obj]

    return obj



class BaseConfig(BaseModel, ABC, Generic[C]):
    @classmethod
    def from_hydra_config(cls, cfg: DictConfig, instantiate: bool = True) -> C:
        logging.info(f"Constructing from Hydra config " + str(cfg))
        raw_dict = OmegaConf.to_container(cfg, resolve=True)

        if not isinstance(raw_dict, dict):
            raise ValueError("Config must be a dictionary")

        if not all(isinstance(k, str) for k in raw_dict):
            raise ValueError("Config keys must be strings")

        raw_dict = cast(Mapping[str, Any], raw_dict)
        
        if instantiate:
            raw_dict = _instantiate_targets(raw_dict)

        # If the root node carried `_target_`, the recursive instantiator
        # already produced a (subclass) instance. Accept it directly so
        # abstract base configs (e.g. DataConfig) can dispatch polymorphically.
        if isinstance(raw_dict, cls):
            return cast(C, raw_dict)

        if not isinstance(raw_dict, dict):
            raise ValueError(
                f"Expected dict or {cls.__name__} instance after instantiation, got {type(raw_dict).__name__}"
            )

        try:
            obj = cls(**raw_dict)

        except ValidationError as e:
            logging.error(f"Attempted to construct {cls.__name__} from dictionary with keys:")
            for k, v in flatten_configs(raw_dict).items():
                logging.error(f"{k}: type: {type(v)}, value: {v}")
            raise e

        return cast(C, obj)

def _resolve_workspace_dir() -> Path:
    curr_path = Path(__file__)
    logging.info(f"Resolving workspace directory, traversing up from starting from {curr_path}")

    # Find the root of the project
    while curr_path.parent != curr_path:
        curr_path = curr_path.parent
        logging.info(f"Current directory: {curr_path}")

        if (curr_path / "pyproject.toml").exists():
            logging.info("Found pyproject.toml")
            with open(curr_path / "pyproject.toml", "rb") as f:
                project_info = tomllib.load(f)

            if (
                "name" in project_info["project"]
                and project_info["project"]["name"] == DEFAULT_PROJECT_NAME
            ):
                return curr_path
            else:
                logging.warning(
                    f"Found pyproject.toml but it does not contain the correct project name. Expected '{DEFAULT_PROJECT_NAME}' but got '{project_info['project']['name']}'"
                )

    raise RuntimeError("Could not find the root of the project")


class GlobalConfig:
    WORKSPACE_DIR: Path = Path(
        os.environ.get("PROJECT_WORKSPACE_DIR", _resolve_workspace_dir())
    )
    CONFIGS_DIR: Path = WORKSPACE_DIR / "src" / "configs"
    DATA_DIR: Path = WORKSPACE_DIR / "data"


_device: torch.device | None = None


@lru_cache(maxsize=1)
def get_device_str() -> str:
    if (override_device := os.environ.get("OVERRIDE_DEVICE")) is not None:
        return override_device
        
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device(get_device_str())

    return _device