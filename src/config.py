from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    resolved = {}
    for key, val in cfg["paths"].items():
        resolved[key] = PROJECT_ROOT / val if key.endswith("_dir") else val
    resolved["panel_path"] = resolved["processed_dir"] / resolved["panel_filename"]
    cfg["resolved_paths"] = resolved
    return cfg
