from typing import Any

from omegaconf import DictConfig, OmegaConf
from word2number import w2n


def cfg_to_dict(cfg_node: DictConfig | None) -> dict[str, Any]:
    """Convert a DictConfig object to a regular dictionary."""
    return {} if cfg_node is None else OmegaConf.to_container(cfg_node, resolve=True)


def intify(result: str | int) -> str:
    """Turns the model output into an integer. Returns -1 if it fails."""
    if isinstance(result, int):
        return str(result)

    if result.startswith("ERROR"):
        return "-1"

    result = result.replace("-", " ").replace(",", "").split(".")[0].lower().strip()
    if result.isdigit():
        return result

    digits = "".join(filter(str.isdigit, result))
    if digits:
        return digits

    try:
        return str(w2n.word_to_num(result))

    except ValueError:
        return "-1"
