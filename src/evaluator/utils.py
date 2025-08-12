from typing import Union, Optional, Dict, Any
from word2number import w2n
from omegaconf import DictConfig, OmegaConf

    
def cfg_to_dict(cfg_node: Optional[DictConfig]) -> Dict[str, Any]:
    """Convert a DictConfig object to a regular dictionary."""
    return {} if cfg_node is None else OmegaConf.to_container(cfg_node, resolve=True)
    
    
def intify(result: Union[str, int]) -> str:
    """Turns the model output into an integer. Returns -1 if it fails."""
    if type(result) == int:
        return str(result)
    
    if result.startswith("ERROR"):
        return '-1'
    
    result = result.replace("-", " ").replace(",", "").split(".")[0].lower().strip()
    if result.isdigit():
        return result
    
    digits = ''.join(filter(str.isdigit, result))
    if digits:
        return digits
    
    try:
        return str(w2n.word_to_num(result))
    
    except ValueError:
        return '-1'