import os

from omegaconf import DictConfig
from openai import OpenAI

from .base import Evaluator
from .hf_model import HuggingFaceModel
from .openai_model import OpenAIModel
from .vllm_model import VLLMGenerateModel, VLLMModel


def load_evaluator(model_cfg: DictConfig) -> Evaluator:
    """
    Load the specified model.
    Args:
        model_cfg (DictConfig): The configuration for the model.
    Returns:
        Evaluator: An instance of the specified model.
    Raises:
        ValueError: If the model name is not supported.
    """
    model_map = {
        "openai": OpenAIModel,
        "google": OpenAIModel,
        "anthropic": OpenAIModel,
        "xai": OpenAIModel,
        "huggingface": HuggingFaceModel,
        "vllm": VLLMModel,
        "vllm_generate": VLLMGenerateModel,
        # Add other models here as needed
    }

    if model_cfg.engine not in model_map:
        raise ValueError(
            f"Backend '{model_cfg.engine}' is not supported. Available backends: {', '.join(model_map.keys())}."
        )

    engine_params = {}
    if model_cfg.engine == "openai":
        engine_params = {"client": OpenAI()}
    elif model_cfg.engine == "google":
        engine_params = {
            "client": OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            "force_download": True,
        }
    elif model_cfg.engine == "anthropic":
        engine_params = {
            "client": OpenAI(api_key=os.getenv("ANTHROPIC_API_KEY"), base_url="https://api.anthropic.com/v1/"),
            "structured_output": False,
        }
    elif model_cfg.engine == "xai":
        engine_params = {"client": OpenAI(api_key=os.getenv("XAI_API_KEY"), base_url="https://api.x.ai/v1")}

    return model_map[model_cfg.engine](**engine_params, model_cfg=model_cfg)
