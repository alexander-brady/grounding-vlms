import os
from openai import OpenAI

from .base import Evaluator
from .hf_model import HuggingFaceModel
from .openai_model import OpenAIModel


def load_evaluator(engine_name: str, **kwargs) -> Evaluator:
    """
    Load the specified model.
    Args:
        engine_name (str): The name of the model to load.
        **kwargs: Additional arguments to pass to the model constructor.
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
        "countgd": CountGDModel,
        # Add other models here as needed
    }
    
    if engine_name not in model_map:
        raise ValueError(f"Backend '{engine_name}' is not supported. Available backends: {', '.join(model_map.keys())}.")
    
    engine_params = {
        "openai": { 'client': OpenAI() },
        "google": {
            'client': OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
            ),
            'force_download': True
        },
        "anthropic": {
            'client': OpenAI(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                base_url='https://api.anthropic.com/v1/'
            ),
            'structured_output': False
        },
        "xai": {
            'client': OpenAI(
                api_key=os.getenv("XAI_API_KEY"),
                base_url='https://api.x.ai/v1'
            )
        },
    }.get(engine_name, {})
    
    return model_map[engine_name](**engine_params, **kwargs)