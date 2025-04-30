from .openai_model import OpenAIModel
from .hf_model import HuggingFaceModel

def load_model(engine_name: str, **kwargs):
    """
    Load the specified model.
    Args:
        engine_name (str): The name of the model to load.
        **kwargs: Additional arguments to pass to the model constructor.
    Returns:
        BaseModel: An instance of the specified model.
    Raises:
        ValueError: If the model name is not supported.
    """
    model_map = {
        "openai": OpenAIModel,
        "huggingface": HuggingFaceModel,
        # Add other models here as needed
    }
    
    if engine_name not in model_map:
        raise ValueError(f"Model '{engine_name}' is not supported. Available models: {list(model_map.keys())}.")
    
    return model_map[engine_name](**kwargs)