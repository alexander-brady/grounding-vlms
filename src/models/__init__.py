from .openai_model import OpenAIModel

def load_model(model_name: str, **kwargs):
    """
    Load the specified model.
    Args:
        model_name (str): The name of the model to load.
        **kwargs: Additional arguments to pass to the model constructor.
    Returns:
        BaseModel: An instance of the specified model.
    Raises:
        ValueError: If the model name is not supported.
    """
    model_map = {
        "openai": OpenAIModel,
        # Add other models here as needed
    }
    
    if model_name not in model_map:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {list(model_map.keys())}.")
    
    return model_map[model_name](**kwargs)