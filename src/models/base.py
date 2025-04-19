class BaseModel:
    def __init__(self):
        pass
    
    def eval(self, prompt: str, image_url: str) -> str:
        raise NotImplementedError("The eval method must be implemented by subclasses.")