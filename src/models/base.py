from dotenv import load_dotenv

class BaseModel:
    def __init__(self):
        load_dotenv()
    
    def eval(self, prompt: str, image_url: str) -> str:
        raise NotImplementedError("The eval method must be implemented by subclasses.")