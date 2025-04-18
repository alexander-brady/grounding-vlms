from .base import BaseModel
from openai import OpenAI

class OpenAIModel(BaseModel):
    def __init__(self, model: str, **params):
        """
        Args:
            model (str): The model to use (e.g. "gpt-4.1").
            **params: Additional arguments for the model (temperature, max_tokens, etc.)
        """
        super().__init__()
        self.client = OpenAI()
        
        self.model = model
        self.params = params
        
        self.system = [{
            "role": "system",
            "content": self.system_prompt,
        }] if 'system_prompt' in params else []
    
        
    def eval(self, prompt: str, image_url: str) -> str:
        """
        Args:
            prompt (str): The prompt to ask the model.
            image_url (str): The url of the image to ask the model about.
            
        Returns:
            str: The model's response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.system + [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }],
                **self.params
            )
        
        except Exception as e:
            raise Exception(f"Error: {e}")
        
        return response.choices[0].message['content']      
        