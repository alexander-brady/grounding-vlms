import json

from .base import BaseModel
from openai import OpenAI
from pathlib import Path

class OpenAIModel(BaseModel):
    '''Evaluation for models using OpenAI-style APIs.'''
    def __init__(self, model: str, engine=OpenAI, **params):
        """
        Args:
            model (str): The model to use (e.g. "gpt-4.1").
            engine (callable): The engine to use for the model (default: OpenAI).
            **params: Additional arguments for the model (temperature, max_tokens, etc.)
        """
        self.client = engine()
        
        self.model = model
        self.params = params
        
        self.system = [{
            "role": "system",
            "content": self.system_prompt,
        }] if 'system_prompt' in params else []
    
        
    def eval(self, prompt: str, image_url: str) -> str:
        """
        Return the model's response to the prompt and image.
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
    
    
    def eval_batch(self, output_dir: Path, prompts: list, image_urls: list) -> list:
        """
        Send a batch request to the model with multiple prompts and image URLs.
        Args:
            output_dir (pathlib.Path): The path to the directory for input.jsonl.
            prompts (list): The prompts to ask the model.
            image_urls (list): The urls of the images to ask the model about.
            
        Returns:
            list: The model's responses.
        """
        if len(prompts) != len(image_urls):
            raise ValueError("The number of prompts and image URLs must be the same.")
        
        with open(output_dir / "input.jsonl", "w") as f:
            for i, (prompt, image_url) in enumerate(zip(prompts, image_urls)):
                message = {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": self.system + [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        }],
                        **self.params
                    }
                }
                f.write(json.dumps(message) + "\n")
                
        batch_input_file = self.client.files.create(
            file=open(output_dir / "input.jsonl", "rb"),
            purpose="batch"
        )
        
        batch_input_file_id = batch_input_file.id
        self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Running evaluation for batch input",
            }
        )
        
        print(
            "Batch input file created and batch started. Batch ID:", 
            batch_input_file_id, "Retrieve it in 24 hours."
        )
        
        return batch_input_file_id