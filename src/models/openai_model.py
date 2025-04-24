import json, base64

from .base import BaseModel
from openai import OpenAI
from pathlib import Path

class OpenAIModel(BaseModel):
    '''Evaluation for models using OpenAI-style APIs.'''
    
    def __str__(self):
        return "openai"
    
        
    def __init__(self, model: str, engine=OpenAI, **params):
        """
        Args:
            model (str): The model to use (e.g. "gpt-4.1").
            engine (callable): The engine to use for the model (default: OpenAI).
            **params: Additional arguments for the model (temperature, max_tokens, etc.)
        """
        super().__init__()
        self.client = engine()
        
        self.model = model
        self.params = params
        
        self.system = [{
            "role": "system",
            "content": params["system_prompt"],
        }] if 'system_prompt' in params else []
    
        
    def eval_single(self, prompt: str, image_url: str) -> str:
        """
        Return the model's response to the prompt and image.
        Args:
            prompt (str): The prompt to ask the model.
            image_url (str): The url of the image to ask the model about.
            
        Returns:
            str: The model's response.
        """
        try:
            if image_url.startswith("data:"):
                img_msg = {"type": "input_image", "image_url": image_url, "detail": "high"}
            else:
                img_msg = {
                    "type": "image_url",
                    "image_url": image_url,
                    "detail": "high"
            }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.system + [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        img_msg,
                    ],
                }],
                **self.params
            )
        except Exception as e:
            raise Exception(f"Error: {e}")
        return response.choices[0].message["content"]


    def eval_batch(self, output_dir: Path, prompts: list, image_urls: list) -> list:
        """
        Send a batch request to the model with multiple prompts and image URLs.
        Args:
            output_dir (pathlib.Path): The path to the directory for input.jsonl.
            prompts (list): The prompts to ask the model.
            image_urls (list): The urls of the images to ask the model about.
            
        Returns:
            []: Empty list, results must be retrieved manually.
        """
        assert len(prompts) == len(image_urls), "Number of prompts and images must match."
        
        with open(output_dir / "input.jsonl", "w") as f:
            for i, (prompt, image_url) in enumerate(zip(prompts, image_urls)):
                if image_url.startswith("data:"):
                    img_msg = {"type": "input_image", "image_url": image_url, "detail": "high"}
                else:
                    img_msg = {
                        "type": "image_url",
                        "image_url": image_url,
                        "detail": "high"
                }

                message = {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": self.system + [{
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                img_msg,
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
        batch = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Running evaluation for batch input",
            }
        )
        
        print(
            "Batch input file created and batch started. Batch ID:", 
            batch.id, "File ID:", batch_input_file_id, "Retrieve it in 24 hours."
        )
        
        return []
    
    
    def process_image(self, dataset_path: Path, image_url: str=None, file_name: str=None) -> str:
        """
        Process the image and return its URL.
        
        Args:
            dataset_path (pathlib.Path): The path to the dataset directory.
            image_url (str): The URL of the image to process.
            file_name (str): The path to the image file to process.
            
        Returns:
            str: The processed image URL.
        """
        if image_url:
            return image_url
        
        elif file_name:
            # look in the `images/` sub‑dir
            file_path = dataset_path / "images" / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Could not find image at {file_path}")
            with open(file_path, "rb") as f:
                return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
            
        else:
            raise ValueError("Either image_url or image_path must be provided.")