import json, base64
from pydantic import BaseModel

from .base import Evaluator
from openai import OpenAI
from pathlib import Path

class ObjectCount(BaseModel):
    count: int

class OpenAIModel(Evaluator):
    '''Evaluation for models using OpenAI-style APIs.'''
    
    def __str__(self):
        return "openai"
    
        
    def __init__(self, model: str, client=OpenAI, system_prompt: str=None, **params):
        """
        Args:
            model (str): The model to use (e.g. "gpt-4.1").
            client (callable): The client to use for the model (default: OpenAI).
            system_prompt (str): The system prompt to use for the model.
            **params: Additional arguments for the model (temperature, max_completion_tokens, etc.)
        """
        super().__init__()
        self.client = client()
        
        openai_params = {
            "temperature", "frequency_penalty", 
            "max_completion_tokens", "top_p",
            "reasoning_effort", "seed", 
        }
        
        self.model = model
        self.params = {
            key: value for key, value 
            in params.items() if key in openai_params
        }
        
        self.system = [{
            "role": "system",
            "content": system_prompt
        }] if system_prompt else []
    
    
    def api_body(self, prompt: str, image_url: str) -> dict:
        """
        Create the body for the API request.
        Args:
            prompt (str): The prompt to ask the model.
            image_url (str): The url of the image to ask the model about.
        Returns:
            list: The messages to send to the model.
        """
        messages = self.system + [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    },
                }
            ],
        }]
        
        return {
            "model": self.model,
            "messages": messages,
            "response_format": ObjectCount,
            **self.params
        }
        
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
            response = self.client.chat.completions.parse(
                **self.api_body(prompt, image_url)
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

        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_file = output_dir / "input.jsonl"
        if batch_file.exists():
            batch_file.unlink()
            
        with open(batch_file, "w") as f:
            for i, (prompt, image_url) in enumerate(zip(prompts, image_urls)):
                message = {
                    "custom_id": str(i),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": self.api_body(prompt, image_url)
                }
                f.write(json.dumps(message) + "\n")
                
        batch_input_file = self.client.files.create(
            file=open(batch_file, "rb"),
            purpose="batch"
        )
        
        batch_file.unlink()
        
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
            "Batch input file created and batch started.",
            "\nBatch ID:", batch.id, 
            "\nFile ID:", batch_input_file_id, 
            "\nRetrieve in 24 hours. (see src/retrieval.ipynb)"
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