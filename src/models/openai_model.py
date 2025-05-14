import base64, time, requests
from pathlib import Path
from pydantic import BaseModel

from .base import Evaluator, BaseDataset


class ObjectCount(BaseModel):
    '''Model response format, constraining output to integers'''
    count: int
    
    
class OpenAIDataset(BaseDataset):
    '''Dataset for OpenAI specific prompts.'''
    def __init__(self, force_download: bool = False, **kwargs):
        """
        Args:
            force_download (bool): Whether to force download the dataset.
            **kwargs: Additional arguments for the dataset.
        """
        super().__init__(**kwargs)
        self.force_download = force_download
    
    def __getitem__(self, idx):
        return (
            self.indices[idx],
            self.system + [{
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompts[idx]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": self.process_image(idx),
                        "detail": "high"
                    },
                }
            ],
        }])
    
    def process_image(self, idx):
        '''Turn image into model readable format.'''
        if self.using_image_urls and not self.force_download:
            return self.images[idx]
        elif self.using_image_urls:
            image_url = self.images[idx]
            response = requests.get(image_url)
            if response.status_code == 200:
                return f"data:image/jpeg;base64,{base64.b64encode(response.content).decode('utf-8')}"
            else:
                return None
        else:
            file_name = self.images[idx]
            with open(self.image_dir / file_name, "rb") as f:
                return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        

class OpenAIModel(Evaluator):
    '''Evaluation for models using OpenAI-style APIs.'''
    
    def __str__(self):
        return "openai"
    
    def max_batch_size(self, items: list) -> int:
        # Limit is 50000 or 209715200b
        return min(45000, len(items))
    
    def __init__(self, model: str, client, force_download: bool = False, **params):
        """
        Args:
            model (str): The model to use (e.g. "gpt-4.1").
            client: The client to use for the model (default: OpenAI, set in __init__.py).
            force_download (bool): Whether to force download the images from url (needed for e.g. Gemini).
            **params: Additional arguments for the model (temperature, max_completion_tokens, etc.)
        """
        super().__init__(params.pop("system_prompt", None))
        self.client = client
        
        openai_params = {
            "temperature", "frequency_penalty", 
            "max_completion_tokens", "top_p",
            "reasoning_effort", "seed", 
        }
        
        self.force_download = force_download
        self.model = model
        self.params = {
            key: value for key, value 
            in params.items() if key in openai_params
        }
    
    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1):
        super().eval(dataset_dir, result_file, batch_size, OpenAIDataset, force_download=self.force_download)
    
    def eval_single(self, prompts: list) -> str:
        """
        Return the model's response to the prompt and image. 
        Args:
            prompts (list): The prompts to ask the model.
            
        Returns:
            str: The model's response.
        """
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=prompts,
                response_format=ObjectCount,
                **self.params
            )
        except Exception as e:
            return  'ERROR: ' + str(e)
        
        if response.refusal:
            return 'ERROR: ' + response.refusal
        
        elif response.choices[0].message.parsed is None:
            return 'ERROR: Output is None'
        
        elif response.choices[0].message.parsed.count is None:
            return 'ERROR: Output count is None'
        
        return response.choices[0].message.parsed.count


    def eval_batch(self, batch: list) -> list:
        '''To work with rate limits, we slow down the batch sending.'''
        curr_time = time.time()
        res = super().eval_batch(batch)
        
        elapsed_time = time.time() - curr_time
        wait_time = max(0, 1 - elapsed_time)
        time.sleep(wait_time)
        return res