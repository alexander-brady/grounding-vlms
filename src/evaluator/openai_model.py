import base64, time, requests
from pathlib import Path
from pydantic import BaseModel

from .base import Evaluator, BaseDataset


class ObjectCount(BaseModel):
    '''Model response format, constraining output to integers'''
    count: int
    
    
class OpenAIDataset(BaseDataset):
    '''Dataset for OpenAI specific prompts.'''
    def __init__(self, df, force_download: bool=False, **kwargs):
        """
        Args:
            df (pd.DataFrame): The DataFrame containing the dataset.
            force_download (bool): Whether to force download the images.
            **kwargs: Additional arguments for the dataset.
        """
        super().__init__(df, **kwargs)
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
    
    def __init__(self, 
        model: str, 
        client: object,
        force_download: bool = False, 
        structured_output: bool = True,
        **params
    ):
        """
        Args:
            model (str): The model to use (e.g. "gpt-4.1").
            client: The client to use for the model (default: OpenAI, set in __init__.py).
            force_download (bool): Whether to force download the images from url (needed for e.g. Gemini).
            structured_output (bool): Whether to use structured output to constrain the output to integers.
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
        self.structured_output = structured_output
        self.model = model
        self.params = {
            key: value for key, value 
            in params.items() if key in openai_params
        }        
        
    @staticmethod
    def max_batch_size(items: list) -> int:
        # Limit is 50000 or 209715200b
        return min(45000, len(items))
    
    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1):
        super().eval(
            dataset_dir, 
            result_file, 
            batch_size,
            Container=OpenAIDataset,
            force_download=self.force_download
        )
    
    def eval_single(self, prompts: list) -> str:
        """
        Return the model's response to the prompt and image. 
        Args:
            prompts (list): The prompts to ask the model.
            
        Returns:
            str: The model's response.
        """
        try:
            if not self.structured_output:
                # If the model does not support structured output,
                # we need to use the default response format.
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=prompts,
                    **self.params
                ).choices[0].message.content               
                
            # Else we use parsing to constrain the output to the expected format.
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=prompts,
                response_format=ObjectCount,
                **self.params
            ).choices[0].message
            
        except Exception as e:
            return  'ERROR: ' + str(e)
        
        if response.refusal:
            return 'ERROR: (Refusal) ' + response.refusal
        
        elif response.parsed is None:
            return 'ERROR: Output is None'
        
        elif response.parsed.count is None:
            return 'ERROR: Output count is None'
        
        return response.parsed.count

    def eval_batch(self, batch: list) -> list:
        '''To work with rate limits, we slow down the batch sending.'''
        curr_time = time.time()
        res = super().eval_batch(batch)
        
        elapsed_time = time.time() - curr_time
        wait_time = max(0, 1 - elapsed_time)
        time.sleep(wait_time)
        return res