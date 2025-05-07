import torch
from transformers import pipeline

from .base import Evaluator

class HuggingFaceModel(Evaluator):
    '''Evaluation for models from huggingface.co.'''
    
    def __str__(self):
        return "huggingface"
    
    def __init__(self, model: str, **params):
        """
        Args:
            model (str): The model to use (e.g. "google/flan-t5-xxl").
            **params: Additional arguments for the model (processor, system_prompt, max_tokens, etc.)
        """
        super().__init__(params.pop("system_prompt", None))
        self.model = pipeline(
            "image-text-to-text",
            model=model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        self.params = params
        
    def eval_batch(self, batch: list) -> list:
        """Evaluate a batch of prompts and images."""
        indices, prompts = zip(*batch)
        outputs = self.model(
            text=prompts,
            batch_size=len(prompts), 
            return_full_text=False, 
            **self.params
        )
        return [ 
                (idx, out[0]['generated_text'])
                for idx, out in zip(indices, outputs)
            ]