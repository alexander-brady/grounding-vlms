import torch
from transformers import pipeline

from .base import Evaluator

class CountGDModel(Evaluator):
    '''Evaluation for the CountGD model from huggingface.co.'''
    
    def __str__(self):
        return "countgd"
    
    def __init__(self, **params):
        """
        Args:
            **params: Additional arguments for the model (processor, system_prompt, max_tokens, etc.)
        """
        super().__init__(params.pop("system_prompt", None))
        model_name = params.pop("model", "nikigoli/CountGD")
        self.model = pipeline(
            "image-text-to-text",
            model=model_name,
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