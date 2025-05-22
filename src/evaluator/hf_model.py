from pathlib import Path
from transformers import pipeline, AutoTokenizer

from .base import Evaluator

class HuggingFaceModel(Evaluator):
    '''Evaluation for models from huggingface.co.'''  
        
    def __init__(self, model: str, **params):
        """
        Args:
            model (str): The 'image-text-to-text' model to use (e.g. "qwen/qwen2-vl-7b-instruct").
            **params: Additional arguments for the model (processor, system_prompt, max_tokens, etc.)
        """
        super().__init__(params.pop("system_prompt", None))
        
        # Initialize tokenizer with left padding
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.padding_side = 'left'
        
        self.model = pipeline(
            "image-text-to-text",
            model=model,
            device_map="auto",
            torch_dtype="auto",
            tokenizer=tokenizer
        )
        
        self.params = params
        
        
    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1):
        super().eval(dataset_dir, result_file, batch_size, pad_batches=True)

        
    def eval_batch(self, prompts: list) -> list:
        """Evaluate a batch of prompts and images."""
        outputs = self.model(
            text=prompts,
            batch_size=len(prompts), 
            return_full_text=False, 
            **self.params
        )
                
        return [
            out[0]['generated_text']
            for out in outputs
        ]