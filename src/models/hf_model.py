import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from .base import Evaluator

class HuggingFaceModel(Evaluator):
    '''Evaluation for models from huggingface.co.'''
    
    def __str__(self):
        return "huggingface"    
    
    
    def __init__(self, model: str, **params):
        """
        Args:
            model (str): The model to use (e.g. "gpt-4.1").
            **params: Additional arguments for the model (processor, system_prompt, max_tokens, etc.)
        """
        super().__init__(device = "cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(params.get("processor", model))
        
        self.params = params
        self.system = [{
            "role": "system",
            "content": params.system_prompt,
        }] if 'system_prompt' in params else []
        
        self.model.eval()


    def eval(self, prompt: str, image: Image) -> str:
        '''
        Evaluate the model with a prompt and an image.
        Args:
            prompt (str): The prompt to evaluate.
            image (PIL.Image): The image to evaluate.
            Returns:
            str: The evaluation result.
        '''
        chat = self.system + [{ "role": "user", "content": prompt, "images": [image]}]        
        inputs = self.processor(chat, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.params)
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    
    def eval_batch(self, prompts: list, images: list) -> list:
        assert len(prompts) == len(images), "Number of prompts and images must match."

        chat_batch = [
            self.system + [{"role": "user", "content": prompt, "images": [image]}]
            for prompt, image in zip(prompts, images)
        ]

        inputs = self.processor(chat_batch, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.params)

        return self.processor.batch_decode(outputs, skip_special_tokens=True)
