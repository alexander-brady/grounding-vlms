from pathlib import Path

from omegaconf import DictConfig
from transformers import pipeline

from .base import Evaluator
from .utils import cfg_to_dict


class HuggingFaceModel(Evaluator):
    """Evaluation for models from huggingface.co."""

    def __init__(self, model_cfg: DictConfig):
        """
        Args:
            model_cfg (DictConfig): The configuration for the model.
        """
        super().__init__(model_cfg.get("system_prompt", None))
        self.model = pipeline("image-text-to-text", model=model_cfg.model, device_map="auto", dtype="auto")

        self.params = cfg_to_dict(model_cfg.get("params", None))

    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1):
        super().eval(dataset_dir, result_file, batch_size, pad_batches=True)

    def eval_batch(self, prompts: list) -> list:
        """Evaluate a batch of prompts and images."""
        outputs = self.model(text=prompts, batch_size=len(prompts), return_full_text=False, **self.params)

        return [out[0]["generated_text"] for out in outputs]
