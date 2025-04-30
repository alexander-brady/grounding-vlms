import torch
import pandas as pd

from pathlib import Path
from transformers import pipeline

from .base import Evaluator, BaseDataset


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
        
    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1, Container: BaseDataset = BaseDataset):
        """
        Evaluate the model with a DataFrame of prompts and images, saving the results to a CSV file.
        
        Args:
            dataset_dir (pathlib.Path): The directory containing the dataset.
            result_file (pathlib.Path): Where to save the results.
            batch_size (int): The number of samples to process in each batch.
            DatasetClass (BaseDataset): The class to use for the dataset (inheriting from BaseDataset).
        """
        df = pd.read_csv(dataset_dir / "dataset.csv")
        dataset = Container(df, image_dir=dataset_dir / "images")
        
        batch_size = batch_size if batch_size > 0 else self.max_batch_size(df)

        with open(result_file, "w") as f:
            f.write("idx,result\n")
            for idx, out in enumerate(self.model(
                dataset,
                batch_size=batch_size,
                total=len(dataset),
                return_full_text=False,
                **self.params            
            )):
                for line_idx, result in enumerate(out, start=idx):
                    f.write(f"{line_idx},{result['generated_text']}\n")