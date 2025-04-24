import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image


class BaseModel:
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def __str__(self):
        return self.__class__.__name__.lower()
    
    def eval(self, dataset_dir: Path, output_dir: Path, batch_size: int = 1):
        """
        Evaluate the model with a DataFrame of prompts and images.
        
        Args:
            dataset_dir (pathlib.Path): The directory containing the dataset.
            output_dir (pathlib.Path): The directory to save the results.
            batch_size (int): The number of samples to process in each batch.
        """
        
        df = pd.read_csv(dataset_dir / "dataset.csv")
        
        assert "prompt" in df.columns, "DataFrame must contain a 'prompt' column."
        assert "image_url" in df.columns or "file_name" in df.columns, "DataFrame must contain an 'image_url' or 'file_name' column."
        
        items = [
            (idx, row["prompt"], self.process_image(
                dataset_dir, 
                image_url=row.get("image_url"), 
                file_name=row.get("file_name")
            ))
            for idx, row in df.iterrows()
        ]
        
        batch_size = batch_size if batch_size > 0 else len(items)
        os.makedirs(output_dir, exist_ok=True)
        
        # write results to “<output_dir>/<model_name>_results.csv”
        result_file = output_dir / f"{self}_results.csv"
        with open(result_file, "w") as f:
            f.write("idx,result\n")
            
            if batch_size == 1:
                for idx, prompt, image in items:
                    result = self.eval_single(prompt, image)
                    if result:
                        f.write(f"{idx},{result}\n")
                    
            else:
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    
                    idxs   = [x[0] for x in batch]
                    prompts = [x[1] for x in batch]
                    images  = [x[2] for x in batch]
                    
                    results = self.eval_batch(output_dir, prompts, images)
                    
                    for idx, result in zip(idxs, results):
                        if result:
                            f.write(f"{idx},{result}\n")
                        
            
    def eval_single(self, prompt: str, image: Image) -> str:
        """Evaluate a single prompt and image. """
        raise NotImplementedError("The eval method must be implemented by subclasses.")
    
    
    def eval_batch(self, prompts: list, images: list) -> list:
        """Evaluate a batch of prompts and images."""
        return [
            self.eval_single(prompt, image)
            for prompt, image in zip(prompts, images)
        ]
            
    
    def process_image(self, dataset_path: Path, image_url: str=None, file_name: str=None) -> Image:
        """
        Process the image from a path/url.
        
        Args:
            dataset_path (pathlib.Path): The path to the dataset directory.
            image_url (str): The URL of the image to process.
            file_name (str): The path to the image file to process.
            
        Returns:
            Image: The processed image.
        """
        if image_url:
            response = requests.get(image_url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                raise ValueError(f"Failed to fetch image from URL: {image_url}")
        
        elif file_name:
            with open(dataset_path / file_name, "rb") as f:
                return Image.open(f).convert("RGB")
            
        else:
            raise ValueError("Either image_url or image_path must be provided.")