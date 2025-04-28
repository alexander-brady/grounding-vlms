from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image


class Evaluator:
    def __init__(self, device: str = "cpu"):
        self.device = device
        
    def __str__(self):
        return self.__class__.__name__.lower()
    
    def max_batch_size(self, df: pd.DataFrame) -> int:
        """
        Returns the maximum batch size for the model.
        
        Returns:
            int: The maximum batch size.
        """
        return len(df)
    
    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1):
        """
        Evaluate the model with a DataFrame of prompts and images.
        
        Args:
            dataset_dir (pathlib.Path): The directory containing the dataset.
            result_file (pathlib.Path): Where to save the results.
            batch_size (int): The number of samples to process in each batch.
        """
        
        df = pd.read_csv(dataset_dir / "dataset.csv")
                
        assert "prompt" in df.columns, "DataFrame must contain a 'prompt' column."
        assert "image_url" in df.columns or "file_name" in df.columns, "DataFrame must contain an 'image_url' or 'file_name' column."

        
        batch_size = batch_size if batch_size > 0 else self.max_batch_size(items)
        
        with open(result_file, "w") as f:
            f.write("idx,result\n")
            
            if batch_size == 1:
                for idx, row in df.iterrows():
                    result = self.eval_single(row["prompt"], self.process_image(
                        dataset_dir, 
                        image_url=row.get("image_url"), 
                        file_name=row.get("file_name")
                    ))
                    if result:
                        f.write(f"{idx},{result}\n")
                    
            else:
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i + batch_size]
                    
                    idxs   = batch.index.tolist()
                    prompts = batch["prompt"].tolist()
                    images  = [ self.process_image(
                        dataset_dir, 
                        image_url=row.get("image_url"), 
                        file_name=row.get("file_name")
                    ) for row in batch.itertuples() ]
                    
                    results = self.eval_batch(i, prompts, images)
                    
                    for idx, result in zip(idxs, results):
                        if result:
                            f.write(f"{idx},{result}\n")
            
    def eval_single(self, prompt: str, image: Image) -> str:
        """Evaluate a single prompt and image. """
        raise NotImplementedError("The eval method must be implemented by subclasses.")
    
    
    def eval_batch(self, batch_index: int, prompts: list, images: list) -> list:
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