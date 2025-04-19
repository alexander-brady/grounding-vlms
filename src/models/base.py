import base64
import pandas as pd
from pathlib import Path


class BaseModel:
    def __init__(self):
        pass
    
    def eval(self, prompt: str, image_url: str) -> str:
        raise NotImplementedError("The eval method must be implemented by subclasses.")
    
    def eval_batch(self, output_dir: Path, prompts: list, image_urls: list):
        pd.DataFrame(
            {
                "prompt": prompts,
                "image_url": image_urls,
            }
        ).apply(
            lambda row: self.eval(row["prompt"], row["image_url"]),
            axis=1,
            result_type="expand",
        ).to_csv(
            output_dir / "results.csv",    
            mode="a",
            header=not (output_dir / "results.csv").exists(),
            index=False,
        )
    
    def process_image(self, dataset_path: Path, image_url: str=None, file_name: str=None) -> str:
        """
        Process the image and return its URL.
        
        Args:
            dataset_path (pathlib.Path): The path to the dataset directory.
            image_url (str): The URL of the image to process.
            file_name (str): The path to the image file to process.
            
        Returns:
            str: The processed image URL.
        """
        if image_url:
            return image_url
        
        elif file_name:
            with open(dataset_path / file_name, "rb") as f:
                return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode("utf-8")}"
            
        else:
            raise ValueError("Either image_url or image_path must be provided.")