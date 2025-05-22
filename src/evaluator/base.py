import requests
from io import BytesIO
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from .utils import intify


class BaseDataset(Dataset):
    '''Turns prompts and images into a usable dataset. Defaults to HuggingFace Standard.'''
    def __init__(
        self, 
        df: pd.DataFrame, 
        image_dir: Path = None, 
        system_prompt: str = str,
        prompt_col: str = "prompt", 
        image_url_col: str = "image_url",
        image_path_col: str = "file_name",
    ):
        """
        Base class for all datasets.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the dataset.
            image_dir (pathlib.Path): The directory containing the images.
            system_prompt (str): The system prompt to use for the model.
            prompt_col (str): The name of the column containing the prompts.
            image_url_col (str): The name of the column containing the image URLs.
            image_path_col (str): The name of the column containing the image file names.
        """
        assert prompt_col in df.columns, f'DataFrame must contain "{prompt_col}" column.'
        assert image_url_col in df.columns or image_path_col in df.columns, f'DataFrame must contain either "{image_url_col}" or "{image_path_col}" column.'
        
        if 'idx' in df.columns:
            self.indices = df['idx'].tolist()
        else:
            self.indices = df.index.tolist()
            
        self.prompts = df[prompt_col].tolist()
        self.system = [{
            "role": "system",
            "content": [{
                'type': "text",
                'text': system_prompt
            }]
        }] if system_prompt else []        
        
        self.using_image_urls = image_url_col in df.columns
        self.images = df[image_url_col].tolist() if self.using_image_urls else df[image_path_col].tolist()
        
        self.image_dir = image_dir
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        image = self.process_image(idx)
        if not image:
            return self.indices[idx], None
        return (
            self.indices[idx],
            self.system + [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {"type": "text", "text": self.prompts[idx]}
            ],
        }])
            
    def process_image(self, idx):
        '''Turn image into model readable format.'''
        if self.using_image_urls:
            image_url = self.images[idx]
            response = requests.get(image_url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                return None
        else:
            file_name = self.images[idx]
            with open(self.image_dir / file_name, "rb") as f:
                return Image.open(f).convert("RGB")
            
    @staticmethod
    def collate_fn(batch):
        '''Collate function to flatten the batch of prompts and images.'''
        valid_indices = [idx for idx, prompt in batch if prompt is not None]
        invalid_indices = [idx for idx, prompt in batch if prompt is None]
        prompts = [prompt for _, prompt in batch if prompt is not None]
        
        return valid_indices, prompts, invalid_indices
    

class Evaluator:
    """Base class for all evaluators."""        
    def __init__(self, system_prompt: str = None):
        self.system = system_prompt
        
    def __str__(self):
        return self.__class__.__name__.lower()
    
    @staticmethod
    def max_batch_size(df: pd.DataFrame) -> int:
        """
        Returns the maximum batch size for the model.
        
        Returns:
            int: The maximum batch size.
        """
        return len(df)
    
    def eval(self, 
        dataset_dir: Path,
        result_file: Path, 
        batch_size: int = 1, 
        pad_batches: bool = False,
        Container: BaseDataset = BaseDataset, 
        **container_kwargs):
        """
        Evaluate the model with a DataFrame of prompts and images, saving the results to a CSV file.
        
        Args:
            dataset_dir (pathlib.Path): The directory containing the dataset.
            result_file (pathlib.Path): Where to save the results.
            batch_size (int): The number of samples to process in each batch.
            pad_batches (bool): Whether to pad the batch with empty samples (ensures all batches are the same size).
            Container (BaseDataset): The class to use for the dataset.
            **container_kwargs: Additional arguments for the dataset class.
        """        
        df = pd.read_csv(dataset_dir / "dataset.csv")
        dataset = Container(df, image_dir=dataset_dir / "images", system_prompt=self.system, **container_kwargs)
        
        batch_size = batch_size if batch_size > 0 else self.max_batch_size(df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False , collate_fn=dataset.collate_fn)
        
        with open(result_file, "w") as f:
            f.write("idx,result,raw_result\n")
            for indices, prompts, failed in dataloader:
                for idx in failed:
                    f.write(f"{idx},-1,ERROR: Image url failed\n")
                
                if pad_batches:
                    prompts += ['Respond with -1'] * (batch_size - len(prompts))
                    
                for idx, count in zip(indices, self.eval_batch(prompts)):
                    f.write(f"{idx},{intify(count)},{count}\n")
                                
                    
    def eval_single(self, prompt: list) -> str:
        """Evaluate a prompt."""
        raise NotImplementedError("The eval method must be implemented by subclasses.")
    
    
    def eval_batch(self, prompts: list) -> list:
        """Evaluate a batch of prompts and images."""
        return [
            self.eval_single(prompt)
            for prompt in prompts
        ]