from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from word2number import w2n


class BaseDataset(Dataset):
    '''Turns prompts and images into a usable dataset. Defaults to HuggingFace Standard.'''
    def __init__(self, 
                 df: pd.DataFrame, 
                 image_dir: Path = None, 
                 system_prompt: str = str,
                 prompt_col: str = "prompt", 
                 image_url_col: str = "image_url",
                 image_path_col: str = "file_name"
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
        
        self.prompts = df[prompt_col].tolist()
        self.system = [{
            "role": "system",
            "content": system_prompt
        }] if system_prompt else []        
        
        self.using_image_urls = image_url_col in df.columns
        self.images = df[image_url_col].tolist() if self.using_image_urls else df[image_path_col].tolist()
        
        self.image_dir = image_dir
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.system + [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": self.process_image(idx),
                },
                {"type": "text", "text": self.prompts[idx]}
            ],
        }]
            
    def process_image(self, idx):
        '''Turn image into model readable format.'''
        if self.using_image_urls:
            image_url = self.images[idx]
            response = requests.get(image_url)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert("RGB")
            else:
                raise ValueError(f"Failed to fetch image from URL: {image_url}")
        else:
            file_name = self.images[idx]
            with open(self.image_dir / file_name, "rb") as f:
                return Image.open(f).convert("RGB")
            
    @staticmethod
    def collate_fn(batch):
        '''Collate function to flatten the batch of prompts and images.'''
        return batch


class Evaluator:
    def __init__(self, system_prompt: str = None):
        """
        Base class for all evaluators.
        """        
        self.system = system_prompt
        
    def __str__(self):
        return self.__class__.__name__.lower()
    
    def max_batch_size(self, df: pd.DataFrame) -> int:
        """
        Returns the maximum batch size for the model.
        
        Returns:
            int: The maximum batch size.
        """
        return len(df)
    
    def intify(self, result: str) -> str:
        '''Turns the model output into an integer. Returns -1 if it fails.'''
        result = result.replace("-", " ").replace(",", "").split(".")[0]
        if result.isdigit():
            return result
        
        digits = ''.join(filter(str.isdigit, result))
        if digits:
            return digits
        
        try:
            return str(w2n.word_to_num(result))
        except ValueError:
            return '-1'        
    
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
        dataset = Container(df, image_dir=dataset_dir / "images", system_prompt=self.system)
        
        batch_size = batch_size if batch_size > 0 else self.max_batch_size(df)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False , collate_fn=dataset.collate_fn)
        
        with open(result_file, "w") as f:
            f.write("idx,result,uncleansed\n")
            for idx, prompt in enumerate(dataloader):
                for line_idx, out in enumerate(self.eval_batch(idx, prompt), start=idx):
                    f.write(f"{line_idx},{self.intify(out)},{out}\n")
                    
    def eval_single(self, prompt: list) -> str:
        """Evaluate a prompt."""
        raise NotImplementedError("The eval method must be implemented by subclasses.")
    
    def eval_batch(self, batch_index: int, prompts: list) -> list:
        """Evaluate a batch of prompts and images."""
        return [
            self.eval_single(prompt)
            for prompt in prompts
        ]