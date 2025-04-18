import os
import yaml
import json
import argparse

from pathlib import Path
from models import load_model

import pandas as pd


def root() -> Path:
    """
    Get the root path of the project.
    """
    return Path(__file__).resolve().parent


def build_config(args: argparse.Namespace) -> dict:
    """
    Build the configuration for the model based on config file and command line overrides.
    Args:
        args (argparse.Namespace): The command line arguments.
    Returns:
        dict: The configuration for the model.
    """    
    if args.config:
        path = root() / "models" / (args.config if args.config.endswith(".yaml") else f"{args.config}.yaml")        
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            
    else:
        config = {
            "type": args.type,
            "model": args.model,
        }
            
    if args.system_prompt:
        config["system_prompt"] = args.system_prompt
        
    if args.params:
        config.update(args.params)
            
    return config



def main(args):
    """
    Run evaluation on a model.
    Args:
        args (argparse.Namespace): The command line arguments.
    """
    config = build_config(args)
    model = load_model(config["type"], **config)
    
    output_dir = root() / args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = args.datasets.split(",")
    for dataset in datasets:
        dataset = dataset.strip()
        dataset_path = root() / "eval" / "datasets" / (dataset if dataset.endswith(".csv") else f"{dataset}.csv")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset} not found at {dataset_path}")
        
        print(f"Evaluating on {dataset}...")
        df = pd.read_csv(dataset_path)
        df.apply(
            lambda row: model.eval(row["prompt"], row["image_url"]),
            axis=1,
            result_type="expand"
        ).to_csv(
            output_dir / f"{dataset}_results.csv",
            index=False
        )
        
        
def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on a model.")
    
    parser.add_argument("--config", help="Path to the config file (e.f. openai/gpt-4.1)")
    
    parser.add_argument("--type", help="Type of model to evaluate (only if config is not provided)")
    parser.add_argument("--model", help="Name of the model to evaluate (only if config is not provided)")
    
    parser.add_argument("--system_prompt", help="System prompt to initialize the model")
    parser.add_argument("--params", type=json.loads, help="Extra params as JSON string")
    
    parser.add_argument(
        "--datasets", 
        help="List of datasets to evaluate on (comma-separated)",
        default="FSC-147, GeckoNum, PixMo_Count, TallyQA"
    )
    
    parser.add_argument("--output_dir", help="Directory to save the evaluation results")
    
    args = parser.parse_args()
    
    if not args.config:
        missing = [ arg for arg in ["type", "model"] if not getattr(args, arg)]
        if missing:
            parser.error(f"Missing required arguments: config or ({' + '.join(missing)})")
            
    if not args.output_dir:
        if args.config:
            args.output_dir = "eval" / "results" / args.config.replace(".yaml", "")
        else:
            args.output_dir = "eval" / "results" / args.type / args.model
            
    return args
        
if __name__ == "__main__":
    args = parse_args()
    main(args)