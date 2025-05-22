import yaml, json, time, argparse

from pathlib import Path
from dotenv import load_dotenv
from evaluator import load_model


def root() -> Path:
    """
    Get the root path of the project.
    """
    return Path(__file__).resolve().parent.parent


def build_config(args: argparse.Namespace) -> dict:
    """
    Build the configuration for the model based on config file and command line overrides.
    Args:
        args (argparse.Namespace): The command line arguments.
    Returns:
        dict: The configuration for the model.
    """    
    if args.config:
        path = root() / "models" / f"{args.config}.yaml"       
        with open(path, "r") as f:
            config = yaml.safe_load(f)
            
    else:
        config = {
            "engine": args.engine,
            "model": args.model,
        }
            
    if args.system_prompt:
        config["system_prompt"] = args.system_prompt
        
    if args.params:
        config.update(args.params)
            
    return config


def main(args):
    """
    Run evaluation on the datasets using the specified model.
    Args:
        args (argparse.Namespace): The command line arguments.
    """
    load_dotenv()
    
    config = build_config(args)
    engine = config.pop("engine")
    batch_size = config.pop("batch_size", 1)
    
    model = load_model(engine, **config)
    
    output_dir = Path(args.output_dir)
    
    # if user passed an absolute path, use it directly; otherwise prepend project root
    if not output_dir.is_absolute():
        output_dir = root() / output_dir
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = output_dir / f"{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = args.datasets.split(",")
    for dataset in datasets:
        dataset = dataset.strip()
        dataset_path = root() / "eval" / "datasets" / dataset.replace(".csv", "")
        
        if not(dataset_path / "dataset.csv").exists():
            raise FileNotFoundError(f"Dataset {dataset} not found at {dataset_path}")
        
        print(f"Evaluating on {dataset}...", flush=True)
        model.eval(dataset_path, output_dir / f"{dataset}.csv", batch_size)
    
    if not any(output_dir.iterdir()):
        output_dir.rmdir()
        
        
def parse_args():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description="Run evaluation on a model.")
    
    parser.add_argument("--config", help="Path to the config file (e.g. openai/gpt-4.1)")
    
    parser.add_argument("--engine", help="Backend of model to evaluate (only if config is not provided)")
    parser.add_argument("--model", help="Name of the model to evaluate (only if config is not provided)")
    
    parser.add_argument("--system_prompt", help="System prompt to initialize the model")
    
    parser.add_argument("--params", type=json.loads, help="Extra params as JSON string")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for evaluation. -1 for all at once')
    
    parser.add_argument(
        "--datasets", 
        help="List of datasets to evaluate on (comma-separated)",
        default="FSC-147, GeckoNum, TallyQA"
    )
    
    parser.add_argument("--output_dir", help="Directory to save the evaluation results")
    
    args = parser.parse_args()
    if args.config:
        args.config = args.config.replace(".yaml", "")
    
    if not args.config:
        missing = [ arg for arg in ["engine", "model"] if not getattr(args, arg)]
        if missing:
            parser.error(f"Missing required arguments: config or ({' + '.join(missing)})")
            
    if not args.output_dir:
        if args.config:
            args.output_dir = f"eval/results/{args.config}"
        else:
            args.output_dir = f"eval/results/{args.engine}/{args.model}"
            
    return args

        
if __name__ == "__main__":
    args = parse_args()
    main(args)