from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from . import load_evaluator
from .results import create_results


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig):
    """
    Run evaluation on the datasets using the specified model.
    Args:
        args (argparse.Namespace): The command line arguments.
    """
    load_dotenv()
    root = Path(__file__).resolve().parent.parent

    model = load_evaluator(cfg.model)
    batch_size = cfg.model.get("batch_size", 1)

    output_dir = Path(cfg.get("output_dir", "."))
    for dataset in cfg.datasets:
        dataset = dataset.strip()
        dataset_path = root / "eval" / "datasets" / dataset.replace(".csv", "")

        if not (dataset_path / "dataset.csv").exists():
            raise FileNotFoundError(f"Dataset {dataset} not found at {dataset_path}")

        print(f"Evaluating on {dataset}...", flush=True)
        model.eval(dataset_path, output_dir / f"{dataset}.csv", batch_size)

    create_results(results_dir=output_dir)


if __name__ == "__main__":
    main()
