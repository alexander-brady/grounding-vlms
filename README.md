# Grounding capabilities of Vision-Language models
Investigate the visual grounding capabilities of state-of-the-art vision–language models (VLMs) using multimodal reasoning benchmarks with data visualizations.

# Table of Contents
- [Setup](#setup)
- [Run Evaluations](#run-evaluations)
  - [Using Config Files](#using-config-files)
  - [Without Config Files](#without-config-files)
  - [Parameters](#parameters)
  - [Supported Backends](#supported-backends)


# Setup

**1. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

# Run Evaluations

> Example: `eval.sh` contains a sample command.

To evaluate a model on the dataset, run:

```bash
python src/run_eval.py
```

## Using Config Files

Place model config files in the `models/` directory. Each config must include:

- `type`: Inference backend type
- `model`: Model name  
- (Additional backend-specific parameters as needed)

Override any config values with `--params`.

```bash
python src/run_eval.py --config path/to/config.yaml
```

## Without Config Files

You can run without a config by specifying `--type` and `--model` directly:

```bash
python src/run_eval.py --type openai --model gpt-4.1
```

## Parameters

**Required**  
Use one of:  
- `--config`: Path to model config file  
**or**  
- `--type`: Backend type  
- `--model`: Model name

**Optional**  
- `--system_prompt`: Custom system prompt  
- `--params`: Override parameters (JSON format), e.g. `'{ "temperature": 0.7, "top_k": 5 }'`  
- `--output_dir`: Output directory (default: `eval/datasets/type/model`)  
- `--datasets`: Comma-separated dataset folders (default: `FSC-147,GeckoNum,PixMo_Count,TallyQA`)  


## Supported Backends
**openai**

For OpenAI API models. Requires `OPENAI_API_KEY` set in `.env`.