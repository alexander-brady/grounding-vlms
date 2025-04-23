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

- `engine`: Inference backend type
- `model`: Model name  
- (Additional backend-specific parameters as needed)

Override any config values with `--params`.

```bash
python src/run_eval.py --config path/to/config.yaml
```

## Without Config Files

You can run without a config by specifying `--engine` and `--model` directly:

```bash
python src/run_eval.py --engine openai --model gpt-4.1
```

## Parameters

**Required**  
Use one of:  
- `--config`: Path to model config file  
**or**  
- `--engine`: Backend type (e.g. openai)
- `--model`: Model name (e.g. gpt-4.1)

**Optional**  
- `--system_prompt`: Custom system prompt  
- `--processor`: Tokenizer (defaults to model for backends that need it).
- `--params`: Override parameters (JSON format), e.g. `'{ "temperature": 0.7, "top_k": 5 }'`  
- `--batch_size`: Batch size for batch execution (default: no batch). Set to -1 to set the whole dataset as a batch.
- `--output_dir`: Output directory (default: `eval/datasets/type/model`)  
- `--datasets`: Comma-separated dataset folders (default: `FSC-147,GeckoNum,PixMo_Count,TallyQA`)  


## Supported Backends

### openai

For OpenAI-style API models. Requires `OPENAI_API_KEY` set in `.env`.

Batch execution can take up to 24 hours and must be retrieved manually. Retrieve the file as below:

```python
from openai import OpenAI

client = OpenAI()

batch = client.batches.retrieve(batch_id)
print(batch) # Check status. Should be 'completed'

file_response = client.files.content(batch_id)
print(file_response.text)
```


### huggingface

For `AutoModelForVision2Seq` models. Set the processor with the `--processor` argument, else will be same as `model`.
