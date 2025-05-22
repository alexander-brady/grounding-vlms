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

The evaluations will be saved by default in `eval/results/` folder. They will be saved in a csv file with the following columns:
- `index`: The index of the image in the dataset
- `result`: The model's prediction, converted to an integer
- `raw_output`: The raw output of the model, which may not be in integer format (e.g. "three", "approximately 32")

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
- `--params`: Override parameters (JSON format), e.g. `'{ "temperature": 0.7, "top_k": 5 }'`  
- `--batch_size`: Batch size for batch execution (default: no batch). Set to -1 to set the whole dataset as a batch.
- `--output_dir`: Output directory (default: `eval/datasets/type/model`)  
- `--datasets`: Comma-separated dataset folders (default: `FSC-147,GeckoNum,PixMo_Count,TallyQA`)


## Supported Backends
> Set the backend using the `engine` parameter

### openai, google, anthropic, xai

For OpenAI-sdk API models. Requires corresponding API key set in `.env`. Uses structured chat completions to get the model's predicted count.
- `openai`: OpenAI models (e.g. gpt-4.1). Requires `OPENAI_API_KEY`.
- `google`: Gemini models (e.g. Gemini-1.5). Requires `GEMINI_API_KEY`.
- `anthropic`: Anthropic models (e.g. Claude-3). Requires `ANTHROPIC_API_KEY`.
- `xai`: xAI models (e.g. Grok). Requires `XAI_API_KEY`.

Supported sampling params:
`temperature`, `frequency_penalty`, `max_completion_tokens`, `reasoning_effort`, `seed`, `top_p`

Additional params:
- `force_download`: Force download of the images instead of passing the url (default: False)
- `structured_output`: Use structured output API for constrained decoding (default: True)

### huggingface

Internally uses `pipeline` for `image-text-to-text` models. 

Some models require `HUGGING_FACE_HUB_TOKEN` environment variable.