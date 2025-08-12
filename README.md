# Can't Count on Vision-Language Models
Vision Language Models can't count very well. We evaluated a bunch of VLMs to prove it. This repo contains an evaluation backend to easily run and benchmark vision language models on counting tasks.

# Table of Contents
- [Setup](#setup)
- [Run Evaluations](#run-evaluations)
  - [Config Management](#using-config-files)
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

> Example: `eval.sh` contains a sample SLURM command.

To evaluate a model on the dataset, run:

```bash
python -m src.evaluator model=openai/gpt-4.1
```

The evaluations will be saved by default in `output` folder (Hydra's working directory). They will be saved in a csv file with the following columns:
- `index`: The index of the image in the dataset
- `result`: The model's prediction, converted to an integer
- `raw_result`: The raw output of the model, which may not be in integer format (e.g. "three", "approximately 32")

## Config Management

This repo uses [Hydra](https://hydra.cc/) for configuration management. Models are specified using config files. Such files specify the model and its parameters. It allows you to easily switch between different models and configurations without modifying the command line arguments.

**Example Config File**

```yaml
engine: openai
model: gpt-4.1
system_prompt: "You are a helpful assistant. Please count the number of objects in the image."
params:
  temperature: 0.7
  top_k: 5
```

Place model config files in the `models/` directory. Each config must include:

- `engine`: Inference backend type
- `model`: Model name  
- (Additional backend-specific parameters as needed)

Config values can be overridden via Hydra CLI.

```bash
python -m src.run_eval model=path/to/config model.params.temperature=0.8 model.params.top_k=1
```

## Without Config Files

You can run without a config by specifying `+model.engine` and `+model.name` directly:

```bash
python -m src.run_eval +model.engine openai +model.name gpt-4.1
```

## Model Parameters

**Required**  
- `--engine`: Backend type (e.g. openai)
- `--model`: Model name (e.g. gpt-4.1)

**Optional**  
- `system_prompt`: Custom system prompt  
- `params`: Override parameters, e.g. `model.params.temperature=0.7 model.params.top_k=5`
- `batch_size`: Batch size for batch execution (default: 1). Set to -1 to use the maximum batch size supported by the backend.
- `output_dir`: Output directory (default: hydra output directory)
- `datasets`: Comma-separated dataset folders (default: `FSC-147, GeckoNum, TallyQA`)


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
