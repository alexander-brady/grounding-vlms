from pathlib import Path

from omegaconf import DictConfig
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from .base import Evaluator, VLLMDataset
from .utils import cfg_to_dict


class VLLMModel(Evaluator):
    """Evaluation for VLMs using vLLM's chat() interface.

    Use this for models with a chat template in their tokenizer config
    (e.g., Qwen-VL, newer models).
    """

    def __init__(self, model_cfg: DictConfig):
        super().__init__(model_cfg.get("system_prompt", None))

        vllm_params = cfg_to_dict(model_cfg.get("vllm_params", {}))
        self.model = LLM(model=model_cfg.model, **vllm_params)

        sampling_params = cfg_to_dict(model_cfg.get("sampling_params", {}))
        structured_outputs_params = StructuredOutputsParams(regex=r"^(0|[1-9]\d*)$")
        self.sampling_params = SamplingParams(**sampling_params, structured_outputs=structured_outputs_params)

    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1):
        super().eval(dataset_dir, result_file, batch_size, pad_batches=False, Container=VLLMDataset)

    def eval_batch(self, prompts: list) -> list:
        """Evaluate a batch of prompts using vLLM's chat interface."""
        outputs = self.model.chat(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


class VLLMGenerateModel(Evaluator):
    """Evaluation for VLMs using vLLM's generate() with explicit prompt formatting.

    Use this for models without a chat template (e.g., LLaVA 1.5).
    """

    DEFAULT_PROMPT_TEMPLATE = "USER: <image>\n{system_prompt}\n{user_prompt}\nASSISTANT:"

    def __init__(self, model_cfg: DictConfig):
        super().__init__(model_cfg.get("system_prompt", None))

        vllm_params = cfg_to_dict(model_cfg.get("vllm_params", {}))
        self.model = LLM(model=model_cfg.model, **vllm_params)

        sampling_params = cfg_to_dict(model_cfg.get("sampling_params", {}))
        structured_outputs_params = StructuredOutputsParams(regex=r"^(0|[1-9]\d*)$")
        self.sampling_params = SamplingParams(**sampling_params, structured_outputs=structured_outputs_params)

        self.prompt_template = model_cfg.get("prompt_template", self.DEFAULT_PROMPT_TEMPLATE)

    def eval(self, dataset_dir: Path, result_file: Path, batch_size: int = 1):
        super().eval(dataset_dir, result_file, batch_size, pad_batches=False)

    def eval_batch(self, prompts: list) -> list:
        """Evaluate a batch of prompts using vLLM's generate()."""
        vllm_inputs = []

        for prompt in prompts:
            image = None
            user_text = ""

            for message in prompt:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content["type"] == "image":
                            image = content["image"]
                        elif content["type"] == "text":
                            user_text = content["text"]

            formatted_prompt = self.prompt_template.format(
                system_prompt=self.system or "",
                user_prompt=user_text,
            )

            vllm_inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": image}})

        outputs = self.model.generate(vllm_inputs, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]
