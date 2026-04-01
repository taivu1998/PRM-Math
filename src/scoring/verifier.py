import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F


DEFAULT_VERIFY_TOKEN = "<|verify|>"
DEFAULT_POSITIVE_LABEL = " +"
DEFAULT_NEGATIVE_LABEL = " -"


@dataclass(frozen=True)
class VerifierContract:
    verify_token: str
    positive_label: str
    negative_label: str
    positive_token_id: int
    negative_token_id: int


def load_verifier_settings(
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Resolve verifier settings from config and, when available, checkpoint metadata.

    Checkpoint metadata wins because it reflects how the model was actually trained.
    """
    settings = {
        "verify_token": DEFAULT_VERIFY_TOKEN,
        "positive_label": DEFAULT_POSITIVE_LABEL,
        "negative_label": DEFAULT_NEGATIVE_LABEL,
    }

    training = (config or {}).get("training", {})
    if training:
        settings["verify_token"] = training.get("response_template", settings["verify_token"])
        settings["positive_label"] = training.get("positive_label", settings["positive_label"])
        settings["negative_label"] = training.get("negative_label", settings["negative_label"])

    if model_path:
        metadata_path = os.path.join(model_path, "training_config.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    training_config = json.load(f)
                training_meta = training_config.get("training", {})
                if training_meta:
                    settings["verify_token"] = training_meta.get(
                        "response_template", settings["verify_token"]
                    )
                    settings["positive_label"] = training_meta.get(
                        "positive_label", settings["positive_label"]
                    )
                    settings["negative_label"] = training_meta.get(
                        "negative_label", settings["negative_label"]
                    )
            except (OSError, json.JSONDecodeError):
                pass

    return settings


def build_verifier_contract(
    tokenizer,
    verify_token: str = DEFAULT_VERIFY_TOKEN,
    positive_label: str = DEFAULT_POSITIVE_LABEL,
    negative_label: str = DEFAULT_NEGATIVE_LABEL,
) -> VerifierContract:
    positive_token_ids = tokenizer.encode(positive_label, add_special_tokens=False)
    negative_token_ids = tokenizer.encode(negative_label, add_special_tokens=False)

    if len(positive_token_ids) != 1 or len(negative_token_ids) != 1:
        raise ValueError(
            "Verifier labels must tokenize to exactly one token each. "
            f"Got {positive_label!r} -> {positive_token_ids} and "
            f"{negative_label!r} -> {negative_token_ids}."
        )
    if positive_token_ids[0] == negative_token_ids[0]:
        raise ValueError("Positive and negative verifier labels must map to different tokens.")

    return VerifierContract(
        verify_token=verify_token,
        positive_label=positive_label,
        negative_label=negative_label,
        positive_token_id=positive_token_ids[0],
        negative_token_id=negative_token_ids[0],
    )


def aggregate_step_scores(step_scores: Iterable[float], aggregation: str = "min") -> float:
    step_scores = list(step_scores)
    if not step_scores:
        return 0.0
    if aggregation == "product":
        return math.prod(step_scores)
    if aggregation == "min":
        return min(step_scores)
    raise ValueError(f"Unsupported aggregation mode: {aggregation}")


def _normalized_binary_score(
    positive_logit_or_prob: float,
    negative_logit_or_prob: float,
    *,
    already_probabilities: bool = False,
) -> float:
    if already_probabilities:
        denominator = positive_logit_or_prob + negative_logit_or_prob
        return positive_logit_or_prob / denominator if denominator > 0 else 0.5

    probs = torch.softmax(
        torch.tensor([positive_logit_or_prob, negative_logit_or_prob], dtype=torch.float32), dim=0
    )
    return probs[0].item()


class TransformerPRMScorer:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        verify_token: str = DEFAULT_VERIFY_TOKEN,
        positive_label: str = DEFAULT_POSITIVE_LABEL,
        negative_label: str = DEFAULT_NEGATIVE_LABEL,
        aggregation: str = "min",
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.aggregation = aggregation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.contract = build_verifier_contract(
            tokenizer,
            verify_token=verify_token,
            positive_label=positive_label,
            negative_label=negative_label,
        )

    def _input_device(self) -> torch.device:
        model_device = getattr(self.model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map:
            for mapped_device in hf_device_map.values():
                if mapped_device not in {"cpu", "disk"}:
                    return torch.device(mapped_device)

        return torch.device(self.device)

    def score_step(self, context: str, step: str) -> float:
        text = f"{context}\n{step}\n{self.contract.verify_token}"
        inputs = self.tokenizer(text, return_tensors="pt")
        input_device = self._input_device()
        inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

        positive_logit = next_token_logits[self.contract.positive_token_id].item()
        negative_logit = next_token_logits[self.contract.negative_token_id].item()
        return _normalized_binary_score(positive_logit, negative_logit)

    def score_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        steps = [step.strip() for step in solution.split("\n") if step.strip()]
        if not steps:
            return {"score": 0.0, "step_scores": [], "steps": []}

        context = f"Problem: {problem}\n\nSolution:"
        step_scores = []
        for step in steps:
            step_scores.append(self.score_step(context, step))
            context = f"{context}\n{step}"

        return {
            "score": aggregate_step_scores(step_scores, self.aggregation),
            "step_scores": step_scores,
            "steps": steps,
        }


class VLLMPRMScorer:
    def __init__(
        self,
        model_path: str,
        *,
        verify_token: str = DEFAULT_VERIFY_TOKEN,
        positive_label: str = DEFAULT_POSITIVE_LABEL,
        negative_label: str = DEFAULT_NEGATIVE_LABEL,
        aggregation: str = "min",
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        logprobs: Optional[int] = None,
    ):
        from vllm import LLM

        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.aggregation = aggregation
        self.contract = build_verifier_contract(
            self.tokenizer,
            verify_token=verify_token,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        tokenizer_vocab_size = getattr(self.tokenizer, "vocab_size", None)
        self.logprobs = logprobs if logprobs is not None else tokenizer_vocab_size
        if self.logprobs is None:
            self.logprobs = 50

    def score_solution(self, problem: str, solution: str) -> Dict[str, Any]:
        from vllm import SamplingParams

        steps = [step.strip() for step in solution.split("\n") if step.strip()]
        if not steps:
            return {"score": 0.0, "step_scores": [], "steps": []}

        prompts = []
        context = f"Problem: {problem}\n\nSolution:"
        for step in steps:
            prompts.append(f"{context}\n{step}\n{self.contract.verify_token}")
            context = f"{context}\n{step}"

        params = SamplingParams(max_tokens=1, temperature=0, logprobs=self.logprobs)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)

        step_scores = []
        for output in outputs:
            if not output.outputs or not output.outputs[0].logprobs:
                step_scores.append(0.5)
                continue

            logprobs = output.outputs[0].logprobs[0]
            positive = logprobs.get(self.contract.positive_token_id)
            negative = logprobs.get(self.contract.negative_token_id)

            if positive is None or negative is None:
                raise RuntimeError(
                    "vLLM did not return verifier label logprobs for both classes. "
                    "Use the transformer verifier path for exact scoring, or increase "
                    "the requested logprobs."
                )

            step_scores.append(
                _normalized_binary_score(
                    positive.logprob,
                    negative.logprob,
                )
            )

        return {
            "score": aggregate_step_scores(step_scores, self.aggregation),
            "step_scores": step_scores,
            "steps": steps,
        }
