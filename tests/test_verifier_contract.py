import json

import torch

from src.scoring.verifier import (
    TransformerPRMScorer,
    aggregate_step_scores,
    build_verifier_contract,
    load_verifier_settings,
)


class FakeTokenizer:
    def __init__(self):
        self.mapping = {
            "<|verify|>": [10],
            " +": [11],
            " -": [12],
        }

    def encode(self, text, add_special_tokens=False):
        return self.mapping.get(text, [1, 2, 3])

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long)}


class FakeModelOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def __call__(self, **inputs):
        logits = torch.zeros((1, inputs["input_ids"].shape[1], 32), dtype=torch.float32)
        logits[0, -1, 11] = 4.0
        logits[0, -1, 12] = 1.0
        return FakeModelOutput(logits)


def test_build_verifier_contract_uses_configured_labels():
    contract = build_verifier_contract(
        FakeTokenizer(),
        verify_token="<|verify|>",
        positive_label=" +",
        negative_label=" -",
    )

    assert contract.verify_token == "<|verify|>"
    assert contract.positive_token_id == 11
    assert contract.negative_token_id == 12


def test_load_verifier_settings_prefers_checkpoint_metadata(tmp_path):
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    training_config = {
        "training": {
            "response_template": "<|verify|>",
            "positive_label": " +",
            "negative_label": " -",
        }
    }
    (model_path / "training_config.json").write_text(json.dumps(training_config))

    settings = load_verifier_settings(
        config={"training": {"response_template": "OVERRIDE", "positive_label": "+", "negative_label": "-"}},
        model_path=str(model_path),
    )

    assert settings["verify_token"] == "<|verify|>"
    assert settings["positive_label"] == " +"
    assert settings["negative_label"] == " -"


def test_transformer_prm_scorer_uses_binary_normalization():
    scorer = TransformerPRMScorer(
        FakeModel(),
        FakeTokenizer(),
        verify_token="<|verify|>",
        positive_label=" +",
        negative_label=" -",
        aggregation="product",
        device="cpu",
    )

    score = scorer.score_step("Problem: 1+1", "Step 1: 2")

    assert 0.5 < score < 1.0


def test_aggregate_step_scores_supports_min_and_product():
    scores = [0.9, 0.8, 0.5]

    assert aggregate_step_scores(scores, "min") == 0.5
    assert aggregate_step_scores(scores, "product") == 0.36000000000000004
