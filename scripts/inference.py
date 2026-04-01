import sys
import os
import argparse
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import setup_logging
from src.config_parser import ConfigParser
from src.scoring.verifier import (
    TransformerPRMScorer,
    VLLMPRMScorer,
    load_verifier_settings,
)


def get_args():
    parser = argparse.ArgumentParser(description="PRM Inference with Best-of-N Search")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned PRM model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model for generation (default: Qwen/Qwen2.5-Math-1.5B-Instruct)",
    )
    parser.add_argument("--problem", type=str, default=None, help="Single problem to solve (demo mode)")
    parser.add_argument(
        "--n_candidates", type=int, default=None, help="Number of candidates (overrides config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Generation temperature (overrides config)",
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM for faster inference (if available)"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="min",
        choices=["min", "product"],
        help="Score aggregation method (min=weakest link, product=cumulative)",
    )
    return parser.parse_args()


class SolutionGenerator:
    """
    Generate candidate solutions using the base math model.
    Uses the original Qwen-Math model (not the fine-tuned PRM).
    """

    def __init__(self, model_name: str, device: str = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading generator model: {model_name}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Generator loaded!")

    def generate(
        self,
        problem: str,
        n_candidates: int = 16,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> list:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful math assistant. Solve the problem step by step and "
                    "provide the final numerical answer."
                ),
            },
            {"role": "user", "content": problem},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"Problem: {problem}\n\nSolution:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        solutions = []
        for _ in range(n_candidates):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            solutions.append(generated.strip())

        return solutions


class PRMVerifier:
    """
    Process Reward Model for scoring solutions.
    Uses the fine-tuned PRM model to score each step.
    """

    def __init__(
        self,
        model_path: str,
        verify_token: str = "<|verify|>",
        device: str = None,
        aggregation: str = "min",
        positive_label: str = " +",
        negative_label: str = " -",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading PRM verifier from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.scorer = TransformerPRMScorer(
            self.model,
            self.tokenizer,
            verify_token=verify_token,
            positive_label=positive_label,
            negative_label=negative_label,
            aggregation=aggregation,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        print(
            "Verifier loaded! "
            f"({self.scorer.contract.positive_label!r}: {self.scorer.contract.positive_token_id}, "
            f"{self.scorer.contract.negative_label!r}: {self.scorer.contract.negative_token_id}, "
            f"aggregation: {aggregation})"
        )

    def score_step(self, context: str, step: str) -> float:
        return self.scorer.score_step(context, step)

    def score_solution(self, problem: str, solution: str) -> dict:
        return self.scorer.score_solution(problem, solution)


class PRMVerifierVLLM:
    """
    PRM Verifier using vLLM for faster batch inference.
    """

    def __init__(
        self,
        model_path: str,
        verify_token: str = "<|verify|>",
        aggregation: str = "min",
        positive_label: str = " +",
        negative_label: str = " -",
    ):
        print(f"Loading PRM verifier with vLLM from: {model_path}")
        self.scorer = VLLMPRMScorer(
            model_path,
            verify_token=verify_token,
            positive_label=positive_label,
            negative_label=negative_label,
            aggregation=aggregation,
        )
        print(
            "vLLM Verifier loaded! "
            f"({self.scorer.contract.positive_label!r}: {self.scorer.contract.positive_token_id}, "
            f"{self.scorer.contract.negative_label!r}: {self.scorer.contract.negative_token_id}, "
            f"aggregation: {aggregation})"
        )

    def score_solution(self, problem: str, solution: str) -> dict:
        return self.scorer.score_solution(problem, solution)


def best_of_n_search(
    generator: SolutionGenerator,
    verifier,
    problem: str,
    n_candidates: int = 16,
    temperature: float = 0.7,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Problem: {problem}")
        print(f"{'=' * 60}")

    if verbose:
        print(f"\nGenerating {n_candidates} candidates (temperature={temperature})...")

    candidates = generator.generate(problem, n_candidates=n_candidates, temperature=temperature)

    if not candidates:
        print("Failed to generate candidates!")
        return {"best": None, "candidates": [], "scores": []}

    if verbose:
        print(f"Generated {len(candidates)} candidates")
        print("Scoring candidates with PRM...")

    scored_candidates = []
    for candidate in tqdm(candidates, disable=not verbose):
        result = verifier.score_solution(problem, candidate)
        scored_candidates.append(
            {
                "solution": candidate,
                "score": result["score"],
                "step_scores": result.get("step_scores", []),
            }
        )

    scored_candidates.sort(key=lambda item: item["score"], reverse=True)

    if verbose:
        print(f"\n{'=' * 60}")
        print("PRM Ranking (Top 3)")
        print(f"{'=' * 60}")
        for idx, item in enumerate(scored_candidates[:3], start=1):
            preview = item["solution"][:100].replace("\n", " ")
            print(f"\nRank {idx} | Score: {item['score']:.4f}")
            print(f"Step scores: {[f'{score:.3f}' for score in item['step_scores'][:5]]}")
            print(f"Preview: {preview}...")

    return {
        "best": scored_candidates[0]["solution"],
        "best_score": scored_candidates[0]["score"],
        "all_results": scored_candidates,
    }


def demo_problems():
    return [
        "What is 15% of 80?",
        "If a train travels at 60 mph, how far does it go in 2.5 hours?",
        "Solve for x: 3x - 7 = 2x + 5",
        "A rectangle has length 12 and width 8. What is its area?",
        "Janet pays $40/hour for 3 hours per week of clarinet lessons. How much does she pay per year?",
    ]


if __name__ == "__main__":
    args = get_args()
    logger = setup_logging()

    if not os.path.exists(args.model_path):
        logger.error(f"PRM model path '{args.model_path}' does not exist.")
        logger.error("Please run training first: make train CONFIG=configs/default.yaml")
        sys.exit(1)

    config = {}
    if os.path.exists(args.config):
        original_argv = sys.argv
        sys.argv = ["inference.py", "--config", args.config]
        config = ConfigParser.get_config()
        sys.argv = original_argv

    n_candidates = args.n_candidates or config.get("inference", {}).get("n_candidates", 16)
    temperature = args.temperature or config.get("inference", {}).get("temperature", 0.7)
    verifier_settings = load_verifier_settings(config, args.model_path)

    logger.info(f"Inference settings: n_candidates={n_candidates}, temperature={temperature}")
    logger.info(f"Base model for generation: {args.base_model}")
    logger.info(f"PRM model for scoring: {args.model_path}")
    logger.info(
        "Verifier settings: token=%r positive=%r negative=%r",
        verifier_settings["verify_token"],
        verifier_settings["positive_label"],
        verifier_settings["negative_label"],
    )

    logger.info("Loading generator (base model)...")
    generator = SolutionGenerator(args.base_model)

    aggregation = args.aggregation
    logger.info(f"Loading verifier (fine-tuned PRM, aggregation: {aggregation})...")
    if args.use_vllm:
        try:
            verifier = PRMVerifierVLLM(
                args.model_path,
                verify_token=verifier_settings["verify_token"],
                aggregation=aggregation,
                positive_label=verifier_settings["positive_label"],
                negative_label=verifier_settings["negative_label"],
            )
        except ImportError:
            logger.warning("vLLM not available, falling back to transformers")
            verifier = PRMVerifier(
                args.model_path,
                verify_token=verifier_settings["verify_token"],
                aggregation=aggregation,
                positive_label=verifier_settings["positive_label"],
                negative_label=verifier_settings["negative_label"],
            )
    else:
        verifier = PRMVerifier(
            args.model_path,
            verify_token=verifier_settings["verify_token"],
            aggregation=aggregation,
            positive_label=verifier_settings["positive_label"],
            negative_label=verifier_settings["negative_label"],
        )

    if args.problem:
        result = best_of_n_search(
            generator,
            verifier,
            args.problem,
            n_candidates=n_candidates,
            temperature=temperature,
        )
        print(f"\n{'=' * 60}")
        print("BEST SOLUTION:")
        print(f"{'=' * 60}")
        print(result["best"])
        print(f"\nPRM Score: {result['best_score']:.4f}")
    else:
        print("\n" + "=" * 60)
        print("Running demo on sample problems...")
        print("(Using BASE model for generation, PRM for scoring)")
        print("=" * 60)

        for problem in demo_problems()[:3]:
            result = best_of_n_search(
                generator,
                verifier,
                problem,
                n_candidates=n_candidates,
                temperature=temperature,
            )
            print(f"\n{'=' * 60}")
            print(f"BEST SOLUTION for: {problem}")
            print(f"{'=' * 60}")
            if result["best"]:
                print(result["best"][:500])
                print(f"\nPRM Score: {result['best_score']:.4f}")
            print()
