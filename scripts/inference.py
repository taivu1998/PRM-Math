import sys
import os
import argparse
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import setup_logging
from src.config_parser import ConfigParser


def get_args():
    parser = argparse.ArgumentParser(description="PRM Inference with Best-of-N Search")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned PRM model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help="Base model for generation (default: Qwen/Qwen2.5-Math-1.5B-Instruct)")
    parser.add_argument("--problem", type=str, default=None, help="Single problem to solve (demo mode)")
    parser.add_argument("--n_candidates", type=int, default=None, help="Number of candidates (overrides config)")
    parser.add_argument("--temperature", type=float, default=None, help="Generation temperature (overrides config)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference (if available)")
    parser.add_argument("--aggregation", type=str, default="min", choices=["min", "product"],
                        help="Score aggregation method (min=weakest link, product=cumulative)")
    return parser.parse_args()


class SolutionGenerator:
    """
    Generate candidate solutions using the base math model.
    Uses the original Qwen-Math model (not the fine-tuned PRM).
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading generator model: {model_name}")
        self.device = device
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

    def generate(self, problem: str, n_candidates: int = 16,
                 temperature: float = 0.7, max_new_tokens: int = 512) -> list:
        """
        Generate multiple candidate solutions for a math problem.
        Uses chat template for better instruction following.
        """
        # Use chat format for better results
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step and provide the final numerical answer."},
            {"role": "user", "content": problem}
        ]

        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"Problem: {problem}\n\nSolution:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        solutions = []
        for i in range(n_candidates):
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
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            solutions.append(generated.strip())

        return solutions


class PRMVerifier:
    """
    Process Reward Model for scoring solutions.
    Uses the fine-tuned PRM model to score each step.
    """

    def __init__(self, model_path: str, verify_token: str = "<|verify|>", device: str = "cuda",
                 aggregation: str = "min"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading PRM verifier from: {model_path}")
        self.device = device
        self.verify_token = verify_token
        self.aggregation = aggregation  # "min" or "product"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

        # Get token IDs for + and -
        self.pos_token_id = self.tokenizer.encode("+", add_special_tokens=False)[0]
        self.neg_token_id = self.tokenizer.encode("-", add_special_tokens=False)[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Verifier loaded! (+ token: {self.pos_token_id}, - token: {self.neg_token_id}, aggregation: {aggregation})")

    def score_step(self, context: str, step: str) -> float:
        """
        Score a single reasoning step.
        Returns probability that the step is correct.
        """
        text = f"{context}\n{step}\n{self.verify_token}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

        probs = F.softmax(next_token_logits, dim=-1)
        pos_prob = probs[self.pos_token_id].item()
        neg_prob = probs[self.neg_token_id].item()

        # Normalize to get P(correct)
        score = pos_prob / (pos_prob + neg_prob) if (pos_prob + neg_prob) > 0 else 0.5
        return score

    def score_solution(self, problem: str, solution: str) -> dict:
        """
        Score an entire solution by scoring each step.
        Supports "min" (weakest link) or "product" (cumulative) aggregation.
        """
        steps = [s.strip() for s in solution.split("\n") if s.strip()]

        if not steps:
            return {"score": 0.0, "step_scores": [], "steps": []}

        context = f"Problem: {problem}\n\nSolution:"
        step_scores = []

        for step in steps:
            score = self.score_step(context, step)
            step_scores.append(score)
            context = f"{context}\n{step}"

        # Aggregation
        if not step_scores:
            final_score = 0.0
        elif self.aggregation == "product":
            final_score = math.prod(step_scores)
        else:  # min (weakest link)
            final_score = min(step_scores)

        return {
            "score": final_score,
            "step_scores": step_scores,
            "steps": steps
        }


class PRMVerifierVLLM:
    """
    PRM Verifier using vLLM for faster batch inference.
    """

    def __init__(self, model_path: str, verify_token: str = "<|verify|>"):
        from vllm import LLM, SamplingParams

        print(f"Loading PRM verifier with vLLM from: {model_path}")
        self.llm = LLM(model=model_path, dtype="float16", tensor_parallel_size=1)
        self.tokenizer = self.llm.get_tokenizer()
        self.verify_token = verify_token

        self.plus_token_ids = self.tokenizer.encode("+", add_special_tokens=False)
        self.minus_token_ids = self.tokenizer.encode("-", add_special_tokens=False)
        print(f"vLLM Verifier loaded! (+ tokens: {self.plus_token_ids})")

    def score_solution(self, problem: str, solution: str) -> dict:
        """Score solution using vLLM."""
        from vllm import SamplingParams

        steps = [s.strip() for s in solution.split('\n') if s.strip()]

        if not steps:
            return {"score": 0.0, "step_scores": [], "steps": []}

        prompts = []
        context = f"Problem: {problem}\n\nSolution:"

        for step in steps:
            prompt = f"{context}\n{step}\n{self.verify_token}"
            prompts.append(prompt)
            context = f"{context}\n{step}"

        params = SamplingParams(max_tokens=1, logprobs=10, temperature=0)

        try:
            outputs = self.llm.generate(prompts, params, use_tqdm=False)
        except Exception as e:
            return {"score": 0.0, "step_scores": [], "steps": steps, "error": str(e)}

        step_scores = []
        for output in outputs:
            if not output.outputs or not output.outputs[0].logprobs:
                step_scores.append(0.0)
                continue

            logprobs = output.outputs[0].logprobs[0]
            score = -100.0
            for tid in self.plus_token_ids:
                if tid in logprobs:
                    score = max(score, logprobs[tid].logprob)

            # Convert logprob to probability-like score
            import math
            step_scores.append(math.exp(score) if score > -100 else 0.0)

        final_score = min(step_scores) if step_scores else 0.0

        return {
            "score": final_score,
            "step_scores": step_scores,
            "steps": steps
        }


def best_of_n_search(generator: SolutionGenerator, verifier, problem: str,
                     n_candidates: int = 16, temperature: float = 0.7,
                     verbose: bool = True) -> dict:
    """
    Performs Best-of-N Search with PRM reranking.

    1. Generate N candidate solutions using the BASE model
    2. Score each candidate using the fine-tuned PRM
    3. Return the best solution based on PRM scores

    Args:
        generator: SolutionGenerator instance (uses base model)
        verifier: PRMVerifier instance (uses fine-tuned model)
        problem: The math problem to solve
        n_candidates: Number of candidates to generate
        temperature: Generation temperature
        verbose: Whether to print progress

    Returns:
        Dictionary with best solution, all candidates, and scores
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}")

    # 1. Generate Candidates using BASE model
    if verbose:
        print(f"\nGenerating {n_candidates} candidates (temperature={temperature})...")

    candidates = generator.generate(problem, n_candidates=n_candidates, temperature=temperature)

    if not candidates:
        print("Failed to generate candidates!")
        return {"best": None, "candidates": [], "scores": []}

    if verbose:
        print(f"Generated {len(candidates)} candidates")

    # 2. Score Candidates using PRM
    if verbose:
        print("Scoring candidates with PRM...")

    scored_candidates = []
    for i, cand in enumerate(tqdm(candidates, disable=not verbose)):
        result = verifier.score_solution(problem, cand)
        scored_candidates.append({
            "solution": cand,
            "score": result["score"],
            "step_scores": result.get("step_scores", [])
        })

    # 3. Rank by score
    scored_candidates.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print(f"\n{'='*60}")
        print("PRM Ranking (Top 3)")
        print(f"{'='*60}")
        for i, item in enumerate(scored_candidates[:3]):
            preview = item["solution"][:100].replace('\n', ' ')
            print(f"\nRank {i+1} | Score: {item['score']:.4f}")
            print(f"Step scores: {[f'{s:.3f}' for s in item['step_scores'][:5]]}")
            print(f"Preview: {preview}...")

    return {
        "best": scored_candidates[0]["solution"],
        "best_score": scored_candidates[0]["score"],
        "all_results": scored_candidates
    }


def demo_problems():
    """Returns a list of demo math problems for testing."""
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

    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"PRM model path '{args.model_path}' does not exist.")
        logger.error("Please run training first: make train CONFIG=configs/default.yaml")
        sys.exit(1)

    # Load config for default values
    config = {}
    if os.path.exists(args.config):
        original_argv = sys.argv
        sys.argv = ['inference.py', '--config', args.config]
        config = ConfigParser.get_config()
        sys.argv = original_argv

    # Get inference parameters (CLI args override config)
    n_candidates = args.n_candidates or config.get('inference', {}).get('n_candidates', 16)
    temperature = args.temperature or config.get('inference', {}).get('temperature', 0.7)
    verify_token = config.get('training', {}).get('response_template', '<|verify|>')

    logger.info(f"Inference settings: n_candidates={n_candidates}, temperature={temperature}")
    logger.info(f"Base model for generation: {args.base_model}")
    logger.info(f"PRM model for scoring: {args.model_path}")

    # Initialize generator (BASE model)
    logger.info("Loading generator (base model)...")
    generator = SolutionGenerator(args.base_model)

    # Initialize verifier (fine-tuned PRM)
    aggregation = args.aggregation
    logger.info(f"Loading verifier (fine-tuned PRM, aggregation: {aggregation})...")
    if args.use_vllm:
        try:
            verifier = PRMVerifierVLLM(args.model_path, verify_token=verify_token)
        except ImportError:
            logger.warning("vLLM not available, falling back to transformers")
            verifier = PRMVerifier(args.model_path, verify_token=verify_token, aggregation=aggregation)
    else:
        verifier = PRMVerifier(args.model_path, verify_token=verify_token, aggregation=aggregation)

    # Run inference
    if args.problem:
        # Single problem mode
        result = best_of_n_search(generator, verifier, args.problem,
                                  n_candidates=n_candidates, temperature=temperature)
        print(f"\n{'='*60}")
        print("BEST SOLUTION:")
        print(f"{'='*60}")
        print(result['best'])
        print(f"\nPRM Score: {result['best_score']:.4f}")
    else:
        # Demo mode - run on sample problems
        print("\n" + "="*60)
        print("Running demo on sample problems...")
        print("(Using BASE model for generation, PRM for scoring)")
        print("="*60)

        for problem in demo_problems()[:3]:
            result = best_of_n_search(generator, verifier, problem,
                                      n_candidates=n_candidates, temperature=temperature)
            print(f"\n{'='*60}")
            print(f"BEST SOLUTION for: {problem}")
            print(f"{'='*60}")
            if result['best']:
                print(result['best'][:500])
                print(f"\nPRM Score: {result['best_score']:.4f}")
            print()
