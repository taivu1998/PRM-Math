"""
Evaluation script for Process Reward Model (PRM).

This script evaluates the PRM on standard math benchmarks (GSM8K, MATH, MATH-500)
and compares different selection strategies:
1. Pass@1 (baseline - single generation)
2. Majority Voting@N
3. PRM Reranking@N (Best-of-N with verifier)
4. PRM-Weighted Majority@N (weighted voting by PRM scores)

IMPORTANT: Uses BASE model for generation and fine-tuned PRM for scoring.

Usage:
    python scripts/evaluate.py --model_path checkpoints/merged_model --dataset gsm8k --n_candidates 16
    python scripts/evaluate.py --model_path checkpoints/merged_model --dataset math500 --n_candidates 8
"""

import sys
import os
import json
import argparse
import re
import math
from collections import Counter
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import setup_logging, seed_everything
from src.config_parser import ConfigParser


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate PRM on math benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned PRM model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help="Base model for generation")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math", "math500"],
                        help="Evaluation dataset (gsm8k, math, or math500)")
    parser.add_argument("--n_candidates", type=int, default=16, help="Number of candidates for Best-of-N")
    parser.add_argument("--n_problems", type=int, default=100, help="Number of problems to evaluate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--aggregation", type=str, default="min", choices=["min", "product"],
                        help="Score aggregation method (min=weakest link, product=cumulative)")
    return parser.parse_args()


def load_gsm8k(n_problems: int = 100) -> List[Dict]:
    """Load GSM8K test set."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")

    problems = []
    for i, entry in enumerate(ds):
        if i >= n_problems:
            break

        # Extract the numeric answer from the solution
        answer_text = entry['answer']
        # GSM8K format: "#### <number>"
        if "####" in answer_text:
            answer = answer_text.split("####")[-1].strip()
            answer = answer.replace(",", "")
        else:
            answer = answer_text.strip()

        problems.append({
            "question": entry['question'],
            "answer": answer,
            "full_solution": answer_text
        })

    return problems


def load_math_dataset(n_problems: int = 100) -> List[Dict]:
    """Load MATH dataset test set."""
    from datasets import load_dataset

    try:
        ds = load_dataset("hendrycks/competition_math", split="test")
    except Exception:
        ds = load_dataset("competition_math", split="test")

    problems = []
    for i, entry in enumerate(ds):
        if i >= n_problems:
            break

        problems.append({
            "question": entry['problem'],
            "answer": entry['solution'],
            "level": entry.get('level', 'unknown'),
            "type": entry.get('type', 'unknown')
        })

    return problems


def load_math500(n_problems: int = 100) -> List[Dict]:
    """Load MATH-500 dataset (competition-level problems)."""
    from datasets import load_dataset

    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        print(f"Loaded MATH-500 with {len(ds)} problems")
    except Exception as e:
        print(f"Error loading MATH-500: {e}")
        print("Trying alternative: lighteval/MATH...")
        try:
            ds = load_dataset("lighteval/MATH", split="test")
        except Exception:
            print("Falling back to hendrycks/competition_math...")
            ds = load_dataset("hendrycks/competition_math", split="test")

    problems = []
    for i, entry in enumerate(ds):
        if i >= n_problems:
            break

        # Handle different column names
        question = entry.get("problem", entry.get("question", ""))

        # Get ground truth answer
        if "answer" in entry:
            answer = entry["answer"]
        elif "solution" in entry:
            answer = extract_math_answer(entry["solution"])
        else:
            answer = ""

        if question and answer:
            problems.append({
                "question": question,
                "answer": answer,
                "level": entry.get('level', 'unknown'),
                "type": entry.get('type', 'unknown')
            })

    return problems


def normalize_math_answer(answer: str) -> Optional[str]:
    """Normalize MATH dataset answers for comparison."""
    if answer is None:
        return None

    answer = str(answer).strip()

    # Remove LaTeX formatting
    answer = answer.replace("\\$", "").replace("$", "")
    answer = answer.replace("\\%", "%").replace("\\!", "")

    # Handle common LaTeX commands
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\textbf\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
    answer = re.sub(r'\\left|\\right', '', answer)

    # Handle fractions: \frac{a}{b}
    frac_match = re.search(r'\\d?frac\{([^}]*)\}\{([^}]*)\}', answer)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                answer = str(num / den)
        except:
            pass

    # Handle sqrt
    sqrt_match = re.search(r'\\sqrt\{([^}]*)\}', answer)
    if sqrt_match:
        try:
            val = float(sqrt_match.group(1))
            answer = str(math.sqrt(val))
        except:
            pass

    # Remove remaining LaTeX commands
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = answer.replace("{", "").replace("}", "").replace(" ", "").strip()

    try:
        return str(float(answer))
    except:
        return answer.lower()


def extract_math_answer(text: str) -> Optional[str]:
    """Extract answer from MATH-style solutions."""
    if not text:
        return None

    # Try \boxed{answer}
    boxed_patterns = [
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\\boxed\{([^}]+)\}',
    ]
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return normalize_math_answer(matches[-1])

    # Try "the answer is X" patterns
    answer_patterns = [
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?([^\$\n]+)\$?',
        r'[Aa]nswer[:\s]*\$?([^\$\n]+)\$?',
        r'=\s*\$?([^\$\n]+)\$?\s*$',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_math_answer(match.group(1))

    # Fallback: last number
    numbers = re.findall(r'([+-]?\d+\.?\d*)', text)
    if numbers:
        return normalize_math_answer(numbers[-1])
    return None


def check_math_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth (for MATH dataset)."""
    if predicted is None or ground_truth is None:
        return False

    pred_norm = normalize_math_answer(predicted)
    gt_norm = normalize_math_answer(ground_truth)

    if pred_norm is None or gt_norm is None:
        return False

    if pred_norm == gt_norm:
        return True

    try:
        return abs(float(pred_norm) - float(gt_norm)) < 1e-4
    except:
        return False


def extract_answer(solution: str) -> Optional[str]:
    """
    Extract the final numeric answer from a solution.
    Handles various formats like "The answer is X", "= X", boxed, etc.
    """
    if not solution:
        return None

    solution = solution.strip()

    # Try common patterns in order of specificity
    patterns = [
        r"\\boxed\{([^}]+)\}",  # LaTeX boxed
        r"\$\\boxed\{([^}]+)\}\$",  # $\boxed{}$
        r"####\s*([+-]?\d+[\d,]*\.?\d*)",  # GSM8K format
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?([+-]?[\d,\.]+)",
        r"[Aa]nswer[:\s]*\$?([+-]?[\d,\.]+)",
        r"=\s*\$?([+-]?[\d,\.]+)\s*$",
        r"[Tt]herefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([+-]?[\d,\.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution, re.IGNORECASE)
        if match:
            answer = match.group(1).replace(",", "").replace("$", "").strip()
            try:
                if "/" in answer:
                    parts = answer.split("/")
                    return str(float(parts[0]) / float(parts[1]))
                return str(float(answer))
            except:
                continue

    # Fallback: find the last number in the solution
    numbers = re.findall(r"([+-]?\d+\.?\d*)", solution)
    if numbers:
        try:
            return str(float(numbers[-1]))
        except:
            pass

    return None


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if predicted is None:
        return False

    pred = str(predicted).strip().replace(",", "").lower()
    gt = str(ground_truth).strip().replace(",", "").lower()

    try:
        pred_num = float(pred)
        gt_num = float(gt)
        return abs(pred_num - gt_num) < 1e-6
    except ValueError:
        pass

    return pred == gt


def majority_vote(answers: List[str]) -> Optional[str]:
    """Return the most common answer from a list."""
    valid_answers = [a for a in answers if a is not None]
    if not valid_answers:
        return None

    counter = Counter(valid_answers)
    return counter.most_common(1)[0][0]


class SolutionGenerator:
    """Generate candidate solutions using the base math model."""

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading generator (base model): {model_name}")
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
                 temperature: float = 0.7, max_new_tokens: int = 512) -> List[str]:
        """Generate multiple candidate solutions."""
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step and provide the final numerical answer."},
            {"role": "user", "content": problem}
        ]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            solutions.append(generated.strip())

        return solutions


class PRMVerifier:
    """PRM Verifier for scoring solutions."""

    def __init__(self, model_path: str, verify_token: str = "<|verify|>", device: str = "cuda",
                 aggregation: str = "min"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading PRM verifier: {model_path}")
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

        self.pos_token_id = self.tokenizer.encode("+", add_special_tokens=False)[0]
        self.neg_token_id = self.tokenizer.encode("-", add_special_tokens=False)[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"PRM Verifier loaded! (+ token: {self.pos_token_id}, aggregation: {aggregation})")

    def score_solution(self, problem: str, solution: str) -> float:
        """Score a solution using step-wise PRM."""
        steps = [s.strip() for s in solution.split("\n") if s.strip()]

        if not steps:
            return 0.0

        context = f"Problem: {problem}\n\nSolution:"
        step_scores = []

        for step in steps:
            text = f"{context}\n{step}\n{self.verify_token}"
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[0, -1, :]

            probs = F.softmax(next_token_logits, dim=-1)
            pos_prob = probs[self.pos_token_id].item()
            neg_prob = probs[self.neg_token_id].item()

            score = pos_prob / (pos_prob + neg_prob) if (pos_prob + neg_prob) > 0 else 0.5
            step_scores.append(score)
            context = f"{context}\n{step}"

        if not step_scores:
            return 0.0

        if self.aggregation == "product":
            return math.prod(step_scores)
        else:  # min (weakest link)
            return min(step_scores)


class Evaluator:
    """Evaluation harness using separate generator and verifier."""

    def __init__(self, generator: SolutionGenerator, verifier: PRMVerifier,
                 use_math_check: bool = False):
        self.generator = generator
        self.verifier = verifier
        self.use_math_check = use_math_check  # Use MATH-style answer checking

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check answer using appropriate method."""
        if self.use_math_check:
            return check_math_answer(predicted, ground_truth)
        return check_answer(predicted, ground_truth)

    def _extract_answer(self, solution: str) -> Optional[str]:
        """Extract answer using appropriate method."""
        if self.use_math_check:
            return extract_math_answer(solution)
        return extract_answer(solution)

    def evaluate_problem(self, problem: Dict, n_candidates: int, temperature: float) -> Dict:
        """Evaluate a single problem with multiple strategies."""
        question = problem['question']
        ground_truth = problem['answer']

        # Generate candidates using BASE model
        solutions = self.generator.generate(question, n_candidates, temperature)

        if not solutions:
            return {
                "pass_at_1": False,
                "majority_vote": False,
                "prm_rerank": False,
                "prm_weighted": False,
                "n_generated": 0
            }

        # Extract answers
        answers = [self._extract_answer(sol) for sol in solutions]

        # Score all solutions
        scores = [self.verifier.score_solution(question, sol) for sol in solutions]

        # 1. Pass@1
        pass_at_1 = self._check_answer(answers[0], ground_truth) if answers else False

        # 2. Majority Vote
        voted_answer = majority_vote(answers)
        majority_correct = self._check_answer(voted_answer, ground_truth)

        # 3. PRM Reranking (best score)
        best_idx = np.argmax(scores)
        prm_answer = answers[best_idx] if best_idx < len(answers) else None
        prm_correct = self._check_answer(prm_answer, ground_truth)

        # 4. PRM-Weighted Majority Voting
        answer_weights = {}
        for ans, score in zip(answers, scores):
            if ans:
                # For MATH dataset, group equivalent answers
                if self.use_math_check:
                    found = False
                    for existing in answer_weights:
                        if check_math_answer(ans, existing):
                            answer_weights[existing] += score
                            found = True
                            break
                    if not found:
                        answer_weights[ans] = score
                else:
                    answer_weights[ans] = answer_weights.get(ans, 0) + score

        prm_weighted_correct = False
        weighted_best = None
        if answer_weights:
            weighted_best = max(answer_weights, key=answer_weights.get)
            prm_weighted_correct = self._check_answer(weighted_best, ground_truth)

        return {
            "pass_at_1": pass_at_1,
            "majority_vote": majority_correct,
            "prm_rerank": prm_correct,
            "prm_weighted": prm_weighted_correct,
            "n_generated": len(solutions),
            "answers": answers,
            "scores": scores,
            "best_score": scores[best_idx] if scores else None,
            "weighted_best": weighted_best
        }


def run_evaluation(args):
    """Run full evaluation."""
    logger = setup_logging(args.output_dir)
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    use_math_check = False
    if args.dataset == "gsm8k":
        problems = load_gsm8k(args.n_problems)
    elif args.dataset == "math500":
        problems = load_math500(args.n_problems)
        use_math_check = True
    else:  # math
        problems = load_math_dataset(args.n_problems)
        use_math_check = True

    logger.info(f"Loaded {len(problems)} problems")

    # Load config for verify token
    verify_token = "<|verify|>"
    if os.path.exists(args.config):
        config = ConfigParser.load_yaml(args.config)
        verify_token = config.get('training', {}).get('response_template', verify_token)

    # Initialize generator (BASE model) and verifier (fine-tuned PRM)
    logger.info(f"Loading base model for generation: {args.base_model}")
    generator = SolutionGenerator(args.base_model)

    logger.info(f"Loading fine-tuned PRM for scoring: {args.model_path}")
    verifier = PRMVerifier(args.model_path, verify_token=verify_token, aggregation=args.aggregation)

    evaluator = Evaluator(generator, verifier, use_math_check=use_math_check)

    # Run evaluation
    results = []
    metrics = {"pass_at_1": 0, "majority_vote": 0, "prm_rerank": 0, "prm_weighted": 0}

    logger.info(f"Evaluating {len(problems)} problems with N={args.n_candidates}...")
    logger.info(f"Using BASE model for generation, PRM for scoring (aggregation: {args.aggregation})")

    for i, problem in enumerate(tqdm(problems)):
        result = evaluator.evaluate_problem(problem, args.n_candidates, args.temperature)
        results.append({
            "problem_idx": i,
            "question": problem['question'],
            "ground_truth": problem['answer'],
            **result
        })

        for key in ["pass_at_1", "majority_vote", "prm_rerank", "prm_weighted"]:
            if result.get(key, False):
                metrics[key] += 1

        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(problems)}")
            for key, val in metrics.items():
                logger.info(f"  {key}: {val}/{i+1} = {val/(i+1)*100:.1f}%")

    # Final metrics
    n_total = len(problems)
    final_metrics = {
        key: {"correct": val, "total": n_total, "accuracy": val / n_total * 100}
        for key, val in metrics.items()
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"N problems: {n_total}")
    print(f"N candidates: {args.n_candidates}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Base model: {args.base_model}")
    print(f"PRM model: {args.model_path}")
    print("-" * 60)
    print(f"Pass@1:           {final_metrics['pass_at_1']['accuracy']:.2f}%")
    print(f"Majority@{args.n_candidates}:      {final_metrics['majority_vote']['accuracy']:.2f}%")
    print(f"PRM Rerank@{args.n_candidates}:    {final_metrics['prm_rerank']['accuracy']:.2f}%")
    print(f"PRM-Weighted@{args.n_candidates}:  {final_metrics['prm_weighted']['accuracy']:.2f}%")
    print("=" * 60)

    baseline = final_metrics['pass_at_1']['accuracy']
    prm_acc = final_metrics['prm_rerank']['accuracy']
    prm_weighted_acc = final_metrics['prm_weighted']['accuracy']
    print(f"\nPRM Rerank improvement over Pass@1: {prm_acc - baseline:+.2f}%")
    print(f"PRM-Weighted improvement over Pass@1: {prm_weighted_acc - baseline:+.2f}%")

    # Save results
    output_file = os.path.join(args.output_dir, f"eval_{args.dataset}_n{args.n_candidates}.json")
    with open(output_file, 'w') as f:
        json.dump({
            "config": vars(args),
            "metrics": final_metrics,
            "results": results
        }, f, indent=2)

    logger.info(f"Results saved to: {output_file}")

    return final_metrics


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.model_path):
        print(f"Error: PRM model path '{args.model_path}' does not exist.")
        print("Please run training first: make train CONFIG=configs/default.yaml")
        sys.exit(1)

    run_evaluation(args)
