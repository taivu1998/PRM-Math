"""
Evaluation script for Process Reward Model (PRM).

This script evaluates the PRM on standard math benchmarks (GSM8K, MATH, MATH-500)
and compares different selection strategies:
1. Pass@1 (baseline - single generation)
2. Majority Voting@N
3. PRM Reranking@N (Best-of-N with verifier)
4. PRM-Weighted Majority@N (weighted voting by PRM scores)

IMPORTANT: Uses BASE model for generation and fine-tuned PRM for scoring.
"""

import sys
import os
import json
import argparse
from collections import Counter
from typing import List, Optional

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from scripts.inference import PRMVerifier, SolutionGenerator
except ModuleNotFoundError:
    from inference import PRMVerifier, SolutionGenerator
from src.config_parser import ConfigParser
from src.eval.benchmarks import BenchmarkAdapter, get_benchmark_adapter, load_benchmark
from src.scoring.verifier import load_verifier_settings
from src.utils import setup_logging, seed_everything


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate PRM on math benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned PRM model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model for generation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "math", "math500"],
        help="Evaluation dataset (gsm8k, math, or math500)",
    )
    parser.add_argument(
        "--n_candidates",
        type=int,
        default=16,
        help="Number of candidates for Best-of-N",
    )
    parser.add_argument(
        "--n_problems",
        type=int,
        default=100,
        help="Number of problems to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--aggregation",
        type=str,
        default="min",
        choices=["min", "product"],
        help="Score aggregation method (min=weakest link, product=cumulative)",
    )
    return parser.parse_args()


def majority_vote(answers: List[str]) -> Optional[str]:
    valid_answers = [answer for answer in answers if answer is not None]
    if not valid_answers:
        return None
    return Counter(valid_answers).most_common(1)[0][0]


class Evaluator:
    def __init__(self, generator: SolutionGenerator, verifier: PRMVerifier, adapter: BenchmarkAdapter):
        self.generator = generator
        self.verifier = verifier
        self.adapter = adapter

    def evaluate_problem(self, problem, n_candidates: int, temperature: float) -> dict:
        question = problem.question
        ground_truth = problem.answer

        solutions = self.generator.generate(question, n_candidates, temperature)
        if not solutions:
            return {
                "pass_at_1": False,
                "majority_vote": False,
                "prm_rerank": False,
                "prm_weighted": False,
                "n_generated": 0,
            }

        answers = [self.adapter.extract_answer(solution) for solution in solutions]
        scores = [self.verifier.score_solution(question, solution)["score"] for solution in solutions]

        pass_at_1 = self.adapter.check_answer(answers[0], ground_truth) if answers else False

        voted_answer = majority_vote(answers)
        majority_correct = self.adapter.check_answer(voted_answer, ground_truth)

        best_idx = int(np.argmax(scores))
        prm_answer = answers[best_idx] if best_idx < len(answers) else None
        prm_correct = self.adapter.check_answer(prm_answer, ground_truth)

        answer_weights = {}
        for answer, score in zip(answers, scores):
            if answer is None:
                continue

            if self.adapter.uses_math_grading:
                matched_key = None
                for existing in answer_weights:
                    if self.adapter.check_answer(answer, existing):
                        matched_key = existing
                        break

                if matched_key is None:
                    answer_weights[answer] = score
                else:
                    answer_weights[matched_key] += score
            else:
                answer_weights[answer] = answer_weights.get(answer, 0.0) + score

        weighted_best = None
        prm_weighted_correct = False
        if answer_weights:
            weighted_best = max(answer_weights, key=answer_weights.get)
            prm_weighted_correct = self.adapter.check_answer(weighted_best, ground_truth)

        return {
            "pass_at_1": pass_at_1,
            "majority_vote": majority_correct,
            "prm_rerank": prm_correct,
            "prm_weighted": prm_weighted_correct,
            "n_generated": len(solutions),
            "answers": answers,
            "scores": scores,
            "best_score": scores[best_idx] if scores else None,
            "weighted_best": weighted_best,
        }


def run_evaluation(args):
    logger = setup_logging(args.output_dir)
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading {args.dataset} dataset...")
    problems = load_benchmark(args.dataset, args.n_problems)
    adapter = get_benchmark_adapter(args.dataset)
    logger.info(f"Loaded {len(problems)} problems")

    config = {}
    if os.path.exists(args.config):
        config = ConfigParser.load_yaml(args.config)
    verifier_settings = load_verifier_settings(config, args.model_path)

    logger.info(f"Loading base model for generation: {args.base_model}")
    generator = SolutionGenerator(args.base_model)

    logger.info(f"Loading fine-tuned PRM for scoring: {args.model_path}")
    verifier = PRMVerifier(
        args.model_path,
        verify_token=verifier_settings["verify_token"],
        aggregation=args.aggregation,
        positive_label=verifier_settings["positive_label"],
        negative_label=verifier_settings["negative_label"],
    )

    evaluator = Evaluator(generator, verifier, adapter)

    results = []
    metrics = {"pass_at_1": 0, "majority_vote": 0, "prm_rerank": 0, "prm_weighted": 0}

    logger.info(f"Evaluating {len(problems)} problems with N={args.n_candidates}...")
    logger.info(
        "Using BASE model for generation, PRM for scoring "
        f"(aggregation: {args.aggregation})"
    )

    for idx, problem in enumerate(tqdm(problems)):
        result = evaluator.evaluate_problem(problem, args.n_candidates, args.temperature)
        results.append(
            {
                "problem_idx": idx,
                "question": problem.question,
                "ground_truth": problem.answer,
                "metadata": problem.metadata,
                **result,
            }
        )

        for key in metrics:
            if result.get(key, False):
                metrics[key] += 1

        if (idx + 1) % 10 == 0:
            logger.info(f"Progress: {idx + 1}/{len(problems)}")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}/{idx + 1} = {value / (idx + 1) * 100:.1f}%")

    n_total = len(problems)
    final_metrics = {
        key: {"correct": value, "total": n_total, "accuracy": value / n_total * 100}
        for key, value in metrics.items()
    }

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

    baseline = final_metrics["pass_at_1"]["accuracy"]
    prm_acc = final_metrics["prm_rerank"]["accuracy"]
    prm_weighted_acc = final_metrics["prm_weighted"]["accuracy"]
    print(f"\nPRM Rerank improvement over Pass@1: {prm_acc - baseline:+.2f}%")
    print(f"PRM-Weighted improvement over Pass@1: {prm_weighted_acc - baseline:+.2f}%")

    output_file = os.path.join(args.output_dir, f"eval_{args.dataset}_n{args.n_candidates}.json")
    with open(output_file, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "verifier_settings": verifier_settings,
                "metrics": final_metrics,
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to: {output_file}")
    return final_metrics


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.model_path):
        print(f"Error: PRM model path '{args.model_path}' does not exist.")
        print("Please run training first: make train CONFIG=configs/default.yaml")
        sys.exit(1)

    run_evaluation(args)
