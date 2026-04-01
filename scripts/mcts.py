"""
MCTS with PRM Value Function for Mathematical Reasoning.

This implementation keeps the existing continuation-based search structure, but
it now uses shared verifier scoring and shared benchmark grading so reported
results are consistent with the main evaluation pipeline.
"""

import sys
import os
import math
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_parser import ConfigParser
from src.eval.answers import extract_numeric_answer
from src.eval.benchmarks import BenchmarkAdapter, BenchmarkExample, get_benchmark_adapter, load_benchmark
from src.scoring.verifier import TransformerPRMScorer, load_verifier_settings
from src.utils import seed_everything, setup_logging


def get_args():
    parser = argparse.ArgumentParser(description="MCTS with PRM Value Function")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned PRM model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model for generation",
    )
    parser.add_argument("--problem", type=str, default=None, help="Single problem to solve")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["gsm8k", "math500"],
        help="Dataset to evaluate on",
    )
    parser.add_argument("--n_problems", type=int, default=50, help="Number of problems to evaluate")
    parser.add_argument("--n_candidates", type=int, default=16, help="Unused legacy arg kept for CLI compatibility")
    parser.add_argument("--simulations", type=int, default=20, help="Number of MCTS simulations")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--c_puct", type=float, default=1.5, help="UCB exploration constant")
    parser.add_argument("--n_expand", type=int, default=3, help="Number of children to expand")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum tree depth")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Optional config file")
    return parser.parse_args()


class MCTSNode:
    def __init__(self, state: str, parent=None, action: str = None, prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self._cached_value: Optional[float] = None

    @property
    def value(self) -> float:
        return self.value_sum / (self.visits + 1e-8)

    def is_fully_expanded(self) -> bool:
        return len(self.children) > 0

    def is_terminal(self) -> bool:
        if self.action is None:
            return False
        return "\\boxed" in self.action or "boxed{" in self.action or "####" in self.action


class MCTSSearchPRM:
    """
    Continuation-based MCTS using generation logprobs as priors and PRM scores
    as node values.
    """

    def __init__(self, base_model, base_tokenizer, prm_scorer: TransformerPRMScorer, config: Dict = None):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.prm_scorer = prm_scorer

        self.config = config or {}
        self.c_puct = self.config.get("c_puct", 1.5)
        self.n_expand = self.config.get("n_expand", 3)
        self.temperature = self.config.get("temperature", 0.8)
        self.max_depth = self.config.get("max_depth", 10)
        self.device = self.config.get("device", "cuda")

    def search_with_checkpoints(
        self,
        problem: str,
        max_simulations: int = 50,
        checkpoints: List[int] = None,
    ) -> Dict[int, str]:
        if checkpoints is None:
            checkpoints = [1, 5, 10, 20, 50]

        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step."},
            {"role": "user", "content": problem},
        ]
        root_state = self.base_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        root = MCTSNode(state=root_state, prior=1.0)

        checkpoints = sorted([count for count in checkpoints if count <= max_simulations])
        if not checkpoints:
            checkpoints = [max_simulations]
        actual_max = max(checkpoints)

        results = {}
        checkpoint_idx = 0

        for sim in range(1, actual_max + 1):
            node = root
            depth = 0

            while node.is_fully_expanded() and not node.is_terminal() and depth < self.max_depth:
                node = self._select_child(node)
                depth += 1

            if not node.is_terminal() and depth < self.max_depth:
                node = self._expand(node)

            value = self._evaluate_prm(node, problem)
            self._backpropagate(node, value)

            if checkpoint_idx < len(checkpoints) and sim == checkpoints[checkpoint_idx]:
                results[sim] = self._get_best_solution(root)
                checkpoint_idx += 1

        return results

    def search(self, problem: str, simulations: int = 10) -> str:
        results = self.search_with_checkpoints(
            problem,
            max_simulations=simulations,
            checkpoints=[simulations],
        )
        return results.get(simulations, "")

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        sqrt_n = math.sqrt(max(1, node.visits))

        def ucb_score(child: MCTSNode) -> float:
            exploitation = child.value
            exploration = self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            return exploitation + exploration

        return max(node.children, key=ucb_score)

    def _expand(self, node: MCTSNode) -> MCTSNode:
        candidates = self._generate_steps_with_logprobs(node.state, n=self.n_expand)
        if not candidates:
            return node

        priors = np.array([candidate[1] for candidate in candidates])
        priors = priors / (priors.sum() + 1e-8)

        for idx, (step_text, _) in enumerate(candidates):
            child_state = node.state + step_text
            child = MCTSNode(child_state, parent=node, action=step_text, prior=priors[idx])
            node.children.append(child)

        return max(node.children, key=lambda child: child.prior)

    def _generate_steps_with_logprobs(self, state: str, n: int = 3) -> List[Tuple[str, float]]:
        candidates = []
        inputs = self.base_tokenizer(state, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            for _ in range(n):
                outputs = self.base_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.base_tokenizer.pad_token_id,
                    eos_token_id=self.base_tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

                generated_ids = outputs.sequences[0, input_length:]
                scores = outputs.scores

                if len(scores) > 0 and len(generated_ids) > 0:
                    total_logprob = 0.0
                    num_tokens = 0

                    for score, token_id in zip(scores, generated_ids):
                        if token_id == self.base_tokenizer.eos_token_id:
                            break
                        if token_id == self.base_tokenizer.pad_token_id:
                            continue

                        probs = F.softmax(score[0] / self.temperature, dim=-1)
                        token_prob = probs[token_id].item()
                        if token_prob > 0:
                            total_logprob += math.log(token_prob)
                            num_tokens += 1

                    prior = math.exp(total_logprob / num_tokens) if num_tokens > 0 else 0.5
                else:
                    prior = 0.5

                generated_text = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
                step = generated_text.strip()
                if step:
                    candidates.append((step, prior))

        if not candidates:
            return [("Let me solve this step by step.", 0.5)]

        return candidates

    def _evaluate_prm(self, node: MCTSNode, problem: str) -> float:
        if node._cached_value is not None:
            return node._cached_value

        solution = node.state
        if "<|im_start|>assistant" in solution:
            solution = solution.split("<|im_start|>assistant")[-1]

        score = self.prm_scorer.score_solution(problem, solution.strip())["score"]
        node._cached_value = score
        return score

    def _backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def _get_best_solution(self, root: MCTSNode) -> str:
        node = root
        solution_text = ""

        while node.children:
            node = max(node.children, key=lambda child: child.visits)
            if node.action:
                solution_text += node.action
            if node.is_terminal():
                break

        if not node.is_terminal() and solution_text:
            solution_text += self._greedy_complete(node.state)

        return solution_text

    def _greedy_complete(self, state: str) -> str:
        inputs = self.base_tokenizer(state, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.base_tokenizer.pad_token_id,
            )

        return self.base_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )


def load_models(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {args.base_model}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    base_model.eval()

    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    print(f"Loading PRM model: {args.model_path}")
    prm_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prm_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    prm_model.eval()

    if prm_tokenizer.pad_token is None:
        prm_tokenizer.pad_token = prm_tokenizer.eos_token

    return base_model, base_tokenizer, prm_model, prm_tokenizer


def evaluate_mcts(
    mcts: MCTSSearchPRM,
    problems: List[BenchmarkExample],
    adapter: BenchmarkAdapter,
    simulations_list: List[int],
) -> Dict:
    results = {f"mcts_{simulations}": 0 for simulations in simulations_list}
    results["total"] = 0

    for idx, problem in enumerate(tqdm(problems)):
        mcts_results = mcts.search_with_checkpoints(
            problem.question,
            max_simulations=max(simulations_list),
            checkpoints=simulations_list,
        )

        for simulations in simulations_list:
            if simulations not in mcts_results:
                continue

            answer = adapter.extract_answer(mcts_results[simulations])
            if adapter.check_answer(answer, problem.answer):
                results[f"mcts_{simulations}"] += 1

        results["total"] += 1

        if idx == 0:
            print("\n--- First Problem ---")
            print(f"Q: {problem.question[:80]}...")
            print(f"GT: {problem.answer}")
            for simulations in simulations_list[:3]:
                if simulations in mcts_results:
                    answer = adapter.extract_answer(mcts_results[simulations])
                    print(f"MCTS@{simulations}: {answer}")

    return results


def main():
    args = get_args()
    logger = setup_logging(args.output_dir)
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model_path):
        print(f"Error: PRM model path '{args.model_path}' does not exist.")
        sys.exit(1)

    config = {}
    if os.path.exists(args.config):
        config = ConfigParser.load_yaml(args.config)

    base_model, base_tokenizer, prm_model, prm_tokenizer = load_models(args)
    verifier_settings = load_verifier_settings(config, args.model_path)

    prm_scorer = TransformerPRMScorer(
        prm_model,
        prm_tokenizer,
        verify_token=verifier_settings["verify_token"],
        positive_label=verifier_settings["positive_label"],
        negative_label=verifier_settings["negative_label"],
        aggregation="product",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    mcts_config = {
        "c_puct": args.c_puct,
        "n_expand": args.n_expand,
        "temperature": args.temperature,
        "max_depth": args.max_depth,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    logger.info(
        "Verifier settings: token=%r positive=%r negative=%r",
        verifier_settings["verify_token"],
        verifier_settings["positive_label"],
        verifier_settings["negative_label"],
    )

    mcts = MCTSSearchPRM(base_model, base_tokenizer, prm_scorer, config=mcts_config)

    if args.problem:
        print(f"\nSolving: {args.problem}")
        print("=" * 60)
        solution = mcts.search(args.problem, simulations=args.simulations)
        print(f"\nSolution (MCTS@{args.simulations}):")
        print(solution)
        print(f"\nExtracted answer: {extract_numeric_answer(solution)}")
        return

    if args.dataset:
        print(f"\nLoading {args.dataset} dataset...")
        problems = load_benchmark(args.dataset, args.n_problems)
        adapter = get_benchmark_adapter(args.dataset)
        print(f"Evaluating {len(problems)} problems...")

        simulations_list = sorted(set([1, 5, 10, 20, args.simulations]))
        results = evaluate_mcts(mcts, problems, adapter, simulations_list)

        print("\n" + "=" * 60)
        print("MCTS EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"Problems: {results['total']}")
        print("-" * 60)

        for simulations in simulations_list:
            key = f"mcts_{simulations}"
            accuracy = results[key] / results["total"] * 100 if results["total"] else 0.0
            print(f"MCTS@{simulations:3d}:  {results[key]}/{results['total']} = {accuracy:.1f}%")

        print("=" * 60)
        return

    demo_problems = [
        "What is 15% of 80?",
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "Janet pays $40/hour for 3 hours per week of lessons. How much per year?",
    ]

    print("\nDemo mode - solving sample problems")
    print("=" * 60)

    for problem in demo_problems:
        print(f"\nProblem: {problem}")
        solution = mcts.search(problem, simulations=args.simulations)
        answer = extract_numeric_answer(solution)
        print(f"Answer: {answer}")
        print(f"Solution preview: {solution[:200]}...")


if __name__ == "__main__":
    main()
