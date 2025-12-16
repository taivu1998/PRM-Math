"""
MCTS with PRM Value Function for Mathematical Reasoning.

Monte Carlo Tree Search using:
- Generation logprobs as prior probabilities
- Trained PRM as value function for node evaluation

Usage:
    python scripts/mcts.py --model_path checkpoints/merged_model --problem "What is 2+2?"
    python scripts/mcts.py --model_path checkpoints/merged_model --dataset gsm8k --n_problems 50
"""

import sys
import os
import math
import argparse
import re
from collections import Counter
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import setup_logging, seed_everything


def get_args():
    parser = argparse.ArgumentParser(description="MCTS with PRM Value Function")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned PRM model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct",
                        help="Base model for generation")
    parser.add_argument("--problem", type=str, default=None, help="Single problem to solve")
    parser.add_argument("--dataset", type=str, default=None, choices=["gsm8k", "math500"],
                        help="Dataset to evaluate on")
    parser.add_argument("--n_problems", type=int, default=50, help="Number of problems to evaluate")
    parser.add_argument("--n_candidates", type=int, default=16, help="Number of candidates for comparison")
    parser.add_argument("--simulations", type=int, default=20, help="Number of MCTS simulations")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--c_puct", type=float, default=1.5, help="UCB exploration constant")
    parser.add_argument("--n_expand", type=int, default=3, help="Number of children to expand")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum tree depth")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, state: str, parent=None, action: str = None, prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
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
    MCTS using generation logprobs as prior and trained PRM as value function.

    The search process:
    1. Selection: Use UCB to select promising nodes
    2. Expansion: Generate candidate next steps with logprob priors
    3. Evaluation: Use PRM to evaluate the current solution quality
    4. Backpropagation: Update node statistics
    """

    def __init__(self, base_model, base_tokenizer, prm_model, prm_tokenizer,
                 pos_token_id: int, neg_token_id: int, verify_token: str = "<|verify|>",
                 config: Dict = None):
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.prm_model = prm_model
        self.prm_tokenizer = prm_tokenizer
        self.pos_token_id = pos_token_id
        self.neg_token_id = neg_token_id
        self.verify_token = verify_token

        self.config = config or {}
        self.c_puct = self.config.get("c_puct", 1.5)
        self.n_expand = self.config.get("n_expand", 3)
        self.temperature = self.config.get("temperature", 0.8)
        self.max_depth = self.config.get("max_depth", 10)
        self.device = self.config.get("device", "cuda")

    def search_with_checkpoints(self, problem: str, max_simulations: int = 50,
                                 checkpoints: List[int] = None) -> Dict[int, str]:
        """
        Run MCTS once and record best solution at each checkpoint.

        This is more efficient than running separate searches because the tree
        is built incrementally.

        Args:
            problem: The math problem to solve
            max_simulations: Maximum number of simulations to run
            checkpoints: List of simulation counts to record results at

        Returns:
            Dict mapping simulation count to best solution at that point
        """
        if checkpoints is None:
            checkpoints = [1, 5, 10, 20, 50]

        # Create root node
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the problem step by step."},
            {"role": "user", "content": problem}
        ]
        root_state = self.base_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        root = MCTSNode(state=root_state, prior=1.0)

        # Sort checkpoints and ensure max_simulations covers all
        checkpoints = sorted([c for c in checkpoints if c <= max_simulations])
        if not checkpoints:
            checkpoints = [max_simulations]
        actual_max = max(checkpoints)

        results = {}
        checkpoint_idx = 0

        for sim in range(1, actual_max + 1):
            node = root

            # Selection
            depth = 0
            while node.is_fully_expanded() and not node.is_terminal() and depth < self.max_depth:
                node = self._select_child(node)
                depth += 1

            # Expansion
            if not node.is_terminal() and depth < self.max_depth:
                node = self._expand(node, problem)

            # Evaluation
            value = self._evaluate_prm(node, problem)

            # Backpropagation
            self._backpropagate(node, value)

            # Check if we hit a checkpoint
            if checkpoint_idx < len(checkpoints) and sim == checkpoints[checkpoint_idx]:
                results[sim] = self._get_best_solution(root, problem)
                checkpoint_idx += 1

        return results

    def search(self, problem: str, simulations: int = 10) -> str:
        """
        Single search returning best solution.

        Args:
            problem: The math problem to solve
            simulations: Number of MCTS simulations

        Returns:
            Best solution found
        """
        results = self.search_with_checkpoints(
            problem, max_simulations=simulations, checkpoints=[simulations]
        )
        return results.get(simulations, "")

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Select child using UCB formula."""
        sqrt_n = math.sqrt(max(1, node.visits))

        def ucb_score(child: MCTSNode) -> float:
            exploitation = child.value
            exploration = self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            return exploitation + exploration

        return max(node.children, key=ucb_score)

    def _expand(self, node: MCTSNode, problem: str) -> MCTSNode:
        """Expand node by generating candidate next steps."""
        candidates = self._generate_steps_with_logprobs(node.state, n=self.n_expand)

        if not candidates:
            return node

        # Normalize priors
        priors = np.array([c[1] for c in candidates])
        priors = priors / (priors.sum() + 1e-8)

        for i, (step_text, _) in enumerate(candidates):
            child_state = node.state + step_text
            child = MCTSNode(child_state, parent=node, action=step_text, prior=priors[i])
            node.children.append(child)

        # Return child with highest prior for immediate evaluation
        return max(node.children, key=lambda c: c.prior)

    def _generate_steps_with_logprobs(self, state: str, n: int = 3) -> List[Tuple[str, float]]:
        """Generate n candidate next steps with logprob-based priors."""
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

                    for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
                        if token_id == self.base_tokenizer.eos_token_id:
                            break
                        if token_id == self.base_tokenizer.pad_token_id:
                            continue

                        probs = F.softmax(score[0] / self.temperature, dim=-1)
                        token_prob = probs[token_id].item()

                        if token_prob > 0:
                            total_logprob += math.log(token_prob)
                            num_tokens += 1

                    if num_tokens > 0:
                        avg_logprob = total_logprob / num_tokens
                        prior = math.exp(avg_logprob)
                    else:
                        prior = 0.5
                else:
                    prior = 0.5

                generated_text = self.base_tokenizer.decode(generated_ids, skip_special_tokens=True)
                step = generated_text.strip()

                if step:
                    candidates.append((step, prior))

        if not candidates:
            candidates = [("Let me solve this step by step.", 0.5)]

        return candidates

    def _evaluate_prm(self, node: MCTSNode, problem: str) -> float:
        """Evaluate node using PRM."""
        if node._cached_value is not None:
            return node._cached_value

        solution = node.state
        # Extract just the assistant response
        if "<|im_start|>assistant" in solution:
            solution = solution.split("<|im_start|>assistant")[-1]

        score = self._prm_score(problem, solution.strip())
        node._cached_value = score
        return score

    def _prm_score(self, problem: str, solution: str) -> float:
        """Score a solution using the PRM with product aggregation."""
        if not solution:
            return 0.0

        steps = [s.strip() for s in solution.split("\n") if s.strip()]
        if not steps:
            return 0.0

        context = f"Problem: {problem}\n\nSolution:"
        step_scores = []

        for step in steps:
            text = f"{context}\n{step}\n{self.verify_token}"
            inputs = self.prm_tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.prm_model(**inputs)
                logits = outputs.logits[0, -1, :]

            probs = F.softmax(logits, dim=-1)
            pos_prob = probs[self.pos_token_id].item()
            neg_prob = probs[self.neg_token_id].item()

            if pos_prob + neg_prob > 0:
                score = pos_prob / (pos_prob + neg_prob)
            else:
                score = 0.5

            step_scores.append(score)
            context = f"{context}\n{step}"

        # Product aggregation
        if step_scores:
            return math.prod(step_scores)
        return 0.0

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent

    def _get_best_solution(self, root: MCTSNode, problem: str) -> str:
        """Extract best solution from tree by following most visited path."""
        node = root
        solution_text = ""

        while node.children:
            node = max(node.children, key=lambda c: c.visits)
            if node.action:
                solution_text += node.action

            if node.is_terminal():
                break

        # If solution incomplete, do greedy completion
        if not node.is_terminal() and solution_text:
            completion = self._greedy_complete(node.state)
            solution_text += completion

        return solution_text

    def _greedy_complete(self, state: str) -> str:
        """Greedily complete a partial solution."""
        inputs = self.base_tokenizer(state, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.base_tokenizer.pad_token_id,
            )

        generated = self.base_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return generated


def extract_answer(text: str) -> Optional[str]:
    """Extract numerical answer from solution."""
    if not text:
        return None

    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?([+-]?[\d,\.]+)",
        r"[Aa]nswer[:\s]*\$?([+-]?[\d,\.]+)",
        r"=\s*\$?([+-]?[\d,\.]+)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).replace(",", "").replace("$", "").strip()
            try:
                return str(float(answer))
            except:
                continue

    # Fallback: last number
    numbers = re.findall(r"([+-]?\d+\.?\d*)", text)
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


def load_models(args):
    """Load base model and PRM model."""
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

    pos_token_id = prm_tokenizer.encode("+", add_special_tokens=False)[0]
    neg_token_id = prm_tokenizer.encode("-", add_special_tokens=False)[0]

    print(f"Token IDs: + = {pos_token_id}, - = {neg_token_id}")

    return base_model, base_tokenizer, prm_model, prm_tokenizer, pos_token_id, neg_token_id


def evaluate_mcts(mcts: MCTSSearchPRM, problems: List[Dict], n_candidates: int,
                  simulations_list: List[int], temperature: float) -> Dict:
    """Evaluate MCTS on a list of problems."""
    from scripts.inference import SolutionGenerator, PRMVerifier

    results = {
        "pass_1": 0,
        "majority": 0,
        "prm_rerank": 0,
    }
    for sims in simulations_list:
        results[f"mcts_{sims}"] = 0

    results["total"] = 0

    for idx, problem_data in enumerate(tqdm(problems)):
        question = problem_data["question"]
        ground_truth = problem_data["answer"]

        # MCTS with checkpoints (single tree build, multiple results)
        mcts_results = mcts.search_with_checkpoints(
            question,
            max_simulations=max(simulations_list),
            checkpoints=simulations_list
        )

        for sims in simulations_list:
            if sims in mcts_results:
                answer = extract_answer(mcts_results[sims])
                if check_answer(answer, ground_truth):
                    results[f"mcts_{sims}"] += 1

        results["total"] += 1

        # Debug first problem
        if idx == 0:
            print(f"\n--- First Problem ---")
            print(f"Q: {question[:80]}...")
            print(f"GT: {ground_truth}")
            for sims in simulations_list[:3]:
                if sims in mcts_results:
                    ans = extract_answer(mcts_results[sims])
                    print(f"MCTS@{sims}: {ans}")

    return results


def main():
    args = get_args()
    logger = setup_logging(args.output_dir)
    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.model_path):
        print(f"Error: PRM model path '{args.model_path}' does not exist.")
        sys.exit(1)

    # Load models
    base_model, base_tokenizer, prm_model, prm_tokenizer, pos_id, neg_id = load_models(args)

    # Create MCTS searcher
    mcts_config = {
        "c_puct": args.c_puct,
        "n_expand": args.n_expand,
        "temperature": args.temperature,
        "max_depth": args.max_depth,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    mcts = MCTSSearchPRM(
        base_model, base_tokenizer,
        prm_model, prm_tokenizer,
        pos_id, neg_id,
        verify_token="<|verify|>",
        config=mcts_config
    )

    if args.problem:
        # Single problem mode
        print(f"\nSolving: {args.problem}")
        print("=" * 60)

        solution = mcts.search(args.problem, simulations=args.simulations)
        print(f"\nSolution (MCTS@{args.simulations}):")
        print(solution)
        print(f"\nExtracted answer: {extract_answer(solution)}")

    elif args.dataset:
        # Dataset evaluation mode
        from datasets import load_dataset

        print(f"\nLoading {args.dataset} dataset...")

        if args.dataset == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            problems = []
            for entry in list(ds)[:args.n_problems]:
                gt_match = re.search(r"####\s*([-\d,\.]+)", entry["answer"])
                if gt_match:
                    problems.append({
                        "question": entry["question"],
                        "answer": str(float(gt_match.group(1).replace(",", "")))
                    })
        elif args.dataset == "math500":
            try:
                ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            except:
                ds = load_dataset("lighteval/MATH", split="test")

            problems = []
            for entry in list(ds)[:args.n_problems]:
                problems.append({
                    "question": entry.get("problem", entry.get("question", "")),
                    "answer": entry.get("answer", "")
                })

        print(f"Evaluating {len(problems)} problems...")

        simulations_list = [1, 5, 10, 20, args.simulations]
        simulations_list = sorted(set(simulations_list))

        results = evaluate_mcts(mcts, problems, args.n_candidates, simulations_list, args.temperature)

        # Print results
        print("\n" + "=" * 60)
        print("MCTS EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset: {args.dataset}")
        print(f"Problems: {results['total']}")
        print("-" * 60)

        for sims in simulations_list:
            key = f"mcts_{sims}"
            if key in results:
                acc = results[key] / results["total"] * 100
                print(f"MCTS@{sims:3d}:  {results[key]}/{results['total']} = {acc:.1f}%")

        print("=" * 60)

    else:
        # Demo mode
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
            answer = extract_answer(solution)
            print(f"Answer: {answer}")
            print(f"Solution preview: {solution[:200]}...")


if __name__ == "__main__":
    main()
