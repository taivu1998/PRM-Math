from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .answers import (
    check_math_answer,
    check_numeric_answer,
    extract_math_answer,
    extract_numeric_answer,
    normalize_math_answer,
)


@dataclass(frozen=True)
class BenchmarkExample:
    question: str
    answer: str
    full_solution: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    extract_answer: Callable[[str], Optional[str]]
    check_answer: Callable[[str, str], bool]
    uses_math_grading: bool


def get_benchmark_adapter(name: str) -> BenchmarkAdapter:
    if name in {"math", "math500"}:
        return BenchmarkAdapter(
            name=name,
            extract_answer=extract_math_answer,
            check_answer=check_math_answer,
            uses_math_grading=True,
        )
    if name == "gsm8k":
        return BenchmarkAdapter(
            name=name,
            extract_answer=extract_numeric_answer,
            check_answer=check_numeric_answer,
            uses_math_grading=False,
        )
    raise ValueError(f"Unsupported benchmark: {name}")


def load_benchmark(name: str, n_problems: int = 100) -> List[BenchmarkExample]:
    if name == "gsm8k":
        return _load_gsm8k(n_problems)
    if name == "math":
        return _load_math(n_problems)
    if name == "math500":
        return _load_math500(n_problems)
    raise ValueError(f"Unsupported benchmark: {name}")


def _load_gsm8k(n_problems: int) -> List[BenchmarkExample]:
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", "main", split="test")
    examples = []

    for entry in dataset.select(range(min(n_problems, len(dataset)))):
        answer = extract_numeric_answer(entry["answer"])
        if answer is None:
            continue
        examples.append(
            BenchmarkExample(
                question=entry["question"],
                answer=answer,
                full_solution=entry["answer"],
            )
        )

    return examples


def _load_math(n_problems: int) -> List[BenchmarkExample]:
    from datasets import load_dataset

    try:
        dataset = load_dataset("hendrycks/competition_math", split="test")
    except Exception:
        dataset = load_dataset("competition_math", split="test")

    examples = []
    for entry in dataset.select(range(min(n_problems, len(dataset)))):
        answer = extract_math_answer(entry["solution"])
        if answer is None:
            continue
        examples.append(
            BenchmarkExample(
                question=entry["problem"],
                answer=answer,
                full_solution=entry["solution"],
                metadata={
                    "level": str(entry.get("level", "unknown")),
                    "type": str(entry.get("type", "unknown")),
                },
            )
        )

    return examples


def _load_math500(n_problems: int) -> List[BenchmarkExample]:
    from datasets import load_dataset

    try:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        try:
            dataset = load_dataset("lighteval/MATH", split="test")
        except Exception:
            dataset = load_dataset("hendrycks/competition_math", split="test")

    examples = []
    for entry in dataset.select(range(min(n_problems, len(dataset)))):
        question = entry.get("problem", entry.get("question", ""))
        raw_answer = entry.get("answer")
        if raw_answer is not None:
            answer = normalize_math_answer(raw_answer)
        else:
            answer = extract_math_answer(entry.get("solution", ""))

        if not question or answer is None:
            continue

        examples.append(
            BenchmarkExample(
                question=question,
                answer=answer,
                full_solution=entry.get("solution"),
                metadata={
                    "level": str(entry.get("level", "unknown")),
                    "type": str(entry.get("type", "unknown")),
                },
            )
        )

    return examples
