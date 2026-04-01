import ast
import math
import re
from fractions import Fraction
from typing import Optional, Tuple


def _read_braced(text: str, start: int) -> Tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise ValueError("Expected braced expression")

    depth = 0
    chars = []
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), idx + 1
            chars.append(char)
        else:
            chars.append(char)

    raise ValueError("Unclosed braced expression")


def _read_latex_argument(text: str, start: int, *, single_token: bool = False) -> Tuple[str, int]:
    if start >= len(text):
        raise ValueError("Missing LaTeX argument")

    if text[start] == "{":
        return _read_braced(text, start)

    if text[start] in "+-":
        sign = text[start]
        value, next_idx = _read_latex_argument(text, start + 1, single_token=single_token)
        return sign + value, next_idx

    if single_token:
        return text[start], start + 1

    match = re.match(r"[A-Za-z0-9.]+", text[start:])
    if match:
        value = match.group(0)
        return value, start + len(value)

    return text[start], start + 1


def _strip_latex_commands(text: str) -> str:
    result = []
    idx = 0

    while idx < len(text):
        if text.startswith("\\left", idx):
            idx += len("\\left")
            continue
        if text.startswith("\\right", idx):
            idx += len("\\right")
            continue

        frac_command = next(
            (command for command in ("\\dfrac", "\\tfrac", "\\frac") if text.startswith(command, idx)),
            None,
        )
        if frac_command is not None:
            idx += len(frac_command)
            numerator, idx = _read_latex_argument(text, idx, single_token=True)
            denominator, idx = _read_latex_argument(text, idx, single_token=True)
            result.append(f"(({_strip_latex_commands(numerator)})/({_strip_latex_commands(denominator)}))")
            continue

        if text.startswith("\\sqrt", idx):
            idx += len("\\sqrt")
            inner, idx = _read_latex_argument(text, idx)
            result.append(f"sqrt({_strip_latex_commands(inner)})")
            continue

        unary_command = next(
            (
                command
                for command in ("\\boxed", "\\text", "\\textbf", "\\mathrm")
                if text.startswith(command, idx)
            ),
            None,
        )
        if unary_command is not None:
            idx += len(unary_command)
            inner, idx = _read_latex_argument(text, idx)
            result.append(_strip_latex_commands(inner))
            continue

        result.append(text[idx])
        idx += 1

    return "".join(result)


def _insert_implicit_multiplication(text: str) -> str:
    text = re.sub(r"(\d)(sqrt|\()", r"\1*\2", text)
    text = re.sub(r"(\))(sqrt|\d|\()", r"\1*\2", text)
    text = re.sub(r"(sqrt\([^)]*\))(\d|\()", r"\1*\2", text)
    return text


def _has_redundant_outer_parentheses(text: str) -> bool:
    if len(text) < 2 or text[0] != "(" or text[-1] != ")":
        return False

    depth = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and idx != len(text) - 1:
                return False
    return depth == 0


def _safe_eval_numeric_expression(expression: str) -> Optional[float]:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Num,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
        ast.Call,
        ast.Name,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return None
        if isinstance(node, ast.Name) and node.id != "sqrt":
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id != "sqrt":
                return None
            if len(node.args) != 1:
                return None

    try:
        value = eval(compile(tree, "<math-normalize>", "eval"), {"__builtins__": {}}, {"sqrt": math.sqrt})
    except Exception:
        return None

    if isinstance(value, complex):
        return None

    return float(value)


def normalize_math_answer(answer: str) -> Optional[str]:
    """Normalize MATH-style answers into a comparison-friendly representation."""
    if answer is None:
        return None

    text = str(answer).strip()
    if not text:
        return None

    text = text.replace("\\$", "").replace("$", "")
    text = text.replace("\\%", "%").replace("\\!", "")
    text = text.replace("\\,", "")
    text = re.sub(r"\\(?:qquad|quad|!|,|;)", "", text)
    text = _strip_latex_commands(text)
    text = text.strip()

    while _has_redundant_outer_parentheses(text):
        text = text[1:-1].strip()

    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace(" ", "")
    text = _insert_implicit_multiplication(text)

    if not text:
        return None

    if re.fullmatch(r"[+-]?\d+/\d+", text):
        try:
            return str(float(Fraction(text)))
        except (ValueError, ZeroDivisionError):
            pass

    numeric_value = _safe_eval_numeric_expression(text)
    if numeric_value is not None:
        return str(float(numeric_value))

    cleaned_text = re.sub(r"\\[a-zA-Z]+", "", text).strip().lower()
    return cleaned_text or None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract and normalize a final answer from MATH-style reasoning traces."""
    if not text:
        return None

    boxed_patterns = [
        r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        r"\\boxed\{([^}]+)\}",
    ]
    for pattern in boxed_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return normalize_math_answer(matches[-1])

    answer_patterns = [
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?([^\$\n]+)\$?",
        r"[Aa]nswer[:\s]*\$?([^\$\n]+)\$?",
        r"=\s*\$?([^\$\n]+)\$?\s*$",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return normalize_math_answer(match.group(1))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        normalized_last_line = normalize_math_answer(lines[-1])
        if normalized_last_line is not None:
            return normalized_last_line

    numbers = re.findall(r"([+-]?\d+\.?\d*(?:/\d+)?)", text)
    if numbers:
        return normalize_math_answer(numbers[-1])

    return None


def check_math_answer(predicted: str, ground_truth: str) -> bool:
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
    except ValueError:
        return False


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from a solution."""
    if not text:
        return None

    text = text.strip()
    patterns = [
        r"\\boxed\{([^}]+)\}",
        r"\$\\boxed\{([^}]+)\}\$",
        r"####\s*([+-]?\d+[\d,]*\.?\d*)",
        r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?([+-]?[\d,./]+)",
        r"[Aa]nswer[:\s]*\$?([+-]?[\d,./]+)",
        r"=\s*\$?([+-]?[\d,./]+)\s*$",
        r"[Tt]herefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([+-]?[\d,./]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        answer = match.group(1).replace(",", "").replace("$", "").strip()
        if "/" in answer and re.fullmatch(r"[+-]?\d+/\d+", answer):
            try:
                return str(float(Fraction(answer)))
            except (ValueError, ZeroDivisionError):
                continue
        try:
            return str(float(answer))
        except ValueError:
            continue

    numbers = re.findall(r"([+-]?\d+\.?\d*)", text)
    if numbers:
        try:
            return str(float(numbers[-1]))
        except ValueError:
            pass

    return None


def check_numeric_answer(predicted: str, ground_truth: str) -> bool:
    if predicted is None or ground_truth is None:
        return False

    pred = str(predicted).strip().replace(",", "").lower()
    gt = str(ground_truth).strip().replace(",", "").lower()

    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred == gt
