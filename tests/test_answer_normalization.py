from src.eval.answers import (
    check_math_answer,
    check_numeric_answer,
    extract_math_answer,
    extract_numeric_answer,
    normalize_math_answer,
)


def test_extract_math_answer_handles_boxed_fraction():
    text = r"The final answer is \boxed{\frac{3}{4}}."
    assert extract_math_answer(text) == "0.75"


def test_check_math_answer_accepts_equivalent_numeric_forms():
    assert check_math_answer(r"\frac{1}{2}", "0.5")
    assert check_math_answer(r"\boxed{2}", "2.0")


def test_check_math_answer_accepts_negative_fraction_and_shorthand_fraction():
    assert check_math_answer(r"\boxed{-\frac{1}{2}}", "-0.5")
    assert check_math_answer(r"\boxed{\frac12}", "0.5")


def test_check_math_answer_accepts_radical_expression():
    assert check_math_answer(r"\boxed{2\sqrt{3}}", str(2 * (3 ** 0.5)))


def test_normalize_math_answer_strips_common_latex_wrappers():
    assert normalize_math_answer(r"\left(\frac{5}{2}\right)") == "2.5"


def test_extract_numeric_answer_supports_fractional_answers():
    assert extract_numeric_answer("Answer: 3/4") == "0.75"


def test_check_numeric_answer_matches_float_equivalence():
    assert check_numeric_answer("2", "2.0")
