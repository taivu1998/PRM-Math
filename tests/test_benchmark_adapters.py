from src.eval.benchmarks import get_benchmark_adapter


def test_math500_adapter_uses_math_grading_for_boxed_answers():
    adapter = get_benchmark_adapter("math500")
    predicted = adapter.extract_answer(r"Therefore, the answer is \boxed{\frac{7}{8}}.")

    assert adapter.uses_math_grading is True
    assert adapter.check_answer(predicted, "0.875")


def test_gsm8k_adapter_uses_numeric_grading():
    adapter = get_benchmark_adapter("gsm8k")
    predicted = adapter.extract_answer("We compute the result. #### 42")

    assert adapter.uses_math_grading is False
    assert adapter.check_answer(predicted, "42.0")
