import subprocess
import sys


def test_evaluate_help_runs_from_documented_entrypoint():
    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout
