import subprocess


def assert_script_runs(command: list[str]):
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0


def test_run_gr():
    assert_script_runs(
        ["python", "scripts/run_generative_rendering.py", "+overrides=test_run_gr"]
    )
