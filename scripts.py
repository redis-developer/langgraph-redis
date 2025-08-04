import subprocess
import sys


def format():
    subprocess.run(["isort", "./langgraph", "./tests/", "--profile", "black"], check=True)
    subprocess.run(["black", "./langgraph", "./tests/"], check=True)


def check_format():
    subprocess.run(["black", "--check", "./langgraph"], check=True)


def sort_imports():
    subprocess.run(["isort", "./langgraph", "./tests/", "--profile", "black"], check=True)


def check_sort_imports():
    subprocess.run(
        ["isort", "./langgraph", "--check-only", "--profile", "black"], check=True
    )


def check_lint():
    subprocess.run(["pylint", "--rcfile=.pylintrc", "./langgraph"], check=True)


def check_mypy():
    subprocess.run(["python", "-m", "mypy", "./langgraph"], check=True)


def test():
    test_cmd = ["python", "-m", "pytest", "--log-level=CRITICAL"]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)


def test_verbose():
    test_cmd = ["python", "-m", "pytest", "-vv", "-s", "--log-level=CRITICAL"]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)


def test_coverage():
    """Run tests with coverage reporting."""
    test_cmd = [
        "python", "-m", "pytest",
        "--cov=langgraph",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--log-level=CRITICAL"
    ]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)


def coverage_report():
    """Generate coverage report without running tests."""
    subprocess.run(["python", "-m", "coverage", "report"], check=True)


def coverage_html():
    """Generate HTML coverage report."""
    subprocess.run(["python", "-m", "coverage", "html"], check=True)
    print("Coverage HTML report generated in htmlcov/")


def find_dead_code():
    """Find dead code using vulture."""
    result = subprocess.run(
        ["python", "-m", "vulture", "langgraph", "--sort-by-size"],
        capture_output=True,
        text=True
    )

    if result.stdout:
        print("Dead code found:")
        print(result.stdout)
    else:
        print("No dead code found!")

    if result.stderr:
        print("Errors:")
        print(result.stderr)

    # Don't fail the build for dead code detection, just report
    return result.returncode
