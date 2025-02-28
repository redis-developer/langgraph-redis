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
    test_cmd = ["python", "-m", "pytest", "-n", "auto", "--log-level=CRITICAL"]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)


def test_verbose():
    test_cmd = ["python", "-m", "pytest", "-n", "auto", "-vv", "-s", "--log-level=CRITICAL"]
    # Get any extra arguments passed to the script
    extra_args = sys.argv[1:]
    if extra_args:
        test_cmd.extend(extra_args)
    subprocess.run(test_cmd, check=True)