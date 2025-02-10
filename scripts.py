import subprocess


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
    subprocess.run(["python", "-m", "pytest", "-n", "auto", "--log-level=CRITICAL"], check=True)


def test_verbose():
    subprocess.run(
        ["python", "-m", "pytest", "-n", "auto", "-vv", "-s", "--log-level=CRITICAL"], check=True
    )
