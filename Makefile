.PHONY: install format lint test test-all test-sentinel clean redis-start redis-stop check-types check docs docs-clean docs-serve

install:
	poetry install --all-extras

redis-start:
	docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest

redis-stop:
	docker stop redis-stack

format:
	poetry run format
	poetry run sort-imports

check-types:
	poetry run check-mypy

lint: format check-types

test:
	poetry run test-verbose

test-all:
	poetry run test-verbose --run-api-tests

test-sentinel:
	poetry run python -m pytest tests/test_sentinel_integration.py -vv -s --run-sentinel-tests

test-coverage:
	poetry run test-coverage

coverage-report:
	poetry run coverage-report

coverage-html:
	poetry run coverage-html

find-dead-code:
	poetry run find-dead-code

check: lint test

docs:
	python docs/copy_notebooks.py
	sphinx-build -b html docs docs/_build/html

docs-clean:
	rm -rf docs/_build
	rm -rf docs/examples/checkpoints docs/examples/human_in_the_loop docs/examples/memory docs/examples/middleware docs/examples/react_agent

docs-serve: docs
	python -m http.server 8085 --directory docs/_build/html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "_build" -exec rm -rf {} +
