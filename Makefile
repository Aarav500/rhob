.PHONY: help install test cov lint format data example clean

help:
	@echo "install  Install the package with dev dependencies"
	@echo "test     Run the test suite"
	@echo "cov      Run tests with coverage"
	@echo "lint     Lint with ruff"
	@echo "format   Format with ruff"
	@echo "data     Generate the GridWorld-Wireheading dataset"
	@echo "example  Run the minimal evaluation example"
	@echo "clean    Remove caches and generated data"

install:
	pip install -e ".[dev]"

test:
	pytest

cov:
	pytest --cov=rhob --cov-report=term-missing

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

data:
	python scripts/generate_gridworld_data.py --output data/gridworld_wireheading.h5

example:
	python examples/minimal_evaluation.py

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
