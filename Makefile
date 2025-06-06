.PHONY: help install install-dev test clean format lint run
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Initial setup - create virtual environment and install dependencies
	python3.11 -m venv .venv
	source .venv/bin/activate && pip install --upgrade pip
	source .venv/bin/activate && pip install -r requirements.txt
	source .venv/bin/activate && pip install -e .

install: ## Install the package in development mode
	source .venv/bin/activate && pip install -e .

install-dev: ## Install development dependencies
	source .venv/bin/activate && pip install -r requirements-dev.txt

test: ## Run tests
	source .venv/bin/activate && pytest tests/ -v

test-cov: ## Run tests with coverage
	source .venv/bin/activate && pytest tests/ --cov=src/cpr --cov-report=html --cov-report=term

format: ## Format code with black
	source .venv/bin/activate && black src/ tests/ --line-length 88

lint: ## Lint code with flake8
	source .venv/bin/activate && flake8 src/ tests/

type-check: ## Type check with mypy
	source .venv/bin/activate && mypy src/cpr/

clean: ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run: ## Run the main analysis
	source .venv/bin/activate && python run_analysis.py

run-sample: ## Run analysis on sample data
	source .venv/bin/activate && python -c "from cpr import EnhancedJosephsonProcessor; processor = EnhancedJosephsonProcessor(); processor.process_single_file('data/Ic/100Ic-.csv')"

deps-update: ## Update dependencies
	source .venv/bin/activate && pip install --upgrade pip
	source .venv/bin/activate && pip install --upgrade -r requirements.txt

build: ## Build distribution packages
	source .venv/bin/activate && python -m build

pre-commit-install: ## Install pre-commit hooks
	source .venv/bin/activate && pre-commit install

pre-commit-run: ## Run pre-commit on all files
	source .venv/bin/activate && pre-commit run --all-files
