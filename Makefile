# clinical-research-assistant — common developer tasks.
# Usage: `make <target>`. Run `make help` to list targets.

.DEFAULT_GOAL := help
PYTHON ?= python
PACKAGE := src/cra

.PHONY: help install lint fmt test eval ingest serve docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff (lint + format check)
	ruff check .
	ruff format --check .

fmt: ## Auto-format with ruff
	ruff check --fix .
	ruff format .

test: ## Run the test suite (LLM mocked)
	CRA_LLM_PROVIDER=fake pytest

ingest: ## Build the FAISS index from data/sample
	$(PYTHON) -m cra.ingest.cli --source data/sample

eval: ## Run evals against the fake LLM and write a markdown report
	CRA_LLM_PROVIDER=fake $(PYTHON) -m cra.evals.runner --golden evals/golden_set.jsonl

eval-bedrock: ## Run evals against real Bedrock (requires AWS creds)
	CRA_LLM_PROVIDER=bedrock $(PYTHON) -m cra.evals.runner --golden evals/golden_set.jsonl --real

serve: ## Run the API locally
	uvicorn cra.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build the Docker image
	docker build -t clinical-research-assistant:latest .

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
