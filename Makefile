# JAVIS - Personal AI Assistant
# Makefile for common development tasks

.PHONY: help install dev test test-fast lint format type-check \
        docker-build docker-up docker-down docker-logs \
        train-local train-modal clean clean-docker clean-all \
        mlflow-ui coverage docs

# Default target
help:
	@echo "JAVIS Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install all dependencies (dev, training, voice, mlops)"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test        Run all tests with coverage"
	@echo "  make test-fast   Run tests without coverage (faster)"
	@echo "  make lint        Run linting checks (ruff)"
	@echo "  make format      Auto-format code with ruff"
	@echo "  make type-check  Run type checking (mypy)"
	@echo "  make coverage    Generate HTML coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services (detached)"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    View service logs"
	@echo ""
	@echo "Training:"
	@echo "  make train-local    Run training locally (requires GPU)"
	@echo "  make train-modal    Run training on Modal.com"
	@echo ""
	@echo "MLOps:"
	@echo "  make mlflow-ui      Start MLflow UI server"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Clean build artifacts"
	@echo "  make clean-docker   Remove Docker containers and volumes"
	@echo "  make clean-all      Clean everything"

# =============================================================================
# Setup
# =============================================================================

install:
	python -m pip install --upgrade pip
	pip install -e .

dev:
	python -m pip install --upgrade pip
	pip install -e ".[dev,training,voice,mlops]"

# =============================================================================
# Testing & Quality
# =============================================================================

test:
	pytest --cov=javis --cov-report=term-missing --cov-report=html --cov-report=xml

test-fast:
	pytest -x -q --no-cov

lint:
	ruff check javis/ tests/

format:
	ruff format javis/ tests/
	ruff check --fix javis/ tests/

type-check:
	mypy javis/ --ignore-missing-imports

coverage:
	pytest --cov=javis --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services started. Run 'make docker-logs' to view logs."
	@echo "API: http://localhost:8000"
	@echo "MLflow: http://localhost:5000"
	@echo "ChromaDB: http://localhost:8001"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-ps:
	docker-compose ps

# Development environment with hot reload
docker-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# =============================================================================
# Training
# =============================================================================

train-local:
	python -m javis.training.finetune --local

train-modal:
	python -m javis.training.finetune --modal

# Run training pipeline
train-pipeline:
	python -c "from javis.training.pipeline import run_pipeline; run_pipeline(force=True)"

# Run DPO training pipeline
train-dpo:
	python -c "from javis.training.pipeline import run_dpo_pipeline; run_dpo_pipeline(force=True)"

# =============================================================================
# MLOps
# =============================================================================

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Run HPO optimization
hpo:
	python -c "from javis.training.hpo import run_optimization; run_optimization()"

# =============================================================================
# Cleanup
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-docker:
	docker-compose down -v --remove-orphans
	docker system prune -f

clean-all: clean clean-docker

# =============================================================================
# Utilities
# =============================================================================

# Start the API server
serve:
	uvicorn javis.interfaces.api:app --reload --host 0.0.0.0 --port 8000

# Interactive CLI
cli:
	python -m javis

# Export training data
export-data:
	python -c "from javis.training.pipeline import TrainingPipeline; p = TrainingPipeline(); p.export_training_data()"

# Show data statistics
data-stats:
	python -c "from javis.training.pipeline import TrainingPipeline; p = TrainingPipeline(); print(p.get_data_stats())"
