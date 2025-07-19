# Victor Python IPC Makefile

.PHONY: help install install-dev test test-cov lint format clean build publish

help:
	@echo "Available targets:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install package with development dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  publish      Publish to PyPI"
	@echo "  run-example  Run the producer example"
	@echo "  run-consumer Run the consumer example"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=victor_ipc --cov-report=html --cov-report=term

test-quick:
	python test_installation.py

lint:
	flake8 src/victor_ipc tests examples
	mypy src/victor_ipc

format:
	black src/victor_ipc tests examples
	isort src/victor_ipc tests examples

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Example targets
run-example:
	python examples/producer_example.py

run-consumer:
	python examples/consumer_example.py

run-ros-demo:
	python examples/ros_to_json_example.py

# CLI targets
cli-help:
	victor-ipc --help

generate-config:
	victor-ipc generate-config config/example_config.json --robot-id example_robot

list-pools:
	victor-ipc list-pools

# Development targets
setup-dev: install-dev
	pre-commit install

check: lint test

check-all: format lint test-cov

# Docker targets (if needed in future)
docker-build:
	docker build -t victor-ipc .

docker-run:
	docker run --rm -it victor-ipc

# Documentation targets
docs:
	@echo "Documentation is in README.md"
	@echo "API docs can be generated with: pdoc victor_ipc"
