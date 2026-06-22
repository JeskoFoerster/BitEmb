.PHONY: install test test-fast lint eval eval-full benchmark visualize all clean

install:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -m "not slow"

lint:
	ruff check .
	mypy bitemb

eval:
	python -m bitemb.eval

eval-full:
	python -m bitemb.eval --full

benchmark:
	python -m bitemb.benchmark

visualize:
	python -m bitemb.visualize

all: lint test eval

clean:
	rm -rf build dist *.egg-info htmlcov .coverage .pytest_cache .mypy_cache
