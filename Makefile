.PHONY: install test test-fast lint clean

install:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest -m "not slow"

lint:
	ruff check bitemb tests
	mypy bitemb

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache __pycache__ results/
