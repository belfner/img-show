.PHONY: clean build upload-test upload install dev test lint format

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:
	uv build

upload-test:
	uv run twine upload --repository testpypi dist/* --verbose

upload:
	uv run twine upload dist/*

install:
	uv pip install -e .

dev:
	uv sync --dev

test:
	uv run python -c "from img_show import show_img; print('Import successful!')"

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

typecheck:
	uv run mypy --strict src/