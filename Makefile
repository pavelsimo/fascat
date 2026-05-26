TOOL_NAME := fascat
MODULE    := fascat

.PHONY: install build test coverage lint fmt fmt-check docs ci publish clean tools help

install: ## Install project and dev dependencies
	uv sync --dev

build: ## Build distribution packages (wheel + sdist)
	uv build

test: ## Run tests with coverage
	uv run pytest

coverage: test ## Open HTML coverage report in browser
	uv run pytest --cov-report=html
	@python3 -m webbrowser -t "htmlcov/index.html" 2>/dev/null || true

lint: ## Run ruff linter and mypy type checker
	uv run ruff check .
	uv run mypy $(MODULE)/

fmt: ## Format code and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

fmt-check: ## Check formatting and linting (CI-safe, exits non-zero if dirty)
	uv run ruff format --check .
	uv run ruff check .

docs: ## Build documentation site to dist/docs-site/
	node scripts/build-docs-site.mjs

ci: fmt-check lint test build ## Full CI gate

publish: ## Publish to PyPI (requires PyPI credentials or OIDC in CI)
	uv publish

clean: ## Remove build artifacts
	rm -rf dist/ htmlcov/ .coverage .mypy_cache .ruff_cache __pycache__ $(MODULE)/__pycache__ tests/__pycache__

tools: ## Install development tools (lefthook)
	pip install lefthook --user 2>/dev/null || brew install lefthook
	lefthook install

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
