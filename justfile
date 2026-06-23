# List available targets.
default:
    @just --list

# Format Python files with Ruff.
format:
    uv run ruff check --select I --fix --exclude "*.ipynb" .
    uv run ruff format --exclude "*.ipynb" .
