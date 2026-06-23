# List available targets.
default:
    @just --list

# Format Python files with Ruff.
format:
    uv run ruff check --select I --fix .
    uv run ruff format .
