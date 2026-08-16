set minimum-version := '1.55.0'
set default-list := true

install:
    uv sync -v

# Format Python files with Ruff.
format:
    uv run ruff check --select I --fix --exclude "*.ipynb" --exclude "vendored"
    uv run ruff format --exclude "*.ipynb" src/ --exclude "vendored"
