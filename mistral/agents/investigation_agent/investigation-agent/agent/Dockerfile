FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project
COPY main.py tool_allowlist.py instructions.md ./
COPY skills ./skills
USER 1000:1000
CMD ["agents", "run-worker"]
