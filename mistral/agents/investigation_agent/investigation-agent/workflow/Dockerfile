FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never PATH="/app/.venv/bin:$PATH" UPLOAD_GRAPH=true
WORKDIR /app
COPY workflow/pyproject.toml ./
RUN uv sync --no-dev --no-install-project
COPY workflow/*.py ./
COPY workflow/activities ./activities
USER 1000:1000
CMD ["python", "worker.py"]
