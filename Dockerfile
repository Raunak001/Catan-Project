FROM python:3.12-slim AS base
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock ./
RUN touch README.md && uv sync --frozen --no-dev

# Copy source, frontend, and trained models
COPY src/ src/
COPY frontend/ frontend/
COPY models/ models/

# Runtime config
ENV DATA_DIR=/app/data
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "catan.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
