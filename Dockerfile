FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY configs/ configs/
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home appuser \
    && chown -R appuser:appuser /app
USER appuser
CMD ["crisislab-score", "--help"]
