# --- Build the frontend -------------------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# The Guide and legal pages read docs/USER_GUIDE.md and docs/legal/ (imported at build time).
COPY docs/USER_GUIDE.md /docs/USER_GUIDE.md
COPY docs/legal/ /docs/legal/
RUN npm run build

# --- Python dependencies, installed without uv's cache --------------------------------------
FROM python:3.12-slim AS deps
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /app
# Optional extras of the app, space-separated. The default image keeps "agent"; the deep vision
# image (114: torch CPU, torchvision, timm; weights downloaded on confirmation, never in the
# image) is built with: docker build --build-arg QUANTILO_EXTRAS="agent vision" -t quantilo:vision .
ARG QUANTILO_EXTRAS="agent"
COPY backend/pyproject.toml backend/uv.lock ./
# Third-party dependencies first (cached layer); the engine package and the app come next.
RUN uv sync --frozen --no-dev --no-install-project --no-install-local \
        $(for extra in $QUANTILO_EXTRAS; do printf -- '--extra %s ' "$extra"; done)
# The engine is a path dependency (../packages): installed as a regular package, not editable.
COPY packages/quantilo/ /packages/quantilo/
COPY backend/ ./
# Libraries' own test suites are never run here (~55 MB).
RUN uv sync --frozen --no-dev --no-editable \
        $(for extra in $QUANTILO_EXTRAS; do printf -- '--extra %s ' "$extra"; done) \
    && find /opt/venv -type d -name tests -prune -exec rm -rf {} +

# --- Runtime: Python backend serving the API and the built frontend (no uv, no build tools) --
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    QUANTILO_HOME=/workspace \
    QUANTILO_STATIC_DIR=/app/static \
    QUANTILO_SAMPLES_DIR=/app/samples \
    QUANTILO_DATA_DIR=/data

# OpenMP runtime for LightGBM / XGBoost (the only system library the image needs).
RUN apt-get update     && apt-get install -y --no-install-recommends libgomp1     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY --from=deps /app /app
COPY --from=web /web/dist ./static
COPY samples/ /app/samples/

RUN useradd --create-home app && mkdir -p /workspace /data && chown app /workspace
USER app

VOLUME /workspace
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
