FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATHASSIST_ENV=production \
    PATHASSIST_HOST=0.0.0.0 \
    PATHASSIST_PORT=8765

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenslide0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-docker.txt

COPY pathassist ./pathassist
COPY demo ./demo
COPY config ./config
COPY scripts/entrypoint.sh ./scripts/entrypoint.sh

RUN chmod +x scripts/entrypoint.sh \
    && useradd --create-home --uid 10001 pathassist \
    && mkdir -p /app/models /app/outputs /app/runs /app/data \
    && chown -R pathassist:pathassist /app

USER pathassist

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health/live')"

ENTRYPOINT ["./scripts/entrypoint.sh"]
