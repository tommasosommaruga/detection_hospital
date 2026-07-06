#!/bin/sh
set -e
cd /app
export PYTHONPATH=/app
exec python -m uvicorn demo.server:app --host "${PATHASSIST_HOST:-0.0.0.0}" --port "${PATHASSIST_PORT:-8765}"
