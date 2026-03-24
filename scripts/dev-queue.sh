#!/usr/bin/env bash
# Start Redis + Celery worker + FastAPI with hot reload for local development.
# All three processes are cleaned up on Ctrl+C / EXIT.
set -e

BROKER="redis://localhost:6379/0"
CELERY_PID=""

cleanup() {
    echo ""
    echo "Shutting down..."
    [ -n "$CELERY_PID" ] && kill "$CELERY_PID" 2>/dev/null || true
    docker stop tts-redis 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── Redis ──────────────────────────────────────────────────────────────────────
echo "→ Starting Redis..."
if docker ps --format '{{.Names}}' | grep -q '^tts-redis$'; then
    echo "  (reusing existing tts-redis container)"
else
    docker run --rm -d --name tts-redis -p 6379:6379 redis:7-alpine > /dev/null
fi

echo "  Waiting for Redis to be ready..."
until docker exec tts-redis redis-cli ping > /dev/null 2>&1; do sleep 0.3; done
echo "  Redis ready."

# ── Celery worker ──────────────────────────────────────────────────────────────
echo "→ Starting Celery worker (concurrency=2)..."
TTS_CELERY_BROKER_URL="$BROKER" \
TTS_CELERY_RESULT_BACKEND_URL="$BROKER" \
uv run celery -A tts_api.workers.celery_app:celery_app worker \
    --loglevel=info --concurrency=2 &
CELERY_PID=$!

# Give the worker a moment to connect before the API starts
sleep 1

# ── FastAPI ────────────────────────────────────────────────────────────────────
echo "→ Starting API server at http://localhost:8000 — Ctrl+C to stop all"
TTS_QUEUE_ENABLED=true \
TTS_CELERY_BROKER_URL="$BROKER" \
TTS_CELERY_RESULT_BACKEND_URL="$BROKER" \
uv run uvicorn tts_api.main:app \
    --reload --host 0.0.0.0 --port 8000 --log-level info
