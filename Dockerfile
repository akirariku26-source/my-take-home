FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python setup ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install uv
RUN pip install uv

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev

# ── Application ───────────────────────────────────────────────────────────────
COPY src/ src/
COPY static/ static/

# Re-sync so the project package itself is installed into the venv,
# then install pip so Kokoro/HuggingFace can call `python -m pip` during model init.
RUN uv sync --no-dev && uv pip install pip

ENV PATH="/app/.venv/bin:$PATH"

# ── Runtime config ────────────────────────────────────────────────────────────
# HuggingFace model cache (mount a volume for persistence across restarts)
ENV HF_HOME=/app/.hf_cache \
    TTS_BACKEND=kokoro \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8000 \
    TTS_MAX_WORKERS=4 \
    PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single-process per container — scale horizontally with replicas
CMD ["uvicorn", "tts_api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
