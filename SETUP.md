# Setup Guide — Fresh Linux

Two modes are available. **Docker** is recommended for a production-like run (all four services). **Local** is faster for development (single process, no queue).

---

## Prerequisites

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y make curl git
```

### 2. Docker (for Docker mode)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in so the group takes effect, then verify:
docker run --rm hello-world
```

Docker Compose v2 is bundled with the above (`docker compose`, not `docker-compose`). No separate install needed.

### 3. uv (Python manager — required for both modes)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or open a new shell
```

`uv` downloads Python 3.11 automatically if it is not present. No separate `python3` install required.

---

## Docker mode (recommended)

Runs all four services: `tts-api`, `celery-worker`, `tts-inference`, `redis`.

```bash
git clone <your-repo-url>
cd my-take-home
cp .env.example .env        # no keys required for basic use
make docker-up              # docker compose up -d
```

**First boot takes 2–5 minutes.** The `tts-inference` container downloads Kokoro model weights (~200 MB) from HuggingFace. The `celery-worker` waits for the healthcheck before starting, so the API will return 503 until the model is ready. Watch progress with:

```bash
docker compose logs -f tts-inference
```

Once ready, the API is at **http://localhost:8000** and the UI at **http://localhost:8000/**.

### Hardware requirements

| Resource | Minimum | Recommended |
|---|---|---|
| RAM | 4 GB | 8 GB |
| CPU | 2 cores | 4 cores |
| Disk | 2 GB free | 5 GB free |

### Verify

```bash
curl http://localhost:8000/health
# {"status":"healthy", ...}
```

### Stop

```bash
make docker-down
```

---

## Local dev mode (no Docker, no queue)

Single uvicorn process with the model running in-process. Faster startup, no Redis or Celery.

```bash
git clone <your-repo-url>
cd my-take-home
cp .env.example .env
make setup                  # creates .venv, installs all deps (~1 min)
make run                    # starts uvicorn on :8000 with hot reload
```

On first request, Kokoro model weights are downloaded (~200 MB). Subsequent starts are instant (weights are cached in `~/.cache/huggingface`).

The API is at **http://localhost:8000**.

---

## Ports used

| Port | Service | Configurable via |
|---|---|---|
| 8000 | TTS API | `TTS_PORT` |
| 6379 | Redis | `TTS_CELERY_BROKER_URL` |
| 50051 | gRPC inference server | `TTS_INFERENCE_PORT` |

---

## Common issues

**`docker compose` not found** — you have Compose v1 (`docker-compose`). Upgrade Docker or install the Compose plugin: `sudo apt-get install docker-compose-plugin`.

**503 on `/health` after `docker-up`** — the inference container is still downloading the model. Wait for `docker compose logs -f tts-inference` to show `inference_server_ready`.

**Out of memory / container OOM killed** — the inference container needs 1–4 GB. Check `docker stats`. On machines with <4 GB RAM, use local dev mode instead.

**Port 8000 already in use** — `lsof -i :8000` to find the process, or set `TTS_PORT=8001` in `.env`.
