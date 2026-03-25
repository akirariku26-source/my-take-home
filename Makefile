.PHONY: setup install dev proto test test-fast loadtest lint lint-fix format format-check typecheck \
        run run-queue run-prod docker-build docker-build-inference docker-up docker-down clean help

UV         := uv
APP_MODULE := tts_api.main:app
DOCKER_IMG := tts-api

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' | sort

# ── Dependency management ─────────────────────────────────────────────────────

setup: ## Create venv and install all dependencies (including dev), warm up model
	$(UV) sync --all-extras
	@echo ""
	@echo "Setup complete. Run 'make run' to start the API server."

install: ## Install production dependencies only
	$(UV) sync

dev: ## Install dev dependencies
	$(UV) sync --all-extras

# ── Proto / gRPC ──────────────────────────────────────────────────────────────

proto: ## Regenerate gRPC Python stubs from proto/tts.proto (requires dev deps)
	$(UV) run python -m grpc_tools.protoc \
	  -I proto \
	  --python_out=src/tts_api/inference \
	  --grpc_python_out=src/tts_api/inference \
	  proto/tts.proto
	# Fix the bare `import tts_pb2` the generator emits → package-qualified import
	sed -i.bak 's/^import tts_pb2/from tts_api.inference import tts_pb2/' \
	  src/tts_api/inference/tts_pb2_grpc.py && \
	rm -f src/tts_api/inference/tts_pb2_grpc.py.bak
	@echo "Stubs written to src/tts_api/inference/"

# ── Code quality ──────────────────────────────────────────────────────────────

lint: ## Run ruff linter
	$(UV) run ruff check src tests

lint-fix: ## Fix auto-fixable lint issues
	$(UV) run ruff check --fix src tests

format: ## Format code with ruff
	$(UV) run ruff format src tests

format-check: ## Check formatting (CI-safe)
	$(UV) run ruff format --check src tests

typecheck: ## Run mypy type checker
	$(UV) run mypy src

# ── Testing ───────────────────────────────────────────────────────────────────

test: ## Run all tests with coverage report
	$(UV) run pytest

test-fast: ## Run tests without coverage, stop on first failure
	$(UV) run pytest --no-cov -x -q

loadtest: ## Run Locust load test (50 users, 60s) against localhost:8000
	@echo "Note: start the server with rate limiting off: TTS_RATE_LIMIT_ENABLED=false make run"
	$(UV) run --with locust locust -f tests/loadtest.py \
	  --headless -u 50 -r 5 -t 60s --host http://localhost:8000

loadtest-server: ## Start dev server with rate limiting disabled (for load testing)
	TTS_RATE_LIMIT_ENABLED=false $(UV) run uvicorn $(APP_MODULE) \
	  --reload --host 0.0.0.0 --port 8000 --log-level warning

# ── Running ───────────────────────────────────────────────────────────────────

run: ## Start dev server with hot reload on :8000 (no queue)
	$(UV) run uvicorn $(APP_MODULE) \
	  --reload --host 0.0.0.0 --port 8000 --log-level info

run-queue: ## Start Redis + Celery worker + API with hot reload (Ctrl+C stops all)
	@bash scripts/dev-queue.sh

run-prod: ## Start production server (4 workers)
	$(UV) run uvicorn $(APP_MODULE) \
	  --host 0.0.0.0 --port 8000 --workers 4

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build API + worker Docker image
	docker build -t $(DOCKER_IMG) .

docker-build-inference: ## Build inference server Docker image
	docker build -f Dockerfile.inference -t $(DOCKER_IMG)-inference .

docker-up: ## Start all services with docker-compose (API, celery-worker, tts-inference, redis)
	docker compose up -d

docker-down: ## Stop docker-compose services
	docker compose down

# ── Housekeeping ──────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov .pytest_cache dist build .mypy_cache .ruff_cache *.egg-info
