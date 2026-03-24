"""
Celery application instance and configuration.

Imported by:
  • tasks.py       — task registration
  • celery_tts.py  — to call .delay() from the FastAPI process

URLs are read from os.environ directly (not via pydantic-settings) to avoid
pulling the full FastAPI app stack into the worker process at import time.
"""

import os

from celery import Celery

broker_url = os.environ.get("TTS_CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend_url = os.environ.get("TTS_CELERY_RESULT_BACKEND_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "tts_workers",
    broker=broker_url,
    backend=result_backend_url,
)

celery_app.autodiscover_tasks(["tts_api.workers"])

celery_app.conf.update(
    # Serialization — JSON is safe and human-readable
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Results
    result_expires=300,  # TTL for results in Redis (5 minutes)
    # Worker behaviour — one task at a time per worker, fair queuing
    worker_prefetch_multiplier=1,
    task_acks_late=True,           # ack only after successful completion
    task_reject_on_worker_lost=True,
    # Timeouts
    task_soft_time_limit=120,  # SIGTERM after 120 s
    task_time_limit=150,       # SIGKILL after 150 s
)
