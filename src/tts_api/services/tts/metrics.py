"""
Shared Prometheus metrics for TTS backends.

Defined here so every backend (Kokoro, future Piper, Coqui, etc.) uses the
same metric names without needing to import from each other.
"""

from prometheus_client import Histogram

MODEL_LATENCY = Histogram(
    "tts_model_inference_seconds",
    "TTS model inference latency — pure model time, excludes HTTP and audio encoding",
    ["voice", "mode"],  # mode: "buffered" | "streaming"
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
