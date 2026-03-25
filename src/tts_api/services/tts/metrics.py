"""
Shared Prometheus metrics for TTS backends.

Defined here so every backend (Kokoro, future Piper, Coqui, etc.) uses the
same metric names without needing to import from each other.
"""

from prometheus_client import Counter, Histogram

MODEL_LATENCY = Histogram(
    "tts_model_inference_seconds",
    "TTS model inference latency — pure model time, excludes HTTP and audio encoding",
    ["voice", "mode"],  # mode: "buffered" | "streaming"
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

REAL_TIME_FACTOR = Histogram(
    "tts_real_time_factor",
    "Real-Time Factor (RTF): inference_time / audio_duration. <1.0 = faster than real-time",
    ["voice", "mode"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)

MODEL_ERRORS = Counter(
    "tts_model_errors_total",
    "TTS model call failures by voice, mode, and error class",
    ["voice", "mode", "error_type"],
    # error_type values:
    #   oom           — out-of-memory (CPU or GPU allocator)
    #   invalid_input — bad voice ID, empty text, or other ValueError
    #   timeout       — thread pool or queue wait exceeded
    #   torch_error   — RuntimeError from PyTorch internals
    #   unknown       — anything else
)


def classify_error(exc: Exception) -> str:
    """Map an exception to a coarse error_type label for MODEL_ERRORS."""
    msg = str(exc).lower()
    if "out of memory" in msg or "not enough memory" in msg:
        return "oom"
    if isinstance(exc, ValueError):
        return "invalid_input"
    # In Python 3.11+ asyncio.TimeoutError is an alias for TimeoutError
    if isinstance(exc, TimeoutError) or "timeout" in msg:
        return "timeout"
    if isinstance(exc, RuntimeError):
        return "torch_error"
    return "unknown"
