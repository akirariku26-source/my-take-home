"""
Locust load test for TTS API.

Usage
─────
# Install (one-time)
  pip install locust

# Run against local server (headless, 60s, 50 users)
  locust -f tests/loadtest.py --headless -u 50 -r 5 -t 60s --host http://localhost:8000

# Run with web UI
  locust -f tests/loadtest.py --host http://localhost:8000
  # → open http://localhost:8089

# Run against a deployed K8s service
  locust -f tests/loadtest.py --headless -u 200 -r 20 -t 300s \
    --host https://tts.my-cluster.example.com \
    --csv results/k8s-run

K8s capacity planning
─────────────────────
The goal is to find the max sustained concurrency for one pod before:
  • p99 latency exceeds your SLA (e.g. 2 s for buffered, 500 ms TTFB for streaming)
  • error rate > 0.1 %
  • CPU saturates (> 90 %)

Recommended procedure:
  1. Deploy one pod with resource limits matching your production spec
     (e.g. 4 CPU / 8 Gi, TTS_MAX_WORKERS=4).
  2. Run this script, ramping from 10 → 200 users over 5 minutes.
  3. Watch:  p50/p95/p99 latency, RPS, error %, pod CPU/mem via kubectl top.
  4. The concurrency at which p99 crosses your SLA is your per-pod capacity.
  5. Divide expected peak traffic by that number → required replica count.

Example:  If one pod handles 20 RPS at p99 < 2 s, and peak is 200 RPS,
          you need ceil(200/20) = 10 replicas + headroom → 12 replicas.

Typical results (Kokoro-82M, CPU only, 4 workers, 4 vCPU):
  • Short text (< 50 chars):   ~15-25 RPS per pod, p99 ~ 0.5 s
  • Medium text (100-300 chars): ~5-10 RPS per pod, p99 ~ 2 s
  • Streaming first-chunk:       p99 ~ 0.3-0.5 s regardless of text length
"""

import random
import string

from locust import HttpUser, between, task

# ── Sample texts of varying lengths ──────────────────────────────────────────

_SHORT_TEXTS = [
    "Hello, how are you?",
    "The weather is nice today.",
    "Please confirm your appointment.",
    "Your order has been shipped.",
    "Thank you for calling.",
]

_MEDIUM_TEXTS = [
    (
        "Welcome to our customer service line. I'd be happy to help you with "
        "your account today. Could you please verify your identity by providing "
        "the last four digits of your social security number?"
    ),
    (
        "The quarterly earnings report shows a fifteen percent increase in "
        "revenue compared to the same period last year. Operating margins "
        "improved by two percentage points, driven primarily by efficiency "
        "gains in our supply chain operations."
    ),
    (
        "Thank you for choosing our text to speech service. This platform "
        "supports multiple voices, adjustable speed, and both streaming and "
        "buffered synthesis modes. You can integrate it into your applications "
        "using our OpenAI-compatible REST API."
    ),
]

_LONG_TEXTS = [
    (
        "In this comprehensive overview, we will discuss the architecture of "
        "modern text to speech systems. These systems typically consist of "
        "three main components: a text analysis frontend that converts raw "
        "text into linguistic features, an acoustic model that predicts "
        "spectral features from those linguistic representations, and a "
        "vocoder that synthesizes the final audio waveform. Recent advances "
        "in neural network architectures have dramatically improved the "
        "naturalness and expressiveness of synthetic speech. Models like "
        "Tacotron, FastSpeech, and VITS have pushed the boundaries of what "
        "is possible with end-to-end synthesis approaches."
    ),
]

_VOICES = ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma"]


class TTSUser(HttpUser):
    """Simulates a client making TTS requests with realistic usage patterns."""

    wait_time = between(0.5, 2.0)

    # ── Buffered (non-streaming) requests ─────────────────────────────────────

    @task(5)
    def synthesize_short(self):
        """Short text, buffered — most common in voice agent greetings."""
        self.client.post(
            "/v1/audio/speech",
            json={
                "input": random.choice(_SHORT_TEXTS),
                "voice": random.choice(_VOICES),
                "speed": 1.0,
                "stream": False,
            },
            name="/v1/audio/speech [short]",
        )

    @task(3)
    def synthesize_medium(self):
        """Medium text, buffered — typical IVR or assistant responses."""
        self.client.post(
            "/v1/audio/speech",
            json={
                "input": random.choice(_MEDIUM_TEXTS),
                "voice": random.choice(_VOICES),
                "speed": round(random.uniform(0.8, 1.3), 1),
                "stream": False,
            },
            name="/v1/audio/speech [medium]",
        )

    @task(1)
    def synthesize_long(self):
        """Long text, buffered — article narration, etc."""
        self.client.post(
            "/v1/audio/speech",
            json={
                "input": random.choice(_LONG_TEXTS),
                "voice": random.choice(_VOICES),
                "speed": 1.0,
                "stream": False,
            },
            name="/v1/audio/speech [long]",
        )

    # ── Streaming requests ────────────────────────────────────────────────────

    @task(4)
    def synthesize_streaming(self):
        """Streaming request — measures time-to-first-byte."""
        with self.client.post(
            "/v1/audio/speech",
            json={
                "input": random.choice(_MEDIUM_TEXTS),
                "voice": random.choice(_VOICES),
                "speed": 1.0,
                "stream": True,
            },
            name="/v1/audio/speech [stream]",
            stream=True,
        ) as resp:
            # Consume the stream so Locust measures full transfer time
            for _ in resp.iter_content(chunk_size=4096):
                pass

    # ── Supporting endpoints ──────────────────────────────────────────────────

    @task(1)
    def list_voices(self):
        self.client.get("/v1/voices", name="/v1/voices")

    @task(1)
    def health_check(self):
        self.client.get("/v1/health", name="/v1/health")
