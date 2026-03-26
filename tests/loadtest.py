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

# Simulate a traffic burst (adaptive concurrency limiter test)
  locust -f tests/loadtest.py --headless -u 100 -r 100 -t 30s \
    --host http://localhost:8000 \
    --class-picker BurstUser

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

Adaptive concurrency limiter
────────────────────────────
BurstUser deliberately generates a spike of back-to-back requests with no
wait time to trigger the AIMD window.  Once in_flight ≥ limit the server
returns 503.  Watch the Prometheus metrics to see the limiter in action:

  # current window and in-flight count
  curl -s http://localhost:8000/metrics | grep tts_concurrency

  Expected behaviour during a burst:
    tts_concurrency_in_flight   rises to match the window
    tts_concurrency_limit       shrinks multiplicatively as latency climbs
    tts_concurrency_rejected_total  counts shed requests (503s)
    tts_concurrency_ewma_latency_seconds  tracks the smoothed latency signal

  A healthy recovery looks like:
    503 rate drops as the burst subsides → EWMA latency falls below target →
    window grows additively back toward its original size.
"""

import random

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
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
            elif resp.raw is not None:
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


class BurstUser(HttpUser):
    """
    Generates a sudden spike of back-to-back synthesis requests with no
    wait time between them.  Use this to observe the adaptive concurrency
    limiter in action:

      • In-flight count climbs to the current window size
      • Once the window is full, requests get 503 (counted as failures by Locust)
      • EWMA latency rises → window shrinks multiplicatively
      • As the burst subsides, window recovers additively

    Run in isolation to get a clean signal:
      locust -f tests/loadtest.py --headless -u 80 -r 80 -t 30s \\
        --host http://localhost:8000 --class-picker BurstUser
    """

    wait_time = between(0, 0.05)  # near-zero think time — maximise concurrency

    @task(8)
    def burst_buffered(self):
        """Dense buffered requests — the primary limiter stress test."""
        with self.client.post(
            "/v1/audio/speech",
            json={
                "input": random.choice(_MEDIUM_TEXTS),
                "voice": random.choice(_VOICES),
                "stream": False,
            },
            name="/v1/audio/speech [burst-buffered]",
            catch_response=True,
        ) as resp:
            # 503 = limiter shed the request (expected under overload, not a bug)
            if resp.status_code == 503:
                resp.success()  # don't count as Locust failure; track via Prometheus

    @task(2)
    def burst_streaming(self):
        """Dense streaming requests — limiter holds the slot for stream duration."""
        with self.client.post(
            "/v1/audio/speech",
            json={
                "input": random.choice(_SHORT_TEXTS),
                "voice": random.choice(_VOICES),
                "stream": True,
            },
            name="/v1/audio/speech [burst-stream]",
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 503:
                resp.success()
                return
            if resp.raw is not None:
                for _ in resp.iter_content(chunk_size=4096):
                    pass
