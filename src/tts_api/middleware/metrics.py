"""
HTTP-layer Prometheus metrics middleware.

Wraps every request regardless of what the application layer does, so it
captures status codes that never reach application logic:
  - 429  from RateLimitMiddleware
  - 401 / 403 from auth dependencies
  - 422  from pydantic request validation
  - 500  from unhandled exceptions

Metrics exported:
  http_requests_total{method, endpoint, status_code}   — counter
  http_request_duration_seconds{method, endpoint}       — histogram

For streaming responses the duration measures time-to-response-headers
(i.e. time until the client starts receiving data), not total transfer time.
Streaming-specific latency is tracked separately in the synthesis layer.
"""

import time

from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests by method, endpoint, and status code",
    ["method", "endpoint", "status_code"],
)

_HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds (time-to-headers for streaming responses)",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Use the raw path — our routes have no path parameters so this is
        # already the canonical template.  Avoids iterating app.routes per request.
        endpoint = request.url.path
        start = time.perf_counter()

        response = await call_next(request)

        _HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        _HTTP_LATENCY.labels(method=request.method, endpoint=endpoint).observe(
            time.perf_counter() - start
        )

        return response
