"""
gRPC health check script for the tts-inference container.

Used by the Docker HEALTHCHECK instruction:
  CMD ["python", "-m", "tts_api.inference.healthcheck"]

Exits 0 if the server is ready, 1 otherwise.
"""

import os
import sys

import grpc

from tts_api.inference import tts_pb2, tts_pb2_grpc


def main() -> None:
    host = os.environ.get("TTS_INFERENCE_HOST", "localhost")
    port = int(os.environ.get("TTS_INFERENCE_PORT", "50051"))
    target = f"{host}:{port}"

    channel = grpc.insecure_channel(target)
    stub = tts_pb2_grpc.TTSInferenceStub(channel)
    try:
        response = stub.HealthCheck(tts_pb2.HealthRequest(), timeout=5)
        if response.ready:
            print(f"healthy: inference server at {target} is ready")
            sys.exit(0)
        else:
            print(f"not ready: inference server at {target} is still loading")
            sys.exit(1)
    except grpc.RpcError as exc:
        print(f"unreachable: {target} — {exc.details()}")
        sys.exit(1)
    finally:
        channel.close()


if __name__ == "__main__":
    main()
