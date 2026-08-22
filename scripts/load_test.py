"""Lightweight concurrent load testing script for MedVision-AI REST API."""

import io
import time
import json
import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import numpy as np
from PIL import Image

# Ensure src/ is in path
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi.testclient import TestClient
from medvision.api.main import app
from medvision.api.services import ModelService
from medvision.utils.logger import get_logger

logger = get_logger("medvision.load_test")


def generate_test_image_bytes() -> bytes:
    """Generate in-memory valid radiograph PNG bytes for load testing."""
    img = np.random.randint(40, 220, size=(224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def run_single_request(client: TestClient, endpoint: str, image_bytes: bytes) -> Dict[str, Any]:
    """Execute a single HTTP request and record latency and status."""
    t_start = time.perf_counter()
    status_code = 0
    success = False
    error_msg = None

    try:
        if endpoint == "/health":
            resp = client.get("/health")
        elif endpoint == "/metadata":
            resp = client.get("/metadata")
        elif endpoint == "/predict":
            files = {"file": ("test.png", image_bytes, "image/png")}
            resp = client.post("/predict?threshold=0.60", files=files)
        elif endpoint == "/explain":
            files = {"file": ("test.png", image_bytes, "image/png")}
            resp = client.post("/explain?alpha=0.40", files=files)
        elif endpoint == "/predict-and-explain":
            files = {"file": ("test.png", image_bytes, "image/png")}
            resp = client.post("/predict-and-explain?threshold=0.60&alpha=0.40", files=files)
        else:
            raise ValueError(f"Unknown endpoint {endpoint}")

        status_code = resp.status_code
        success = (status_code == 200)
        if not success:
            error_msg = resp.text
    except Exception as e:
        error_msg = str(e)

    latency_ms = (time.perf_counter() - t_start) * 1000.0
    return {
        "success": success,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "error": error_msg,
    }


def run_benchmark(
    endpoint: str,
    total_requests: int = 50,
    concurrency: int = 5,
    output_dir: str = "artifacts/load_test",
) -> Dict[str, Any]:
    """Execute concurrent load test against the specified endpoint."""
    logger.info(f"Starting load test on '{endpoint}': {total_requests} requests, concurrency={concurrency}...")

    # Ensure model is initialized
    service = ModelService.get_instance()
    service.initialize()

    client = TestClient(app)
    image_bytes = generate_test_image_bytes()

    latencies: List[float] = []
    successes = 0
    failures = 0
    errors: List[str] = []

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(run_single_request, client, endpoint, image_bytes)
            for _ in range(total_requests)
        ]
        for f in as_completed(futures):
            res = f.result()
            latencies.append(res["latency_ms"])
            if res["success"]:
                successes += 1
            else:
                failures += 1
                if res["error"]:
                    errors.append(res["error"])

    total_wall_time_s = time.perf_counter() - wall_start
    rps = total_requests / total_wall_time_s if total_wall_time_s > 0 else 0

    lat_arr = np.array(latencies)
    results = {
        "endpoint": endpoint,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successful_requests": successes,
        "failed_requests": failures,
        "success_rate_percent": round((successes / total_requests) * 100.0, 2),
        "total_duration_sec": round(total_wall_time_s, 3),
        "throughput_req_per_sec": round(rps, 2),
        "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "latency_p90_ms": round(float(np.percentile(lat_arr, 90)), 2),
        "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "latency_mean_ms": round(float(np.mean(lat_arr)), 2),
        "latency_min_ms": round(float(np.min(lat_arr)), 2),
        "latency_max_ms": round(float(np.max(lat_arr)), 2),
    }

    print("\n" + "=" * 60)
    print(f"MEDVISION-AI LOAD TEST REPORT: {endpoint}")
    print("=" * 60)
    print(f"Total Requests      : {results['total_requests']}")
    print(f"Concurrency Level   : {results['concurrency']}")
    print(f"Success Rate        : {results['success_rate_percent']}% ({successes}/{total_requests})")
    print(f"Total Wall Time     : {results['total_duration_sec']}s")
    print(f"Throughput          : {results['throughput_req_per_sec']} req/s")
    print(f"Latency Mean        : {results['latency_mean_ms']} ms")
    print(f"Latency p50 (Median): {results['latency_p50_ms']} ms")
    print(f"Latency p95         : {results['latency_p95_ms']} ms")
    print(f"Latency p99         : {results['latency_p99_ms']} ms")
    print("=" * 60)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    sanitized_ep = endpoint.replace("/", "_").strip("_")
    report_file = out_p / f"load_test_{sanitized_ep}.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Load test report saved to: {report_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MedVision-AI API Load Testing Engine")
    parser.add_argument("--endpoint", type=str, default="/predict", choices=["/health", "/metadata", "/predict", "/explain", "/predict-and-explain"])
    parser.add_argument("--requests", type=int, default=30, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent worker threads")
    parser.add_argument("--output-dir", type=str, default="artifacts/load_test")
    args = parser.parse_args()

    run_benchmark(
        endpoint=args.endpoint,
        total_requests=args.requests,
        concurrency=args.concurrency,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
