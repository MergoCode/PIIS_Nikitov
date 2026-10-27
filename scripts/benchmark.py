import argparse
import os
import sys
import time
import numpy as np
import requests


def run_benchmark(
    url: str,
    audio_path: str,
    n_requests: int = 100,
    warmup: int = 10,
):
    if not os.path.exists(audio_path):
        print(f"Error: audio file '{audio_path}' does not exist.")
        sys.exit(1)

    predict_url = f"{url.rstrip('/')}/predict"
    health_url = f"{url.rstrip('/')}/health"

    print(f"Connecting to service at {health_url}...")
    try:
        r = requests.get(health_url, timeout=5)
        if r.status_code != 200:
            print(f"Healthcheck failed with code {r.status_code}: {r.text}")
            sys.exit(1)
        print("Service is healthy!")
    except Exception as e:
        print(f"Failed to connect to service: {e}")
        sys.exit(1)

    with open(audio_path, "rb") as f:
        file_bytes = f.read()

    filename = os.path.basename(audio_path)

    print(f"\nRunning {warmup} warmup requests...")
    for _ in range(warmup):
        files = {"file": (filename, file_bytes, "audio/wav")}
        requests.post(predict_url, files=files, timeout=10)

    print(f"Running benchmark with {n_requests} requests...")
    client_latencies = []
    server_latencies = []
    status_codes = []

    for i in range(n_requests):
        files = {"file": (filename, file_bytes, "audio/wav")}
        t_start = time.perf_counter()
        resp = requests.post(predict_url, files=files, timeout=10)
        t_client = (time.perf_counter() - t_start) * 1000.0

        status_codes.append(resp.status_code)
        if resp.status_code == 200:
            client_latencies.append(t_client)
            data = resp.json()
            if "latency_ms" in data:
                server_latencies.append(data["latency_ms"])
        else:
            print(f"Request {i+1} failed with status {resp.status_code}")

    if not client_latencies:
        print("All benchmark requests failed!")
        sys.exit(1)

    c_mean = np.mean(client_latencies)
    c_p50 = np.percentile(client_latencies, 50)
    c_p95 = np.percentile(client_latencies, 95)

    s_mean = np.mean(server_latencies) if server_latencies else 0.0
    s_p50 = np.percentile(server_latencies, 50) if server_latencies else 0.0
    s_p95 = np.percentile(server_latencies, 95) if server_latencies else 0.0

    print("\n" + "=" * 55)
    print("           BENCHMARK RESULTS")
    print("=" * 55)
    print(f"Target URL:         {predict_url}")
    print(f"Sample File:        {audio_path}")
    print(f"Total Requests:     {n_requests} (warmup: {warmup})")
    print(f"Successful:         {len(client_latencies)}/{n_requests}")
    print("-" * 55)
    print(f"{'Metric':<25} | {'Server (ms)':<12} | {'Client e2e (ms)':<14}")
    print("-" * 55)
    print(f"{'Mean':<25} | {s_mean:<12.2f} | {c_mean:<14.2f}")
    print(f"{'p50 (Median)':<25} | {s_p50:<12.2f} | {c_p50:<14.2f}")
    print(f"{'p95':<25} | {s_p95:<12.2f} | {c_p95:<14.2f}")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark KWS REST API latency")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL of service")
    parser.add_argument("--file", type=str, default="samples/yes.wav", help="Path to sample .wav file")
    parser.add_argument("-n", "--requests", type=int, default=100, help="Number of benchmark requests")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup requests")

    args = parser.parse_args()
    run_benchmark(
        url=args.url,
        audio_path=args.file,
        n_requests=args.requests,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main()
