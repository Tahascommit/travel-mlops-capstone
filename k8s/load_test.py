"""Small load test for the flight price API, used to watch the HPA scale the deployment.

    python k8s/load_test.py --minutes 4 --workers 16
    kubectl -n travel-mlops get hpa -w        (in another terminal)

Only uses the standard library so it runs without the project venv.
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CITIES = ["Aracaju (SE)", "Brasilia (DF)", "Campo Grande (MS)", "Florianopolis (SC)", "Natal (RN)",
          "Recife (PE)", "Sao Paulo (SP)"]
OPTIONS = [("economic", "CloudFy"), ("premium", "Rainbow"), ("firstClass", "FlyingDrops")]


def make_batch(size):
    batch = []
    for _ in range(size):
        origin, dest = random.sample(CITIES, 2)
        flight_type, agency = random.choice(OPTIONS)
        batch.append({"from": origin, "to": dest, "flightType": flight_type, "agency": agency})
    return json.dumps(batch).encode()


def worker(url, end, batch_size):
    ok = failed = 0
    while time.time() < end:
        req = urllib.request.Request(url, data=make_batch(batch_size), headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=30).read()
            ok += 1
        except Exception:
            failed += 1
    return ok, failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--minutes", type=float, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch", type=int, default=100, help="flights per request")
    args = parser.parse_args()

    end = time.time() + args.minutes * 60
    start = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda _: worker(args.url, end, args.batch), range(args.workers)))

    ok = sum(r[0] for r in results)
    failed = sum(r[1] for r in results)
    elapsed = time.time() - start
    print(f"{ok} requests ok, {failed} failed in {elapsed:.0f}s "
          f"({ok / elapsed:.1f} req/s, {ok * args.batch / elapsed:.0f} predictions/s)")


if __name__ == "__main__":
    main()
