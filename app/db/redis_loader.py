import os
import csv
import json
import redis
from uuid import uuid4
from .utils import parse_penguin_row

def load_into_redis(csv_path="/app/penguins_size.csv", redis_url=None):
    # Default to 'redis' hostname for Docker, fallback to localhost
    default_host = os.getenv("REDIS_HOST", "redis")
    redis_url = redis_url or os.getenv("REDIS_URL", f"redis://{default_host}:6379/0")
    r = redis.from_url(redis_url)

    count = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc = parse_penguin_row(row)
            key = f"penguin:{doc['penguin_id']}"
            r.set(key, json.dumps(doc))
            count += 1

    print(f"Inserted {count} records into Redis ({redis_url})")


if __name__ == "__main__":
    load_into_redis()
