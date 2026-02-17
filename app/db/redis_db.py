import redis
import os
import json
from .base import Database

class RedisDB(Database):
    def connect(self):
        # Use 'redis' as hostname if inside Docker, else localhost
        # This handles both local dev (if port forwarded) and docker-compose
        redis_host = os.getenv("REDIS_HOST", "redis") 
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_url = f"redis://{redis_host}:{redis_port}/0"
        
        try:
            self.client = redis.from_url(self.redis_url)
            self.client.ping() # Check connection
            print(f"Connected to Redis at {self.redis_url}")
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis at {self.redis_url}: {e}")
            raise e

    def insert(self, data):
        # Generate a unique ID if not present
        if "penguin_id" not in data:
            import uuid
            data["penguin_id"] = str(uuid.uuid4())
            
        key = f"penguin:{data['penguin_id']}"
        self.client.set(key, json.dumps(data))
        return data["penguin_id"]

    def find(self, query):
        # Basic implementation: return all penguins if query is empty
        # Redis isn't great for complex queries without extra indexing
        # For this project, we'll iterate keys matching "penguin:*"
        
        keys = self.client.keys("penguin:*")
        results = []
        for key in keys:
            raw = self.client.get(key)
            if raw:
                try:
                    doc = json.loads(raw)
                    # Simple filter: if query is empty, match all. 
                    # If query has keys, check if doc matches values.
                    match = True
                    for k, v in query.items():
                        if doc.get(k) != v:
                            match = False
                            break
                    if match:
                        results.append(doc)
                except json.JSONDecodeError:
                    continue
        return results

    def close(self):
        self.client.close()
