from flask import Blueprint, jsonify, request
import time
import numpy as np
from db.get_db import get_database

benchmark_bp = Blueprint("benchmark", __name__)

@benchmark_bp.route("/run_benchmark", methods=["POST"])
def run_benchmark():
    count = int(request.json.get("count", 100))
    
    db = get_database()
    
    results = {
        "count": count,
        "write_time": 0,
        "read_time": 0,
        "throughput_write": 0,
        "throughput_read": 0
    }
    
    # -----------------------------------------------------
    # 1. Ingest Data (Write)
    # -----------------------------------------------------
    # Generate fake data
    from uuid import uuid4
    fake_data = []
    for i in range(count):
        fake_data.append({
            "penguin_id": str(uuid4()), # Required by Mongo schema
            "species": np.random.choice(["Adelie", "Chinstrap", "Gentoo"]),
            "island": np.random.choice(["Torgersen", "Biscoe", "Dream"]),
            "sex": np.random.choice(["MALE", "FEMALE"]),
            "features": {
                "bill_length_mm": float(np.random.normal(40, 5)),
                "bill_depth_mm": float(np.random.normal(17, 3)),
                "flipper_length_mm": int(np.random.normal(200, 20)),
                "body_mass_g": int(np.random.normal(4000, 500)),
            }
        })
        
    try:
        start_time = time.time()
        for doc in fake_data:
            db.insert(doc)
        end_time = time.time()
        
        results["write_time"] = round(end_time - start_time, 4)
        results["throughput_write"] = round(count / results["write_time"], 2) if results["write_time"] > 0 else 0
        
        # -----------------------------------------------------
        # 2. Read Data (Read)
        # -----------------------------------------------------
        start_time = time.time()
        raw_data = db.find({})
        # Consume generator/cursor to ensure data is actually fetched
        _ = list(raw_data)
        end_time = time.time()
        
        results["read_time"] = round(end_time - start_time, 4)
        results["throughput_read"] = round(len(_) / results["read_time"], 2) if results["read_time"] > 0 else 0
        
        return jsonify(results)
    except Exception as e:
        print(f"Benchmark Error: {e}") # Log to container output
        return jsonify({"error": str(e)}), 500
    
@benchmark_bp.route("/memory_stats", methods=["GET"])
def memory_stats():
    db = get_database()
    stats = {"memory": "N/A"}
    
    from db.get_db import get_db_type
    db_type = get_db_type()
    
    try:
        if db_type == "redis":
            info = db.client.info("memory")
            stats["memory"] = info.get("used_memory_human", "N/A")
        elif db_type == "mongo":
            # For Mongo, we can use db.command("dbstats")
            # need to access client from the wrapper
            info = db.db.command("dbstats")
            # dataSize or storageSize
            size_mb = info.get("dataSize", 0) / (1024 * 1024)
            stats["memory"] = f"{size_mb:.2f} MB"
        # Cassandra difficult to get memory via driver
    except Exception as e:
        stats["error"] = str(e)
        
    return jsonify(stats)
