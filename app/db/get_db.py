import os
from .mongo_db import MongoDB
from .cassandra_db import CassandraDB

def get_database():
    db_type = os.getenv("DATABASE_TYPE", "mongo").lower()

    if db_type == "mongo":
        return MongoDB()
    elif db_type == "cassandra":
        return CassandraDB()
    else:
        raise ValueError("Unknown database type: " + db_type)