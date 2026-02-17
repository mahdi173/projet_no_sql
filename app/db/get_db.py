import os
from .mongo_db import MongoDB
from .cassandra_db import CassandraDB
from .redis_db import RedisDB

_current_db_instance = None
_current_db_type = None

def get_database():
    global _current_db_instance
    
    # Initialize if not already set
    if _current_db_instance is None:
        default_type = os.getenv("DATABASE_TYPE", "mongo").lower()
        set_db_type(default_type)
        
    return _current_db_instance

def get_db_type():
    global _current_db_type
    if _current_db_type is None:
        get_database() # triggers init
    return _current_db_type

def set_db_type(db_type):
    global _current_db_instance, _current_db_type
    
    db_type = db_type.lower()
    
    # Close existing connection if any
    if _current_db_instance:
        try:
            _current_db_instance.close()
        except Exception as e:
            print(f"Error closing DB: {e}")
            
    print(f"Switching database to: {db_type}")
    
    if db_type == "mongo":
        _current_db_instance = MongoDB()
    elif db_type == "cassandra":
        _current_db_instance = CassandraDB()
    elif db_type == "redis":
        _current_db_instance = RedisDB()
    else:
        # Fallback or error? Let's raise error to be clear
        raise ValueError("Unknown database type: " + db_type)

    _current_db_instance.connect()
    _current_db_type = db_type
    return _current_db_type