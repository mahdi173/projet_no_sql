import multiprocessing
import os

# Gunicorn configuration
bind = f"0.0.0.0:{os.getenv('PORT', '5622')}"
workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
timeout = 120
keepalive = 5

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Security
# In production, it's better to run as a non-privileged user
# but we'll stick to Docker defaults for now unless specified.
