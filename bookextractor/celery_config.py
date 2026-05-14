import os

# Broker and Backend
broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Task settings
task_always_eager = False  # Switch from sync to async
task_track_started = True
task_time_limit = 3600  # 1 hour max per task
task_soft_time_limit = 3000  # 50 min soft limit

# Worker configuration
worker_prefetch_multiplier = 1  # One task per worker at a time
worker_max_tasks_per_child = 100  # Restart worker after 100 tasks

# Result settings
result_expires = 86400  # 24 hours

# Queue and Routing Configuration
task_default_queue = "default_queue"
task_queues = {
    "default_queue": {
        "exchange": "default_queue",
        "routing_key": "default_queue",
    },
    "vlm_queue": {
        "exchange": "vlm_queue",
        "routing_key": "vlm_queue",
    },
}

# Mapping specific tasks to default queues.
# Image tasks are routed at runtime based on use_vlm flag (see main.py:extract_async).
task_routes = {
    "bookextractor.extract_pdf": {"queue": "vlm_queue"},
    "bookextractor.extract_text": {"queue": "vlm_queue"},
}
