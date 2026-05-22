from .config import settings

# Broker and Backend
broker_url = settings.CELERY_BROKER_URL
result_backend = settings.CELERY_RESULT_BACKEND

# Task settings
task_always_eager = settings.CELERY_TASK_ALWAYS_EAGER
task_track_started = True
task_time_limit = settings.CELERY_TASK_TIME_LIMIT
task_soft_time_limit = settings.CELERY_TASK_SOFT_TIME_LIMIT

# Worker configuration
worker_prefetch_multiplier = settings.CELERY_WORKER_PREFETCH_MULTIPLIER
worker_max_tasks_per_child = settings.CELERY_WORKER_MAX_TASKS_PER_CHILD

# Result settings
result_expires = settings.CELERY_RESULT_EXPIRES

# Queue and Routing Configuration
task_default_queue = settings.CELERY_DEFAULT_QUEUE
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

# Image tasks are routed at runtime based on use_vlm flag.
task_routes = {
    "metaextractor.extract_pdf": {"queue": "vlm_queue"},
    "metaextractor.extract_text": {"queue": "vlm_queue"},
}
